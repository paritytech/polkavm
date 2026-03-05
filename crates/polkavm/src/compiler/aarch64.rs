use core::sync::atomic::Ordering;

use polkavm_assembler::aarch64::*;
use polkavm_assembler::Label;

use polkavm_common::cast::cast;
use polkavm_common::program::{ProgramCounter, RawReg, Reg};
use polkavm_common::utils::GasVisitorT;
use polkavm_common::zygote::VmCtx;

use crate::compiler::{ArchVisitor, Bitness, BitnessT, SandboxKind};
use crate::config::GasMeteringKind;
use crate::sandbox::Sandbox;

/// The register used for the generic sandbox to hold the base address of the guest's linear memory.
const GENERIC_SANDBOX_MEMORY_REG: polkavm_assembler::aarch64::Reg = AUX_TMP_REG;

use polkavm_common::regmap::to_native_reg as conv_reg_const;
use polkavm_common::regmap::{AUX_TMP_REG, TMP_REG};

polkavm_common::static_assert!(polkavm_common::regmap::to_guest_reg(TMP_REG).is_none());
polkavm_common::static_assert!(polkavm_common::regmap::to_guest_reg(AUX_TMP_REG).is_none());

type NativeReg = polkavm_assembler::aarch64::Reg;

#[derive(Copy, Clone)]
pub enum RegImm {
    Reg(RawReg),
    Imm(u32),
}

impl From<RawReg> for RegImm {
    #[inline]
    fn from(reg: RawReg) -> Self {
        RegImm::Reg(reg)
    }
}

impl From<u32> for RegImm {
    #[inline]
    fn from(value: u32) -> Self {
        RegImm::Imm(value)
    }
}

static REG_MAP: [NativeReg; 16] = {
    let mut output = [conv_reg_const(Reg::T2); 16];
    let mut index = 0;
    while index < Reg::ALL.len() {
        assert!(Reg::ALL[index] as usize == index);
        output[index] = conv_reg_const(Reg::ALL[index]);
        index += 1;
    }
    output
};

#[inline]
fn conv_reg(reg: RawReg) -> NativeReg {
    let native_reg = REG_MAP[reg.raw_unparsed() as usize & 0b1111];
    debug_assert_eq!(native_reg, conv_reg_const(reg.get()));
    native_reg
}

#[test]
fn test_conv_reg() {
    for reg in Reg::ALL {
        assert_eq!(conv_reg(reg.into()), conv_reg_const(reg));
    }
}

#[derive(Copy, Clone)]
enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Copy, Clone)]
enum DivRem {
    Div,
    Rem,
}

enum ShiftKind {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
}

/// Convert RegSize to MemSize
fn reg_to_mem_size(rs: RegSize) -> MemSize {
    match rs {
        RegSize::W32 => MemSize::B32,
        RegSize::X64 => MemSize::B64,
    }
}

/// Emit `dst = src + offset` (32-bit), handling offsets >= 4096.
fn emit_add_offset_32(asm: &mut polkavm_assembler::Assembler, dst: NativeReg, src: NativeReg, offset: u32) {
    if offset == 0 {
        if dst != src {
            asm.push(mov_reg(RegSize::W32, dst, src));
        }
    } else if offset < 4096 {
        asm.push(add_imm(RegSize::W32, dst, src, offset));
    } else {
        emit_load_imm32(asm, dst, offset);
        asm.push(add(RegSize::W32, dst, dst, src));
    }
}

/// Load a 32-bit immediate into a register using MOVZ + MOVK.
fn emit_load_imm32(asm: &mut polkavm_assembler::Assembler, rd: NativeReg, value: u32) {
    let lo = (value & 0xFFFF) as u16;
    let hi = ((value >> 16) & 0xFFFF) as u16;
    if hi == 0 {
        asm.push(movz(RegSize::W32, rd, lo, 0));
    } else if lo == 0 {
        asm.push(movz(RegSize::W32, rd, hi, 16));
    } else {
        asm.push(movz(RegSize::W32, rd, lo, 0));
        asm.push(movk(RegSize::W32, rd, hi, 16));
    }
}

/// Load a 64-bit immediate into a register using MOVZ + MOVK.
fn emit_load_imm64(asm: &mut polkavm_assembler::Assembler, rd: NativeReg, value: u64) {
    if value == 0 {
        asm.push(movz(RegSize::X64, rd, 0, 0));
        return;
    }

    let chunks = [
        (value & 0xFFFF) as u16,
        ((value >> 16) & 0xFFFF) as u16,
        ((value >> 32) & 0xFFFF) as u16,
        ((value >> 48) & 0xFFFF) as u16,
    ];

    // Try MOVZ approach (for values with few non-zero chunks)
    let nonzero_count = chunks.iter().filter(|&&c| c != 0).count();
    let not_value = !value;
    let not_chunks = [
        (not_value & 0xFFFF) as u16,
        ((not_value >> 16) & 0xFFFF) as u16,
        ((not_value >> 32) & 0xFFFF) as u16,
        ((not_value >> 48) & 0xFFFF) as u16,
    ];
    let not_nonzero_count = not_chunks.iter().filter(|&&c| c != 0).count();

    if nonzero_count <= not_nonzero_count + 1 {
        // Use MOVZ + MOVK
        let mut first = true;
        for (i, &chunk) in chunks.iter().enumerate() {
            if chunk != 0 || (first && i == 3) {
                let shift = (i as u32) * 16;
                if first {
                    asm.push(movz(RegSize::X64, rd, chunk, shift));
                    first = false;
                } else {
                    asm.push(movk(RegSize::X64, rd, chunk, shift));
                }
            }
        }
        if first {
            asm.push(movz(RegSize::X64, rd, 0, 0));
        }
    } else {
        // Use MOVN + MOVK (better for values with many 1-bits).
        // MOVN Xd, #imm16, LSL #shift  =>  Xd = ~(imm16 << shift)
        // Find the first non-0xFFFF chunk and use MOVN to set it (MOVN inverts).
        // Then use MOVK for any other non-0xFFFF chunks.
        let first_diff = not_chunks.iter().position(|&c| c != 0);
        if let Some(fi) = first_diff {
            // MOVN with the inverted chunk value; MOVN will re-invert it,
            // giving us 0xFFFF for all other chunks and the correct value for this one.
            asm.push(movn(RegSize::X64, rd, not_chunks[fi], (fi as u32) * 16));
            for (i, &nc) in not_chunks.iter().enumerate() {
                if i != fi && nc != 0 {
                    // This chunk differs from 0xFFFF; patch it with the actual value.
                    asm.push(movk(RegSize::X64, rd, chunks[i], (i as u32) * 16));
                }
            }
        } else {
            // All not_chunks are zero, meaning value is all 1s (0xFFFFFFFFFFFFFFFF)
            asm.push(movn(RegSize::X64, rd, 0, 0));
        }
    }
}

/// Offset of gas metering cost within the gas stub, in bytes from the start of the stub.
/// The gas stub on AArch64 (generic sandbox) looks like:
///   sub x16, x17, #1, LSL #12         ; 4 bytes (load_vmctx_base)
///   ldr x9, [x16, #gas_offset]        ; 4 bytes
///   sub x9, x9, #COST                 ; 4 bytes (cost in bits [21:10])
///   str x9, [x16, #gas_offset]        ; 4 bytes
///   [sync only: cmp x9, #0 / b.ge ok / brk / ok:]
/// Total (async): 16 bytes. Total (sync): 28 bytes. The cost is at offset 8.
const GAS_METERING_STUB_SIZE: usize = 16;
const GAS_COST_OFFSET: usize = 8;

// The trap offset for gas metering: from the brk instruction back to the start of the stub.
const GAS_METERING_TRAP_OFFSET: u64 = 24;

fn are_we_executing_memset<S>(compiled_module: &crate::compiler::CompiledModule<S>, machine_code_offset: u64) -> bool
where
    S: Sandbox,
{
    machine_code_offset >= compiled_module.memset_trampoline_start && machine_code_offset < compiled_module.memset_trampoline_end
}

fn set_program_counter_after_interruption<S>(
    compiled_module: &crate::compiler::CompiledModule<S>,
    machine_code_offset: u64,
    vmctx: &VmCtx,
) -> Result<ProgramCounter, &'static str>
where
    S: Sandbox,
{
    let Some(program_counter) = compiled_module.program_counter_by_native_code_offset(machine_code_offset, false) else {
        return Err("internal error: failed to find the program counter based on the native program counter when handling a page fault");
    };
    vmctx.program_counter.store(program_counter.0, Ordering::Relaxed);
    vmctx.next_program_counter.store(program_counter.0, Ordering::Relaxed);
    Ok(program_counter)
}

impl<'r, 'a, S, B, G> ArchVisitor<'r, 'a, S, B, G>
where
    S: Sandbox,
    B: BitnessT,
    G: GasVisitorT,
{
    // NOP = 0xD503201F, but padding byte is a single byte, so we use 0x00
    // (which forms part of a UDF instruction for trapping on accidental execution).
    pub const PADDING_BYTE: u8 = 0x00;

    #[inline(always)]
    fn push<T>(&mut self, inst: polkavm_assembler::Instruction<T>)
    where
        T: core::fmt::Display,
    {
        self.0.asm.push(inst);
    }

    #[allow(clippy::unused_self)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn reg_size(&self) -> RegSize {
        match B::BITNESS {
            Bitness::B32 => RegSize::W32,
            Bitness::B64 => RegSize::X64,
        }
    }

    /// Load the address of a VmCtx field into TMP_REG and return the offset
    /// that can be used with LDR/STR [TMP_REG, #offset].
    fn load_vmctx_base(&mut self) -> NativeReg {
        match S::KIND {
            SandboxKind::Linux => {
                // On Linux sandbox, AUX_TMP_REG holds vmctx address directly.
                AUX_TMP_REG
            }
            SandboxKind::Generic => {
                #[cfg(feature = "generic-sandbox")]
                {
                    // VmCtx is at GUEST_MEMORY_TO_VMCTX_OFFSET (-4096) from the memory base.
                    // 4096 doesn't fit in a 12-bit immediate (max 4095), but with the
                    // shift bit set, sub_imm_lsl12 subtracts (imm12 << 12) = 1 << 12 = 4096.
                    let offset = crate::sandbox::generic::GUEST_MEMORY_TO_VMCTX_OFFSET;
                    let abs_offset = (-offset) as u32;
                    if abs_offset < 4096 {
                        self.push(sub_imm(RegSize::X64, TMP_REG, GENERIC_SANDBOX_MEMORY_REG, abs_offset));
                    } else if abs_offset % 4096 == 0 && abs_offset / 4096 < 4096 {
                        self.push(sub_imm_lsl12(RegSize::X64, TMP_REG, GENERIC_SANDBOX_MEMORY_REG, abs_offset / 4096));
                    } else {
                        emit_load_imm32(&mut self.0.asm, TMP_REG, abs_offset);
                        self.push(sub(RegSize::X64, TMP_REG, GENERIC_SANDBOX_MEMORY_REG, TMP_REG));
                    }
                    TMP_REG
                }
                #[cfg(not(feature = "generic-sandbox"))]
                {
                    unreachable!();
                }
            }
        }
    }

    /// Emit a load from a VmCtx field (64-bit).
    fn load_vmctx_field_u64(&mut self, dst: NativeReg, field_offset: usize) {
        let vmctx_base = self.load_vmctx_base();
        self.push(ldr_imm(MemSize::B64, dst, vmctx_base, field_offset as u32));
    }

    /// Emit a store to a VmCtx field (64-bit).
    fn store_vmctx_field_u64(&mut self, src: NativeReg, field_offset: usize) {
        let vmctx_base = self.load_vmctx_base();
        self.push(str_imm(MemSize::B64, src, vmctx_base, field_offset as u32));
    }

    /// Emit a store of a 32-bit immediate to a VmCtx field.
    fn store_vmctx_field_imm32(&mut self, field_offset: usize, value: u32) {
        // Load vmctx base first (uses TMP_REG for generic sandbox),
        // then load the value into x9 to avoid clobbering.
        let vmctx_base = self.load_vmctx_base();
        emit_load_imm32(&mut self.0.asm, x9, value);
        self.push(str_imm(MemSize::B32, x9, vmctx_base, field_offset as u32));
    }

    /// Emit a store of zero to a VmCtx field (64-bit).
    fn store_vmctx_field_zero_u64(&mut self, field_offset: usize) {
        let vmctx_base = self.load_vmctx_base();
        self.push(str_imm(MemSize::B64, ZR, vmctx_base, field_offset as u32));
    }

    /// Load a 32-bit immediate into a native register.
    fn emit_imm32(&mut self, rd: NativeReg, value: u32) {
        emit_load_imm32(&mut self.0.asm, rd, value);
    }

    /// Load a 64-bit immediate into a native register.
    fn emit_imm64(&mut self, rd: NativeReg, value: u64) {
        emit_load_imm64(&mut self.0.asm, rd, value);
    }

    /// Load a sign-extended 32-bit immediate into a register (for B64 mode).
    fn emit_imm_bitness(&mut self, rd: NativeReg, value: u32) {
        match B::BITNESS {
            Bitness::B32 => emit_load_imm32(&mut self.0.asm, rd, value),
            Bitness::B64 => {
                let value64 = value as i32 as i64 as u64;
                emit_load_imm64(&mut self.0.asm, rd, value64);
            }
        }
    }

    fn save_registers_to_vmctx(&mut self) {
        let msize = match B::BITNESS {
            Bitness::B32 => MemSize::B32,
            Bitness::B64 => MemSize::B64,
        };
        let vmctx_base = self.load_vmctx_base();
        for (nth, reg) in Reg::ALL.iter().copied().enumerate() {
            let offset = (S::offset_table().regs + nth * 8) as u32;
            self.push(str_imm(msize, conv_reg(reg.into()), vmctx_base, offset));
        }
    }

    fn restore_registers_from_vmctx(&mut self) {
        let msize = match B::BITNESS {
            Bitness::B32 => MemSize::B32,
            Bitness::B64 => MemSize::B64,
        };
        let vmctx_base = self.load_vmctx_base();
        for (nth, reg) in Reg::ALL.iter().copied().enumerate() {
            let offset = (S::offset_table().regs + nth * 8) as u32;
            self.push(ldr_imm(msize, conv_reg(reg.into()), vmctx_base, offset));
        }
    }

    fn save_return_address_to_vmctx(&mut self) {
        // On AArch64, the return address is in LR (x30) after a BL.
        self.store_vmctx_field_u64(lr, S::offset_table().next_native_program_counter);
    }

    pub(crate) fn emit_sysenter(&mut self) -> Label {
        log::trace!("Emitting trampoline: sysenter");
        let label = self.asm.create_label();
        self.restore_registers_from_vmctx();
        // Jump to the address stored in vmctx.next_native_program_counter
        self.load_vmctx_field_u64(TMP_REG, S::offset_table().next_native_program_counter);
        self.push(br(TMP_REG));
        label
    }

    pub(crate) fn emit_sysreturn(&mut self) -> Label {
        log::trace!("Emitting trampoline: sysreturn");
        let label = self.asm.create_label();
        // Clear next_native_program_counter
        self.store_vmctx_field_zero_u64(S::offset_table().next_native_program_counter);
        self.save_registers_to_vmctx();
        // Jump to syscall_return
        self.emit_imm64(TMP_REG, S::address_table().syscall_return);
        self.push(br(TMP_REG));
        label
    }

    pub(crate) fn emit_ecall_trampoline(&mut self) {
        log::trace!("Emitting trampoline: ecall");
        let label = self.ecall_label;
        self.define_label(label);
        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.emit_imm64(TMP_REG, S::address_table().syscall_hostcall);
        self.push(br(TMP_REG));
    }

    pub(crate) fn emit_step_trampoline(&mut self) {
        log::trace!("Emitting trampoline: step");
        let label = self.step_label;
        self.define_label(label);
        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.emit_imm64(TMP_REG, S::address_table().syscall_step);
        self.push(br(TMP_REG));
    }

    pub(crate) fn emit_trap_trampoline(&mut self) {
        log::trace!("Emitting trampoline: trap");
        let label = self.trap_label;
        self.define_label(label);
        self.save_registers_to_vmctx();
        self.store_vmctx_field_zero_u64(S::offset_table().next_native_program_counter);
        self.emit_imm64(TMP_REG, S::address_table().syscall_trap);
        self.push(br(TMP_REG));
    }

    pub(crate) fn emit_sbrk_trampoline(&mut self) {
        log::trace!("Emitting trampoline: sbrk");
        let label = self.sbrk_label;
        self.define_label(label);

        // Save LR and GENERIC_SANDBOX_MEMORY_REG (x17) on stack.
        // blr will clobber LR; the C callee may clobber x16/x17 (IP0/IP1).
        self.push(stp_pre(MemSize::B64, lr, GENERIC_SANDBOX_MEMORY_REG, sp, -16));

        // Save sbrk size argument (in TMP_REG) to x9 before save clobbers TMP_REG
        self.push(mov_reg(RegSize::X64, x9, TMP_REG));

        // Save all guest registers (clobbers TMP_REG via load_vmctx_base)
        self.save_registers_to_vmctx();

        // Move sbrk size argument to x0 (first argument for the syscall)
        self.push(mov_reg(RegSize::X64, x0, x9));

        // Call the sbrk syscall
        self.emit_imm64(TMP_REG, S::address_table().syscall_sbrk);
        self.push(blr(TMP_REG));

        // Save result (in x0) to x9 before restore clobbers TMP_REG
        self.push(mov_reg(RegSize::X64, x9, x0));

        // Restore LR and GENERIC_SANDBOX_MEMORY_REG from stack
        self.push(ldp_post(MemSize::B64, lr, GENERIC_SANDBOX_MEMORY_REG, sp, 16));

        // Restore guest registers (clobbers TMP_REG via load_vmctx_base)
        self.restore_registers_from_vmctx();

        // Move result to TMP_REG
        self.push(mov_reg(RegSize::X64, TMP_REG, x9));

        // Return
        self.push(ret_lr());
    }

    pub(crate) fn emit_memset_trampoline(&mut self) {
        log::trace!("Emitting trampoline: memset");
        let label = self.memset_label;
        self.define_label(label);

        // On AArch64, the memset trampoline handles gas-limited memset (slow path).
        // Called when gas went negative after pre-charging count bytes.
        // Registers: A0 (x0) = dst, A1 (x1) = value, A2 (x2) = count
        //
        // Strategy: calculate how many bytes we can write with remaining gas.
        // Use TMP_REG (x16) as loop counter so the signal handler saves it
        // to vmctx.tmp_reg on OOB fault (parallels x86's rcx/rep stosb).

        let dst = conv_reg(Reg::A0.into());
        let val = conv_reg(Reg::A1.into());
        let count = conv_reg(Reg::A2.into());

        // Set vmctx.arg = 2 to mark "trampoline memset in progress" BEFORE using x9/TMP_REG.
        // (Inline memset uses arg=1; the handler uses 2 to know it must add tmp_reg to A2.)
        // Must be done first since store_vmctx_field_imm32 clobbers both TMP_REG and x9.
        self.store_vmctx_field_imm32(S::offset_table().arg, 2);

        let vmctx_base = self.load_vmctx_base(); // TMP_REG = vmctx

        // Load gas counter into x9, zero the gas field
        self.push(ldr_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
        self.push(str_imm(MemSize::B64, ZR, vmctx_base, S::offset_table().gas as u32));

        // Calculate bytes we can memset: bytes_available = gas + count (gas is negative)
        self.push(add(RegSize::X64, x9, x9, count));

        // A2 = remaining after gas-limited write = count - bytes_available
        self.push(sub(RegSize::X64, count, count, x9));

        // Move bytes_available to TMP_REG (x16) for the loop.
        // On page fault, the signal handler saves x16 to vmctx.tmp_reg.
        self.push(mov_reg(RegSize::X64, TMP_REG, x9));

        // Memory base register for sandbox-relative addressing
        let mem_base = match S::KIND {
            SandboxKind::Linux => AUX_TMP_REG,
            SandboxKind::Generic => {
                #[cfg(feature = "generic-sandbox")]
                { GENERIC_SANDBOX_MEMORY_REG }
                #[cfg(not(feature = "generic-sandbox"))]
                { unreachable!() }
            }
        };

        // Execute byte-store loop: TMP_REG = bytes to write
        let label_loop = self.asm.forward_declare_label();
        let label_done = self.asm.forward_declare_label();

        self.push(cbz_label(RegSize::X64, TMP_REG, label_done));
        self.asm.define_label(label_loop);
        self.push(str_reg_uxtw(MemSize::B8, val, mem_base, dst));
        self.push(add_imm(RegSize::W32, dst, dst, 1));
        self.push(sub_imm(RegSize::X64, TMP_REG, TMP_REG, 1));
        self.push(cbnz_label(RegSize::X64, TMP_REG, label_loop));

        self.asm.define_label(label_done);
        // Clear memset flag
        self.store_vmctx_field_imm32(S::offset_table().arg, 0);
        // Out of gas
        self.save_registers_to_vmctx();
        self.emit_imm64(TMP_REG, S::address_table().syscall_not_enough_gas);
        self.push(br(TMP_REG));
    }

    fn emit_divrem_trampoline_common(&mut self, reg_size: RegSize, div_rem: DivRem, kind: Signedness) {
        let label = match (reg_size, div_rem, kind) {
            (RegSize::W32, DivRem::Div, Signedness::Unsigned) => { log::trace!("Emitting trampoline: divu32"); self.div32u_label }
            (RegSize::W32, DivRem::Div, Signedness::Signed) => { log::trace!("Emitting trampoline: divs32"); self.div32s_label }
            (RegSize::W32, DivRem::Rem, Signedness::Unsigned) => { log::trace!("Emitting trampoline: remu32"); self.rem32u_label }
            (RegSize::W32, DivRem::Rem, Signedness::Signed) => { log::trace!("Emitting trampoline: rems32"); self.rem32s_label }
            (RegSize::X64, DivRem::Div, Signedness::Unsigned) => { log::trace!("Emitting trampoline: divu64"); self.div64u_label }
            (RegSize::X64, DivRem::Div, Signedness::Signed) => { log::trace!("Emitting trampoline: divs64"); self.div64s_label }
            (RegSize::X64, DivRem::Rem, Signedness::Unsigned) => { log::trace!("Emitting trampoline: remu64"); self.rem64u_label }
            (RegSize::X64, DivRem::Rem, Signedness::Signed) => { log::trace!("Emitting trampoline: rems64"); self.rem64s_label }
        };
        self.define_label(label);

        // Convention: dividend in x9 (caller saves it), divisor in TMP_REG (x16).
        // Result returned in TMP_REG.
        // We use x9, x10, x11 as scratch (not guest regs since caller saves).
        let dividend = x9;
        let divisor = TMP_REG;
        let result_reg = TMP_REG;

        // Save callee-saved registers we'll use as scratch
        self.push(stp_pre(MemSize::B64, x9, x10, sp, -16));
        self.push(stp_pre(MemSize::B64, x11, x12, sp, -16));

        // dividend is already loaded by caller into x9
        // divisor is in TMP_REG (x16)
        self.push(mov_reg(RegSize::X64, x10, divisor)); // x10 = divisor (save before we clobber TMP)

        // Check for division by zero
        let label_not_zero = self.asm.forward_declare_label();
        self.push(cbnz_label(reg_size, x10, label_not_zero));

        // Division by zero: return -1 for div, dividend for rem
        match div_rem {
            DivRem::Div => {
                // Return all-ones (-1)
                self.push(movn(reg_size, result_reg, 0, 0));
            }
            DivRem::Rem => {
                // Return dividend
                self.push(mov_reg(reg_size, result_reg, dividend));
            }
        }
        let label_done = self.asm.forward_declare_label();
        self.push(b_label(label_done));

        self.asm.define_label(label_not_zero);

        if matches!(kind, Signedness::Signed) {
            // Check for signed overflow: dividend == MIN && divisor == -1
            let label_no_overflow = self.asm.forward_declare_label();

            // Check divisor == -1
            self.push(cmn(reg_size, x10, x10)); // Won't work, use add
            // Actually: CMN Xn, #1 is ADDS XZR, Xn, #1
            self.push(add_imm(reg_size, x11, x10, 1)); // x11 = divisor + 1
            self.push(cbnz_label(reg_size, x11, label_no_overflow));

            // Divisor is -1. Check if dividend is MIN.
            let min_val = match reg_size {
                RegSize::W32 => i32::MIN as u32 as u64,
                RegSize::X64 => i64::MIN as u64,
            };

            match reg_size {
                RegSize::W32 => {
                    emit_load_imm32(&mut self.0.asm, x11, min_val as u32);
                    self.push(cmp(reg_size, dividend, x11));
                }
                RegSize::X64 => {
                    emit_load_imm64(&mut self.0.asm, x11, min_val);
                    self.push(cmp(reg_size, dividend, x11));
                }
            }
            self.push(b_cond_label(Condition::NE, label_no_overflow));

            // Signed overflow: return MIN for div, 0 for rem
            match div_rem {
                DivRem::Div => {
                    self.push(mov_reg(reg_size, result_reg, x11)); // x11 still has MIN
                }
                DivRem::Rem => {
                    self.push(movz(reg_size, result_reg, 0, 0));
                }
            }
            self.push(b_label(label_done));
            self.asm.define_label(label_no_overflow);
        }

        // Normal division
        match kind {
            Signedness::Unsigned => {
                self.push(udiv(reg_size, x11, dividend, x10));
            }
            Signedness::Signed => {
                self.push(sdiv(reg_size, x11, dividend, x10));
            }
        }

        match div_rem {
            DivRem::Div => {
                self.push(mov_reg(reg_size, result_reg, x11));
            }
            DivRem::Rem => {
                // remainder = dividend - quotient * divisor
                self.push(msub(reg_size, result_reg, x11, x10, dividend));
            }
        }

        self.asm.define_label(label_done);

        // Sign-extend 32-bit result to 64-bit if needed
        if reg_size == RegSize::W32 && B::BITNESS == Bitness::B64 {
            self.push(sxtw(result_reg, result_reg));
        }

        // Restore and return
        self.push(ldp_post(MemSize::B64, x11, x12, sp, 16));
        self.push(ldp_post(MemSize::B64, x9, x10, sp, 16));
        self.push(ret_lr());
    }

    pub(crate) fn emit_divrem_trampoline(&mut self) {
        log::trace!("Emitting trampoline: divrem");

        self.emit_divrem_trampoline_common(RegSize::W32, DivRem::Div, Signedness::Unsigned);
        self.emit_divrem_trampoline_common(RegSize::W32, DivRem::Rem, Signedness::Unsigned);
        self.emit_divrem_trampoline_common(RegSize::W32, DivRem::Div, Signedness::Signed);
        self.emit_divrem_trampoline_common(RegSize::W32, DivRem::Rem, Signedness::Signed);

        if B::BITNESS == Bitness::B64 {
            self.emit_divrem_trampoline_common(RegSize::X64, DivRem::Div, Signedness::Unsigned);
            self.emit_divrem_trampoline_common(RegSize::X64, DivRem::Rem, Signedness::Unsigned);
            self.emit_divrem_trampoline_common(RegSize::X64, DivRem::Div, Signedness::Signed);
            self.emit_divrem_trampoline_common(RegSize::X64, DivRem::Rem, Signedness::Signed);
        }
    }

    pub(crate) fn trace_execution(&mut self, code_offset: Option<u32>) {
        let step_label = self.step_label;
        let origin = self.asm.len();

        if let Some(code_offset) = code_offset {
            self.store_vmctx_field_imm32(S::offset_table().program_counter, code_offset);
            self.store_vmctx_field_imm32(S::offset_table().next_program_counter, code_offset);
        } else {
            // Emit padding NOPs to maintain consistent size
            self.push(nop());
            self.push(nop());
            self.push(nop());
            self.push(nop());
        }

        self.push(bl_label(step_label));
        let _ = origin; // step_prelude_length validated externally
    }

    pub(crate) fn emit_gas_metering_stub(&mut self, kind: GasMeteringKind) {
        let _origin = self.asm.len();
        let vmctx_base = self.load_vmctx_base();

        // Load gas counter into x9 (not TMP_REG, to avoid clobbering vmctx_base)
        self.push(ldr_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
        // Subtract cost (placeholder — will be patched by emit_weight)
        // Use sub_imm with max value (0xFFF = 4095) as placeholder
        self.push(sub_imm(RegSize::X64, x9, x9, 0xFFF));
        // Store back (vmctx_base = TMP_REG is still valid since we used x9 for gas)
        self.push(str_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));

        if matches!(kind, GasMeteringKind::Sync) {
            // If gas went negative, trap
            self.push(cmp_imm(RegSize::X64, x9, 0));
            // UDF for trap (SIGILL handler will catch this)
            let label_ok = self.asm.forward_declare_label();
            self.push(b_cond_label(Condition::GE, label_ok));
            self.push(udf(0));
            self.asm.define_label(label_ok);
        }
    }

    pub(crate) fn emit_weight(&mut self, offset: usize, cost: u32) {
        // Patch the sub_imm instruction at GAS_COST_OFFSET from the stub start.
        // The sub_imm instruction has the immediate in bits [21:10].
        let sub_offset = offset + GAS_COST_OFFSET;
        let code = self.asm.code_mut();
        let mut inst_bytes = [code[sub_offset], code[sub_offset + 1], code[sub_offset + 2], code[sub_offset + 3]];
        let mut inst_word = u32::from_le_bytes(inst_bytes);

        // Clear old immediate and set new one
        // sub_imm format: sf_1_0_100010_sh_imm12_Rn_Rd
        // imm12 is bits [21:10]
        inst_word &= !(0xFFF << 10);
        inst_word |= (cost & 0xFFF) << 10;

        inst_bytes = inst_word.to_le_bytes();
        code[sub_offset..sub_offset + 4].copy_from_slice(&inst_bytes);
    }

    // ── Load/Store helpers ────────────────────────────────────────────────

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn store_to_memory(&mut self, src: impl Into<RegImm>, base: Option<RawReg>, offset: u32, msize: MemSize) {
        let src = src.into();
        // For 64-bit stores of immediates, sign-extend the u32 to i64.
        let load_imm_for_store = |asm: &mut polkavm_assembler::Assembler, reg: NativeReg, v: u32| {
            if msize == MemSize::B64 {
                let value64 = v as i32 as i64 as u64;
                emit_load_imm64(asm, reg, value64);
            } else {
                emit_load_imm32(asm, reg, v);
            }
        };
        match S::KIND {
            SandboxKind::Linux => {
                // Linux sandbox: direct memory access
                let src_reg = match src {
                    RegImm::Reg(r) => conv_reg(r),
                    RegImm::Imm(v) => {
                        load_imm_for_store(&mut self.0.asm, TMP_REG, v);
                        TMP_REG
                    }
                };
                if let Some(base) = base {
                    if offset != 0 {
                        emit_add_offset_32(&mut self.0.asm, TMP_REG, conv_reg(base), offset);
                        self.push(str_reg(msize, src_reg, AUX_TMP_REG, TMP_REG));
                    } else {
                        self.push(str_reg(msize, src_reg, AUX_TMP_REG, conv_reg(base)));
                    }
                } else {
                    emit_load_imm32(&mut self.0.asm, TMP_REG, offset);
                    self.push(str_reg(msize, src_reg, AUX_TMP_REG, TMP_REG));
                }
            }
            SandboxKind::Generic => {
                // Compute effective guest address in x9
                let addr_reg = x9;
                if let Some(base) = base {
                    emit_add_offset_32(&mut self.0.asm, addr_reg, conv_reg(base), offset);
                } else {
                    emit_load_imm32(&mut self.0.asm, addr_reg, offset);
                }

                // Load source value
                let src_reg = match src {
                    RegImm::Reg(r) => conv_reg(r),
                    RegImm::Imm(v) => {
                        load_imm_for_store(&mut self.0.asm, TMP_REG, v);
                        TMP_REG
                    }
                };
                self.push(str_reg_uxtw(msize, src_reg, GENERIC_SANDBOX_MEMORY_REG, addr_reg));
            }
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn load_from_memory(&mut self, dst: RawReg, base: Option<RawReg>, offset: u32, msize: MemSize, sign_extend: bool) {
        let dst_native = conv_reg(dst);
        let target_size = self.reg_size();

        let addr_reg = if let Some(base) = base {
            emit_add_offset_32(&mut self.0.asm, TMP_REG, conv_reg(base), offset);
            TMP_REG
        } else {
            emit_load_imm32(&mut self.0.asm, TMP_REG, offset);
            TMP_REG
        };

        match S::KIND {
            SandboxKind::Linux => {
                if sign_extend {
                    match msize {
                        MemSize::B8 => self.push(ldrsb_reg(target_size, dst_native, AUX_TMP_REG, addr_reg)),
                        MemSize::B16 => self.push(ldrsh_reg(target_size, dst_native, AUX_TMP_REG, addr_reg)),
                        MemSize::B32 => self.push(ldrsw_reg(dst_native, AUX_TMP_REG, addr_reg)),
                        MemSize::B64 => self.push(ldr_reg(msize, dst_native, AUX_TMP_REG, addr_reg)),
                    }
                } else {
                    self.push(ldr_reg(msize, dst_native, AUX_TMP_REG, addr_reg));
                }
            }
            SandboxKind::Generic => {
                if sign_extend {
                    match msize {
                        MemSize::B8 => self.push(ldrsb_reg_uxtw(target_size, dst_native, GENERIC_SANDBOX_MEMORY_REG, addr_reg)),
                        MemSize::B16 => self.push(ldrsh_reg_uxtw(target_size, dst_native, GENERIC_SANDBOX_MEMORY_REG, addr_reg)),
                        MemSize::B32 => self.push(ldrsw_reg_uxtw(dst_native, GENERIC_SANDBOX_MEMORY_REG, addr_reg)),
                        MemSize::B64 => self.push(ldr_reg_uxtw(msize, dst_native, GENERIC_SANDBOX_MEMORY_REG, addr_reg)),
                    }
                } else {
                    self.push(ldr_reg_uxtw(msize, dst_native, GENERIC_SANDBOX_MEMORY_REG, addr_reg));
                }
            }
        }
    }

    // ── Shift helpers ─────────────────────────────────────────────────────

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn shift_imm_op(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, mut s2: u32, kind: ShiftKind) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        s2 &= match reg_size { RegSize::W32 => 31, RegSize::X64 => 63 };

        match kind {
            ShiftKind::LogicalLeft => self.push(lsl_imm(reg_size, d, s1, s2)),
            ShiftKind::LogicalRight => self.push(lsr_imm(reg_size, d, s1, s2)),
            ShiftKind::ArithmeticRight => self.push(asr_imm(reg_size, d, s1, s2)),
        }

        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn shift_reg_op(&mut self, reg_size: RegSize, d: RawReg, s1: impl Into<RegImm>, s2: RawReg, kind: ShiftKind) {
        let d = conv_reg(d);
        let mut s2_native = conv_reg(s2);

        // If d == s2, loading s1 into d would clobber the shift amount.
        // Save s2 to TMP_REG first.
        if d == s2_native {
            let s1_val = s1.into();
            let needs_save = match s1_val {
                RegImm::Reg(r) => conv_reg(r) != d,
                RegImm::Imm(_) => true,
            };
            if needs_save {
                self.push(mov_reg(reg_size, TMP_REG, s2_native));
                s2_native = TMP_REG;
            }
            // Now load s1 into d
            match s1_val {
                RegImm::Reg(r) => {
                    let s1 = conv_reg(r);
                    if d != s1 { self.push(mov_reg(reg_size, d, s1)); }
                }
                RegImm::Imm(v) => {
                    self.emit_imm_bitness(d, v);
                }
            }
        } else {
            match s1.into() {
                RegImm::Reg(r) => {
                    let s1 = conv_reg(r);
                    if d != s1 { self.push(mov_reg(reg_size, d, s1)); }
                }
                RegImm::Imm(v) => {
                    self.emit_imm_bitness(d, v);
                }
            }
        }

        match kind {
            ShiftKind::LogicalLeft => self.push(lsl(reg_size, d, d, s2_native)),
            ShiftKind::LogicalRight => self.push(lsr(reg_size, d, d, s2_native)),
            ShiftKind::ArithmeticRight => self.push(asr(reg_size, d, d, s2_native)),
        }

        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    // ── Compare helpers ───────────────────────────────────────────────────

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn compare_reg_reg(&mut self, d: RawReg, s1: RawReg, s2: RawReg, condition: Condition) {
        let reg_size = self.reg_size();
        self.push(cmp(reg_size, conv_reg(s1), conv_reg(s2)));
        self.push(cset(RegSize::X64, conv_reg(d), condition));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn compare_reg_imm(&mut self, d: RawReg, s1: RawReg, s2: u32, condition: Condition) {
        let reg_size = self.reg_size();
        if s2 < 4096 {
            self.push(cmp_imm(reg_size, conv_reg(s1), s2));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(cmp(reg_size, conv_reg(s1), TMP_REG));
        }
        self.push(cset(RegSize::X64, conv_reg(d), condition));
    }

    // ── Branch helper ─────────────────────────────────────────────────────

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn branch(&mut self, s1: RawReg, s2: impl Into<RegImm>, target: u32, condition: Condition) {
        let reg_size = self.reg_size();
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);

        match s2.into() {
            RegImm::Reg(s2) => self.push(cmp(reg_size, conv_reg(s1), conv_reg(s2))),
            RegImm::Imm(s2) => {
                if s2 < 4096 {
                    self.push(cmp_imm(reg_size, conv_reg(s1), s2));
                } else {
                    self.emit_imm_bitness(TMP_REG, s2);
                    self.push(cmp(reg_size, conv_reg(s1), TMP_REG));
                }
            }
        }
        self.push(b_cond_label(condition, label));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn cmov(&mut self, d: RawReg, s: RawReg, c: RawReg, condition: Condition) {
        if d == s { return; }
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s = conv_reg(s);
        let c = conv_reg(c);
        self.push(cmp_imm(reg_size, c, 0));
        self.push(csel(reg_size, d, s, d, condition));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn cmov_imm(&mut self, d: RawReg, s: u32, c: RawReg, condition: Condition) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let c = conv_reg(c);
        self.emit_imm_bitness(TMP_REG, s);
        self.push(cmp_imm(reg_size, c, 0));
        self.push(csel(reg_size, d, TMP_REG, d, condition));
    }

    fn jump_to_label(&mut self, label: Label) {
        self.push(b_label(label));
    }

    fn call_to_label(&mut self, label: Label) {
        self.push(bl_label(label));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn jump_indirect_impl(&mut self, load_imm: Option<(RawReg, u32)>, base: RawReg, offset: u32) {
        match S::KIND {
            SandboxKind::Linux => {
                emit_add_offset_32(&mut self.0.asm, TMP_REG, conv_reg(base), offset);
                // Multiply by 8 (jump table entry size)
                self.push(lsl_imm(RegSize::X64, TMP_REG, TMP_REG, 3));
                // Load from vmctx jump table
                self.push(add(RegSize::X64, TMP_REG, TMP_REG, AUX_TMP_REG));
                self.push(ldr_imm(MemSize::B64, TMP_REG, TMP_REG, 0));

                if let Some((ra, value)) = load_imm {
                    self.emit_imm_bitness(conv_reg(ra), value);
                }
                self.push(br(TMP_REG));
            }
            SandboxKind::Generic => {
                // Save next_native_program_counter for Rosetta 2 compatibility
                #[cfg(target_os = "macos")]
                {
                    let label_here = self.asm.create_label();
                    self.push(adr_label(x9, label_here));
                    self.store_vmctx_field_u64(x9, S::offset_table().next_native_program_counter);
                }

                // Compute jump table offset
                emit_add_offset_32(&mut self.0.asm, TMP_REG, conv_reg(base), offset);
                self.push(lsl_imm(RegSize::X64, TMP_REG, TMP_REG, 3));

                // Load jump table base from the label
                let jump_table_label = self.jump_table_label;
                self.push(adr_label(x9, jump_table_label));
                self.push(add(RegSize::X64, TMP_REG, TMP_REG, x9));
                self.push(ldr_imm(MemSize::B64, TMP_REG, TMP_REG, 0));

                if let Some((ra, value)) = load_imm {
                    self.emit_imm_bitness(conv_reg(ra), value);
                }
                self.push(br(TMP_REG));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Public instruction methods (called by the bytecode visitor)
    // ═══════════════════════════════════════════════════════════════════════

    #[inline(always)]
    pub fn trap(&mut self, code_offset: u32) {
        let trap_label = self.trap_label;
        self.store_vmctx_field_imm32(S::offset_table().program_counter, code_offset);
        self.call_to_label(trap_label);
    }

    #[inline(always)]
    pub fn trap_without_modifying_program_counter(&mut self) {
        let trap_label = self.trap_label;
        self.call_to_label(trap_label);
    }

    #[inline(always)]
    pub fn invalid(&mut self, code_offset: u32) {
        log::debug!("Encountered invalid instruction");
        self.trap(code_offset);
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    #[inline(always)]
    pub fn fallthrough(&mut self) {}

    #[inline(always)]
    pub fn load_imm(&mut self, dst: RawReg, s2: u32) {
        self.emit_imm_bitness(conv_reg(dst), s2);
    }

    #[inline(always)]
    pub fn load_imm64(&mut self, dst: RawReg, s2: u64) {
        assert_eq!(B::BITNESS, Bitness::B64);
        emit_load_imm64(&mut self.0.asm, conv_reg(dst), s2);
    }

    #[inline(always)]
    pub fn move_reg(&mut self, d: RawReg, s: RawReg) {
        self.push(mov_reg(self.reg_size(), conv_reg(d), conv_reg(s)));
    }

    // ── ALU: add/sub ──────────────────────────────────────────────────────

    fn add_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        self.push(add(reg_size, d, s1, s2));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)]
    pub fn add_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.add_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)]
    pub fn add_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.add_generic(RegSize::X64, d, s1, s2); }

    fn sub_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        self.push(sub(reg_size, d, s1, s2));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)]
    pub fn sub_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.sub_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)]
    pub fn sub_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.sub_generic(RegSize::X64, d, s1, s2); }

    fn add_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: u32) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        if s2 < 4096 {
            self.push(add_imm(reg_size, d, s1, s2));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(add(reg_size, d, s1, TMP_REG));
        }
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)]
    pub fn add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.add_imm_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)]
    pub fn add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.add_imm_generic(RegSize::X64, d, s1, s2); }

    fn negate_and_add_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: u32) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        // d = s2 - s1
        if s2 == 0 {
            self.push(neg(reg_size, d, s1));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(sub(reg_size, d, TMP_REG, s1));
        }
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)]
    pub fn negate_and_add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.negate_and_add_imm_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)]
    pub fn negate_and_add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.negate_and_add_imm_generic(RegSize::X64, d, s1, s2); }

    // ── ALU: mul ──────────────────────────────────────────────────────────

    fn mul_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        self.push(mul(reg_size, d, conv_reg(s1), conv_reg(s2)));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)]
    pub fn mul_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.mul_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)]
    pub fn mul_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.mul_generic(RegSize::X64, d, s1, s2); }

    #[inline(always)]
    pub fn mul_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        emit_load_imm32(&mut self.0.asm, TMP_REG, s2);
        self.push(mul(RegSize::W32, conv_reg(d), conv_reg(s1), TMP_REG));
        if B::BITNESS == Bitness::B64 {
            self.push(sxtw(conv_reg(d), conv_reg(d)));
        }
    }

    #[inline(always)]
    pub fn mul_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.emit_imm_bitness(TMP_REG, s2);
        self.push(mul(RegSize::X64, conv_reg(d), conv_reg(s1), TMP_REG));
    }

    #[inline(always)]
    pub fn mul_upper_signed_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        match B::BITNESS {
            Bitness::B32 => {
                // 32×32 signed → 64 bits, take upper 32
                self.push(smull(d, conv_reg(s1), conv_reg(s2)));
                self.push(lsr_imm(RegSize::X64, d, d, 32));
            }
            Bitness::B64 => {
                self.push(smulh(d, conv_reg(s1), conv_reg(s2)));
            }
        }
    }

    #[inline(always)]
    pub fn mul_upper_unsigned_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        match B::BITNESS {
            Bitness::B32 => {
                self.push(umull(d, conv_reg(s1), conv_reg(s2)));
                self.push(lsr_imm(RegSize::X64, d, d, 32));
            }
            Bitness::B64 => {
                self.push(umulh(d, conv_reg(s1), conv_reg(s2)));
            }
        }
    }

    #[inline(always)]
    pub fn mul_upper_signed_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        match B::BITNESS {
            Bitness::B32 => {
                // sign-extend s1 to 64, zero-extend s2 to 64, multiply, shift
                self.push(sxtw(TMP_REG, s1));
                // s2 is already zero-extended (W-reg ops zero upper bits)
                self.push(mul(RegSize::X64, d, TMP_REG, s2));
                self.push(lsr_imm(RegSize::X64, d, d, 32));
            }
            Bitness::B64 => {
                // mulhsu(a, b) = umulh(a, b) + (a >> 63) * b
                // umulh treats both as unsigned. When a is negative (signed),
                // the unsigned interpretation overestimates by b in the upper bits.
                // asr gives -1 when a < 0, so (-1) * b + umulh = umulh - b.
                self.push(umulh(d, s1, s2));
                self.push(asr_imm(RegSize::X64, TMP_REG, s1, 63));
                self.push(madd(RegSize::X64, d, TMP_REG, s2, d));
            }
        }
    }

    // ── ALU: div/rem ──────────────────────────────────────────────────────

    fn divrem(&mut self, reg_size: RegSize, div_rem: DivRem, kind: Signedness, d: RawReg, s1: RawReg, s2: RawReg) {
        let label = match (reg_size, div_rem, kind) {
            (RegSize::W32, DivRem::Div, Signedness::Unsigned) => self.div32u_label,
            (RegSize::W32, DivRem::Div, Signedness::Signed) => self.div32s_label,
            (RegSize::W32, DivRem::Rem, Signedness::Unsigned) => self.rem32u_label,
            (RegSize::W32, DivRem::Rem, Signedness::Signed) => self.rem32s_label,
            (RegSize::X64, DivRem::Div, Signedness::Unsigned) => self.div64u_label,
            (RegSize::X64, DivRem::Div, Signedness::Signed) => self.div64s_label,
            (RegSize::X64, DivRem::Rem, Signedness::Unsigned) => self.rem64u_label,
            (RegSize::X64, DivRem::Rem, Signedness::Signed) => self.rem64s_label,
        };

        // Convention: dividend in x9, divisor in TMP_REG
        self.push(mov_reg(reg_size, TMP_REG, conv_reg(s2)));
        self.push(mov_reg(reg_size, x9, conv_reg(s1)));
        self.call_to_label(label);
        self.push(mov_reg(RegSize::X64, conv_reg(d), TMP_REG));
    }

    #[inline(always)] pub fn div_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.divrem(RegSize::W32, DivRem::Div, Signedness::Unsigned, d, s1, s2); }
    #[inline(always)] pub fn div_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.divrem(RegSize::X64, DivRem::Div, Signedness::Unsigned, d, s1, s2); }
    #[inline(always)] pub fn div_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.divrem(RegSize::W32, DivRem::Div, Signedness::Signed, d, s1, s2); }
    #[inline(always)] pub fn div_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.divrem(RegSize::X64, DivRem::Div, Signedness::Signed, d, s1, s2); }
    #[inline(always)] pub fn rem_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.divrem(RegSize::W32, DivRem::Rem, Signedness::Unsigned, d, s1, s2); }
    #[inline(always)] pub fn rem_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.divrem(RegSize::X64, DivRem::Rem, Signedness::Unsigned, d, s1, s2); }
    #[inline(always)] pub fn rem_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.divrem(RegSize::W32, DivRem::Rem, Signedness::Signed, d, s1, s2); }
    #[inline(always)] pub fn rem_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.divrem(RegSize::X64, DivRem::Rem, Signedness::Signed, d, s1, s2); }

    // ── ALU: bitwise ──────────────────────────────────────────────────────

    #[inline(always)]
    pub fn and(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(and(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn or(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(orr(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn xor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(eor(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn and_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = s1 & ~s2
        self.push(bic(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn or_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = s1 | ~s2
        self.push(orn(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn xnor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = ~(s1 ^ s2)
        self.push(eon(self.reg_size(), conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }

    #[inline(always)]
    pub fn and_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        let value = match reg_size { RegSize::W32 => s2 as u64, RegSize::X64 => s2 as i32 as i64 as u64 };
        if let Some((n, immr, imms)) = encode_logical_imm(reg_size, value) {
            self.push(and_imm(reg_size, conv_reg(d), conv_reg(s1), n, immr, imms));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(and(reg_size, conv_reg(d), conv_reg(s1), TMP_REG));
        }
    }

    #[inline(always)]
    pub fn or_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        let value = match reg_size { RegSize::W32 => s2 as u64, RegSize::X64 => s2 as i32 as i64 as u64 };
        if let Some((n, immr, imms)) = encode_logical_imm(reg_size, value) {
            self.push(orr_imm(reg_size, conv_reg(d), conv_reg(s1), n, immr, imms));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(orr(reg_size, conv_reg(d), conv_reg(s1), TMP_REG));
        }
    }

    #[inline(always)]
    pub fn xor_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        if s2 == !0u32 {
            self.push(mvn(reg_size, conv_reg(d), conv_reg(s1)));
            return;
        }
        let value = match reg_size { RegSize::W32 => s2 as u64, RegSize::X64 => s2 as i32 as i64 as u64 };
        if let Some((n, immr, imms)) = encode_logical_imm(reg_size, value) {
            self.push(eor_imm(reg_size, conv_reg(d), conv_reg(s1), n, immr, imms));
        } else {
            self.emit_imm_bitness(TMP_REG, s2);
            self.push(eor(reg_size, conv_reg(d), conv_reg(s1), TMP_REG));
        }
    }

    // ── ALU: shifts ───────────────────────────────────────────────────────

    #[inline(always)] pub fn shift_logical_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_logical_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_arithmetic_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::ArithmeticRight); }
    #[inline(always)] pub fn shift_arithmetic_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::ArithmeticRight); }

    #[inline(always)] pub fn shift_logical_left_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_left_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_logical_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_arithmetic_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::W32, d, s1, s2, ShiftKind::ArithmeticRight); }
    #[inline(always)] pub fn shift_arithmetic_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.shift_imm_op(RegSize::X64, d, s1, s2, ShiftKind::ArithmeticRight); }

    // Shift imm alt: d = s1 << s2 where s1 is the immediate
    #[inline(always)] pub fn shift_logical_left_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_left_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalLeft); }
    #[inline(always)] pub fn shift_logical_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_logical_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::LogicalRight); }
    #[inline(always)] pub fn shift_arithmetic_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::W32, d, s1, s2, ShiftKind::ArithmeticRight); }
    #[inline(always)] pub fn shift_arithmetic_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) { self.shift_reg_op(RegSize::X64, d, s1, s2, ShiftKind::ArithmeticRight); }

    // ── ALU: rotates ──────────────────────────────────────────────────────

    fn rotate_left_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        // ROL = ROR by (width - amount)
        let width = match reg_size { RegSize::W32 => 32u32, RegSize::X64 => 64 };
        self.emit_imm32(TMP_REG, width);
        self.push(sub(reg_size, TMP_REG, TMP_REG, s2));
        self.push(ror(reg_size, d, s1, TMP_REG));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)] pub fn rotate_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.rotate_left_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)] pub fn rotate_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.rotate_left_generic(RegSize::X64, d, s1, s2); }

    fn rotate_right_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        self.push(ror(reg_size, d, conv_reg(s1), conv_reg(s2)));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)] pub fn rotate_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.rotate_right_generic(RegSize::W32, d, s1, s2); }
    #[inline(always)] pub fn rotate_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { assert_eq!(B::BITNESS, Bitness::B64); self.rotate_right_generic(RegSize::X64, d, s1, s2); }

    fn rotate_right_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s: RawReg, c: u32) {
        let d = conv_reg(d);
        self.push(ror_imm(reg_size, d, conv_reg(s), c));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)] pub fn rotate_right_imm_32(&mut self, d: RawReg, s: RawReg, c: u32) { self.rotate_right_imm_generic(RegSize::W32, d, s, c); }
    #[inline(always)] pub fn rotate_right_imm_64(&mut self, d: RawReg, s: RawReg, c: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.rotate_right_imm_generic(RegSize::X64, d, s, c); }

    fn rotate_right_imm_alt_generic(&mut self, reg_size: RegSize, d: RawReg, s: RawReg, c: u32) {
        // d = c ROR s
        let d = conv_reg(d);
        self.emit_imm_bitness(d, c);
        self.push(ror(reg_size, d, d, conv_reg(s)));
        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::W32) {
            self.push(sxtw(d, d));
        }
    }

    #[inline(always)] pub fn rotate_right_imm_alt_32(&mut self, d: RawReg, s: RawReg, c: u32) { self.rotate_right_imm_alt_generic(RegSize::W32, d, s, c); }
    #[inline(always)] pub fn rotate_right_imm_alt_64(&mut self, d: RawReg, s: RawReg, c: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.rotate_right_imm_alt_generic(RegSize::X64, d, s, c); }

    // ── ALU: min/max ──────────────────────────────────────────────────────

    fn min_max_generic(&mut self, c: Condition, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        self.push(cmp(reg_size, s1, s2));
        self.push(csel(reg_size, d, s1, s2, c));
    }

    #[inline(always)] pub fn maximum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.min_max_generic(Condition::GT, d, s1, s2); }
    #[inline(always)] pub fn maximum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.min_max_generic(Condition::HI, d, s1, s2); }
    #[inline(always)] pub fn minimum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.min_max_generic(Condition::LT, d, s1, s2); }
    #[inline(always)] pub fn minimum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.min_max_generic(Condition::LO, d, s1, s2); }

    // ── Comparisons ───────────────────────────────────────────────────────

    #[inline(always)] pub fn set_less_than_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.compare_reg_reg(d, s1, s2, Condition::LO); }
    #[inline(always)] pub fn set_less_than_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) { self.compare_reg_reg(d, s1, s2, Condition::LT); }
    #[inline(always)] pub fn set_less_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.compare_reg_imm(d, s1, s2, Condition::LO); }
    #[inline(always)] pub fn set_less_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.compare_reg_imm(d, s1, s2, Condition::LT); }
    #[inline(always)] pub fn set_greater_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.compare_reg_imm(d, s1, s2, Condition::HI); }
    #[inline(always)] pub fn set_greater_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) { self.compare_reg_imm(d, s1, s2, Condition::GT); }

    // ── Conditional moves ─────────────────────────────────────────────────

    #[inline(always)] pub fn cmov_if_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) { self.cmov(d, s, c, Condition::EQ); }
    #[inline(always)] pub fn cmov_if_not_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) { self.cmov(d, s, c, Condition::NE); }
    #[inline(always)] pub fn cmov_if_zero_imm(&mut self, d: RawReg, c: RawReg, s: u32) { self.cmov_imm(d, s, c, Condition::EQ); }
    #[inline(always)] pub fn cmov_if_not_zero_imm(&mut self, d: RawReg, c: RawReg, s: u32) { self.cmov_imm(d, s, c, Condition::NE); }

    // ── Sign/zero extend + bit ops ────────────────────────────────────────

    #[inline(always)] pub fn sign_extend_8(&mut self, d: RawReg, s: RawReg) { self.push(sxtb(self.reg_size(), conv_reg(d), conv_reg(s))); }
    #[inline(always)] pub fn sign_extend_16(&mut self, d: RawReg, s: RawReg) { self.push(sxth(self.reg_size(), conv_reg(d), conv_reg(s))); }
    #[inline(always)] pub fn zero_extend_16(&mut self, d: RawReg, s: RawReg) { self.push(uxth(conv_reg(d), conv_reg(s))); }

    #[inline(always)]
    pub fn reverse_byte(&mut self, d: RawReg, s: RawReg) {
        self.push(rev(self.reg_size(), conv_reg(d), conv_reg(s)));
    }

    #[inline(always)] pub fn count_leading_zero_bits_32(&mut self, d: RawReg, s: RawReg) { self.push(clz(RegSize::W32, conv_reg(d), conv_reg(s))); }
    #[inline(always)] pub fn count_leading_zero_bits_64(&mut self, d: RawReg, s: RawReg) { self.push(clz(self.reg_size(), conv_reg(d), conv_reg(s))); }

    #[inline(always)]
    pub fn count_trailing_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
        // CTZ = RBIT + CLZ
        self.push(rbit(RegSize::W32, conv_reg(d), conv_reg(s)));
        self.push(clz(RegSize::W32, conv_reg(d), conv_reg(d)));
    }

    #[inline(always)]
    pub fn count_trailing_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
        let rs = self.reg_size();
        self.push(rbit(rs, conv_reg(d), conv_reg(s)));
        self.push(clz(rs, conv_reg(d), conv_reg(d)));
    }

    #[inline(always)]
    pub fn count_set_bits_32(&mut self, d: RawReg, s: RawReg) {
        // Kernighan's bit counting: loop { x &= (x-1); count++; } until x == 0
        let d = conv_reg(d);
        let s = conv_reg(s);
        self.push(mov_reg(RegSize::W32, TMP_REG, s));
        self.push(movz(RegSize::W32, d, 0, 0));
        let label_loop = self.asm.forward_declare_label();
        let label_done = self.asm.forward_declare_label();
        self.push(cbz_label(RegSize::W32, TMP_REG, label_done));
        self.asm.define_label(label_loop);
        self.push(sub_imm(RegSize::W32, x9, TMP_REG, 1));
        self.push(and(RegSize::W32, TMP_REG, TMP_REG, x9));
        self.push(add_imm(RegSize::W32, d, d, 1));
        self.push(cbnz_label(RegSize::W32, TMP_REG, label_loop));
        self.asm.define_label(label_done);
    }

    #[inline(always)]
    pub fn count_set_bits_64(&mut self, d: RawReg, s: RawReg) {
        let d = conv_reg(d);
        let s = conv_reg(s);
        let rs = self.reg_size();
        self.push(mov_reg(rs, TMP_REG, s));
        self.push(movz(RegSize::X64, d, 0, 0));
        let label_loop = self.asm.forward_declare_label();
        let label_done = self.asm.forward_declare_label();
        self.push(cbz_label(RegSize::X64, TMP_REG, label_done));
        self.asm.define_label(label_loop);
        self.push(sub_imm(RegSize::X64, x9, TMP_REG, 1));
        self.push(and(RegSize::X64, TMP_REG, TMP_REG, x9));
        self.push(add_imm(RegSize::X64, d, d, 1));
        self.push(cbnz_label(RegSize::X64, TMP_REG, label_loop));
        self.asm.define_label(label_done);
    }

    // ── Memory: store ─────────────────────────────────────────────────────

    #[inline(always)] pub fn store_u8(&mut self, src: RawReg, offset: u32) { self.store_to_memory(src, None, offset, MemSize::B8); }
    #[inline(always)] pub fn store_u16(&mut self, src: RawReg, offset: u32) { self.store_to_memory(src, None, offset, MemSize::B16); }
    #[inline(always)] pub fn store_u32(&mut self, src: RawReg, offset: u32) { self.store_to_memory(src, None, offset, MemSize::B32); }
    #[inline(always)] pub fn store_u64(&mut self, src: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.store_to_memory(src, None, offset, MemSize::B64); }

    #[inline(always)] pub fn store_indirect_u8(&mut self, src: RawReg, base: RawReg, offset: u32) { self.store_to_memory(src, Some(base), offset, MemSize::B8); }
    #[inline(always)] pub fn store_indirect_u16(&mut self, src: RawReg, base: RawReg, offset: u32) { self.store_to_memory(src, Some(base), offset, MemSize::B16); }
    #[inline(always)] pub fn store_indirect_u32(&mut self, src: RawReg, base: RawReg, offset: u32) { self.store_to_memory(src, Some(base), offset, MemSize::B32); }
    #[inline(always)] pub fn store_indirect_u64(&mut self, src: RawReg, base: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.store_to_memory(src, Some(base), offset, MemSize::B64); }

    #[inline(always)] pub fn store_imm_u8(&mut self, offset: u32, value: u32) { self.store_to_memory(value, None, offset, MemSize::B8); }
    #[inline(always)] pub fn store_imm_u16(&mut self, offset: u32, value: u32) { self.store_to_memory(value, None, offset, MemSize::B16); }
    #[inline(always)] pub fn store_imm_u32(&mut self, offset: u32, value: u32) { self.store_to_memory(value, None, offset, MemSize::B32); }
    #[inline(always)] pub fn store_imm_u64(&mut self, offset: u32, value: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.store_to_memory(value, None, offset, MemSize::B64); }

    #[inline(always)] pub fn store_imm_indirect_u8(&mut self, base: RawReg, offset: u32, value: u32) { self.store_to_memory(value, Some(base), offset, MemSize::B8); }
    #[inline(always)] pub fn store_imm_indirect_u16(&mut self, base: RawReg, offset: u32, value: u32) { self.store_to_memory(value, Some(base), offset, MemSize::B16); }
    #[inline(always)] pub fn store_imm_indirect_u32(&mut self, base: RawReg, offset: u32, value: u32) { self.store_to_memory(value, Some(base), offset, MemSize::B32); }
    #[inline(always)] pub fn store_imm_indirect_u64(&mut self, base: RawReg, offset: u32, value: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.store_to_memory(value, Some(base), offset, MemSize::B64); }

    // ── Memory: load ──────────────────────────────────────────────────────

    #[inline(always)] pub fn load_u8(&mut self, dst: RawReg, offset: u32) { self.load_from_memory(dst, None, offset, MemSize::B8, false); }
    #[inline(always)] pub fn load_i8(&mut self, dst: RawReg, offset: u32) { self.load_from_memory(dst, None, offset, MemSize::B8, true); }
    #[inline(always)] pub fn load_u16(&mut self, dst: RawReg, offset: u32) { self.load_from_memory(dst, None, offset, MemSize::B16, false); }
    #[inline(always)] pub fn load_i16(&mut self, dst: RawReg, offset: u32) { self.load_from_memory(dst, None, offset, MemSize::B16, true); }
    #[inline(always)]
    pub fn load_i32(&mut self, dst: RawReg, offset: u32) {
        let sign = matches!(B::BITNESS, Bitness::B64);
        self.load_from_memory(dst, None, offset, MemSize::B32, sign);
    }
    #[inline(always)] pub fn load_u32(&mut self, dst: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.load_from_memory(dst, None, offset, MemSize::B32, false); }
    #[inline(always)] pub fn load_u64(&mut self, dst: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.load_from_memory(dst, None, offset, MemSize::B64, false); }

    #[inline(always)] pub fn load_indirect_u8(&mut self, dst: RawReg, base: RawReg, offset: u32) { self.load_from_memory(dst, Some(base), offset, MemSize::B8, false); }
    #[inline(always)] pub fn load_indirect_i8(&mut self, dst: RawReg, base: RawReg, offset: u32) { self.load_from_memory(dst, Some(base), offset, MemSize::B8, true); }
    #[inline(always)] pub fn load_indirect_u16(&mut self, dst: RawReg, base: RawReg, offset: u32) { self.load_from_memory(dst, Some(base), offset, MemSize::B16, false); }
    #[inline(always)] pub fn load_indirect_i16(&mut self, dst: RawReg, base: RawReg, offset: u32) { self.load_from_memory(dst, Some(base), offset, MemSize::B16, true); }
    #[inline(always)]
    pub fn load_indirect_i32(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        let sign = matches!(B::BITNESS, Bitness::B64);
        self.load_from_memory(dst, Some(base), offset, MemSize::B32, sign);
    }
    #[inline(always)] pub fn load_indirect_u32(&mut self, dst: RawReg, base: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.load_from_memory(dst, Some(base), offset, MemSize::B32, false); }
    #[inline(always)] pub fn load_indirect_u64(&mut self, dst: RawReg, base: RawReg, offset: u32) { assert_eq!(B::BITNESS, Bitness::B64); self.load_from_memory(dst, Some(base), offset, MemSize::B64, false); }

    // ── Branches ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn branch_less_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::LO); }
    #[inline(always)] pub fn branch_less_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::LT); }
    #[inline(always)] pub fn branch_greater_or_equal_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::HS); }
    #[inline(always)] pub fn branch_greater_or_equal_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::GE); }
    #[inline(always)] pub fn branch_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::EQ); }
    #[inline(always)] pub fn branch_not_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) { self.branch(s1, s2, target, Condition::NE); }

    #[inline(always)] pub fn branch_eq_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::EQ); }
    #[inline(always)] pub fn branch_not_eq_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::NE); }
    #[inline(always)] pub fn branch_less_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::LO); }
    #[inline(always)] pub fn branch_less_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::LT); }
    #[inline(always)] pub fn branch_greater_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::HS); }
    #[inline(always)] pub fn branch_greater_or_equal_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::GE); }
    #[inline(always)] pub fn branch_less_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::LS); }
    #[inline(always)] pub fn branch_less_or_equal_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::LE); }
    #[inline(always)] pub fn branch_greater_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::HI); }
    #[inline(always)] pub fn branch_greater_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) { self.branch(s1, s2, target, Condition::GT); }

    #[inline(always)]
    pub fn jump(&mut self, target: u32) {
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);
        self.jump_to_label(label);
    }

    #[inline(always)]
    pub fn load_imm_and_jump(&mut self, ra: RawReg, value: u32, target: u32) {
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);
        self.emit_imm_bitness(conv_reg(ra), value);
        self.jump_to_label(label);
    }

    #[inline(always)]
    pub fn jump_indirect(&mut self, base: RawReg, offset: u32) {
        self.jump_indirect_impl(None, base, offset);
    }

    #[inline(always)]
    pub fn load_imm_and_jump_indirect(&mut self, ra: RawReg, base: RawReg, value: u32, offset: u32) {
        self.jump_indirect_impl(Some((ra, value)), base, offset);
    }

    // ── Special ───────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn ecalli(&mut self, code_offset: u32, args_length: u32, imm: u32) {
        if let Some(ref custom_codegen) = self.0.custom_codegen {
            if !custom_codegen.should_emit_ecalli(imm, &mut self.0.asm) {
                return;
            }
        }

        let ecall_label = self.ecall_label;
        // Load vmctx base once and reuse for all three stores.
        // For generic sandbox load_vmctx_base() returns TMP_REG, while
        // emit_load_imm32 uses x9, so vmctx_base stays valid across stores.
        let vmctx_base = self.load_vmctx_base();
        emit_load_imm32(&mut self.0.asm, x9, imm);
        self.push(str_imm(MemSize::B32, x9, vmctx_base, S::offset_table().arg as u32));
        emit_load_imm32(&mut self.0.asm, x9, code_offset);
        self.push(str_imm(MemSize::B32, x9, vmctx_base, S::offset_table().program_counter as u32));
        emit_load_imm32(&mut self.0.asm, x9, code_offset + args_length + 1);
        self.push(str_imm(MemSize::B32, x9, vmctx_base, S::offset_table().next_program_counter as u32));
        self.call_to_label(ecall_label);
    }

    #[inline(always)]
    pub fn sbrk(&mut self, dst: RawReg, size: RawReg) {
        let label_bump_only = self.asm.forward_declare_label();
        let label_continue = self.asm.forward_declare_label();
        let sbrk_label = self.sbrk_label;

        let dst_n = conv_reg(dst);
        let size_n = conv_reg(size);
        if dst_n != size_n {
            self.push(mov_reg(RegSize::W32, dst_n, size_n));
        }

        let offset = S::offset_table().heap_info;
        let vmctx_base = self.load_vmctx_base(); // TMP_REG = vmctx

        // Load current top-of-heap into x9 (keep vmctx_base in TMP_REG)
        self.push(ldr_imm(MemSize::B64, x9, vmctx_base, offset as u32));
        // Calculate new top: new_top = old_top + size
        self.push(add(RegSize::X64, dst_n, dst_n, x9));
        // Load threshold into x9 (vmctx_base = TMP_REG still valid)
        self.push(ldr_imm(MemSize::B64, x9, vmctx_base, (offset + 8) as u32));
        // Compare new_top <= threshold?
        self.push(cmp(RegSize::X64, dst_n, x9));
        self.push(b_cond_label(Condition::LS, label_bump_only));

        // Need to allocate more memory
        self.push(mov_reg(RegSize::X64, TMP_REG, dst_n));
        self.call_to_label(sbrk_label);
        self.push(mov_reg(RegSize::W32, dst_n, TMP_REG));
        self.push(b_label(label_continue));

        // Bump only
        self.asm.define_label(label_bump_only);
        let vmctx_base2 = self.load_vmctx_base();
        self.push(str_imm(MemSize::B64, dst_n, vmctx_base2, offset as u32));

        self.asm.define_label(label_continue);
    }

    #[inline(always)]
    pub fn memset(&mut self) {
        let _reg_size = self.reg_size();

        let dst = conv_reg(Reg::A0.into()); // x0
        let val = conv_reg(Reg::A1.into()); // x1
        let count = conv_reg(Reg::A2.into()); // x2

        // Store restart address
        let label_repeat = self.asm.create_label();
        self.push(adr_label(x9, label_repeat));
        self.store_vmctx_field_u64(x9, S::offset_table().next_native_program_counter);

        // Zero-extend count to 64 bits
        self.push(mov_reg(RegSize::W32, count, count));

        // Memory base register for sandbox-relative addressing
        let mem_base = match S::KIND {
            SandboxKind::Linux => AUX_TMP_REG,
            SandboxKind::Generic => {
                #[cfg(feature = "generic-sandbox")]
                { GENERIC_SANDBOX_MEMORY_REG }
                #[cfg(not(feature = "generic-sandbox"))]
                { unreachable!() }
            }
        };

        // Set vmctx.arg = 1 to mark "memset in progress" for page fault handler.
        // This allows the handler to refund pre-charged gas by remaining count (A2).
        if self.gas_metering.is_some() {
            self.store_vmctx_field_imm32(S::offset_table().arg, 1);
        }

        match self.gas_metering {
            None => {
                let label_done = self.asm.forward_declare_label();
                self.push(cbz_label(RegSize::X64, count, label_done));
                let label_loop = self.asm.create_label();
                self.push(str_reg_uxtw(MemSize::B8, val, mem_base, dst));
                self.push(add_imm(RegSize::W32, dst, dst, 1));
                self.push(sub_imm(RegSize::X64, count, count, 1));
                self.push(cbnz_label(RegSize::X64, count, label_loop));
                self.asm.define_label(label_done);
            }
            Some(GasMeteringKind::Sync) => {
                let vmctx_base = self.load_vmctx_base();
                self.push(ldr_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
                self.push(sub(RegSize::X64, x9, x9, count));
                self.push(str_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
                self.push(cmp_imm(RegSize::X64, x9, 0));
                let label_slow = self.memset_label;
                self.push(b_cond_label(Condition::LT, label_slow));
                let label_done = self.asm.forward_declare_label();
                self.push(cbz_label(RegSize::X64, count, label_done));
                let label_loop = self.asm.create_label();
                self.push(str_reg_uxtw(MemSize::B8, val, mem_base, dst));
                self.push(add_imm(RegSize::W32, dst, dst, 1));
                self.push(sub_imm(RegSize::X64, count, count, 1));
                self.push(cbnz_label(RegSize::X64, count, label_loop));
                self.asm.define_label(label_done);
            }
            Some(GasMeteringKind::Async) => {
                let vmctx_base = self.load_vmctx_base();
                self.push(ldr_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
                self.push(sub(RegSize::X64, x9, x9, count));
                self.push(str_imm(MemSize::B64, x9, vmctx_base, S::offset_table().gas as u32));
                let label_done = self.asm.forward_declare_label();
                self.push(cbz_label(RegSize::X64, count, label_done));
                let label_loop = self.asm.create_label();
                self.push(str_reg_uxtw(MemSize::B8, val, mem_base, dst));
                self.push(add_imm(RegSize::W32, dst, dst, 1));
                self.push(sub_imm(RegSize::X64, count, count, 1));
                self.push(cbnz_label(RegSize::X64, count, label_loop));
                self.asm.define_label(label_done);
            }
        }

        // Clear memset flag and restart address after loop completes normally.
        if self.gas_metering.is_some() {
            self.store_vmctx_field_imm32(S::offset_table().arg, 0);
        }
        self.store_vmctx_field_zero_u64(S::offset_table().next_native_program_counter);
    }
}

// ── Module-level functions ────────────────────────────────────────────────────

pub fn on_signal_trap<S>(
    compiled_module: &crate::compiler::CompiledModule<S>,
    is_gas_metering_enabled: bool,
    machine_code_offset: u64,
    vmctx: &VmCtx,
) -> Result<bool, &'static str>
where
    S: Sandbox,
{
    if are_we_executing_memset(compiled_module, machine_code_offset) {
        // Memset interruption
        set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;
        vmctx.next_native_program_counter.store(0, Ordering::Relaxed);
        return Ok(false);
    }

    if is_gas_metering_enabled && vmctx.gas.load(Ordering::Relaxed) < 0 {
        // Gas exhaustion (BRK trap)
        let Some(offset) = machine_code_offset.checked_sub(GAS_METERING_TRAP_OFFSET) else {
            return Err("internal error: address underflow after a trap");
        };

        vmctx.next_native_program_counter.store(compiled_module.native_code_origin + offset, Ordering::Relaxed);
        let program_counter = set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;

        // Read back the gas cost from the sub instruction
        let cost_offset = offset as usize + GAS_COST_OFFSET;
        let Some(inst_bytes) = compiled_module.machine_code().get(cost_offset..cost_offset + 4) else {
            return Err("internal error: failed to read back the gas cost from the machine code");
        };
        let inst_word = u32::from_le_bytes([inst_bytes[0], inst_bytes[1], inst_bytes[2], inst_bytes[3]]);
        let gas_cost = (inst_word >> 10) & 0xFFF;

        let gas = vmctx.gas.fetch_add(i64::from(gas_cost), Ordering::Relaxed);
        log::trace!(
            "Out of gas; program counter = {program_counter}, reverting gas: {gas} -> {new_gas} (gas cost: {gas_cost})",
            new_gas = gas + i64::from(gas_cost)
        );

        Ok(true)
    } else {
        // Regular trap
        set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;
        vmctx.next_native_program_counter.store(0, Ordering::Relaxed);
        Ok(false)
    }
}

pub fn on_page_fault<S>(
    compiled_module: &crate::compiler::CompiledModule<S>,
    is_gas_metering_enabled: bool,
    machine_code_address: u64,
    machine_code_offset: u64,
    vmctx: &VmCtx,
) -> Result<(), &'static str>
where
    S: Sandbox,
{
    if are_we_executing_memset(compiled_module, machine_code_offset) {
        // Memset page fault
        let bytes_remaining = vmctx.tmp_reg.load(Ordering::Relaxed);
        vmctx.regs[Reg::A2 as usize].fetch_add(bytes_remaining, Ordering::Relaxed);
        if is_gas_metering_enabled {
            vmctx.gas.fetch_add(cast(bytes_remaining).to_signed(), Ordering::Relaxed);
        }
        let original_offset = vmctx.next_native_program_counter.load(Ordering::Relaxed) - compiled_module.native_code_origin;
        set_program_counter_after_interruption(compiled_module, original_offset, vmctx)?;
    } else {
        set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;
        vmctx.next_native_program_counter.store(machine_code_address, Ordering::Relaxed);
    }
    Ok(())
}

pub fn extract_gas_cost<S>(machine_code: &[u8], basic_block_machine_code_offset: usize) -> u32
where
    S: Sandbox,
{
    let offset = basic_block_machine_code_offset + GAS_COST_OFFSET;
    let inst_bytes = &machine_code[offset..offset + 4];
    let inst_word = u32::from_le_bytes([inst_bytes[0], inst_bytes[1], inst_bytes[2], inst_bytes[3]]);
    // Extract imm12 from sub_imm: bits [21:10]
    (inst_word >> 10) & 0xFFF
}

#[inline(always)]
pub fn step_prelude_length<S>() -> usize
where
    S: Sandbox,
{
    // trace_execution emits: 4 instructions (store PC + store next_PC or 4 NOPs) + BL = 5 instructions = 20 bytes
    // But the actual size depends on vmctx field stores which vary by sandbox kind.
    // For now, use a generous estimate. The exact value will be validated by the compiler.
    match S::KIND {
        SandboxKind::Linux => 24,
        SandboxKind::Generic => 32,
    }
}
