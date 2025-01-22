use core::sync::atomic::Ordering;

use polkavm_assembler::amd64::addr::*;
use polkavm_assembler::amd64::inst::*;
use polkavm_assembler::amd64::Reg::rsp;
use polkavm_assembler::amd64::RegIndex as NativeReg;
use polkavm_assembler::amd64::RegIndex::*;
use polkavm_assembler::amd64::{Condition, LoadKind, MemOp, RegSize, Size};
use polkavm_assembler::{Label, NonZero, ReservedAssembler, U1, U2, U3, U4};

use polkavm_common::cast::cast;
use polkavm_common::program::{ProgramCounter, RawReg, Reg};
use polkavm_common::zygote::{VmCtx, VM_ADDR_VMCTX};

use crate::compiler::{ArchVisitor, Bitness, CompilerBitness, SandboxKind};
use crate::config::GasMeteringKind;
use crate::sandbox::Sandbox;
use crate::utils::RegImm;

/// The register used for the embedded sandbox to hold the base address of the guest's linear memory.
const GENERIC_SANDBOX_MEMORY_REG: NativeReg = AUX_TMP_REG;

/// The register used for the linux sandbox to hold the address of the VM context.
const LINUX_SANDBOX_VMCTX_REG: NativeReg = AUX_TMP_REG;

use polkavm_common::regmap::to_native_reg as conv_reg_const;
use polkavm_common::regmap::{AUX_TMP_REG, TMP_REG};

polkavm_common::static_assert!(polkavm_common::regmap::to_guest_reg(TMP_REG).is_none());
polkavm_common::static_assert!(polkavm_common::regmap::to_guest_reg(AUX_TMP_REG).is_none());

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

macro_rules! with_sandbox_kind {
    ($input:expr, |$kind:ident| $body:expr) => {
        match $input {
            SandboxKind::Linux => {
                #[allow(non_upper_case_globals)]
                const $kind: SandboxKind = SandboxKind::Linux;
                $body
            }
            SandboxKind::Generic => {
                #[allow(non_upper_case_globals)]
                const $kind: SandboxKind = SandboxKind::Generic;
                $body
            }
        }
    };
}

macro_rules! load_store_operand {
    ($self:ident, $kind:expr, $base:ident, $offset:expr, |$op:ident| $body:expr) => {
        with_sandbox_kind!($kind, |sandbox_kind| {
            match sandbox_kind {
                SandboxKind::Linux => {
                    if let Some($base) = $base {
                        // [address + offset]
                        let $op = reg_indirect(RegSize::R32, conv_reg($base) + $offset as i32);
                        $body
                    } else {
                        // [address] = ..
                        let $op = abs(RegSize::R32, $offset as i32);
                        $body
                    }
                }
                SandboxKind::Generic => {
                    match ($base, $offset) {
                        // [address] = ..
                        // (address is in the lower 2GB of the address space)
                        (None, _) if $offset as i32 >= 0 => {
                            let $op = reg_indirect(RegSize::R64, GENERIC_SANDBOX_MEMORY_REG + $offset as i32);
                            $body
                        }

                        // [address] = ..
                        (None, _) => {
                            $self.push(mov_imm(TMP_REG, imm32($offset)));
                            let $op = base_index(RegSize::R64, GENERIC_SANDBOX_MEMORY_REG, TMP_REG);
                            $body
                        }

                        // [base] = ..
                        (Some($base), 0) => {
                            // NOTE: This assumes that `base` has its upper 32-bits clear.
                            let $op = base_index(RegSize::R64, GENERIC_SANDBOX_MEMORY_REG, conv_reg($base));
                            $body
                        }

                        // [base + offset] = ..
                        (Some($base), _) => {
                            $self.push(lea(
                                RegSize::R32,
                                TMP_REG,
                                reg_indirect(RegSize::R32, conv_reg($base) + $offset as i32),
                            ));
                            let $op = base_index(RegSize::R64, GENERIC_SANDBOX_MEMORY_REG, TMP_REG);
                            $body
                        }
                    }
                }
            }
        })
    };
}

enum Signedness {
    Signed,
    Unsigned,
}

enum DivRem {
    Div,
    Rem,
}

enum ShiftKind {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn calculate_label_offset(asm_len: usize, rel8_len: usize, rel32_len: usize, offset: isize) -> Result<i8, i32> {
    let offset_near = offset - (asm_len as isize + rel8_len as isize);
    if offset_near <= i8::MAX as isize && offset_near >= i8::MIN as isize {
        Ok(offset_near as i8)
    } else {
        let offset = offset - (asm_len as isize + rel32_len as isize);
        Err(offset as i32)
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn branch_to_label<R>(asm: ReservedAssembler<R>, condition: Condition, label: Label) -> ReservedAssembler<R::Next>
where
    R: NonZero,
{
    if let Some(offset) = asm.get_label_origin_offset(label) {
        let offset = calculate_label_offset(
            asm.len(),
            jcc_rel8(condition, i8::MAX).len(),
            jcc_rel32(condition, i32::MAX).len(),
            offset,
        );

        match offset {
            Ok(offset) => asm.push(jcc_rel8(condition, offset)),
            Err(offset) => asm.push(jcc_rel32(condition, offset)),
        }
    } else {
        // For invalid jumps this will emit:
        //   0f 84 fc ff ff ff    je near 2
        // so if the branch triggers this will jump into itself:
        //   fc                   cld
        //   ff ff                invalid
        // The `cld` here is harmless, and the 'ff' is the opcode shared by single register 'inc/dec/call/jmp/push' instructions,
        // however the /7 opext it has here is currently undefined so this will trap. Same as for gas, this is technically
        // a forward compatibility hazard.
        asm.push(jcc_label32_default(condition, label, -4))
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn jump_to_label<R>(asm: ReservedAssembler<R>, label: Label) -> ReservedAssembler<R::Next>
where
    R: NonZero,
{
    if let Some(offset) = asm.get_label_origin_offset(label) {
        let offset = calculate_label_offset(asm.len(), jmp_rel8(i8::MAX).len(), jmp_rel32(i32::MAX).len(), offset);

        match offset {
            Ok(offset) => asm.push(jmp_rel8(offset)),
            Err(offset) => asm.push(jmp_rel32(offset)),
        }
    } else {
        asm.push(jmp_label32(label))
    }
}

const GAS_METERING_TRAP_OFFSET: u64 = 9;
const GAS_COST_OFFSET: usize = 3;
const REP_STOSB_MACHINE_CODE: &[u8] = &[0xf3, 0xaa];

#[derive(Copy, Clone)]
enum MemsetKind {
    Inline,
    Trampoline,
}

fn are_we_executing_memset<S>(compiled_module: &crate::compiler::CompiledModule<S>, machine_code_offset: u64) -> Option<MemsetKind>
where
    S: Sandbox,
{
    if machine_code_offset >= compiled_module.memset_trampoline_start && machine_code_offset < compiled_module.memset_trampoline_end {
        Some(MemsetKind::Trampoline)
    } else if compiled_module
        .machine_code()
        .get(machine_code_offset as usize..)
        .map_or(false, |slice| slice.starts_with(REP_STOSB_MACHINE_CODE))
    {
        Some(MemsetKind::Inline)
    } else {
        None
    }
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

fn handle_interruption_during_memset<S>(
    memset_kind: MemsetKind,
    compiled_module: &crate::compiler::CompiledModule<S>,
    is_gas_metering_enabled: bool,
    machine_code_offset: u64,
    vmctx: &VmCtx,
) -> Result<(), &'static str>
where
    S: Sandbox,
{
    let bytes_remaining = vmctx.tmp_reg.load(Ordering::Relaxed);

    // Move the count into the non-temporary register.
    match memset_kind {
        MemsetKind::Inline => vmctx.regs[Reg::A2 as usize].store(bytes_remaining, Ordering::Relaxed),
        MemsetKind::Trampoline => {
            vmctx.regs[Reg::A2 as usize].fetch_add(bytes_remaining, Ordering::Relaxed);
        }
    }

    if is_gas_metering_enabled {
        // Give back the gas that we've pre-charged.
        vmctx.gas.fetch_add(cast(bytes_remaining).to_signed(), Ordering::Relaxed);
    }

    let original_offset = match memset_kind {
        MemsetKind::Inline => machine_code_offset,
        MemsetKind::Trampoline => vmctx.next_native_program_counter.load(Ordering::Relaxed) - compiled_module.native_code_origin,
    };

    set_program_counter_after_interruption(compiled_module, original_offset, vmctx)?;

    Ok(())
}

impl<'r, 'a, S, B> ArchVisitor<'r, 'a, S, B>
where
    S: Sandbox,
    B: CompilerBitness,
{
    pub const PADDING_BYTE: u8 = 0x90; // NOP

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
            Bitness::B32 => RegSize::R32,
            Bitness::B64 => RegSize::R64,
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn store(&mut self, src: impl Into<RegImm>, base: Option<RawReg>, offset: u32, kind: Size) {
        let src = src.into();
        load_store_operand!(self, S::KIND, base, offset, |dst| {
            match src {
                RegImm::Reg(src) => self.push(store(kind, dst, conv_reg(src))),
                RegImm::Imm(value) => match kind {
                    Size::U8 => self.push(mov_imm(dst, imm8(value as u8))),
                    Size::U16 => self.push(mov_imm(dst, imm16(value as u16))),
                    Size::U32 => self.push(mov_imm(dst, imm32(value))),
                    Size::U64 => {
                        assert_eq!(B::BITNESS, Bitness::B64);
                        self.push(mov_imm(dst, imm64(cast(value).to_signed())));
                    }
                },
            }
        });
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn load(&mut self, dst: RawReg, base: Option<RawReg>, offset: u32, kind: LoadKind) {
        load_store_operand!(self, S::KIND, base, offset, |src| {
            self.push(load(kind, conv_reg(dst), src));
        });
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn clear_reg(&mut self, reg: RawReg) {
        let reg = conv_reg(reg);
        self.push(xor((RegSize::R32, reg, reg)));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn fill_with_ones(&mut self, reg: RawReg) {
        match self.reg_size() {
            RegSize::R32 => {
                self.push(mov_imm(conv_reg(reg), imm32(0xffffffff)));
            }
            RegSize::R64 => {
                self.push(mov_imm(conv_reg(reg), imm64(cast(0xffffffff_u32).to_signed())));
            }
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn compare_reg_reg(&mut self, d: RawReg, s1: RawReg, s2: RawReg, condition: Condition) {
        let reg_size = self.reg_size();
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        let d = conv_reg(d);
        let asm = self.asm.reserve::<U3>();
        if d == s1 || d == s2 {
            let asm = asm.push(cmp((reg_size, s1, s2)));
            let asm = asm.push(setcc(condition, d));
            asm.push(and((d, imm32(1))));
        } else {
            let asm = asm.push(xor((RegSize::R32, d, d)));
            let asm = asm.push(cmp((reg_size, s1, s2)));
            asm.push(setcc(condition, d));
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn compare_reg_imm(&mut self, d: RawReg, s1: RawReg, s2: u32, condition: Condition) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U4>();
        let asm = if d != s1 {
            asm.push(xor((RegSize::R32, d, d)))
        } else {
            asm.push_none()
        };

        let asm = if condition == Condition::Below && s2 == 1 {
            // d = s1 <u 1  =>  d = s1 == 0
            let asm = asm.push(test((reg_size, s1, s1)));
            asm.push(setcc(Condition::Equal, d))
        } else if condition == Condition::Above && s2 == 0 {
            // d = s1 >u 0  =>  d = s1 != 0
            let asm = asm.push(test((reg_size, s1, s1)));
            asm.push(setcc(Condition::NotEqual, d))
        } else {
            let asm = match reg_size {
                RegSize::R32 => asm.push(cmp((s1, imm32(s2)))),
                RegSize::R64 => asm.push(cmp((s1, imm64(s2 as i32)))),
            };
            asm.push(setcc(condition, d))
        };

        let asm = asm.push_if(d == s1, and((d, imm32(1))));
        asm.assert_reserved_exactly_as_needed();
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn shift_imm(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, mut s2: u32, kind: ShiftKind) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let asm = self.asm.reserve::<polkavm_assembler::U3>();

        s2 &= match reg_size {
            RegSize::R32 => 32 - 1,
            RegSize::R64 => 64 - 1,
        };

        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));

        // d = d << s2
        let asm = match kind {
            ShiftKind::LogicalLeft => asm.push(shl_imm(reg_size, d, s2 as u8)),
            ShiftKind::LogicalRight => asm.push(shr_imm(reg_size, d, s2 as u8)),
            ShiftKind::ArithmeticRight => asm.push(sar_imm(reg_size, d, s2 as u8)),
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn shift(&mut self, reg_size: RegSize, d: RawReg, s1: impl Into<RegImm>, s2: RawReg, kind: ShiftKind) {
        let d = conv_reg(d);
        let s2 = conv_reg(s2);
        let asm = self.asm.reserve::<polkavm_assembler::U4>();

        // TODO: Consider using shlx/shrx/sarx when BMI2 is available.
        let asm = asm.push(mov(reg_size, rcx, s2));
        let asm = match s1.into() {
            RegImm::Reg(s1) => {
                let s1 = conv_reg(s1);
                asm.push_if(d != s1, mov(reg_size, d, s1))
            }
            RegImm::Imm(s1) => match reg_size {
                RegSize::R32 => asm.push(mov_imm(d, imm32(s1))),
                RegSize::R64 => asm.push(mov_imm(d, imm64(cast(s1).to_signed()))),
            },
        };

        // d = d << s2
        let asm = match kind {
            ShiftKind::LogicalLeft => asm.push(shl_cl(reg_size, d)),
            ShiftKind::LogicalRight => asm.push(shr_cl(reg_size, d)),
            ShiftKind::ArithmeticRight => asm.push(sar_cl(reg_size, d)),
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed()
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn mov(&mut self, dst: RawReg, src: RawReg) {
        self.push(mov(self.reg_size(), conv_reg(dst), conv_reg(src)))
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn jump_to_label(&mut self, label: Label) {
        let asm = self.asm.reserve::<U1>();
        jump_to_label(asm, label).assert_reserved_exactly_as_needed();
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn call_to_label(&mut self, label: Label) {
        self.push(call_label32(label));
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn branch(&mut self, s1: RawReg, s2: impl Into<RegImm>, target: u32, condition: Condition) {
        let reg_size = self.reg_size();
        let s1 = conv_reg(s1);
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);

        let asm = self.asm.reserve::<U2>();
        let asm = match s2.into() {
            RegImm::Reg(s2) => asm.push(cmp((reg_size, s1, conv_reg(s2)))),
            RegImm::Imm(s2) => match reg_size {
                RegSize::R32 => asm.push(cmp((s1, imm32(s2)))),
                RegSize::R64 => asm.push(cmp((s1, imm64(cast(s2).to_signed())))),
            },
        };

        branch_to_label(asm, condition, label).assert_reserved_exactly_as_needed();
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn cmov(&mut self, d: RawReg, s: RawReg, c: RawReg, condition: Condition) {
        if d == s {
            return;
        }

        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s = conv_reg(s);
        let c = conv_reg(c);

        let asm = self.asm.reserve::<U2>();
        let asm = asm.push(test((reg_size, c, c)));
        let asm = asm.push(cmov(condition, reg_size, d, s));
        asm.assert_reserved_exactly_as_needed();
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn cmov_imm(&mut self, d: RawReg, s: u32, c: RawReg, condition: Condition) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let c = conv_reg(c);

        let asm = self.asm.reserve::<U3>();
        let asm = asm.push(test((reg_size, c, c)));
        let asm = match reg_size {
            RegSize::R32 => asm.push(mov_imm(TMP_REG, imm32(s))),
            RegSize::R64 => asm.push(mov_imm(TMP_REG, imm64(cast(s).to_signed()))),
        };
        let asm = asm.push(cmov(condition, reg_size, d, TMP_REG));
        asm.assert_reserved_exactly_as_needed();
    }

    fn div_rem(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg, div_rem: DivRem, kind: Signedness) {
        // Unlike most other architectures RISC-V doesn't trap on division by zero
        // nor on division with overflow, and has well defined results in such cases.

        let label_divisor_is_zero = self.asm.forward_declare_label();
        let label_next = self.asm.forward_declare_label();

        self.push(test((reg_size, conv_reg(s2), conv_reg(s2))));
        self.push(jcc_label8(Condition::Equal, label_divisor_is_zero));

        if matches!(kind, Signedness::Signed) {
            let label_normal = self.asm.forward_declare_label();
            match reg_size {
                RegSize::R32 => {
                    self.push(cmp((conv_reg(s1), imm32(cast(i32::MIN).to_unsigned()))));
                    self.push(jcc_label8(Condition::NotEqual, label_normal));
                }
                RegSize::R64 => {
                    self.push(mov(RegSize::R64, TMP_REG, conv_reg(s1)));
                    self.push(cmp((TMP_REG, imm32(0))));
                    self.push(jcc_label8(Condition::NotEqual, label_normal));
                    self.push(shr_imm(RegSize::R64, TMP_REG, 32));
                    self.push(cmp((TMP_REG, imm32(cast(i32::MIN).to_unsigned()))));
                    self.push(jcc_label8(Condition::NotEqual, label_normal));
                }
            }

            match reg_size {
                RegSize::R32 => self.push(cmp((conv_reg(s2), imm32(cast(-1_i32).to_unsigned())))),
                RegSize::R64 => self.push(cmp((conv_reg(s2), imm64(-1_i32)))),
            }
            self.push(jcc_label8(Condition::NotEqual, label_normal));

            match div_rem {
                DivRem::Div => self.mov(d, s1),
                DivRem::Rem => self.clear_reg(d),
            }
            self.push(jmp_label8(label_next));
            self.define_label(label_normal);
        }

        // The division instruction always accepts the dividend and returns the result in rdx:rax.
        // This isn't great because we're using these registers for the VM's registers, so we need
        // to do all of this in such a way that those won't be accidentally overwritten.

        const _: () = {
            assert!(TMP_REG as u32 != rdx as u32);
            assert!(TMP_REG as u32 != rax as u32);
        };

        // Push the registers that will be clobbered.
        self.push(push(rdx));
        self.push(push(rax));

        // Push the operands.
        self.push(push(conv_reg(s1)));
        self.push(push(conv_reg(s2)));

        // Pop the divisor.
        self.push(pop(TMP_REG));

        // Pop the dividend.
        self.push(xor((RegSize::R32, rdx, rdx)));
        self.push(pop(rax));

        match kind {
            Signedness::Unsigned => self.push(div(reg_size, TMP_REG)),
            Signedness::Signed => {
                match reg_size {
                    RegSize::R32 => self.push(cdq()),
                    RegSize::R64 => self.push(cqo()),
                }
                self.push(idiv(reg_size, TMP_REG))
            }
        }

        // Move the result to the temporary register.
        match div_rem {
            DivRem::Div => self.push(mov(reg_size, TMP_REG, rax)),
            DivRem::Rem => self.push(mov(reg_size, TMP_REG, rdx)),
        }

        // Restore the original registers.
        self.push(pop(rax));
        self.push(pop(rdx));

        // Move the output into the destination registers.
        self.push(mov(reg_size, conv_reg(d), TMP_REG));

        // Go to the next instruction.
        self.push(jmp_label8(label_next));

        self.define_label(label_divisor_is_zero);
        match div_rem {
            DivRem::Div => self.fill_with_ones(d),
            DivRem::Rem if d == s1 => {}
            DivRem::Rem => self.mov(d, s1),
        }

        self.define_label(label_next);

        if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            self.push(movsxd_32_to_64(conv_reg(d), conv_reg(d)))
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn vmctx_field(offset: usize) -> MemOp {
        match S::KIND {
            SandboxKind::Linux => reg_indirect(RegSize::R64, LINUX_SANDBOX_VMCTX_REG + offset as i32),
            SandboxKind::Generic => {
                #[cfg(feature = "generic-sandbox")]
                {
                    let offset = crate::sandbox::generic::GUEST_MEMORY_TO_VMCTX_OFFSET as i32 + offset as i32;
                    reg_indirect(RegSize::R64, GENERIC_SANDBOX_MEMORY_REG + offset)
                }

                #[cfg(not(feature = "generic-sandbox"))]
                {
                    unreachable!();
                }
            }
        }
    }

    fn load_vmctx_field_address(&mut self, offset: usize) -> NativeReg {
        if offset == 0 && matches!(S::KIND, SandboxKind::Linux) {
            LINUX_SANDBOX_VMCTX_REG
        } else {
            self.push(lea(RegSize::R64, TMP_REG, Self::vmctx_field(offset)));
            TMP_REG
        }
    }

    fn save_registers_to_vmctx(&mut self) {
        let kind = match B::BITNESS {
            Bitness::B32 => Size::U32,
            Bitness::B64 => Size::U64,
        };

        for (nth, reg) in Reg::ALL.iter().copied().enumerate() {
            self.push(store(
                kind,
                Self::vmctx_field(S::offset_table().regs + nth * 8),
                conv_reg(reg.into()),
            ));
        }
    }

    fn restore_registers_from_vmctx(&mut self) {
        let kind = match B::BITNESS {
            Bitness::B32 => LoadKind::U32,
            Bitness::B64 => LoadKind::U64,
        };

        for (nth, reg) in Reg::ALL.iter().copied().enumerate() {
            self.push(load(
                kind,
                conv_reg(reg.into()),
                Self::vmctx_field(S::offset_table().regs + nth * 8),
            ));
        }
    }

    fn save_return_address_to_vmctx(&mut self) {
        self.push(load(LoadKind::U64, TMP_REG, reg_indirect(RegSize::R64, rsp)));
        self.push(store(
            Size::U64,
            Self::vmctx_field(S::offset_table().next_native_program_counter),
            TMP_REG,
        ));
    }

    pub(crate) fn emit_sysenter(&mut self) -> Label {
        log::trace!("Emitting trampoline: sysenter");
        let label = self.asm.create_label();

        if matches!(S::KIND, SandboxKind::Linux) {
            self.push(mov_imm64(LINUX_SANDBOX_VMCTX_REG, VM_ADDR_VMCTX));
        }
        self.restore_registers_from_vmctx();
        self.push(jmp(Self::vmctx_field(S::offset_table().next_native_program_counter)));

        label
    }

    pub(crate) fn emit_sysreturn(&mut self) -> Label {
        log::trace!("Emitting trampoline: sysreturn");
        let label = self.asm.create_label();

        self.push(mov_imm(Self::vmctx_field(S::offset_table().next_native_program_counter), imm64(0)));
        self.save_registers_to_vmctx();
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_return));
        self.push(jmp(TMP_REG));

        label
    }

    pub(crate) fn emit_ecall_trampoline(&mut self) {
        log::trace!("Emitting trampoline: ecall");
        let label = self.ecall_label;
        self.define_label(label);

        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_hostcall));
        self.push(jmp(TMP_REG));
    }

    pub(crate) fn emit_step_trampoline(&mut self) {
        log::trace!("Emitting trampoline: step");
        let label = self.step_label;
        self.define_label(label);

        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_step));
        self.push(jmp(TMP_REG));
    }

    pub(crate) fn emit_trap_trampoline(&mut self) {
        log::trace!("Emitting trampoline: trap");
        let label = self.trap_label;
        self.define_label(label);

        self.save_registers_to_vmctx();
        self.push(mov_imm(Self::vmctx_field(S::offset_table().next_native_program_counter), imm64(0)));
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_trap));
        self.push(jmp(TMP_REG));
    }

    pub(crate) fn emit_sbrk_trampoline(&mut self) {
        log::trace!("Emitting trampoline: sbrk");
        let label = self.sbrk_label;
        self.define_label(label);

        self.push(push(TMP_REG));
        self.save_registers_to_vmctx();
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_sbrk));
        self.push(pop(rdi));
        self.push(call(TMP_REG));
        self.push(push(rax));
        self.restore_registers_from_vmctx();
        self.push(pop(TMP_REG));
        self.push(ret());
    }

    // This is a slower memset implementation when we're running with gas metering enabled
    // and we don't have enough gas to finish running the whole memset.
    pub(crate) fn emit_memset_trampoline(&mut self) {
        log::trace!("Emitting trampoline: memset");
        let label = self.memset_label;
        self.define_label(label);

        let count = conv_reg(Reg::A2.into());

        // // Store the address to restart the memset in case we trigger a page fault.
        // self.push(store(
        //     RegSize::R64,
        //     Self::vmctx_field(S::offset_table().next_native_program_counter),
        //     rcx,
        // ));

        // Grab the amount of gas we have (this will always be negative), and zero the gas counter.
        // (We assume the memset will consume all of the gas.)
        self.push(xor((RegSize::R32, rcx, rcx)));
        self.push(xchg_mem(RegSize::R64, rcx, Self::vmctx_field(S::offset_table().gas)));

        // Calculate the number of bytes we can memset given our gas budget.
        self.push(add((RegSize::R64, rcx, count)));

        // Set the final counter to how much bytes will be remaining when we run out of gas.
        self.push(sub((RegSize::R64, count, rcx)));

        // Stash the counter, in case we page fault in the middle of the memset.
        self.push(store(RegSize::R64, Self::vmctx_field(S::offset_table().arg), rcx));

        // Execute the memset.
        self.asm.push_raw(REP_STOSB_MACHINE_CODE);

        // We've successfully finished memset without page faulting, so we can run out of gas.
        self.save_registers_to_vmctx();
        self.push(mov_imm64(TMP_REG, S::address_table().syscall_not_enough_gas));
        self.push(jmp(TMP_REG));
    }

    pub(crate) fn trace_execution(&mut self, code_offset: u32) {
        let step_label = self.step_label;
        let asm = self.asm.reserve::<U3>();
        let asm = asm.push(mov_imm(Self::vmctx_field(S::offset_table().program_counter), imm32(code_offset)));
        let asm = asm.push(mov_imm(
            Self::vmctx_field(S::offset_table().next_program_counter),
            imm32(code_offset),
        ));
        let asm = asm.push(call_label32(step_label));
        asm.assert_reserved_exactly_as_needed();
    }

    pub(crate) fn emit_gas_metering_stub(&mut self, kind: GasMeteringKind) {
        let origin = self.asm.len();

        self.push(sub((Self::vmctx_field(S::offset_table().gas), imm64(i32::MAX))));
        debug_assert_eq!(GAS_COST_OFFSET, self.asm.len() - origin - 4); // Offset to bring us from the start of the stub to the gas cost.

        if matches!(kind, GasMeteringKind::Sync) {
            // 49833F00             cmp qword [r15],0
            self.push(cmp((Self::vmctx_field(S::offset_table().gas), imm64(0))));

            // This will jump two bytes backwards to "3f00", and 3f corresponds to the AAS instruction
            // which is invalid in 64-bit, so it will trap with an SIGILL.
            //
            // Note that this is technically a forward-compatibility hazard as this opcode could arguably
            // be reused for something in the future.
            assert_eq!(Self::vmctx_field(S::offset_table().gas), reg_indirect(RegSize::R64, r15 + 0)); // Sanity check.
            debug_assert!(self.asm.code_mut().ends_with(&[0x49, 0x83, 0x3F, 0x00]));
            // Offset to bring us from where the trap will trigger to the beginning of the stub.
            debug_assert_eq!(GAS_METERING_TRAP_OFFSET, (self.asm.len() - origin - 2) as u64);
            self.asm.push_raw(&[0x78, 0xfc]);
        }
    }

    pub(crate) fn emit_weight(&mut self, offset: usize, cost: u32) {
        let length = sub((Self::vmctx_field(S::offset_table().gas), imm64(i32::MAX))).len();
        let xs = cost.to_le_bytes();
        self.asm.code_mut()[offset + length - 4..offset + length].copy_from_slice(&xs);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn jump_indirect_impl(&mut self, load_imm: Option<(RawReg, u32)>, base: RawReg, offset: u32) {
        match S::KIND {
            SandboxKind::Linux => {
                use polkavm_assembler::amd64::{Scale, SegReg};

                let asm = self.asm.reserve::<U3>();
                let (asm, target) = if offset != 0 || load_imm.map_or(false, |(t, _)| t == base) {
                    let asm = asm.push(lea(
                        RegSize::R32,
                        TMP_REG,
                        reg_indirect(RegSize::R32, conv_reg(base) + offset as i32),
                    ));
                    (asm, TMP_REG)
                } else {
                    (asm.push_none(), conv_reg(base))
                };

                let asm = if let Some((return_register, return_address)) = load_imm {
                    match B::BITNESS {
                        Bitness::B32 => asm.push(mov_imm(conv_reg(return_register), imm32(return_address))),
                        Bitness::B64 => asm.push(mov_imm(conv_reg(return_register), imm64(cast(return_address).to_signed()))),
                    }
                } else {
                    asm.push_none()
                };

                let asm = asm.push(jmp(MemOp::IndexScaleOffset(Some(SegReg::gs), RegSize::R64, target, Scale::x8, 0)));
                asm.assert_reserved_exactly_as_needed();
            }
            SandboxKind::Generic => {
                // TODO: This also could be more efficient.
                self.push(lea_rip_label(TMP_REG, self.jump_table_label));
                self.push(push(conv_reg(base)));
                self.push(shl_imm(RegSize::R64, conv_reg(base), 3));
                if offset > 0 {
                    let offset = offset.wrapping_mul(8);
                    self.push(add((conv_reg(base), imm32(offset))));
                }
                self.push(add((RegSize::R64, TMP_REG, conv_reg(base))));
                self.push(pop(conv_reg(base)));
                self.push(load(LoadKind::U64, TMP_REG, reg_indirect(RegSize::R64, TMP_REG)));

                if let Some((return_register, return_address)) = load_imm {
                    match B::BITNESS {
                        Bitness::B32 => self.push(mov_imm(conv_reg(return_register), imm32(return_address))),
                        Bitness::B64 => self.push(mov_imm(conv_reg(return_register), imm64(cast(return_address).to_signed()))),
                    }
                }

                self.push(jmp(TMP_REG));
            }
        }
    }

    #[inline(always)]
    pub fn trap(&mut self, code_offset: u32) {
        let trap_label = self.trap_label;
        let asm = self.asm.reserve::<U2>();
        let asm = asm.push(mov_imm(Self::vmctx_field(S::offset_table().program_counter), imm32(code_offset)));
        let asm = asm.push(call_label32(trap_label));
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn trap_without_modifying_program_counter(&mut self) {
        let trap_label = self.trap_label;
        let asm = self.asm.reserve::<U1>();
        let asm = asm.push(call_label32(trap_label));
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn and_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        // todo: change this with ANDN instruction

        let asm = self.asm.reserve::<U3>();
        if d == s1 {
            // d = d & ~s2
            let asm = asm.push(mov(reg_size, TMP_REG, s2));
            let asm = asm.push(not(reg_size, TMP_REG));
            asm.push(and((reg_size, d, TMP_REG)))
        } else if d == s2 {
            // d = s1 & ~d
            let asm = asm.push(not(reg_size, s2));
            let asm = asm.push(and((reg_size, d, s1)));
            asm.push_none()
        } else {
            // d = s1 & ~s2
            let asm = asm.push(mov(reg_size, d, s2));
            let asm = asm.push(not(reg_size, d));
            asm.push(and((reg_size, d, s1)))
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn or_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U3>();
        if d == s1 {
            // d = d & ~s2
            let asm = asm.push(mov(reg_size, TMP_REG, s2));
            let asm = asm.push(not(reg_size, TMP_REG));
            asm.push(or((reg_size, d, TMP_REG)))
        } else if d == s2 {
            // d = s1 & ~d
            let asm = asm.push(not(reg_size, s2));
            let asm = asm.push(or((reg_size, d, s1)));
            asm.push_none()
        } else {
            // d = s1 & ~s2
            let asm = asm.push(mov(reg_size, d, s2));
            let asm = asm.push(not(reg_size, d));
            asm.push(or((reg_size, d, s1)))
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn xnor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        self.xor(d, s1, s2);
        let asm = self.asm.reserve::<U1>();
        asm.push(not(reg_size, conv_reg(d))).assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn min_max_generic(&mut self, c: Condition, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U3>();
        if d == s1 {
            let asm = asm.push(cmp((reg_size, s2, s1)));
            asm.push(cmov(c, reg_size, d, s2)).push_none()
        } else if d == s2 {
            let asm = asm.push(cmp((reg_size, s1, s2)));
            asm.push(cmov(c, reg_size, d, s1)).push_none()
        } else {
            let asm = asm.push(mov(reg_size, d, s1));
            let asm = asm.push(cmp((reg_size, s2, s1)));
            asm.push(cmov(c, reg_size, d, s2))
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn maximum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.min_max_generic(Condition::Greater, d, s1, s2);
    }

    #[inline(always)]
    pub fn maximum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.min_max_generic(Condition::Above, d, s1, s2);
    }

    #[inline(always)]
    pub fn minimum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.min_max_generic(Condition::Less, d, s1, s2);
    }

    #[inline(always)]
    pub fn minimum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.min_max_generic(Condition::Below, d, s1, s2);
    }

    #[inline(always)]
    pub fn rotate_left_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<polkavm_assembler::U4>();
        let asm = asm.push(mov(reg_size, rcx, s2));
        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));
        let asm = asm.push(rol_cl(reg_size, d));

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn rotate_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.rotate_left_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn rotate_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.rotate_left_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    pub fn rotate_right_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<polkavm_assembler::U4>();
        let asm = asm.push(mov(reg_size, rcx, s2));
        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));
        let asm = asm.push(ror_cl(reg_size, d));

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn rotate_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.rotate_right_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn rotate_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.rotate_right_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    #[cold]
    pub fn invalid(&mut self, code_offset: u32) {
        log::debug!("Encountered invalid instruction");
        self.trap(code_offset)
    }

    #[allow(clippy::unused_self)]
    #[inline(always)]
    pub fn fallthrough(&mut self) {}

    #[inline(always)]
    pub fn sbrk(&mut self, dst: RawReg, size: RawReg) {
        let label_bump_only = self.asm.forward_declare_label();
        let label_continue = self.asm.forward_declare_label();
        let sbrk_label = self.sbrk_label;

        let dst = conv_reg(dst);
        let size = conv_reg(size);
        if dst != size {
            self.push(mov(RegSize::R32, dst, size));
        }

        let offset = S::offset_table().heap_info;
        let heap_info_base = self.load_vmctx_field_address(offset);

        // Calculate new top-of-the-heap pointer.
        self.push(add((RegSize::R64, dst, reg_indirect(RegSize::R64, heap_info_base))));
        // Compare it to the current threshold.
        self.push(cmp((RegSize::R64, dst, reg_indirect(RegSize::R64, heap_info_base + 8))));
        // If it was less or equal to the threshold then no extra action is necessary (bump only!).
        self.push(jcc_label8(Condition::BelowOrEqual, label_bump_only));

        // The new top-of-the-heap pointer crossed the threshold, so more involved handling is necessary.
        // We'll either allocate new memory, or return a null pointer.
        self.push(mov(RegSize::R64, TMP_REG, dst));
        self.call_to_label(sbrk_label);
        self.push(mov(RegSize::R32, dst, TMP_REG));
        // Note: `dst` can be zero here, which is why we do the pointer bump from within the handler.
        self.push(jmp_label8(label_continue));

        self.define_label(label_bump_only);
        // Only a bump was necessary, so just updated the pointer and continue.
        self.push(store(RegSize::R64, reg_indirect(RegSize::R64, heap_info_base), dst));

        self.define_label(label_continue);
    }

    #[inline(always)]
    pub fn memset(&mut self) {
        let reg_size = self.reg_size();

        const _: () = {
            assert!(TMP_REG as u32 == rcx as u32);
            assert!(conv_reg_const(Reg::A0) as u32 == rdi as u32);
            assert!(conv_reg_const(Reg::A1) as u32 == rax as u32);
        };

        let label_repeat = self.asm.create_label();
        self.asm.push(lea_rip_label(rcx, label_repeat));

        // Store the address to restart the memset in case we trigger a page fault.
        self.push(store(
            RegSize::R64,
            Self::vmctx_field(S::offset_table().next_native_program_counter),
            rcx,
        ));

        let count = conv_reg(Reg::A2.into());
        self.asm.push(mov(RegSize::R32, count, count));

        match self.gas_metering {
            None => {
                self.asm.push(mov(reg_size, rcx, count));
                // rep stosb, rdi is destination pointer, rcx is count, rax is the value
                self.asm.push_raw(REP_STOSB_MACHINE_CODE);
                self.asm.push(mov(reg_size, count, rcx));
            }
            Some(GasMeteringKind::Sync) => {
                // Pre charge the gas cost of the memset.
                self.asm.push(sub((RegSize::R64, Self::vmctx_field(S::offset_table().gas), count)));
                // Will we have enough gas to finish the operation?
                self.asm.push(cmp((Self::vmctx_field(S::offset_table().gas), imm64(0))));
                // If no - jump to a slower version of the routine.
                let label_slow = self.memset_label;
                branch_to_label(self.asm.reserve::<U1>(), Condition::Less, label_slow);
                // If yes - do it the fast way.
                self.asm.push(mov(reg_size, rcx, count));
                self.asm.push_raw(REP_STOSB_MACHINE_CODE);
                self.asm.push(mov(reg_size, count, rcx));
            }
            Some(GasMeteringKind::Async) => {
                self.asm.push(sub((RegSize::R64, Self::vmctx_field(S::offset_table().gas), count)));
                self.asm.push(mov(reg_size, rcx, count));
                self.asm.push_raw(REP_STOSB_MACHINE_CODE);
                self.asm.push(mov(reg_size, count, rcx));
            }
        }
    }

    pub fn on_signal_trap(
        compiled_module: &crate::compiler::CompiledModule<S>,
        is_gas_metering_enabled: bool,
        machine_code_offset: u64,
        vmctx: &VmCtx,
    ) -> Result<bool, &'static str> {
        enum TrapKind {
            Memset { kind: MemsetKind },
            NotEnoughGas,
            Trap,
        }

        let trap_kind = if let Some(kind) = are_we_executing_memset(compiled_module, machine_code_offset) {
            TrapKind::Memset { kind }
        } else if is_gas_metering_enabled && vmctx.gas.load(Ordering::Relaxed) < 0 {
            TrapKind::NotEnoughGas
        } else {
            TrapKind::Trap
        };

        match trap_kind {
            TrapKind::NotEnoughGas => {
                let Some(offset) = machine_code_offset.checked_sub(GAS_METERING_TRAP_OFFSET) else {
                    return Err("internal error: address underflow after a trap");
                };

                // If we restart we want to check the gas again, so set the address to point there.
                vmctx
                    .next_native_program_counter
                    .store(compiled_module.native_code_origin + offset, Ordering::Relaxed);

                // Also set the logical program counter, in case the user wants to stash the location
                // somewhere and continue some other time.
                let program_counter = set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;

                // The gas counter is now negative; refund the amount of gas consumed so that it's positive again.
                let offset = offset as usize + GAS_COST_OFFSET;
                let Some(gas_cost) = &compiled_module.machine_code().get(offset..offset + 4) else {
                    return Err("internal error: failed to read back the gas cost from the machine code");
                };

                let gas_cost = u32::from_le_bytes([gas_cost[0], gas_cost[1], gas_cost[2], gas_cost[3]]);
                let gas = vmctx.gas.fetch_add(i64::from(gas_cost), Ordering::Relaxed);
                log::trace!(
                    "Out of gas; program counter = {program_counter}, reverting gas: {gas} -> {new_gas} (gas cost: {gas_cost})",
                    new_gas = gas + i64::from(gas_cost)
                );

                Ok(true)
            }
            TrapKind::Trap => {
                set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;

                // Traps are unrecoverable.
                vmctx.next_native_program_counter.store(0, Ordering::Relaxed);

                Ok(false)
            }
            // Did we prematurely trigger a page fault during a memset?
            TrapKind::Memset { kind } => {
                handle_interruption_during_memset(kind, compiled_module, is_gas_metering_enabled, machine_code_offset, vmctx)?;

                // We can only be here if dynamic page faulting is disabled, so this situation is unrecoverable.
                vmctx.next_native_program_counter.store(0, Ordering::Relaxed);

                Ok(false)
            }
        }
    }

    pub fn on_page_fault(
        compiled_module: &crate::compiler::CompiledModule<S>,
        is_gas_metering_enabled: bool,
        machine_code_address: u64,
        machine_code_offset: u64,
        vmctx: &VmCtx,
    ) -> Result<(), &'static str> {
        if let Some(kind) = are_we_executing_memset(compiled_module, machine_code_offset) {
            handle_interruption_during_memset(kind, compiled_module, is_gas_metering_enabled, machine_code_offset, vmctx)?;
        } else {
            set_program_counter_after_interruption(compiled_module, machine_code_offset, vmctx)?;
            vmctx.next_native_program_counter.store(machine_code_address, Ordering::Relaxed);
        }

        Ok(())
    }

    #[inline(always)]
    pub fn ecalli(&mut self, code_offset: u32, args_length: u32, imm: u32) {
        let ecall_label = self.ecall_label;
        let asm = self.asm.reserve::<U4>();
        let asm = asm.push(mov_imm(Self::vmctx_field(S::offset_table().arg), imm32(imm)));
        let asm = asm.push(mov_imm(Self::vmctx_field(S::offset_table().program_counter), imm32(code_offset)));
        let asm = asm.push(mov_imm(
            Self::vmctx_field(S::offset_table().next_program_counter),
            imm32(code_offset + args_length + 1),
        ));
        let asm = asm.push(call_label32(ecall_label));
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn set_less_than_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.compare_reg_reg(d, s1, s2, Condition::Below);
    }

    #[inline(always)]
    pub fn set_less_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.compare_reg_imm(d, s1, s2, Condition::Below);
    }

    #[inline(always)]
    pub fn set_greater_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.compare_reg_imm(d, s1, s2, Condition::Above);
    }

    #[inline(always)]
    pub fn set_less_than_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.compare_reg_reg(d, s1, s2, Condition::Less);
    }

    #[inline(always)]
    pub fn set_less_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.compare_reg_imm(d, s1, s2, Condition::Less);
    }

    #[inline(always)]
    pub fn set_greater_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.compare_reg_imm(d, s1, s2, Condition::Greater);
    }

    #[inline(always)]
    pub fn shift_logical_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_logical_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_logical_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn shift_logical_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn shift_logical_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_logical_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_logical_left_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R32, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn shift_logical_left_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: u32) {
        self.shift(RegSize::R64, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn xor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U2>();
        match (d, s1, s2) {
            // d = d ^ s2
            (_, _, _) if d == s1 => asm.push(xor((reg_size, d, s2))).push_none(),
            // d = s1 ^ d
            (_, _, _) if d == s2 => asm.push(xor((reg_size, d, s1))).push_none(),
            // d = s1 ^ s2
            _ => {
                let asm = asm.push(mov(reg_size, d, s1));
                asm.push(xor((reg_size, d, s2)))
            }
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn and(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U2>();
        match (d, s1, s2) {
            // d = d & s2
            (_, _, _) if d == s1 => asm.push(and((reg_size, d, s2))).push_none(),
            // d = s1 & d
            (_, _, _) if d == s2 => asm.push(and((reg_size, d, s1))).push_none(),
            // d = s1 & s2
            _ => {
                let asm = asm.push(mov(reg_size, d, s1));
                asm.push(and((reg_size, d, s2)))
            }
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn or(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U2>();
        match (d, s1, s2) {
            // d = d | s2
            (_, _, _) if d == s1 => asm.push(or((reg_size, d, s2))).push_none(),
            // d = s1 | d
            (_, _, _) if d == s2 => asm.push(or((reg_size, d, s1))).push_none(),
            // d = s1 | s2
            _ => {
                let asm = asm.push(mov(reg_size, d, s1));
                asm.push(or((reg_size, d, s2)))
            }
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    fn add_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U3>();
        let asm = match (d, s1, s2) {
            // d = d + s2
            (_, _, _) if d == s1 => asm.push(add((reg_size, d, s2))).push_none(),
            // d = s1 + d
            (_, _, _) if d == s2 => asm.push(add((reg_size, d, s1))).push_none(),
            // d = s1 + s2
            _ => {
                let asm = asm.push_if(d != s1, mov(reg_size, d, s1));
                asm.push(add((reg_size, d, s2)))
            }
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn add_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.add_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn add_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.add_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    fn sub_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U3>();
        let asm = match (d, s1, s2) {
            // d = d - s2
            (_, _, _) if d == s1 => asm.push(sub((reg_size, d, s2))).push_none(),
            // d = s1 - d
            (_, _, _) if d == s2 => {
                let asm = asm.push(neg(reg_size, d));
                asm.push(add((reg_size, d, s1)))
            }
            // d = s1 - s2
            _ => {
                let asm = asm.push(mov(reg_size, d, s1));
                asm.push(sub((reg_size, d, s2)))
            }
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn sub_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.sub_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn sub_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.sub_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    fn negate_and_add_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: u32) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U3>();
        let asm = if d == s1 {
            // d = -d + s2
            let asm = asm.push(neg(reg_size, d));
            if s2 != 0 {
                match reg_size {
                    RegSize::R32 => asm.push(add((d, imm32(s2)))),
                    RegSize::R64 => asm.push(add((d, imm64(cast(s2).to_signed())))),
                }
            } else {
                asm.push_none()
            }
        } else {
            // d = -s1 + s2  =>  d = s2 - s1
            if s2 == 0 {
                let asm = asm.push(mov(reg_size, d, s1));
                asm.push(neg(reg_size, d))
            } else {
                let asm = match reg_size {
                    RegSize::R32 => asm.push(mov_imm(d, imm32(s2))),
                    RegSize::R64 => asm.push(mov_imm(d, imm64(cast(s2).to_signed()))),
                };
                asm.push(sub((reg_size, d, s1)))
            }
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn negate_and_add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.negate_and_add_imm_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn negate_and_add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.negate_and_add_imm_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    fn mul_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);

        let asm = self.asm.reserve::<U3>();
        let asm = if d == s1 {
            // d = d * s2
            asm.push(imul(reg_size, d, s2)).push_none()
        } else if d == s2 {
            // d = s1 * d
            asm.push(imul(reg_size, d, s1)).push_none()
        } else {
            // d = s1 * s2
            let asm = asm.push(mov(reg_size, d, s1));
            asm.push(imul(reg_size, d, s2))
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn mul_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.mul_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn mul_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.mul_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    pub fn mul_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U2>();
        let asm = asm.push(imul_imm(RegSize::R32, d, s1, s2 as i32));

        let asm = if B::BITNESS == Bitness::B64 {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn mul_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.push(imul_imm(RegSize::R64, conv_reg(d), conv_reg(s1), s2 as i32));
    }

    #[inline(always)]
    pub fn mul_upper_signed_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        match B::BITNESS {
            Bitness::B32 => {
                let asm = self.asm.reserve::<U4>();
                let asm = asm.push(movsxd_32_to_64(TMP_REG, s2));
                let asm = asm.push(movsxd_32_to_64(d, s1));
                let asm = asm.push(imul(RegSize::R64, d, TMP_REG));
                let asm = asm.push(shr_imm(RegSize::R64, d, 32));
                asm.assert_reserved_exactly_as_needed();
            }
            Bitness::B64 => {
                const _: () = {
                    assert!(TMP_REG as u32 != rdx as u32);
                    assert!(TMP_REG as u32 != rax as u32);
                };

                self.push(push(rax));
                self.push(push(rdx));
                match (s1 == rax, s2 == rax) {
                    (true, true) => self.push(imul_dx_ax(RegSize::R64, rax)),
                    (false, true) => self.push(imul_dx_ax(RegSize::R64, s1)),
                    (true, false) => self.push(imul_dx_ax(RegSize::R64, s2)),
                    (false, false) => {
                        self.push(mov(RegSize::R64, rax, s1));
                        self.push(imul_dx_ax(RegSize::R64, s2));
                    }
                }
                self.push(mov(RegSize::R64, TMP_REG, rdx));
                self.push(pop(rdx));
                self.push(pop(rax));
                self.push(mov(RegSize::R64, d, TMP_REG));
            }
        }
    }

    #[inline(always)]
    pub fn mul_upper_unsigned_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        match B::BITNESS {
            Bitness::B32 => {
                let asm = self.asm.reserve::<U3>();
                let asm = if d == s1 {
                    // d = d * s2
                    asm.push(imul(RegSize::R64, d, s2)).push_none()
                } else if d == s2 {
                    // d = s1 * d
                    asm.push(imul(RegSize::R64, d, s1)).push_none()
                } else {
                    // d = s1 * s2
                    let asm = asm.push(mov(RegSize::R32, d, s1));
                    asm.push(imul(RegSize::R64, d, s2))
                };

                let asm = asm.push(shr_imm(RegSize::R64, d, 32));
                asm.assert_reserved_exactly_as_needed();
            }
            Bitness::B64 => {
                const _: () = {
                    assert!(TMP_REG as u32 != rdx as u32);
                    assert!(TMP_REG as u32 != rax as u32);
                };

                self.push(push(rax));
                self.push(push(rdx));
                match (s1 == rax, s2 == rax) {
                    (true, true) => self.push(mul_dx_ax(RegSize::R64, rax)),
                    (false, true) => self.push(mul_dx_ax(RegSize::R64, s1)),
                    (true, false) => self.push(mul_dx_ax(RegSize::R64, s2)),
                    (false, false) => {
                        self.push(mov(RegSize::R64, rax, s1));
                        self.push(mul(RegSize::R64, s2));
                    }
                }
                self.push(mov(RegSize::R64, TMP_REG, rdx));
                self.push(pop(rdx));
                self.push(pop(rax));
                self.push(mov(RegSize::R64, d, TMP_REG));
            }
        }
    }

    #[inline(always)]
    pub fn mul_upper_signed_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // This instruction is equivalent to:
        //   let s1: i32;
        //   let s2: u32;
        //   let s1: i64 = s1 as i64;
        //   let s2: i64 = s2 as u64 as i64;
        //   let d: u32 = ((s1 * s2) >> 32) as u32;
        //
        // So, basically:
        //   1) sign-extend the s1 to 64-bits,
        //   2) zero-extend the s2 to 64-bits,
        //   3) multiply,
        //   4) return the upper 32-bits.

        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        match B::BITNESS {
            Bitness::B32 => {
                let asm = self.asm.reserve::<U4>();
                let asm = if d == s2 {
                    // d = s1 * d
                    let asm = asm.push(mov(RegSize::R32, TMP_REG, s2));
                    let asm = asm.push(movsxd_32_to_64(d, s1));
                    asm.push(imul(RegSize::R64, d, TMP_REG))
                } else {
                    // d = s1 * s2
                    let asm = asm.push(movsxd_32_to_64(d, s1));
                    let asm = asm.push(imul(RegSize::R64, d, s2));
                    asm.push_none()
                };

                let asm = asm.push(shr_imm(RegSize::R64, d, 32));
                asm.assert_reserved_exactly_as_needed();
            }
            Bitness::B64 => {
                const _: () = {
                    assert!(TMP_REG as u32 != rdx as u32);
                    assert!(TMP_REG as u32 != rax as u32);
                };

                // TODO: This is not the most efficient implementation. We can optimize this.
                self.push(push(AUX_TMP_REG));
                self.push(push(rax));
                self.push(push(rdx));

                self.push(mov(RegSize::R64, TMP_REG, s1));
                self.push(mov(RegSize::R64, AUX_TMP_REG, s2));

                self.push(mov(RegSize::R64, rax, s2));
                self.push(mul(RegSize::R64, TMP_REG));
                self.push(sar_imm(RegSize::R64, TMP_REG, 63));
                self.push(imul(RegSize::R64, TMP_REG, AUX_TMP_REG));
                self.push(lea(RegSize::R64, TMP_REG, base_index(RegSize::R64, TMP_REG, rdx)));

                self.push(pop(rdx));
                self.push(pop(rax));
                self.push(pop(AUX_TMP_REG));
                self.push(mov(RegSize::R64, d, TMP_REG));
            }
        }
    }

    #[inline(always)]
    pub fn div_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.div_rem(RegSize::R32, d, s1, s2, DivRem::Div, Signedness::Unsigned);
    }

    #[inline(always)]
    pub fn div_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.div_rem(RegSize::R64, d, s1, s2, DivRem::Div, Signedness::Unsigned);
    }

    #[inline(always)]
    pub fn div_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.div_rem(RegSize::R32, d, s1, s2, DivRem::Div, Signedness::Signed);
    }

    #[inline(always)]
    pub fn div_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.div_rem(RegSize::R64, d, s1, s2, DivRem::Div, Signedness::Signed);
    }

    #[inline(always)]
    pub fn rem_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.div_rem(RegSize::R32, d, s1, s2, DivRem::Rem, Signedness::Unsigned);
    }

    #[inline(always)]
    pub fn rem_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.div_rem(RegSize::R64, d, s1, s2, DivRem::Rem, Signedness::Unsigned);
    }

    #[inline(always)]
    pub fn rem_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.div_rem(RegSize::R32, d, s1, s2, DivRem::Rem, Signedness::Signed);
    }

    #[inline(always)]
    pub fn rem_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.div_rem(RegSize::R64, d, s1, s2, DivRem::Rem, Signedness::Signed);
    }

    #[inline(always)]
    pub fn shift_logical_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R32, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_logical_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R64, d, s1, s2, ShiftKind::LogicalRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R32, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_arithmetic_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R64, d, s1, s2, ShiftKind::ArithmeticRight);
    }

    #[inline(always)]
    pub fn shift_logical_left_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R32, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn shift_logical_left_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.shift_imm(RegSize::R64, d, s1, s2, ShiftKind::LogicalLeft);
    }

    #[inline(always)]
    pub fn or_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U2>();
        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));

        // d = s1 | s2
        let asm = match reg_size {
            RegSize::R32 => asm.push(or((d, imm32(s2)))),
            RegSize::R64 => asm.push(or((d, imm64(cast(s2).to_signed())))),
        };
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn and_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U2>();
        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));

        // d = s1 & s2
        let asm = match reg_size {
            RegSize::R32 => asm.push(and((d, imm32(s2)))),
            RegSize::R64 => asm.push(and((d, imm64(cast(s2).to_signed())))),
        };
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn xor_imm(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        let reg_size = self.reg_size();
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let asm = self.asm.reserve::<U2>();
        let asm = asm.push_if(d != s1, mov(reg_size, d, s1));

        if s2 != !0 {
            // d = s1 ^ s2
            match reg_size {
                RegSize::R32 => asm.push(xor((d, imm32(s2)))),
                RegSize::R64 => asm.push(xor((d, imm64(cast(s2).to_signed())))),
            }
        } else {
            // d = s1 ^ 0xfffffff
            asm.push(not(reg_size, d))
        }
        .assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn load_imm(&mut self, dst: RawReg, s2: u32) {
        match B::BITNESS {
            Bitness::B32 => self.push(mov_imm(conv_reg(dst), imm32(s2))),
            Bitness::B64 => self.push(mov_imm(conv_reg(dst), imm64(cast(s2).to_signed()))),
        }
    }

    #[inline(always)]
    pub fn load_imm64(&mut self, dst: RawReg, s2: u64) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.push(mov_imm64(conv_reg(dst), s2));
    }

    #[inline(always)]
    pub fn move_reg(&mut self, d: RawReg, s: RawReg) {
        self.mov(d, s);
    }

    #[inline(always)]
    pub fn count_leading_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
        self.push(lzcnt(RegSize::R32, conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn count_leading_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.push(lzcnt(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn count_trailing_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
        self.push(tzcnt(RegSize::R32, conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn count_trailing_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.push(tzcnt(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn count_set_bits_32(&mut self, d: RawReg, s: RawReg) {
        self.push(popcnt(RegSize::R32, conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn count_set_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.push(popcnt(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn sign_extend_8(&mut self, d: RawReg, s: RawReg) {
        self.push(movsx_8_to_64(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn sign_extend_16(&mut self, d: RawReg, s: RawReg) {
        self.push(movsx_16_to_64(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn zero_extend_16(&mut self, d: RawReg, s: RawReg) {
        self.push(movzx_16_to_64(self.reg_size(), conv_reg(d), conv_reg(s)))
    }

    #[inline(always)]
    pub fn reverse_byte(&mut self, d: RawReg, s: RawReg) {
        let reg_size = self.reg_size();
        let asm = self.asm.reserve::<U2>();
        let asm = asm.push_if(d != s, mov(reg_size, conv_reg(d), conv_reg(s)));
        let asm = asm.push(bswap(reg_size, conv_reg(d)));
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn cmov_if_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
        self.cmov(d, s, c, Condition::Equal);
    }

    #[inline(always)]
    pub fn cmov_if_not_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
        self.cmov(d, s, c, Condition::NotEqual);
    }

    #[inline(always)]
    pub fn cmov_if_zero_imm(&mut self, d: RawReg, c: RawReg, s: u32) {
        self.cmov_imm(d, s, c, Condition::Equal);
    }

    #[inline(always)]
    pub fn cmov_if_not_zero_imm(&mut self, d: RawReg, c: RawReg, s: u32) {
        self.cmov_imm(d, s, c, Condition::NotEqual);
    }

    #[inline(always)]
    pub fn rotate_right_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s: RawReg, c: u32) {
        let d = conv_reg(d);
        let s = conv_reg(s);

        let asm = self.asm.reserve::<polkavm_assembler::U4>();
        let asm = asm.push(mov_imm(rcx, imm32(c)));
        let asm = asm.push_if(d != s, mov(reg_size, d, s));
        let asm = asm.push(ror_cl(reg_size, d));

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn rotate_right_32_imm(&mut self, d: RawReg, s: RawReg, c: u32) {
        self.rotate_right_imm_generic(RegSize::R32, d, s, c);
    }

    #[inline(always)]
    pub fn rotate_right_64_imm(&mut self, d: RawReg, s: RawReg, c: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.rotate_right_imm_generic(RegSize::R64, d, s, c);
    }

    #[inline(always)]
    pub fn rotate_right_imm_alt_generic(&mut self, reg_size: RegSize, d: RawReg, s: RawReg, c: u32) {
        let d = conv_reg(d);
        let s = conv_reg(s);

        let asm = self.asm.reserve::<polkavm_assembler::U4>();
        let asm = asm.push(mov(reg_size, rcx, s));
        let asm = asm.push(mov_imm(d, imm32(c)));
        let asm = asm.push(ror_cl(reg_size, d));

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn rotate_right_32_imm_alt(&mut self, d: RawReg, s: RawReg, c: u32) {
        self.rotate_right_imm_alt_generic(RegSize::R32, d, s, c);
    }

    #[inline(always)]
    pub fn rotate_right_64_imm_alt(&mut self, d: RawReg, s: RawReg, c: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.rotate_right_imm_alt_generic(RegSize::R64, d, s, c);
    }

    #[inline(always)]
    fn add_imm_generic(&mut self, reg_size: RegSize, d: RawReg, s1: RawReg, s2: u32) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);

        let asm = self.asm.reserve::<U2>();
        let asm = if d == s1 {
            if s2 == 1 {
                asm.push(inc(reg_size, d))
            } else {
                match reg_size {
                    RegSize::R32 => asm.push(add((d, imm32(s2)))),
                    RegSize::R64 => asm.push(add((d, imm64(cast(s2).to_signed())))),
                }
            }
        } else {
            asm.push(lea(reg_size, d, reg_indirect(reg_size, s1 + s2 as i32)))
        };

        let asm = if (B::BITNESS, reg_size) == (Bitness::B64, RegSize::R32) {
            asm.push(movsxd_32_to_64(d, d))
        } else {
            asm.push_none()
        };

        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        self.add_imm_generic(RegSize::R32, d, s1, s2);
    }

    #[inline(always)]
    pub fn add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.add_imm_generic(RegSize::R64, d, s1, s2);
    }

    #[inline(always)]
    pub fn store_u8(&mut self, src: RawReg, offset: u32) {
        self.store(src, None, offset, Size::U8);
    }

    #[inline(always)]
    pub fn store_u16(&mut self, src: RawReg, offset: u32) {
        self.store(src, None, offset, Size::U16);
    }

    #[inline(always)]
    pub fn store_u32(&mut self, src: RawReg, offset: u32) {
        self.store(src, None, offset, Size::U32);
    }

    #[inline(always)]
    pub fn store_u64(&mut self, src: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.store(src, None, offset, Size::U64);
    }

    #[inline(always)]
    pub fn store_indirect_u8(&mut self, src: RawReg, base: RawReg, offset: u32) {
        self.store(src, Some(base), offset, Size::U8);
    }

    #[inline(always)]
    pub fn store_indirect_u16(&mut self, src: RawReg, base: RawReg, offset: u32) {
        self.store(src, Some(base), offset, Size::U16);
    }

    #[inline(always)]
    pub fn store_indirect_u32(&mut self, src: RawReg, base: RawReg, offset: u32) {
        self.store(src, Some(base), offset, Size::U32);
    }

    #[inline(always)]
    pub fn store_indirect_u64(&mut self, src: RawReg, base: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.store(src, Some(base), offset, Size::U64);
    }

    #[inline(always)]
    pub fn store_imm_indirect_u8(&mut self, base: RawReg, offset: u32, value: u32) {
        self.store(value, Some(base), offset, Size::U8);
    }

    #[inline(always)]
    pub fn store_imm_indirect_u16(&mut self, base: RawReg, offset: u32, value: u32) {
        self.store(value, Some(base), offset, Size::U16);
    }

    #[inline(always)]
    pub fn store_imm_indirect_u32(&mut self, base: RawReg, offset: u32, value: u32) {
        self.store(value, Some(base), offset, Size::U32);
    }

    #[inline(always)]
    pub fn store_imm_indirect_u64(&mut self, base: RawReg, offset: u32, value: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.store(value, Some(base), offset, Size::U64);
    }

    #[inline(always)]
    pub fn store_imm_u8(&mut self, offset: u32, value: u32) {
        self.store(value, None, offset, Size::U8);
    }

    #[inline(always)]
    pub fn store_imm_u16(&mut self, offset: u32, value: u32) {
        self.store(value, None, offset, Size::U16);
    }

    #[inline(always)]
    pub fn store_imm_u32(&mut self, offset: u32, value: u32) {
        self.store(value, None, offset, Size::U32);
    }

    #[inline(always)]
    pub fn store_imm_u64(&mut self, offset: u32, value: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.store(value, None, offset, Size::U64);
    }

    #[inline(always)]
    pub fn load_indirect_u8(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        self.load(dst, Some(base), offset, LoadKind::U8);
    }

    #[inline(always)]
    pub fn load_indirect_i8(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        self.load(dst, Some(base), offset, LoadKind::I8);
    }

    #[inline(always)]
    pub fn load_indirect_u16(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        self.load(dst, Some(base), offset, LoadKind::U16);
    }

    #[inline(always)]
    pub fn load_indirect_i16(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        self.load(dst, Some(base), offset, LoadKind::I16);
    }

    #[inline(always)]
    pub fn load_indirect_u32(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        // NOTE: For 32-bit the 'LoadKind::U32' is deliberate.
        self.load(dst, Some(base), offset, LoadKind::U32);
    }

    #[inline(always)]
    pub fn load_indirect_i32(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        let kind = match B::BITNESS {
            Bitness::B32 => LoadKind::U32,
            Bitness::B64 => LoadKind::I32,
        };
        self.load(dst, Some(base), offset, kind);
    }

    #[inline(always)]
    pub fn load_indirect_u64(&mut self, dst: RawReg, base: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.load(dst, Some(base), offset, LoadKind::U64);
    }

    #[inline(always)]
    pub fn load_u8(&mut self, dst: RawReg, offset: u32) {
        self.load(dst, None, offset, LoadKind::U8);
    }

    #[inline(always)]
    pub fn load_i8(&mut self, dst: RawReg, offset: u32) {
        self.load(dst, None, offset, LoadKind::I8);
    }

    #[inline(always)]
    pub fn load_u16(&mut self, dst: RawReg, offset: u32) {
        self.load(dst, None, offset, LoadKind::U16);
    }

    #[inline(always)]
    pub fn load_i16(&mut self, dst: RawReg, offset: u32) {
        self.load(dst, None, offset, LoadKind::I16);
    }

    #[inline(always)]
    pub fn load_i32(&mut self, dst: RawReg, offset: u32) {
        let kind = match B::BITNESS {
            Bitness::B32 => LoadKind::U32,
            Bitness::B64 => LoadKind::I32,
        };
        self.load(dst, None, offset, kind);
    }

    #[inline(always)]
    pub fn load_u32(&mut self, dst: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.load(dst, None, offset, LoadKind::U32);
    }

    #[inline(always)]
    pub fn load_u64(&mut self, dst: RawReg, offset: u32) {
        assert_eq!(B::BITNESS, Bitness::B64);
        self.load(dst, None, offset, LoadKind::U64);
    }

    #[inline(always)]
    pub fn branch_less_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::Below);
    }

    #[inline(always)]
    pub fn branch_less_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::Less);
    }

    #[inline(always)]
    pub fn branch_greater_or_equal_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::AboveOrEqual);
    }

    #[inline(always)]
    pub fn branch_greater_or_equal_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::GreaterOrEqual);
    }

    #[inline(always)]
    pub fn branch_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::Equal);
    }

    #[inline(always)]
    pub fn branch_not_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch(s1, s2, target, Condition::NotEqual);
    }

    #[inline(always)]
    pub fn branch_eq_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::Equal);
    }

    #[inline(always)]
    pub fn branch_not_eq_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::NotEqual);
    }

    #[inline(always)]
    pub fn branch_less_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::Below);
    }

    #[inline(always)]
    pub fn branch_less_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::Less);
    }

    #[inline(always)]
    pub fn branch_greater_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::AboveOrEqual);
    }

    #[inline(always)]
    pub fn branch_greater_or_equal_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::GreaterOrEqual);
    }

    #[inline(always)]
    pub fn branch_less_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::BelowOrEqual);
    }

    #[inline(always)]
    pub fn branch_less_or_equal_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::LessOrEqual);
    }

    #[inline(always)]
    pub fn branch_greater_unsigned_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::Above);
    }

    #[inline(always)]
    pub fn branch_greater_signed_imm(&mut self, s1: RawReg, s2: u32, target: u32) {
        self.branch(s1, s2, target, Condition::Greater);
    }

    #[inline(always)]
    pub fn jump(&mut self, target: u32) {
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);
        self.jump_to_label(label);
    }

    #[inline(always)]
    pub fn load_imm_and_jump(&mut self, ra: RawReg, value: u32, target: u32) {
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.invalid_jump_label);
        let asm = self.asm.reserve::<U2>();
        let asm = match B::BITNESS {
            Bitness::B32 => asm.push(mov_imm(conv_reg(ra), imm32(value))),
            Bitness::B64 => asm.push(mov_imm(conv_reg(ra), imm64(cast(value).to_signed()))),
        };
        let asm = jump_to_label(asm, label);
        asm.assert_reserved_exactly_as_needed();
    }

    #[inline(always)]
    pub fn jump_indirect(&mut self, base: RawReg, offset: u32) {
        self.jump_indirect_impl(None, base, offset)
    }

    #[inline(always)]
    pub fn load_imm_and_jump_indirect(&mut self, ra: RawReg, base: RawReg, value: u32, offset: u32) {
        self.jump_indirect_impl(Some((ra, value)), base, offset)
    }
}
