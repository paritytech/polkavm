//! AArch64 (ARMv8.2-A) code generation backend; ARM counterpart to [`crate::compiler::amd64`].

use polkavm_assembler::aarch64 as a64;
use polkavm_assembler::Label;

use polkavm_common::program::{RawReg, Reg};
use polkavm_common::regmap::{to_native_reg, NativeReg, AUX_TMP_REG, TMP_REG};
use polkavm_common::utils::GasVisitorT;

use crate::compiler::{ArchVisitor, Bitness, BitnessT, SandboxKind};
use crate::config::GasMeteringKind;
use crate::sandbox::Sandbox;

/// Converts a guest register reference into its native home register.
#[inline]
fn conv_reg(reg: RawReg) -> NativeReg {
    to_native_reg(reg.get())
}

/// Guest linear-memory base (generic sandbox); the vmctx is at a fixed negative offset from it.
const MEMORY_BASE_REG: NativeReg = AUX_TMP_REG; // x14

// Codegen scratch (not guest regs x0..x12, TMP_REG x13 or MEMORY_BASE_REG x14).
const SCRATCH0: NativeReg = NativeReg::X16;
const SCRATCH1: NativeReg = NativeReg::X17;
const SCRATCH2: NativeReg = NativeReg::X15;

/// Hardware stack pointer (encoding 31, shared with `XZR`); only for load/store and add/sub-immediate.
const SP: NativeReg = NativeReg::XZR;

// Byte offsets within the gas-metering stub (see emit_gas_metering_stub).
const GAS_COST_OFFSET: usize = 8; // the embedded 32-bit cost literal
const GAS_METERING_TRAP_OFFSET: u64 = 36; // the `udf` that fires on gas underflow

impl<'r, 'a, S, B, G> ArchVisitor<'r, 'a, S, B, G>
where
    S: Sandbox,
    B: BitnessT,
    G: GasVisitorT,
{
    // Code/jump-table padding (never executed); `0x00` decodes as `udf #0` so any fall-through traps.
    pub const PADDING_BYTE: u8 = 0x00;

    #[inline(always)]
    fn push<T>(&mut self, inst: polkavm_assembler::Instruction<T>)
    where
        T: core::fmt::Display,
    {
        self.0.asm.push(inst);
    }

    #[allow(clippy::unused_self)]
    fn reg_size(&self) -> a64::RegSize {
        match B::BITNESS {
            Bitness::B32 => a64::RegSize::W,
            Bitness::B64 => a64::RegSize::X,
        }
    }

    // ---- Core code-generation helpers ----

    /// Materializes a 64-bit constant via `movz`/`movn` + `movk` (picks the fewer-`movk` base).
    fn mov_imm(&mut self, rd: NativeReg, value: u64) {
        use a64::RegSize::X;
        let halves = [value as u16, (value >> 16) as u16, (value >> 32) as u16, (value >> 48) as u16];
        let ones = halves.iter().filter(|&&h| h == 0xffff).count();
        let zeros = halves.iter().filter(|&&h| h == 0).count();

        if ones > zeros {
            // `movn rd, #imm, lsl s` sets that halfword and fills the rest with ones.
            let first = halves.iter().position(|&h| h != 0xffff).unwrap_or(0);
            self.push(a64::movn(X, rd, !halves[first], first as u32));
            for (i, &half) in halves.iter().enumerate() {
                if i != first && half != 0xffff {
                    self.push(a64::movk(X, rd, half, i as u32));
                }
            }
        } else {
            let first = halves.iter().position(|&h| h != 0).unwrap_or(0);
            self.push(a64::movz(X, rd, halves[first], first as u32));
            for (i, &half) in halves.iter().enumerate() {
                if i > first && half != 0 {
                    self.push(a64::movk(X, rd, half, i as u32));
                }
            }
        }
    }

    /// `rd = rn + value` (signed); single add/sub immediate (incl. `lsl #12`) when it fits, else materialize.
    fn add_const(&mut self, rd: NativeReg, rn: NativeReg, value: u64) {
        use a64::RegSize::X;
        let signed = value as i64;
        if signed == 0 {
            if !rd.equals(rn) {
                self.push(a64::mov_reg(X, rd, rn));
            }
            return;
        }

        let sub = signed < 0;
        let mag = signed.unsigned_abs();
        if mag < 0x1000 {
            let m = mag as u32;
            if sub {
                self.push(a64::sub_imm(X, rd, rn, m, false));
            } else {
                self.push(a64::add_imm(X, rd, rn, m, false));
            }
        } else if mag & 0xfff == 0 && (mag >> 12) < 0x1000 {
            let m = (mag >> 12) as u32;
            if sub {
                self.push(a64::sub_imm(X, rd, rn, m, true));
            } else {
                self.push(a64::add_imm(X, rd, rn, m, true));
            }
        } else {
            self.mov_imm(SCRATCH1, value);
            self.push(a64::add_reg(X, rd, rn, SCRATCH1));
        }
    }

    /// Address of a VM-context field into `dst` (generic sandbox only; vmctx sits below the memory base).
    fn vmctx_addr(&mut self, dst: NativeReg, field_offset: usize) {
        debug_assert!(matches!(S::KIND, SandboxKind::Generic));
        #[cfg(feature = "generic-sandbox")]
        let off = (crate::sandbox::generic::GUEST_MEMORY_TO_VMCTX_OFFSET as i64 + field_offset as i64) as u64;
        #[cfg(not(feature = "generic-sandbox"))]
        let off = field_offset as u64;
        self.add_const(dst, MEMORY_BASE_REG, off);
    }

    fn ld_vmctx(&mut self, dst: NativeReg, field_offset: usize) {
        self.vmctx_addr(SCRATCH0, field_offset);
        self.push(a64::ldr_imm(a64::Size::U64, dst, SCRATCH0, 0));
    }

    fn st_vmctx(&mut self, src: NativeReg, field_offset: usize) {
        self.vmctx_addr(SCRATCH0, field_offset);
        self.push(a64::str_imm(a64::Size::U64, src, SCRATCH0, 0));
    }

    fn st_vmctx_imm(&mut self, value: u64, field_offset: usize) {
        if value == 0 {
            self.vmctx_addr(SCRATCH0, field_offset);
            self.push(a64::str_imm(a64::Size::U64, NativeReg::XZR, SCRATCH0, 0));
        } else {
            self.mov_imm(SCRATCH2, value);
            self.vmctx_addr(SCRATCH0, field_offset);
            self.push(a64::str_imm(a64::Size::U64, SCRATCH2, SCRATCH0, 0));
        }
    }

    fn st_vmctx_imm32(&mut self, value: u32, field_offset: usize) {
        if value == 0 {
            self.vmctx_addr(SCRATCH0, field_offset);
            self.push(a64::str_imm(a64::Size::U32, NativeReg::XZR, SCRATCH0, 0));
        } else {
            self.mov_imm(SCRATCH2, u64::from(value));
            self.vmctx_addr(SCRATCH0, field_offset);
            self.push(a64::str_imm(a64::Size::U32, SCRATCH2, SCRATCH0, 0));
        }
    }

    fn save_registers_to_vmctx(&mut self) {
        self.vmctx_addr(SCRATCH0, S::offset_table().regs);
        for (nth, reg) in Reg::ALL.iter().enumerate() {
            self.push(a64::str_imm(a64::Size::U64, conv_reg((*reg).into()), SCRATCH0, nth as u32));
        }
    }

    fn restore_registers_from_vmctx(&mut self) {
        self.vmctx_addr(SCRATCH0, S::offset_table().regs);
        for (nth, reg) in Reg::ALL.iter().enumerate() {
            self.push(a64::ldr_imm(a64::Size::U64, conv_reg((*reg).into()), SCRATCH0, nth as u32));
        }
    }

    /// Stores the link register (trampoline return address) into the VM context for host resumption.
    fn save_return_address_to_vmctx(&mut self) {
        self.st_vmctx(NativeReg::X30, S::offset_table().next_native_program_counter);
    }

    /// Sign-extends a 32-bit result in `d` to 64 bits (RV64 word-op semantics); no-op in 32-bit mode.
    fn finish32(&mut self, d: RawReg) {
        if matches!(B::BITNESS, Bitness::B64) {
            let d = conv_reg(d);
            self.push(a64::sbfm(a64::RegSize::X, d, d, 0, 31));
        }
    }

    /// Immediate into a scratch register, sign-extended (RISC-V immediates are signed; 32-bit consumers read the low half).
    fn imm_reg(&mut self, value: u32) -> NativeReg {
        self.mov_imm(SCRATCH0, i64::from(value as i32) as u64);
        SCRATCH0
    }

    /// `dst = MEMORY_BASE + ((base + offset) as u32)`; the 32-bit wrap bounds accesses to the guarded 4 GiB region.
    fn guest_addr(&mut self, dst: NativeReg, base: Option<RawReg>, offset: u32) {
        use a64::RegSize::{W, X};
        match base {
            Some(b) => {
                let b = conv_reg(b);
                if offset == 0 {
                    self.push(a64::mov_reg(W, dst, b));
                } else if offset < 0x1000 {
                    self.push(a64::add_imm(W, dst, b, offset, false));
                } else if offset & 0xfff == 0 && (offset >> 12) < 0x1000 {
                    self.push(a64::add_imm(W, dst, b, offset >> 12, true));
                } else {
                    self.mov_imm(dst, u64::from(offset));
                    self.push(a64::add_reg(W, dst, b, dst));
                }
                self.push(a64::add_reg(X, dst, MEMORY_BASE_REG, dst));
            }
            None => self.add_const(dst, MEMORY_BASE_REG, u64::from(offset)),
        }
    }

    fn emit_guest_load(&mut self, dst: RawReg, base: Option<RawReg>, offset: u32, size: a64::Size, signed: bool) {
        self.guest_addr(SCRATCH0, base, offset);
        let d = conv_reg(dst);
        // A signed word load only widens to 64 bits; in 32-bit mode a word load already fills the reg.
        let plain = !signed || (matches!(size, a64::Size::U32) && matches!(B::BITNESS, Bitness::B32));
        if plain {
            self.push(a64::ldr_imm(size, d, SCRATCH0, 0));
        } else {
            match B::BITNESS {
                Bitness::B32 => self.push(a64::ldrs32_imm(size, d, SCRATCH0, 0)),
                Bitness::B64 => self.push(a64::ldrs64_imm(size, d, SCRATCH0, 0)),
            }
        }
    }

    fn emit_guest_store_reg(&mut self, src: RawReg, base: Option<RawReg>, offset: u32, size: a64::Size) {
        self.guest_addr(SCRATCH0, base, offset);
        self.push(a64::str_imm(size, conv_reg(src), SCRATCH0, 0));
    }

    fn emit_guest_store_imm(&mut self, value: u64, base: Option<RawReg>, offset: u32, size: a64::Size) {
        if value == 0 {
            self.guest_addr(SCRATCH0, base, offset);
            self.push(a64::str_imm(size, NativeReg::XZR, SCRATCH0, 0));
        } else {
            self.mov_imm(SCRATCH2, value);
            self.guest_addr(SCRATCH0, base, offset);
            self.push(a64::str_imm(size, SCRATCH2, SCRATCH0, 0));
        }
    }

    /// `rd = rn << n` (logical shift left by a constant), via the UBFM alias.
    fn shl_imm(&mut self, size: a64::RegSize, rd: NativeReg, rn: NativeReg, n: u32) {
        let bits = match size {
            a64::RegSize::W => 32,
            a64::RegSize::X => 64,
        };
        self.push(a64::ubfm(size, rd, rn, (bits - n) % bits, bits - 1 - n));
    }

    /// Branch to `jump_table[(base + offset) as u32]` (opt. loading a return addr first). An invalid entry
    /// faults at the target with PC lost, so we stash this site in next_native_program_counter for the handler.
    fn jump_indirect_impl(&mut self, load_imm: Option<(RawReg, u32)>, base: RawReg, offset: u32) {
        use a64::RegSize::{W, X};
        let here = self.0.asm.create_label();
        self.push(a64::adr(SCRATCH2, here)); // SCRATCH2: st_vmctx's addressing never clobbers it
        self.st_vmctx(SCRATCH2, S::offset_table().next_native_program_counter);

        // index = (base + offset) as u32
        let b = conv_reg(base);
        if offset == 0 {
            self.push(a64::mov_reg(W, SCRATCH2, b));
        } else if offset < 0x1000 {
            self.push(a64::add_imm(W, SCRATCH2, b, offset, false));
        } else {
            self.mov_imm(SCRATCH2, u64::from(offset));
            self.push(a64::add_reg(W, SCRATCH2, b, SCRATCH2));
        }
        self.shl_imm(X, SCRATCH2, SCRATCH2, 3); // index * 8

        let jump_table_label = self.jump_table_label;
        self.push(a64::adr(SCRATCH1, jump_table_label));
        self.push(a64::add_reg(X, SCRATCH1, SCRATCH1, SCRATCH2));
        self.push(a64::ldr_imm(a64::Size::U64, SCRATCH1, SCRATCH1, 0));
        if let Some((ra, value)) = load_imm {
            self.load_imm(ra, value as i32);
        }
        self.push(a64::br(SCRATCH1));
    }

    /// Branches to a (possibly not-yet-defined) label, choosing the widest-range encoding needed.
    fn jump_to_label(&mut self, label: Label) {
        self.push(a64::b(label));
    }

    /// Conditional branch to `label`: inverted `B.cond` over an unconditional `B` (±128MB range).
    fn branch_to_label(&mut self, cond: a64::Cond, label: Label) {
        let skip = self.0.asm.forward_declare_label();
        self.push(a64::b_cond(cond.invert(), skip));
        self.push(a64::b(label));
        self.0.asm.define_label(skip);
    }

    /// Resolve a jump target to its label. An invalid target (e.g. mid-block) has no defined label and
    /// would patch to a branch-to-self, so route it to the trap trampoline (traps when reached).
    fn jump_target_label(&mut self, target: u32) -> Label {
        if self.is_jump_target_valid(target) {
            self.get_or_forward_declare_label(target).unwrap_or(self.trap_label)
        } else {
            self.trap_label
        }
    }

    /// `cmp s1, s2; b.<cond> target` (register/register form). `cond` is in terms of `s1 <cond> s2`.
    fn branch_rr(&mut self, s1: RawReg, s2: RawReg, target: u32, cond: a64::Cond) {
        if !self.is_jump_target_valid(target) {
            // Invalid target: the instruction traps unconditionally (matches the interpreter), whether
            // or not the branch would be taken.
            let trap = self.trap_label;
            self.jump_to_label(trap);
            return;
        }
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.trap_label);
        self.branch_to_label(cond, label);
    }

    /// `cmp s1, #imm; b.<cond> target` (register/immediate form).
    fn branch_ri(&mut self, s1: RawReg, imm: u32, target: u32, cond: a64::Cond) {
        if !self.is_jump_target_valid(target) {
            let trap = self.trap_label;
            self.jump_to_label(trap);
            return;
        }
        let sz = self.reg_size();
        let r = self.imm_reg(imm);
        self.push(a64::cmp_reg(sz, conv_reg(s1), r));
        let label = self.get_or_forward_declare_label(target).unwrap_or(self.trap_label);
        self.branch_to_label(cond, label);
    }

    pub fn add_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::add_reg(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn add_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::add_reg(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::add_reg(a64::RegSize::W, conv_reg(d), conv_reg(s1), r));
        self.finish32(d);
    }
    pub fn add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        // Sign-extend the 32-bit immediate to 64 bits (matches the x86 backend's `imm64(sign_extend)`).
        self.mov_imm(SCRATCH0, i64::from(s2 as i32) as u64);
        self.push(a64::add_reg(a64::RegSize::X, conv_reg(d), conv_reg(s1), SCRATCH0));
    }
    pub fn and(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::and_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn and_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::and_reg(sz, conv_reg(d), conv_reg(s1), r));
    }
    pub fn and_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = s1 & ~s2
        let sz = self.reg_size();
        self.push(a64::bic_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn branch_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Eq);
    }
    pub fn branch_eq_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Eq);
    }
    pub fn branch_not_eq(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Ne);
    }
    pub fn branch_not_eq_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Ne);
    }
    pub fn branch_less_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Cc);
    }
    pub fn branch_less_unsigned_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Cc);
    }
    pub fn branch_less_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Lt);
    }
    pub fn branch_less_signed_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Lt);
    }
    pub fn branch_greater_or_equal_unsigned(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Cs);
    }
    pub fn branch_greater_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Cs);
    }
    pub fn branch_greater_or_equal_signed(&mut self, s1: RawReg, s2: RawReg, target: u32) {
        self.branch_rr(s1, s2, target, a64::Cond::Ge);
    }
    pub fn branch_greater_or_equal_signed_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Ge);
    }
    pub fn branch_greater_unsigned_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Hi);
    }
    pub fn branch_greater_signed_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Gt);
    }
    pub fn branch_less_or_equal_unsigned_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Ls);
    }
    pub fn branch_less_or_equal_signed_imm(&mut self, s1: RawReg, s2: i32, target: u32) {
        let s2 = s2 as u32;
        self.branch_ri(s1, s2, target, a64::Cond::Le);
    }
    pub fn cmov_if_not_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_imm(sz, conv_reg(c), 0, false));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s), conv_reg(d), a64::Cond::Ne));
    }
    pub fn cmov_if_not_zero_imm(&mut self, d: RawReg, c: RawReg, s: i32) {
        let s = s as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s);
        self.push(a64::cmp_imm(sz, conv_reg(c), 0, false));
        self.push(a64::csel(sz, conv_reg(d), r, conv_reg(d), a64::Cond::Ne));
    }
    pub fn cmov_if_zero(&mut self, d: RawReg, s: RawReg, c: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_imm(sz, conv_reg(c), 0, false));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s), conv_reg(d), a64::Cond::Eq));
    }
    pub fn cmov_if_zero_imm(&mut self, d: RawReg, c: RawReg, s: i32) {
        let s = s as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s);
        self.push(a64::cmp_imm(sz, conv_reg(c), 0, false));
        self.push(a64::csel(sz, conv_reg(d), r, conv_reg(d), a64::Cond::Eq));
    }
    pub fn count_leading_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
        self.push(a64::clz(a64::RegSize::W, conv_reg(d), conv_reg(s)));
        self.finish32(d);
    }
    pub fn count_leading_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.push(a64::clz(a64::RegSize::X, conv_reg(d), conv_reg(s)));
    }
    pub fn count_trailing_zero_bits_32(&mut self, d: RawReg, s: RawReg) {
        // ctz(x) = clz(rbit(x))
        self.push(a64::rbit(a64::RegSize::W, conv_reg(d), conv_reg(s)));
        self.push(a64::clz(a64::RegSize::W, conv_reg(d), conv_reg(d)));
        self.finish32(d);
    }
    pub fn count_trailing_zero_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.push(a64::rbit(a64::RegSize::X, conv_reg(d), conv_reg(s)));
        self.push(a64::clz(a64::RegSize::X, conv_reg(d), conv_reg(d)));
    }
    // popcount via NEON (no scalar instruction): move to v0, count per byte, sum. Result is small and
    // positive, so no 32-bit sign-extension is needed.
    fn emit_popcount(&mut self, d: RawReg, s: RawReg, is_64: bool) {
        let d = conv_reg(d);
        let s = conv_reg(s);
        if is_64 {
            self.push(a64::fmov_gpr_to_d(0, s));
        } else {
            self.push(a64::fmov_gpr_to_s(0, s));
        }
        self.push(a64::cnt_8b(0, 0));
        self.push(a64::addv_8b(0, 0));
        self.push(a64::fmov_s_to_gpr(d, 0));
    }
    pub fn count_set_bits_32(&mut self, d: RawReg, s: RawReg) {
        self.emit_popcount(d, s, false);
    }
    pub fn count_set_bits_64(&mut self, d: RawReg, s: RawReg) {
        self.emit_popcount(d, s, true);
    }
    // RISC-V vs AArch64 differ only on divide-by-zero (RISC-V: quotient -1, remainder = dividend).
    // `msub` already yields the dividend when the quotient is 0; only the quotient needs the -1 fix-up.
    fn emit_div(&mut self, d: RawReg, s1: RawReg, s2: RawReg, size: a64::RegSize, signed: bool) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        if signed {
            self.push(a64::sdiv(size, SCRATCH0, s1, s2));
        } else {
            self.push(a64::udiv(size, SCRATCH0, s1, s2));
        }
        self.push(a64::cmp_imm(size, s2, 0, false));
        // s2 != 0 ? quotient : ~xzr (= -1)
        self.push(a64::csinv(size, d, SCRATCH0, NativeReg::XZR, a64::Cond::Ne));
    }

    fn emit_rem(&mut self, d: RawReg, s1: RawReg, s2: RawReg, size: a64::RegSize, signed: bool) {
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        if signed {
            self.push(a64::sdiv(size, SCRATCH0, s1, s2));
        } else {
            self.push(a64::udiv(size, SCRATCH0, s1, s2));
        }
        // remainder = s1 - quotient * s2; on a zero divisor the quotient is 0, so this yields s1.
        self.push(a64::msub(size, d, SCRATCH0, s2, s1));
    }

    pub fn div_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_div(d, s1, s2, a64::RegSize::W, true);
        self.finish32(d);
    }
    pub fn div_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_div(d, s1, s2, a64::RegSize::X, true);
    }
    pub fn div_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_div(d, s1, s2, a64::RegSize::W, false);
        self.finish32(d);
    }
    pub fn div_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_div(d, s1, s2, a64::RegSize::X, false);
    }
    pub fn ecalli(&mut self, code_offset: u32, args_length: u32, imm: i32) {
        let imm = imm as u32;
        if let Some(ref custom_codegen) = self.0.custom_codegen {
            if !custom_codegen.should_emit_ecalli(imm, &mut self.0.asm) {
                return;
            }
        }
        self.st_vmctx_imm32(imm, S::offset_table().arg);
        self.st_vmctx_imm32(code_offset, S::offset_table().program_counter);
        self.st_vmctx_imm32(code_offset + args_length + 1, S::offset_table().next_program_counter);
        let ecall_label = self.ecall_label;
        self.push(a64::bl(ecall_label));
    }
    pub(crate) fn emit_divrem_trampoline(&mut self) {
        // div/rem are handled inline; these labels are unreachable but defined so they aren't dangling.
        let labels = [
            self.div32u_label,
            self.div32s_label,
            self.div64u_label,
            self.div64s_label,
            self.rem32u_label,
            self.rem32s_label,
            self.rem64u_label,
            self.rem64s_label,
        ];
        for label in labels {
            self.0.asm.define_label(label);
            self.push(a64::udf(0));
        }
    }
    pub(crate) fn emit_ecall_trampoline(&mut self) {
        let label = self.ecall_label;
        self.0.asm.define_label(label);
        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.mov_imm(TMP_REG, S::address_table().syscall_hostcall);
        self.push(a64::br(TMP_REG));
    }
    pub(crate) fn emit_gas_metering_stub(&mut self, kind: GasMeteringKind) {
        let origin = self.0.asm.len();
        assert_eq!(S::offset_table().gas, 0x60);

        // Gas counter sits a fixed displacement below the memory base: one `sub` immediate finds it.
        #[cfg(feature = "generic-sandbox")]
        let disp = -(crate::sandbox::generic::GUEST_MEMORY_TO_VMCTX_OFFSET as i64 + S::offset_table().gas as i64);
        #[cfg(not(feature = "generic-sandbox"))]
        let disp = 0i64;
        debug_assert!(disp > 0 && disp < 0x1000, "gas displacement does not fit a sub-immediate");

        // Fixed layout (offsets fixed by GAS_COST_OFFSET / GAS_METERING_TRAP_OFFSET).
        self.0.asm.push_raw(&0x1800_0050u32.to_le_bytes()); // +0  ldr w16, #8 (load cost)
        self.0.asm.push_raw(&0x1400_0002u32.to_le_bytes()); // +4  b #8 (skip literal)
        self.0.asm.push_raw(&i32::MAX.to_le_bytes()); // +8  cost literal (patched by emit_weight)
        self.push(a64::sub_imm(a64::RegSize::X, SCRATCH1, MEMORY_BASE_REG, disp as u32, false)); // +12 &gas
        self.push(a64::ldr_imm(a64::Size::U64, SCRATCH2, SCRATCH1, 0)); // +16 gas
        self.push(a64::sub_reg(a64::RegSize::X, SCRATCH2, SCRATCH2, SCRATCH0)); // +20 gas -= cost
        self.push(a64::str_imm(a64::Size::U64, SCRATCH2, SCRATCH1, 0)); // +24

        if matches!(kind, GasMeteringKind::Sync) {
            self.push(a64::cmp_imm(a64::RegSize::X, SCRATCH2, 0, false)); // +28
            self.0.asm.push_raw(&0x5400_004Au32.to_le_bytes()); // +32 b.ge #8 (skip trap if gas >= 0)
            self.push(a64::udf(0)); // +36 underflow -> SIGILL
            debug_assert_eq!(GAS_METERING_TRAP_OFFSET, (self.0.asm.len() - origin - 4) as u64);
        }
        debug_assert_eq!(GAS_COST_OFFSET, 8);
    }
    pub(crate) fn emit_memset_trampoline(&mut self) {
        // Out-of-gas path for `memset`: A0/A2 already reflect progress, so just report NotEnoughGas.
        let label = self.memset_label;
        self.0.asm.define_label(label);
        self.save_registers_to_vmctx();
        self.mov_imm(TMP_REG, S::address_table().syscall_not_enough_gas);
        self.push(a64::br(TMP_REG));
    }
    pub(crate) fn emit_sbrk_trampoline(&mut self) {
        let label = self.sbrk_label;
        self.0.asm.define_label(label);

        // Arg in TMP_REG, return addr in x30. Save x30 across the C call to syscall_sbrk; result in x0.
        self.push(a64::sub_imm(a64::RegSize::X, SP, SP, 16, false));
        self.push(a64::str_imm(a64::Size::U64, NativeReg::X30, SP, 0));
        self.push(a64::mov_reg(a64::RegSize::X, SCRATCH2, TMP_REG)); // preserve arg across save_registers
        self.save_registers_to_vmctx();
        self.push(a64::mov_reg(a64::RegSize::X, NativeReg::X0, SCRATCH2)); // arg -> x0
        self.mov_imm(TMP_REG, S::address_table().syscall_sbrk);
        self.push(a64::blr(TMP_REG));
        self.push(a64::mov_reg(a64::RegSize::X, SCRATCH2, NativeReg::X0)); // stash result
        self.restore_registers_from_vmctx();
        self.push(a64::mov_reg(a64::RegSize::X, TMP_REG, SCRATCH2)); // result -> TMP_REG for the handler
        self.push(a64::ldr_imm(a64::Size::U64, NativeReg::X30, SP, 0));
        self.push(a64::add_imm(a64::RegSize::X, SP, SP, 16, false));
        self.push(a64::ret(NativeReg::X30));
    }
    pub(crate) fn emit_step_trampoline(&mut self) {
        let label = self.step_label;
        self.0.asm.define_label(label);
        self.save_return_address_to_vmctx();
        self.save_registers_to_vmctx();
        self.mov_imm(TMP_REG, S::address_table().syscall_step);
        self.push(a64::br(TMP_REG));
    }
    pub(crate) fn emit_sysenter(&mut self) -> Label {
        let label = self.0.asm.create_label();
        self.restore_registers_from_vmctx();
        self.ld_vmctx(TMP_REG, S::offset_table().next_native_program_counter);
        self.push(a64::br(TMP_REG));
        label
    }

    pub(crate) fn emit_sysreturn(&mut self) -> Label {
        let label = self.0.asm.create_label();
        self.st_vmctx_imm(0, S::offset_table().next_native_program_counter);
        self.save_registers_to_vmctx();
        self.mov_imm(TMP_REG, S::address_table().syscall_return);
        self.push(a64::br(TMP_REG));
        label
    }
    pub(crate) fn emit_trap_trampoline(&mut self) {
        let label = self.trap_label;
        self.0.asm.define_label(label);
        self.save_registers_to_vmctx();
        self.st_vmctx_imm(0, S::offset_table().next_native_program_counter);
        self.mov_imm(TMP_REG, S::address_table().syscall_trap);
        self.push(a64::br(TMP_REG));
    }
    pub(crate) fn emit_weight(&mut self, offset: usize, cost: u32) {
        // Patch the gas-cost literal embedded in the metering stub at `offset`.
        let p = offset + GAS_COST_OFFSET;
        self.0.asm.code_mut()[p..p + 4].copy_from_slice(&cost.to_le_bytes());
    }
    pub fn fallthrough(&mut self) {}
    pub fn invalid(&mut self, code_offset: u32) {
        log::debug!("Encountered invalid instruction");
        self.trap(code_offset);
    }
    pub fn jump(&mut self, target: u32) {
        let label = self.jump_target_label(target);
        self.jump_to_label(label);
    }
    pub fn jump_indirect(&mut self, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.jump_indirect_impl(None, base, offset);
    }
    pub fn load_u8(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U8, false);
    }
    pub fn load_i8(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U8, true);
    }
    pub fn load_u16(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U16, false);
    }
    pub fn load_i16(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U16, true);
    }
    pub fn load_u32(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U32, false);
    }
    pub fn load_i32(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U32, true);
    }
    pub fn load_u64(&mut self, dst: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, None, offset, a64::Size::U64, false);
    }
    pub fn load_indirect_u8(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U8, false);
    }
    pub fn load_indirect_i8(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U8, true);
    }
    pub fn load_indirect_u16(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U16, false);
    }
    pub fn load_indirect_i16(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U16, true);
    }
    pub fn load_indirect_u32(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U32, false);
    }
    pub fn load_indirect_i32(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U32, true);
    }
    pub fn load_indirect_u64(&mut self, dst: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_load(dst, Some(base), offset, a64::Size::U64, false);
    }
    pub fn load_imm(&mut self, dst: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let value = match B::BITNESS {
            Bitness::B32 => u64::from(s2),
            Bitness::B64 => i64::from(s2 as i32) as u64,
        };
        self.mov_imm(conv_reg(dst), value);
    }
    pub fn load_imm_and_jump(&mut self, ra: RawReg, value: i32, target: u32) {
        self.load_imm(ra, value);
        let label = self.jump_target_label(target);
        self.jump_to_label(label);
    }
    pub fn load_imm_and_jump_indirect(&mut self, ra: RawReg, base: RawReg, value: i32, offset: i32) {
        let value = value as u32;
        let offset = offset as u32;
        self.jump_indirect_impl(Some((ra, value)), base, offset);
    }
    pub fn load_imm64(&mut self, dst: RawReg, s2: u64) {
        self.mov_imm(conv_reg(dst), s2);
    }
    pub fn maximum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s1), conv_reg(s2), a64::Cond::Gt));
    }
    pub fn maximum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s1), conv_reg(s2), a64::Cond::Hi));
    }
    pub fn minimum(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s1), conv_reg(s2), a64::Cond::Lt));
    }
    pub fn minimum_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.push(a64::csel(sz, conv_reg(d), conv_reg(s1), conv_reg(s2), a64::Cond::Cc));
    }
    pub fn memset(&mut self) {
        // Fill A2 bytes at A0 with A1's low byte, 1 gas/byte; A0/A2 update in place to track progress on fault/exhaustion.
        use a64::RegSize::{W, X};
        let dst = conv_reg(Reg::A0.into());
        let val = conv_reg(Reg::A1.into());
        let cnt = conv_reg(Reg::A2.into());
        let metering = self.gas_metering.is_some();

        let memset_label = self.memset_label;
        let sz = self.reg_size();
        // The count is always 32-bit (the interpreter reads A2 as u32); clear its upper bits so a
        // dirty A2 doesn't run an over-long fill and so A2 is written back zero-extended on exit.
        self.push(a64::mov_reg(W, cnt, cnt));
        // 32-bit mode: clear A0's upper bits for the address calc. 64-bit mode: A0 is a full pointer, leave it.
        if matches!(B::BITNESS, Bitness::B32) {
            self.push(a64::mov_reg(W, dst, dst));
        }

        let done = self.0.asm.forward_declare_label();
        let loop_top = self.0.asm.create_label();
        self.push(a64::cbz(X, cnt, done));
        if metering {
            // On gas exhaustion branch to the (shared) trampoline, which reports NotEnoughGas.
            self.ld_vmctx(SCRATCH2, S::offset_table().gas);
            let has_gas = self.0.asm.forward_declare_label();
            self.push(a64::cbnz(X, SCRATCH2, has_gas));
            self.push(a64::b(memset_label));
            self.0.asm.define_label(has_gas);
        }
        // Guest addresses are 32-bit; use A0's low 32 bits (matches the interpreter's u32 destination).
        self.push(a64::mov_reg(W, SCRATCH0, dst));
        self.push(a64::add_reg(X, SCRATCH0, MEMORY_BASE_REG, SCRATCH0));
        self.push(a64::str_imm(a64::Size::U8, val, SCRATCH0, 0));
        if metering {
            self.push(a64::sub_imm(X, SCRATCH2, SCRATCH2, 1, false));
            self.st_vmctx(SCRATCH2, S::offset_table().gas);
        }
        self.push(a64::add_imm(sz, dst, dst, 1, false));
        self.push(a64::sub_imm(X, cnt, cnt, 1, false));
        self.jump_to_label(loop_top);
        self.0.asm.define_label(done);
    }
    pub fn move_reg(&mut self, d: RawReg, s: RawReg) {
        let sz = self.reg_size();
        self.push(a64::mov_reg(sz, conv_reg(d), conv_reg(s)));
    }
    pub fn mul_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::madd(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2), NativeReg::XZR));
        self.finish32(d);
    }
    pub fn mul_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::madd(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2), NativeReg::XZR));
    }
    pub fn mul_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::madd(a64::RegSize::W, conv_reg(d), conv_reg(s1), r, NativeReg::XZR));
        self.finish32(d);
    }
    pub fn mul_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        self.mov_imm(SCRATCH0, i64::from(s2 as i32) as u64);
        self.push(a64::madd(a64::RegSize::X, conv_reg(d), conv_reg(s1), SCRATCH0, NativeReg::XZR));
    }
    pub fn mul_upper_signed_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        use a64::RegSize::X;
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        if matches!(B::BITNESS, Bitness::B32) {
            // High 32 of a signed 32x32 product: widen (sxtw), 64-bit multiply, take bits [63:32].
            self.push(a64::sbfm(X, SCRATCH0, s1, 0, 31));
            self.push(a64::sbfm(X, SCRATCH1, s2, 0, 31));
            self.push(a64::madd(X, d, SCRATCH0, SCRATCH1, NativeReg::XZR));
            self.push(a64::ubfm(X, d, d, 32, 63)); // lsr #32
        } else {
            self.push(a64::smulh(d, s1, s2));
        }
    }
    pub fn mul_upper_unsigned_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        use a64::RegSize::X;
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        if matches!(B::BITNESS, Bitness::B32) {
            // High 32 of an unsigned 32x32 product: widen (uxtw), 64-bit multiply, take bits [63:32].
            self.push(a64::ubfm(X, SCRATCH0, s1, 0, 31));
            self.push(a64::ubfm(X, SCRATCH1, s2, 0, 31));
            self.push(a64::madd(X, d, SCRATCH0, SCRATCH1, NativeReg::XZR));
            self.push(a64::ubfm(X, d, d, 32, 63)); // lsr #32
        } else {
            self.push(a64::umulh(d, s1, s2));
        }
    }
    pub fn mul_upper_signed_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        use a64::RegSize::X;
        let d = conv_reg(d);
        let s1 = conv_reg(s1);
        let s2 = conv_reg(s2);
        if matches!(B::BITNESS, Bitness::B32) {
            // High 32 of signed(s1) * unsigned(s2): sxtw s1, uxtw s2, 64-bit multiply, bits [63:32].
            self.push(a64::sbfm(X, SCRATCH0, s1, 0, 31));
            self.push(a64::ubfm(X, SCRATCH1, s2, 0, 31));
            self.push(a64::madd(X, d, SCRATCH0, SCRATCH1, NativeReg::XZR));
            self.push(a64::ubfm(X, d, d, 32, 63)); // lsr #32
        } else {
            // High64 of signed(s1) * unsigned(s2) = umulh(s1,s2) - (s1<0 ? s2 : 0).
            self.push(a64::umulh(SCRATCH0, s1, s2)); // SCRATCH0 = unsigned high
            self.push(a64::sbfm(X, SCRATCH1, s1, 63, 63)); // asr #63: sign mask
            self.push(a64::and_reg(X, SCRATCH1, SCRATCH1, s2));
            self.push(a64::sub_reg(X, d, SCRATCH0, SCRATCH1));
        }
    }
    pub fn negate_and_add_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        // d = s2 - s1
        let r = self.imm_reg(s2);
        self.push(a64::sub_reg(a64::RegSize::W, conv_reg(d), r, conv_reg(s1)));
        self.finish32(d);
    }
    pub fn negate_and_add_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        self.mov_imm(SCRATCH0, i64::from(s2 as i32) as u64);
        self.push(a64::sub_reg(a64::RegSize::X, conv_reg(d), SCRATCH0, conv_reg(s1)));
    }
    pub fn or(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::orr_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn or_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::orr_reg(sz, conv_reg(d), conv_reg(s1), r));
    }
    pub fn or_inverted(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = s1 | ~s2
        let sz = self.reg_size();
        self.push(a64::orn_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn rem_signed_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_rem(d, s1, s2, a64::RegSize::W, true);
        self.finish32(d);
    }
    pub fn rem_signed_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_rem(d, s1, s2, a64::RegSize::X, true);
    }
    pub fn rem_unsigned_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_rem(d, s1, s2, a64::RegSize::W, false);
        self.finish32(d);
    }
    pub fn rem_unsigned_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.emit_rem(d, s1, s2, a64::RegSize::X, false);
    }
    pub fn reverse_byte(&mut self, d: RawReg, s: RawReg) {
        match B::BITNESS {
            Bitness::B32 => self.push(a64::rev32_word(conv_reg(d), conv_reg(s))),
            Bitness::B64 => self.push(a64::rev64(conv_reg(d), conv_reg(s))),
        }
    }
    pub fn rotate_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // rol(x, n) = ror(x, -n); AArch64 only has ror.
        self.push(a64::sub_reg(a64::RegSize::W, SCRATCH0, NativeReg::XZR, conv_reg(s2)));
        self.push(a64::rorv(a64::RegSize::W, conv_reg(d), conv_reg(s1), SCRATCH0));
        self.finish32(d);
    }
    pub fn rotate_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::sub_reg(a64::RegSize::X, SCRATCH0, NativeReg::XZR, conv_reg(s2)));
        self.push(a64::rorv(a64::RegSize::X, conv_reg(d), conv_reg(s1), SCRATCH0));
    }
    pub fn rotate_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::rorv(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn rotate_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::rorv(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn rotate_right_imm_32(&mut self, d: RawReg, s: RawReg, c: i32) {
        let c = c as u32;
        let r = self.imm_reg(c);
        self.push(a64::rorv(a64::RegSize::W, conv_reg(d), conv_reg(s), r));
        self.finish32(d);
    }
    pub fn rotate_right_imm_64(&mut self, d: RawReg, s: RawReg, c: i32) {
        let c = c as u32;
        let r = self.imm_reg(c);
        self.push(a64::rorv(a64::RegSize::X, conv_reg(d), conv_reg(s), r));
    }
    pub fn rotate_right_imm_alt_32(&mut self, d: RawReg, s: RawReg, c: i32) {
        let c = c as u32;
        // d = ror(c, s): the immediate is the value, the register is the amount.
        let r = self.imm_reg(c);
        self.push(a64::rorv(a64::RegSize::W, conv_reg(d), r, conv_reg(s)));
        self.finish32(d);
    }
    pub fn rotate_right_imm_alt_64(&mut self, d: RawReg, s: RawReg, c: i32) {
        let c = c as u32;
        self.mov_imm(SCRATCH0, i64::from(c as i32) as u64);
        self.push(a64::rorv(a64::RegSize::X, conv_reg(d), SCRATCH0, conv_reg(s)));
    }
    pub fn sbrk(&mut self, dst: RawReg, size: RawReg) {
        use a64::RegSize::{W, X};
        // pending heap top = heap_top + size; if it stays within the threshold just bump the pointer,
        // otherwise call the trampoline (which maps memory or returns 0). Mirrors the x86 backend.
        let dst = conv_reg(dst);
        let size = conv_reg(size);
        let heap = S::offset_table().heap_info;

        self.push(a64::mov_reg(W, SCRATCH1, size)); // zero-extend the 32-bit size
        self.ld_vmctx(SCRATCH2, heap); // heap_top
        self.push(a64::add_reg(X, SCRATCH2, SCRATCH2, SCRATCH1)); // pending = heap_top + size
        self.ld_vmctx(SCRATCH1, heap + 8); // heap_threshold
        self.push(a64::cmp_reg(X, SCRATCH2, SCRATCH1));

        let bump_only = self.0.asm.forward_declare_label();
        let cont = self.0.asm.forward_declare_label();
        self.branch_to_label(a64::Cond::Ls, bump_only); // pending <= threshold

        // Crossed the threshold: the trampoline maps memory and returns the new top (or 0) in TMP_REG.
        self.push(a64::mov_reg(X, TMP_REG, SCRATCH2));
        let sbrk_label = self.sbrk_label;
        self.push(a64::bl(sbrk_label));
        self.push(a64::mov_reg(X, dst, TMP_REG));
        self.jump_to_label(cont);

        self.0.asm.define_label(bump_only);
        self.st_vmctx(SCRATCH2, heap); // heap_top = pending
        self.push(a64::mov_reg(X, dst, SCRATCH2));
        self.0.asm.define_label(cont);
    }
    // `cset d, cond` is `csinc d, xzr, xzr, invert(cond)`; we pass the already-inverted condition.
    fn set_cond(&mut self, d: RawReg, cond_when_true: a64::Cond) {
        let sz = self.reg_size();
        self.push(a64::csinc(sz, conv_reg(d), NativeReg::XZR, NativeReg::XZR, cond_when_true.invert()));
    }

    pub fn set_less_than_unsigned(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.set_cond(d, a64::Cond::Cc);
    }
    pub fn set_less_than_signed(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::cmp_reg(sz, conv_reg(s1), conv_reg(s2)));
        self.set_cond(d, a64::Cond::Lt);
    }
    pub fn set_less_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::cmp_reg(sz, conv_reg(s1), r));
        self.set_cond(d, a64::Cond::Cc);
    }
    pub fn set_less_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::cmp_reg(sz, conv_reg(s1), r));
        self.set_cond(d, a64::Cond::Lt);
    }
    pub fn set_greater_than_unsigned_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::cmp_reg(sz, conv_reg(s1), r));
        self.set_cond(d, a64::Cond::Hi);
    }
    pub fn set_greater_than_signed_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::cmp_reg(sz, conv_reg(s1), r));
        self.set_cond(d, a64::Cond::Gt);
    }
    pub fn shift_arithmetic_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::asrv(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_arithmetic_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::asrv(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn shift_arithmetic_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::asrv(a64::RegSize::W, conv_reg(d), conv_reg(s1), r));
        self.finish32(d);
    }
    pub fn shift_arithmetic_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::asrv(a64::RegSize::X, conv_reg(d), conv_reg(s1), r));
    }
    pub fn shift_arithmetic_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        let r = self.imm_reg(s1);
        self.push(a64::asrv(a64::RegSize::W, conv_reg(d), r, conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_arithmetic_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        self.mov_imm(SCRATCH0, i64::from(s1 as i32) as u64);
        self.push(a64::asrv(a64::RegSize::X, conv_reg(d), SCRATCH0, conv_reg(s2)));
    }
    pub fn shift_logical_left_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::lslv(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_logical_left_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::lslv(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn shift_logical_left_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::lslv(a64::RegSize::W, conv_reg(d), conv_reg(s1), r));
        self.finish32(d);
    }
    pub fn shift_logical_left_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::lslv(a64::RegSize::X, conv_reg(d), conv_reg(s1), r));
    }
    pub fn shift_logical_left_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        let r = self.imm_reg(s1);
        self.push(a64::lslv(a64::RegSize::W, conv_reg(d), r, conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_logical_left_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        self.mov_imm(SCRATCH0, i64::from(s1 as i32) as u64);
        self.push(a64::lslv(a64::RegSize::X, conv_reg(d), SCRATCH0, conv_reg(s2)));
    }
    pub fn shift_logical_right_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::lsrv(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_logical_right_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::lsrv(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn shift_logical_right_imm_32(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::lsrv(a64::RegSize::W, conv_reg(d), conv_reg(s1), r));
        self.finish32(d);
    }
    pub fn shift_logical_right_imm_64(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let r = self.imm_reg(s2);
        self.push(a64::lsrv(a64::RegSize::X, conv_reg(d), conv_reg(s1), r));
    }
    pub fn shift_logical_right_imm_alt_32(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        let r = self.imm_reg(s1);
        self.push(a64::lsrv(a64::RegSize::W, conv_reg(d), r, conv_reg(s2)));
        self.finish32(d);
    }
    pub fn shift_logical_right_imm_alt_64(&mut self, d: RawReg, s2: RawReg, s1: i32) {
        let s1 = s1 as u32;
        self.mov_imm(SCRATCH0, i64::from(s1 as i32) as u64);
        self.push(a64::lsrv(a64::RegSize::X, conv_reg(d), SCRATCH0, conv_reg(s2)));
    }
    pub fn sign_extend_8(&mut self, d: RawReg, s: RawReg) {
        let sz = self.reg_size();
        self.push(a64::sbfm(sz, conv_reg(d), conv_reg(s), 0, 7));
    }
    pub fn sign_extend_16(&mut self, d: RawReg, s: RawReg) {
        let sz = self.reg_size();
        self.push(a64::sbfm(sz, conv_reg(d), conv_reg(s), 0, 15));
    }
    pub fn store_u8(&mut self, src: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, None, offset, a64::Size::U8);
    }
    pub fn store_u16(&mut self, src: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, None, offset, a64::Size::U16);
    }
    pub fn store_u32(&mut self, src: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, None, offset, a64::Size::U32);
    }
    pub fn store_u64(&mut self, src: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, None, offset, a64::Size::U64);
    }
    pub fn store_indirect_u8(&mut self, src: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, Some(base), offset, a64::Size::U8);
    }
    pub fn store_indirect_u16(&mut self, src: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, Some(base), offset, a64::Size::U16);
    }
    pub fn store_indirect_u32(&mut self, src: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, Some(base), offset, a64::Size::U32);
    }
    pub fn store_indirect_u64(&mut self, src: RawReg, base: RawReg, offset: i32) {
        let offset = offset as u32;
        self.emit_guest_store_reg(src, Some(base), offset, a64::Size::U64);
    }
    pub fn store_imm_u8(&mut self, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), None, offset, a64::Size::U8);
    }
    pub fn store_imm_u16(&mut self, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), None, offset, a64::Size::U16);
    }
    pub fn store_imm_u32(&mut self, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), None, offset, a64::Size::U32);
    }
    pub fn store_imm_u64(&mut self, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(value as i64 as u64, None, offset, a64::Size::U64);
    }
    pub fn store_imm_indirect_u8(&mut self, base: RawReg, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), Some(base), offset, a64::Size::U8);
    }
    pub fn store_imm_indirect_u16(&mut self, base: RawReg, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), Some(base), offset, a64::Size::U16);
    }
    pub fn store_imm_indirect_u32(&mut self, base: RawReg, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(u64::from(value as u32), Some(base), offset, a64::Size::U32);
    }
    pub fn store_imm_indirect_u64(&mut self, base: RawReg, offset: i32, value: i32) {
        let offset = offset as u32;
        self.emit_guest_store_imm(value as i64 as u64, Some(base), offset, a64::Size::U64);
    }
    pub fn sub_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::sub_reg(a64::RegSize::W, conv_reg(d), conv_reg(s1), conv_reg(s2)));
        self.finish32(d);
    }
    pub fn sub_64(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.push(a64::sub_reg(a64::RegSize::X, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub(crate) fn trace_execution(&mut self, code_offset: Option<u32>) {
        // Fixed-length prelude (like the x86 backend) so `step_prelude_length` is exact: the segfault
        // resume path skips it by that amount. `code_offset` is materialized with a forced movz+movk.
        let step_label = self.step_label;
        let origin = self.asm.len();
        let has_offset = code_offset.is_some();
        if let Some(code_offset) = code_offset {
            use a64::RegSize::X;
            self.push(a64::movz(X, SCRATCH2, code_offset as u16, 0));
            self.push(a64::movk(X, SCRATCH2, (code_offset >> 16) as u16, 1));
            self.vmctx_addr(SCRATCH0, S::offset_table().program_counter);
            self.push(a64::str_imm(a64::Size::U32, SCRATCH2, SCRATCH0, 0));
            self.vmctx_addr(SCRATCH0, S::offset_table().next_program_counter);
            self.push(a64::str_imm(a64::Size::U32, SCRATCH2, SCRATCH0, 0));
        }
        self.push(a64::bl(step_label));
        if has_offset {
            debug_assert_eq!(self.asm.len() - origin, step_prelude_length::<S>());
        }
    }
    pub fn trap(&mut self, code_offset: u32) {
        self.st_vmctx_imm32(code_offset, S::offset_table().program_counter);
        let trap_label = self.trap_label;
        self.push(a64::bl(trap_label));
    }
    pub fn xnor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        // d = ~(s1 ^ s2) = s1 ^ ~s2
        let sz = self.reg_size();
        self.push(a64::eon_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn xor(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        let sz = self.reg_size();
        self.push(a64::eor_reg(sz, conv_reg(d), conv_reg(s1), conv_reg(s2)));
    }
    pub fn xor_imm(&mut self, d: RawReg, s1: RawReg, s2: i32) {
        let s2 = s2 as u32;
        let sz = self.reg_size();
        let r = self.imm_reg(s2);
        self.push(a64::eor_reg(sz, conv_reg(d), conv_reg(s1), r));
    }
    pub fn zero_extend_16(&mut self, d: RawReg, s: RawReg) {
        let sz = self.reg_size();
        self.push(a64::ubfm(sz, conv_reg(d), conv_reg(s), 0, 15));
    }
}

// ---- Free functions used by the sandbox (parallel to amd64.rs) ----
// `on_signal_trap`/`on_page_fault` are zygote-only; the generic sandbox recovers from signals inline in generic.rs.

/// Reads the 32-bit gas cost embedded in a block's metering stub.
pub fn extract_gas_cost<S>(machine_code: &[u8], basic_block_machine_code_offset: usize) -> u32
where
    S: Sandbox,
{
    let p = basic_block_machine_code_offset + GAS_COST_OFFSET;
    let xs = &machine_code[p..p + 4];
    u32::from_le_bytes([xs[0], xs[1], xs[2], xs[3]])
}

/// Byte length of a block's gas-metering stub (see `emit_gas_metering_stub`): Sync adds a `cmp`/`b.ge`/`udf`.
pub fn gas_metering_stub_length(kind: GasMeteringKind) -> usize {
    match kind {
        GasMeteringKind::Sync => GAS_METERING_TRAP_OFFSET as usize + 4, // up to and including the `udf`
        GasMeteringKind::Async => 28,                                   // ends after the `str` at +24
    }
}

#[inline(always)]
pub fn step_prelude_length<S>() -> usize
where
    S: Sandbox,
{
    // 7 fixed-length instructions; see `trace_execution`, whose debug_assert verifies this.
    7 * 4
}
