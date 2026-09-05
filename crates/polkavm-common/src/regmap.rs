use crate::program::Reg;

#[cfg(target_arch = "x86_64")]
pub use polkavm_assembler::amd64::RegIndex as NativeReg;

#[cfg(target_arch = "aarch64")]
pub use polkavm_assembler::aarch64::Reg as NativeReg;

#[cfg(target_arch = "x86_64")]
use polkavm_assembler::amd64::RegIndex::*;

#[cfg(target_arch = "x86_64")]
#[inline]
pub const fn to_native_reg(reg: Reg) -> NativeReg {
    // NOTE: This is sorted roughly in the order of which registers are more commonly used.
    // We try to assign registers which result in more compact code to the more common RISC-V registers.
    match reg {
        Reg::A0 => rdi,
        Reg::A1 => rax,
        Reg::SP => rsi,
        Reg::RA => rbx,
        Reg::A2 => rdx,
        Reg::A3 => r14,
        Reg::S0 => r8,
        Reg::S1 => r9,
        Reg::A4 => r10,
        Reg::A5 => r11,
        Reg::T0 => r15,
        Reg::T1 => rbp,
        Reg::T2 => r12,
    }
}

#[cfg(target_arch = "x86_64")]
/// A temporary register which can be freely used.
pub const TMP_REG: NativeReg = rcx;

#[cfg(target_arch = "x86_64")]
/// A temporary register which must be saved/restored.
pub const AUX_TMP_REG: NativeReg = r13;

// ── AArch64 register mapping ─────────────────────────────────────────────────
//
// Guest registers → AArch64 native registers.
// We use callee-saved registers (x19-x28) for most guest registers so that
// trampolines don't need to save/restore them across host calls.
//
// Guest       → AArch64   Rationale
// ─────────────────────────────────────
// A0          → x0        argument reg (matches calling convention)
// A1          → x1        argument reg
// A2          → x2        argument reg
// A3          → x3        argument reg
// A4          → x4        argument reg
// A5          → x5        argument reg
// RA          → x19       callee-saved
// SP          → x20       callee-saved
// S0          → x21       callee-saved
// S1          → x22       callee-saved
// T0          → x23       callee-saved
// T1          → x24       callee-saved
// T2          → x25       callee-saved
// TMP_REG     → x16       scratch (IP0, linker veneer scratch)
// AUX_TMP_REG → x17       scratch (IP1) — used for guest memory base

#[cfg(target_arch = "aarch64")]
#[inline]
pub const fn to_native_reg(reg: Reg) -> NativeReg {
    use polkavm_assembler::aarch64::Reg as A;
    match reg {
        Reg::A0 => A::x0,
        Reg::A1 => A::x1,
        Reg::A2 => A::x2,
        Reg::A3 => A::x3,
        Reg::A4 => A::x4,
        Reg::A5 => A::x5,
        Reg::RA => A::x19,
        Reg::SP => A::x20,
        Reg::S0 => A::x21,
        Reg::S1 => A::x22,
        Reg::T0 => A::x23,
        Reg::T1 => A::x24,
        Reg::T2 => A::x25,
    }
}

#[cfg(target_arch = "aarch64")]
/// A temporary register which can be freely used (IP0).
pub const TMP_REG: NativeReg = polkavm_assembler::aarch64::Reg::x16;

#[cfg(target_arch = "aarch64")]
/// A temporary register which must be saved/restored (IP1).
/// On the generic sandbox, this holds the guest memory base address.
pub const AUX_TMP_REG: NativeReg = polkavm_assembler::aarch64::Reg::x17;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
pub const fn to_guest_reg(reg: NativeReg) -> Option<Reg> {
    let mut index = 0;
    while index < Reg::ALL.len() {
        let guest_reg = Reg::ALL[index];
        if to_native_reg(guest_reg) as u32 == reg as u32 {
            return Some(guest_reg);
        }

        index += 1;
    }

    None
}
