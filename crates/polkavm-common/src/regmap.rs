use crate::program::Reg;

#[cfg(target_arch = "x86_64")]
pub use polkavm_assembler::amd64::RegIndex as NativeReg;
#[cfg(target_arch = "x86_64")]
use polkavm_assembler::amd64::RegIndex::*;

#[cfg(target_arch = "aarch64")]
pub use polkavm_assembler::aarch64::Reg as NativeReg;
#[cfg(target_arch = "aarch64")]
use polkavm_assembler::aarch64::Reg::*;

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
        Reg::A3 => rbp,
        Reg::S0 => r8,
        Reg::S1 => r9,
        Reg::A4 => r10,
        Reg::A5 => r11,
        Reg::T0 => r13,
        Reg::T1 => r14,
        Reg::T2 => r12,
    }
}

/// A temporary register which can be freely used.
#[cfg(target_arch = "x86_64")]
pub const TMP_REG: NativeReg = rcx;

/// A temporary register which must be saved/restored.
#[cfg(target_arch = "x86_64")]
pub const AUX_TMP_REG: NativeReg = r15;

#[cfg(target_arch = "aarch64")]
#[inline]
pub const fn to_native_reg(reg: Reg) -> NativeReg {
    // The 13 guest registers are kept live in native registers during guest execution (loaded at
    // sysenter, stored back at sysreturn). They are placed in the caller-saved bank (x0..x12): this
    // lets the sandbox entry inline assembly clobber them via `clobber_abi("C")` without having to
    // touch the callee-saved registers x19..x28, which Rust/LLVM forbids from appearing as inline-asm
    // operands (x19 in particular is reserved by LLVM). We deliberately avoid x18 (the platform
    // register, reserved on Darwin), x29 (frame pointer), x30 (link register) and the stack pointer;
    // x15..x17 are left free as codegen scratch.
    match reg {
        Reg::A0 => X0,
        Reg::A1 => X1,
        Reg::A2 => X2,
        Reg::A3 => X3,
        Reg::A4 => X4,
        Reg::A5 => X5,
        Reg::S0 => X6,
        Reg::S1 => X7,
        Reg::SP => X8,
        Reg::RA => X9,
        Reg::T0 => X10,
        Reg::T1 => X11,
        Reg::T2 => X12,
    }
}

/// A temporary register which can be freely used (analogous to `rcx` on x86-64).
#[cfg(target_arch = "aarch64")]
pub const TMP_REG: NativeReg = X13;

/// A temporary register which must be saved/restored; on the generic sandbox it holds the base
/// address of the guest's linear memory (analogous to `r15` on x86-64).
#[cfg(target_arch = "aarch64")]
pub const AUX_TMP_REG: NativeReg = X14;

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
