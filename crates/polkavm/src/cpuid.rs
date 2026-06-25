use core::arch::x86_64::__cpuid_count;

/// Returns the `EBX` register of `CPUID.(EAX=7, ECX=0)`, or zero if that leaf is unsupported.
#[inline]
fn structured_extended_feature_flags_ebx() -> u32 {
    // SAFETY: CPUID is always available.
    let max_leaf = unsafe { __cpuid_count(0, 0) }.eax;
    if max_leaf < 7 {
        // The leaf is out of range; querying it would return data for some other leaf.
        return 0;
    }

    // SAFETY: CPUID is always available on x86-64, and we've checked that leaf 7 is supported.
    unsafe { __cpuid_count(7, 0) }.ebx
}

pub fn is_bmi2_supported() -> bool {
    // CPUID.(EAX=7, ECX=0):EBX.BMI2[bit 8]
    structured_extended_feature_flags_ebx() & (1 << 8) != 0
}

pub fn is_avx2_supported() -> bool {
    // Using AVX2 safely requires three separate things, mirroring what
    // `std::is_x86_feature_detected!("avx2")` checks:
    //   1. the CPU supports AVX and the OS has enabled `XSAVE` (`OSXSAVE`),
    //   2. the OS actually preserves the `XMM`/`YMM` registers across context switches (`XCR0`),
    //   3. the CPU supports AVX2.
    // Skipping (1) and (2) would let us emit AVX2 instructions on a kernel that doesn't preserve
    // the wide registers, silently corrupting state.

    const OSXSAVE: u32 = 1 << 27;
    const AVX: u32 = 1 << 28;

    // SAFETY: CPUID is always available on x86-64; leaf 1 always exists.
    let feature_ecx = unsafe { __cpuid_count(1, 0) }.ecx;
    if feature_ecx & (OSXSAVE | AVX) != (OSXSAVE | AVX) {
        return false;
    }

    // Bit 1 = `XMM` (SSE) state, bit 2 = `YMM` (AVX) state; the OS must save/restore both.
    const XMM_AND_YMM: u64 = (1 << 1) | (1 << 2);

    // SAFETY: `OSXSAVE` is set, so `xgetbv` is guaranteed to be available.
    if unsafe { xcr0() } & XMM_AND_YMM != XMM_AND_YMM {
        return false;
    }

    // CPUID.(EAX=7, ECX=0):EBX.AVX2[bit 5]
    structured_extended_feature_flags_ebx() & (1 << 5) != 0
}

/// Reads the `XCR0` extended control register via `xgetbv`.
///
/// # Safety
///
/// The `OSXSAVE` CPUID bit must be set, otherwise `xgetbv` triggers an invalid opcode exception.
#[inline]
unsafe fn xcr0() -> u64 {
    let eax: u32;
    let edx: u32;
    // SAFETY: The caller guarantees `xgetbv` is available. `xgetbv` reads `ECX` (the register
    // number, 0 = `XCR0`) and writes the result into `EDX:EAX`; it touches no memory, stack or
    // flags.
    unsafe {
        core::arch::asm!(
            "xgetbv",
            in("ecx") 0_u32,
            out("eax") eax,
            out("edx") edx,
            options(nomem, nostack, preserves_flags),
        );
    }
    (u64::from(edx) << 32) | u64::from(eax)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #[test]
    fn matches_std_detection() {
        assert_eq!(super::is_bmi2_supported(), std::is_x86_feature_detected!("bmi2"));
        assert_eq!(super::is_avx2_supported(), std::is_x86_feature_detected!("avx2"));
    }
}
