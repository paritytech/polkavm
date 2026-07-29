//! blake2b-256 with a hand-scheduled RISC-V assembly compression function
//! (`blake2b_compress.S`, generated).
//!
//! Rationale: the recompiled output of the portable Rust implementation is
//! issue-bound — LLVM's register allocation for the 13-register rv64e target
//! spills far more than necessary and its 3-operand code forces the
//! recompiler to emit register-copy `mov`s. The assembly version pins the
//! state's a/b rows (v0..v7) in registers, streams the c/d rows through a
//! 64-byte stack frame, uses strictly two-operand form (no recompiler movs),
//! and bakes the message schedule into constant load offsets.
//!
//! On non-riscv64 targets (host builds, riscv32) this falls back to the
//! compact Rust implementation so the export exists everywhere and
//! checksums stay comparable.

#[cfg(target_arch = "riscv64")]
core::arch::global_asm!(include_str!("blake2b_compress.S"));

#[cfg(target_arch = "riscv64")]
extern "C" {
    fn blake2b_compress_asm(h: *mut u64, block: *const u8, t: u64, iv: *const u64, last: u64);
}

#[cfg(target_arch = "riscv64")]
pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    use crate::compact_blake2b::IV;

    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 32;

    let mut t: u64 = 0;
    let mut offset = 0;
    while input.len() - offset > 128 {
        t += 128;
        unsafe { blake2b_compress_asm(h.as_mut_ptr(), input.as_ptr().add(offset), t, IV.as_ptr(), 0) };
        offset += 128;
    }

    let rem = &input[offset..];
    let mut block = [0u8; 128];
    block[..rem.len()].copy_from_slice(rem);
    t += rem.len() as u64;
    unsafe { blake2b_compress_asm(h.as_mut_ptr(), block.as_ptr(), t, IV.as_ptr(), 1) };

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..][..8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

#[cfg(not(target_arch = "riscv64"))]
pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    crate::compact_blake2b::blake2b_256(input)
}
