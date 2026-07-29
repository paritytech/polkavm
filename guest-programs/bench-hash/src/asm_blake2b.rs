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
    /// Processes `nblocks` consecutive 128-byte blocks; the byte counter for
    /// block k is `t_base + 128*k` (wrapping). `last != 0` finalizes every
    /// block in the batch, so callers pass `nblocks = 1` for the last block.
    fn blake2b_compress_many(h: *mut u64, data: *const u8, nblocks: u64, t_base: u64, iv: *const u64, last: u64);
}

#[cfg(target_arch = "riscv64")]
pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    use crate::compact_blake2b::IV;

    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 32;

    // All full blocks except a full last block (the last block is always
    // finalized separately below).
    let full = input.len().saturating_sub(1) / 128;
    if full > 0 {
        unsafe { blake2b_compress_many(h.as_mut_ptr(), input.as_ptr(), full as u64, 0, IV.as_ptr(), 0) };
    }

    let rem = &input[full * 128..];
    let mut block = [0u8; 128];
    block[..rem.len()].copy_from_slice(rem);
    let t_base = (input.len() as u64).wrapping_sub(128);
    unsafe { blake2b_compress_many(h.as_mut_ptr(), block.as_ptr(), 1, t_base, IV.as_ptr(), 1) };

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
