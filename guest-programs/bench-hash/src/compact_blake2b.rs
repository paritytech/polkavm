//! A code-size-compact blake2b-256: a single round body in a loop, instead
//! of the fully-unrolled form used by `blake2b_simd` (and RustCrypto
//! `blake2`).
//!
//! Rationale: on PVM the recompiled unrolled compression carries ~2x the
//! instructions of native (spills + 3-op to 2-op translation), overflowing
//! the host CPU's uop cache and capping the front-end at legacy-decoder
//! width. Rolling the rounds into a loop keeps the hot code cache-resident
//! while the state still lives in registers; the costs are the loop control
//! and SIGMA-indexed (instead of constant-offset) message loads.
//!
//! The loop bound is laundered through `black_box` so the compiler cannot
//! unroll it back.

pub(crate) const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[u8; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

#[inline(never)]
fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, chunk) in block.chunks_exact(8).enumerate() {
        m[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }

    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }

    // One round body; the opaque bound keeps it a real loop. The sigma row
    // is selected with a rotating index (no modulo — an opaque `r % 10`
    // would compile to a real division per round), and the message indices
    // are masked so the loads are provably in-bounds (no bounds checks).
    let mut s_idx = 0;
    for _ in 0..core::hint::black_box(12usize) {
        let s = &SIGMA[s_idx];
        s_idx += 1;
        if s_idx == 10 {
            s_idx = 0;
        }
        g(&mut v, 0, 4, 8, 12, m[s[0] as usize & 15], m[s[1] as usize & 15]);
        g(&mut v, 1, 5, 9, 13, m[s[2] as usize & 15], m[s[3] as usize & 15]);
        g(&mut v, 2, 6, 10, 14, m[s[4] as usize & 15], m[s[5] as usize & 15]);
        g(&mut v, 3, 7, 11, 15, m[s[6] as usize & 15], m[s[7] as usize & 15]);
        g(&mut v, 0, 5, 10, 15, m[s[8] as usize & 15], m[s[9] as usize & 15]);
        g(&mut v, 1, 6, 11, 12, m[s[10] as usize & 15], m[s[11] as usize & 15]);
        g(&mut v, 2, 7, 8, 13, m[s[12] as usize & 15], m[s[13] as usize & 15]);
        g(&mut v, 3, 4, 9, 14, m[s[14] as usize & 15], m[s[15] as usize & 15]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 32; // digest_length = 32, key = 0, fanout = 1, depth = 1

    let mut t: u128 = 0;
    let mut offset = 0;
    while input.len() - offset > 128 {
        let block: &[u8; 128] = input[offset..offset + 128].try_into().unwrap();
        t += 128;
        compress(&mut h, block, t, false);
        offset += 128;
    }

    let rem = &input[offset..];
    let mut block = [0u8; 128];
    block[..rem.len()].copy_from_slice(rem);
    t += rem.len() as u128;
    compress(&mut h, &block, t, true);

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..][..8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}
