// Hand-scheduled blake2b compression for PVM (rv64e + zbb).
// Generated - see the report / bench-hash sources for context.
// fn blake2b_compress_asm(h: *mut u64 [a0], block: *const u8 [a1],
//                         t: u64 [a2], iv: *const u64 [a3], last: u64 [a4])
.section .text.blake2b_compress_asm,"ax"
.globl blake2b_compress_asm
blake2b_compress_asm:
    addi sp, sp, -96
    sd   ra, 72(sp)
    sd   s0, 80(sp)
    sd   s1, 88(sp)
    sd   a0, 64(sp)
    // v8..v15 = IV[0..7] (+ counter / finalization flag)
    ld   t1, 0(a3)
    sd   t1, 0(sp)
    ld   t1, 8(a3)
    sd   t1, 8(sp)
    ld   t1, 16(a3)
    sd   t1, 16(sp)
    ld   t1, 24(a3)
    sd   t1, 24(sp)
    ld   t1, 32(a3)
    xor  t1, t1, a2
    sd   t1, 32(sp)
    ld   t1, 40(a3)
    sd   t1, 40(sp)
    ld   t1, 48(a3)
    neg  t2, a4
    xor  t1, t1, t2
    sd   t1, 48(sp)
    ld   t1, 56(a3)
    sd   t1, 56(sp)
    mv   t0, a1
    // v0..v7 = h[0..7] (a0 loaded last - it is the source pointer)
    ld   s1, 56(a0)
    ld   s0, 48(a0)
    ld   a5, 40(a0)
    ld   a4, 32(a0)
    ld   a3, 24(a0)
    ld   a2, 16(a0)
    ld   a1, 8(a0)
    ld   a0, 0(a0)
    // round 0
    ld   t1, 0(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 8(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 16(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 24(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 32(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 40(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 48(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 56(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 64(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 72(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 80(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 88(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 96(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 104(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 112(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 120(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 1
    ld   t1, 112(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 80(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 32(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 64(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 72(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 120(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 104(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 48(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 8(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 96(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 0(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 16(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 88(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 56(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 40(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 24(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 2
    ld   t1, 88(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 64(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 96(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 0(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 40(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 16(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 120(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 104(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 80(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 112(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 24(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 48(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 56(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 8(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 72(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 32(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 3
    ld   t1, 56(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 72(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 24(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 8(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 104(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 96(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 88(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 112(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 16(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 48(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 40(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 80(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 32(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 0(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 120(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 64(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 4
    ld   t1, 72(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 0(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 40(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 56(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 16(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 32(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 80(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 120(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 112(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 8(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 88(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 96(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 48(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 64(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 24(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 104(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 5
    ld   t1, 16(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 96(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 48(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 80(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 0(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 88(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 64(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 24(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 32(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 104(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 56(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 40(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 120(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 112(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 8(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 72(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 6
    ld   t1, 96(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 40(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 8(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 120(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 112(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 104(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 32(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 80(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 0(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 56(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 48(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 24(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 72(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 16(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 64(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 88(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 7
    ld   t1, 104(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 88(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 56(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 112(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 96(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 8(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 24(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 72(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 40(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 0(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 120(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 32(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 64(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 48(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 16(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 80(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 8
    ld   t1, 48(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 120(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 112(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 72(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 88(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 24(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 0(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 64(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 96(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 16(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 104(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 56(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 8(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 32(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 80(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 40(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 9
    ld   t1, 80(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 16(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 64(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 32(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 56(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 48(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 8(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 40(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 120(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 88(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 72(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 112(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 24(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 96(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 104(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 0(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 10
    ld   t1, 0(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 8(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 16(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 24(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 32(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 40(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 48(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 56(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 64(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 72(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 80(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 88(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 96(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 104(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 112(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 120(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // round 11
    ld   t1, 112(t0)
    add  a0, a0, a4
    add  a0, a0, t1
    ld   t2, 32(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 80(t0)
    add  a0, a0, a4
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    ld   t1, 32(t0)
    add  a1, a1, a5
    add  a1, a1, t1
    ld   t2, 40(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 64(t0)
    add  a1, a1, a5
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 72(t0)
    add  a2, a2, s0
    add  a2, a2, t1
    ld   t2, 48(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 120(t0)
    add  a2, a2, s0
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 104(t0)
    add  a3, a3, s1
    add  a3, a3, t1
    ld   t2, 56(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 48(t0)
    add  a3, a3, s1
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 8(t0)
    add  a0, a0, a5
    add  a0, a0, t1
    ld   t2, 56(sp)
    xor  t2, t2, a0
    rori t2, t2, 32
    ld   t1, 16(sp)
    add  t1, t1, t2
    xor  a5, a5, t1
    rori a5, a5, 24
    ld   ra, 96(t0)
    add  a0, a0, a5
    add  a0, a0, ra
    xor  t2, t2, a0
    rori t2, t2, 16
    sd   t2, 56(sp)
    add  t1, t1, t2
    sd   t1, 16(sp)
    xor  a5, a5, t1
    rori a5, a5, 63
    ld   t1, 0(t0)
    add  a1, a1, s0
    add  a1, a1, t1
    ld   t2, 32(sp)
    xor  t2, t2, a1
    rori t2, t2, 32
    ld   t1, 24(sp)
    add  t1, t1, t2
    xor  s0, s0, t1
    rori s0, s0, 24
    ld   ra, 16(t0)
    add  a1, a1, s0
    add  a1, a1, ra
    xor  t2, t2, a1
    rori t2, t2, 16
    sd   t2, 32(sp)
    add  t1, t1, t2
    sd   t1, 24(sp)
    xor  s0, s0, t1
    rori s0, s0, 63
    ld   t1, 88(t0)
    add  a2, a2, s1
    add  a2, a2, t1
    ld   t2, 40(sp)
    xor  t2, t2, a2
    rori t2, t2, 32
    ld   t1, 0(sp)
    add  t1, t1, t2
    xor  s1, s1, t1
    rori s1, s1, 24
    ld   ra, 56(t0)
    add  a2, a2, s1
    add  a2, a2, ra
    xor  t2, t2, a2
    rori t2, t2, 16
    sd   t2, 40(sp)
    add  t1, t1, t2
    sd   t1, 0(sp)
    xor  s1, s1, t1
    rori s1, s1, 63
    ld   t1, 40(t0)
    add  a3, a3, a4
    add  a3, a3, t1
    ld   t2, 48(sp)
    xor  t2, t2, a3
    rori t2, t2, 32
    ld   t1, 8(sp)
    add  t1, t1, t2
    xor  a4, a4, t1
    rori a4, a4, 24
    ld   ra, 24(t0)
    add  a3, a3, a4
    add  a3, a3, ra
    xor  t2, t2, a3
    rori t2, t2, 16
    sd   t2, 48(sp)
    add  t1, t1, t2
    sd   t1, 8(sp)
    xor  a4, a4, t1
    rori a4, a4, 63
    // h[i] ^= v[i] ^ v[i+8]
    ld   t0, 64(sp)
    ld   t1, 0(t0)
    xor  t1, t1, a0
    ld   t2, 0(sp)
    xor  t1, t1, t2
    sd   t1, 0(t0)
    ld   t1, 8(t0)
    xor  t1, t1, a1
    ld   t2, 8(sp)
    xor  t1, t1, t2
    sd   t1, 8(t0)
    ld   t1, 16(t0)
    xor  t1, t1, a2
    ld   t2, 16(sp)
    xor  t1, t1, t2
    sd   t1, 16(t0)
    ld   t1, 24(t0)
    xor  t1, t1, a3
    ld   t2, 24(sp)
    xor  t1, t1, t2
    sd   t1, 24(t0)
    ld   t1, 32(t0)
    xor  t1, t1, a4
    ld   t2, 32(sp)
    xor  t1, t1, t2
    sd   t1, 32(t0)
    ld   t1, 40(t0)
    xor  t1, t1, a5
    ld   t2, 40(sp)
    xor  t1, t1, t2
    sd   t1, 40(t0)
    ld   t1, 48(t0)
    xor  t1, t1, s0
    ld   t2, 48(sp)
    xor  t1, t1, t2
    sd   t1, 48(t0)
    ld   t1, 56(t0)
    xor  t1, t1, s1
    ld   t2, 56(sp)
    xor  t1, t1, t2
    sd   t1, 56(t0)
    ld   ra, 72(sp)
    ld   s0, 80(sp)
    ld   s1, 88(sp)
    addi sp, sp, 96
    ret
