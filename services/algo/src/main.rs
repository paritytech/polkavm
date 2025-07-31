#![no_std]
#![no_main]
#![feature(asm_const)]
#![allow(non_snake_case)]

extern crate alloc;
use alloc::collections::BinaryHeap;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;
//use core::mem::transmute;
use core::primitive::u64;
use hashbrown::HashMap;

const SIZE0: usize = 0x10000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1: usize = 0x10000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use core::slice;
use core::sync::atomic::{AtomicU64, Ordering};
use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{call_log, parse_accumulate_args, parse_refine_args, parse_transfer_args, parse_accumulate_operand_args, write_result};
use utils::host_functions::{fetch, gas, write};

/// A simple xorshift64* PRNG
#[derive(Clone, Copy)]
pub struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    pub const fn new(seed: u64) -> Self {
        // seed must be nonzero
        XorShift64Star { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

// we’ll put the RNG state in an AtomicU64 to avoid `unsafe` in get_random_number
static RNG_STATE: AtomicU64 = AtomicU64::new(0x_1234_5678_9ABC_DEF1);

/// Initialize the global PRNG (call once at startup, with any nonzero seed)
pub fn seed_rng(seed: u64) {
    assert!(seed != 0);
    RNG_STATE.store(seed, Ordering::SeqCst);
}

/// Returns the next random `u64`
pub fn get_random_number() -> u64 {
    // load-and-update atomically
    let mut state = RNG_STATE.load(Ordering::SeqCst);
    // xorshift64* step
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let out = state.wrapping_mul(0x2545F4914F6CDD1D);
    RNG_STATE.store(state, Ordering::SeqCst);
    out
}

#[allow(dead_code)]
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

// GCD
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

// Extended GCD
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a.abs(), a.signum(), 0)
    } else {
        let (g, x1, y1) = extended_gcd(b, a % b);
        (g, y1, x1 - (a / b) * y1)
    }
}

// 2. Extended GCD → modular inverse
fn mod_inv(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = extended_gcd(a, m);
    if g != 1 {
        None
    } else {
        Some((x % m + m) % m)
    }
}

// Integer √ (binary search)
fn integer_sqrt(n: u64) -> u64 {
    let mut lo = 0;
    let mut hi = n;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let sq = mid.saturating_mul(mid);
        if sq == n {
            return mid;
        } else if sq < n {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    hi
}

// integer nth root
fn integer_nth_root(n: u64, k: u32) -> u64 {
    let mut lo = 0u64;
    let mut hi = n.min(1 << (64 / k)) + 1;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        let mut acc = 1u64;
        for _ in 0..k {
            acc = acc.saturating_mul(mid as u64);
            if acc > n as u64 {
                break;
            }
        }
        if acc <= n as u64 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

// binomial for Narayana
fn binomial(n: u64, k: u64) -> u64 {
    let k = k.min(n - k);
    let mut res = 1u64;
    for i in 1..=k {
        res = res * (n + 1 - i) as u64 / i as u64;
    }
    res
}

fn jacobi(mut a: u64, mut n: u64) -> i64 {
    if n == 0 || n & 1 == 0 {
        return 0;
    }
    let mut t = 1;
    a %= n;
    while a != 0 {
        while a & 1 == 0 {
            a >>= 1;
            let r = n % 8;
            if r == 3 || r == 5 {
                t = -t;
            }
        }
        mem::swap(&mut a, &mut n);
        if a & n & 2 != 0 {
            t = -t;
        }
        a %= n;
    }
    if n == 1 {
        t
    } else {
        0
    }
}

#[allow(dead_code)]
fn dist2(a: (i64, i64), b: (i64, i64)) -> u64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy) as u64
}
// Next prime ≥ n
fn next_prime(mut n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    n += 1;
    while !is_prime(n) {
        n += 1;
    }
    n
}
/*
fn primes_up_to(n: usize) -> Vec<u64> {
    let mut sieve = vec![true; n+1];
    let mut p = Vec::new();
    for i in 2..=n {
        if sieve[i] {
            p.push(i as u64);
            let mut j = i*i;
            while j <= n {
                sieve[j] = false;
                j += i;
            }
        }
    }
    p
}
     */

// Prime factors
fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut f = Vec::new();
    while n % 2 == 0 {
        f.push(2);
        n /= 2;
    }
    let mut p = 3;
    while p * p <= n {
        while n % p == 0 {
            f.push(p);
            n /= p;
        }
        p += 2;
    }
    if n > 1 {
        f.push(n);
    }
    f
}
// Primitive root mod p
fn primitive_root(p: u64) -> u64 {
    let phi = p - 1;
    let mut pf = prime_factors(phi);
    pf.sort();
    pf.dedup();
    'outer: for g in 2..p {
        for &q in &pf {
            if mod_exp(g, phi / q, p) == 1 {
                continue 'outer;
            }
        }
        return g;
    }
    0
}

// Primitive Root Test (for prime p)
/*
fn primitive_root_test(p: u64) -> Option<u64> {
    if p < 2 { return None; }
    let phi = p - 1;
    let mut facs = trial_division_wheel(phi);
    facs.sort(); facs.dedup();
    for g in 2..p {
        if facs.iter().all(|&q| mod_exp(g, phi / q, p) != 1) {
            return Some(g);
        }
    }
    None
}

 */
// 3. Modular exponentiation
fn mod_exp(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut res = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            res = res.wrapping_mul(base) % m;
        }
        base = base.wrapping_mul(base) % m;
        exp >>= 1;
    }
    res
}

// 4. CRT for two congruences
fn crt2(a1: i64, n1: i64, a2: i64, n2: i64) -> i64 {
    let inv = mod_inv(n1, n2).unwrap();
    let t = ((a2 - a1).rem_euclid(n2) * inv).rem_euclid(n2);
    a1 + n1 * t
}
// 5. Garner’s general CRT
fn garner(a: &[i64], n: &[i64]) -> i64 {
    let mut x = a[0];
    let mut prod = n[0];
    for i in 1..a.len() {
        let inv = mod_inv(prod.rem_euclid(n[i]), n[i]).unwrap();
        let t = ((a[i] - x).rem_euclid(n[i]) * inv).rem_euclid(n[i]);
        x += prod * t;
        prod *= n[i];
    }
    x
}

// 7. Floor log₂
//fn floor_log2(n: u64) -> u32 { 63 - n.leading_zeros() }

// 8. CLZ, CTZ, Popcount, Parity
#[allow(dead_code)]
fn clz(n: u64) -> u32 {
    n.leading_zeros()
}

#[allow(dead_code)]
fn ctz(n: u64) -> u32 {
    n.trailing_zeros()
}

#[allow(dead_code)]
fn popcount(n: u64) -> u32 {
    n.count_ones()
}
#[allow(dead_code)]
fn parity(n: u64) -> u32 {
    (n.count_ones() & 1) as u32
}
// 9. Bit reversal & Gray code
fn reverse_bits32(n: u32) -> u32 {
    n.reverse_bits()
}
fn gray_encode(n: u64) -> u64 {
    n ^ (n >> 1)
}
fn gray_decode(mut g: u64) -> u64 {
    let mut n = g;
    while g > 0 {
        g >>= 1;
        n ^= g;
    }
    n
}
// 10. Sieve & segmented sieve
fn sieve(n: usize) -> Vec<usize> {
    let mut is_prime = vec![true; n + 1];
    let mut p = Vec::new();
    for i in 2..=n {
        if is_prime[i] {
            p.push(i);
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    p
}

fn is_prime_small(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}
/*
fn pollards_rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }

    let mut rng = thread_rng();
    let c: u64 = rng.gen_range(1..n);
    let mut x: u64 = rng.gen_range(2..n);
    let mut y = x;

    // annotate v as u64 (and return type u64 if you like)
    let f = |v: u64| -> u64 { (v.wrapping_mul(v).wrapping_add(c)) % n };

    let mut d = 1;
    while d == 1 {
        x = f(x);
        y = f(f(y));
        d = gcd((x as i64 - y as i64).abs() as u64, n);
        if d == n {
            // retry with a new random constant
            return pollards_rho(n);
        }
    }
    d
}

fn factor(n: u64) -> Vec<u64> {
    if n == 1 {
        return vec![];
    }
    if is_prime_small(n) {
        return vec![n];
    }
    let d = pollards_rho(n);
    let mut l = factor(d);
    let mut r = factor(n / d);
    l.append(&mut r);
    l.sort_unstable();
    l
}
*/

// 12. Fast Fibonacci (fast doubling)
fn fib(n: u64) -> (u64, u64) {
    if n == 0 {
        return (0, 1);
    }
    let (a, b) = fib(n >> 1);
    let c = a.wrapping_mul(b.wrapping_mul(2).wrapping_sub(a));
    let d = a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b));
    if n & 1 == 0 {
        (c, d)
    } else {
        (d, c.wrapping_add(d))
    }
}
// 13–14. Factorial & binomial & Catalan
fn catalan(n: u64) -> u64 {
    (1..=n).fold(1u64, |c, i| c.wrapping_mul(4 * i as u64 - 2) / (i as u64 + 1))
}
// 15. Karatsuba 64×64→128
fn karatsuba(x: u64, y: u64) -> u128 {
    // mask and all intermediates promoted to u128
    let mask = (1u128 << 32) - 1;
    let x0 = (x as u128) & mask;
    let x1 = (x as u128) >> 32;
    let y0 = (y as u128) & mask;
    let y1 = (y as u128) >> 32;

    let z0 = x0 * y0;
    let z2 = x1 * y1;
    let z1 = (x0 + x1) * (y0 + y1) - z0 - z2;

    // now shifts are < 128, so no overflow
    (z2 << 64) | (z1 << 32) | z0
}

// 16–17. MP add/sub/mul naive (2‑limb)
fn mp_add(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let (r0, carry) = a[0].overflowing_add(b[0]);
    let (r1, _) = a[1].overflowing_add(b[1].wrapping_add(carry as u64));
    [r0, r1]
}
fn mp_sub(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let (r0, bor) = a[0].overflowing_sub(b[0]);
    let (r1, _) = a[1].overflowing_sub(b[1].wrapping_add(bor as u64));
    [r0, r1]
}
fn mp_mul_naive(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let mut res = [0u64; 4];

    for i in 0..2 {
        // carry must be big enough to hold the high half
        let mut carry: u128 = 0;

        for j in 0..2 {
            let idx = i + j;
            // promote everything to u128
            let t: u128 = (a[i] as u128) * (b[j] as u128) + (res[idx] as u128) + carry;

            // low 64 bits back into res
            res[idx] = t as u64;
            // high 64 bits become the new carry
            carry = t >> 64;
        }

        // leftover carry goes in the next limb
        res[i + 2] = carry as u64;
    }

    res
}

// 18–19 Montgomery & Barrett (32‑bit)
fn mont_mul32(a: u32, b: u32, m: u32, m_prime: u32) -> u32 {
    let t = a as u64 * b as u64;
    let m0 = (t as u32).wrapping_mul(m_prime);
    let tmp = t.wrapping_add((m0 as u64).wrapping_mul(m as u64));
    let u = (tmp >> 32) as u32;
    if u >= m {
        u - m
    } else {
        u
    }
}
fn mont_reduce32(t: u64, m: u32, m_prime: u32) -> u32 {
    let m0 = (t as u32).wrapping_mul(m_prime);
    let mut u = ((t + m0 as u64 * m as u64) >> 32) as u32;
    if u >= m {
        u -= m
    }
    u
}
fn barrett_reduce(x: u64, m: u32, mu: u64) -> u32 {
    // do the 64×64→128 multiplication in u128
    let prod: u128 = (x as u128) * (mu as u128);
    // now shifting right by 64 is fine on a 128‑bit value
    let q = (prod >> 64) as u64;

    // compute the remainder
    let mut r = x.wrapping_sub(q.wrapping_mul(m as u64));
    // one subtraction should suffice if mu was precomputed correctly
    if r >= m as u64 {
        r -= m as u64;
    }

    r as u32
}

// 20. NTT (mod 17, N=8)
const MOD_NTT: u64 = 17;
const NTT_N: usize = 8;
const ROOT: u64 = 3;
fn ntt(a: &[u64; NTT_N]) -> [u64; NTT_N] {
    let w = mod_exp(ROOT, (MOD_NTT - 1) / NTT_N as u64, MOD_NTT);
    let mut out = [0u64; NTT_N];
    for k in 0..NTT_N {
        let mut sum = 0;
        for j in 0..NTT_N {
            sum = (sum + a[j] * mod_exp(w, (j * k) as u64, MOD_NTT)) % MOD_NTT;
        }
        out[k] = sum;
    }
    out
}
// 21. CORDIC Q16
fn cordic(angle: i32) -> (i32, i32) {
    const ANG: [i32; 16] = [51471, 30385, 16054, 8153, 4097, 2049, 1025, 512, 256, 128, 64, 32, 16, 8, 4, 2];
    const K: i32 = 39797;
    let mut x = K;
    let mut y = 0;
    let mut z = angle;
    for i in 0..16 {
        let dx = x >> i;
        let dy = y >> i;
        if z >= 0 {
            x -= dy;
            y += dx;
            z -= ANG[i];
        } else {
            x += dy;
            y -= dx;
            z += ANG[i];
        }
    }
    (x, y)
}
// 22. Fixed-point Q16
fn fix_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}
fn fix_div(a: i32, b: i32) -> i32 {
    // avoid divide-by-zero
    if b == 0 {
        return 0;
    }
    (((a as i64) << 16) / (b as i64)) as i32
}
// 23. LCG
struct Lcg {
    state: u32,
    a: u32,
    c: u32,
}
impl Lcg {
    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(self.a).wrapping_add(self.c);
        self.state
    }
}
// 24. Xorshift64
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
// 25. PCG32
struct Pcg {
    state: u64,
    inc: u64,
}
impl Pcg {
    fn next(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc | 1);
        let x = ((old >> 18) ^ old) >> 27;
        let r = (old >> 59) as u32;
        (x as u32).rotate_right(r)
    }
}
// 26. MWC
struct Mwc {
    state: u64,
    carry: u64,
}
impl Mwc {
    fn next(&mut self) -> u32 {
        let t = self.state.wrapping_mul(4294957665).wrapping_add(self.carry);
        self.state = t & 0xFFFF_FFFF;
        self.carry = t >> 32;
        self.state as u32
    }
}
// 27–29. CRC32, Adler-32, FNV-1a
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    const MODAD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);

    for &byte in data {
        let v = byte as u32; // promote u8 → u32
        a = (a + v) % MODAD;
        b = (b + a) % MODAD;
    }

    (b << 16) | a
}

fn fnv1a(data: &[u8]) -> u32 {
    let mut h = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

// ─── 41–50 ─────────────────────────────────────────────────────────────────

// 30. Murmur3 finalizer
fn murmur3_finalizer(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}
// 31. Jenkins
fn jenkins(data: &[u8]) -> u32 {
    let mut h = 0u32;
    for &b in data {
        h = h.wrapping_add(b as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}
// 32–33. Bresenham line & circle
/*
fn bresenham_line(x0:i32,y0:i32,x1:i32,y1:i32)->Vec<(i32,i32)>{
    let (dx,dy)=((x1-x0).abs(),-(y1-y0).abs());
    let (sx,sy)=(if x0<x1{1}else{-1},if y0<y1{1}else{-1});
    let mut err=dx+dy; let (mut x,mut y)=(x0,y0);
    let mut pts=Vec::new();
    loop {
        pts.push((x,y));
        if x==x1 && y==y1 { break }
        let e2=2*err;
        if e2>=dy { err+=dy; x+=sx; }
        if e2<=dx { err+=dx; y+=sy; }
    }
    pts
}
fn bresenham_circle(cx:i32,cy:i32,r:i32)->Vec<(i32,i32)>{
    let(mut x,mut y)=(0,r);
    let mut d=3-2*r;
    let mut pts=Vec::new();
    while y>=x {
        for &(dx,dy) in &[
            ( x, y),( -x, y),( x, -y),( -x, -y),
            ( y, x),( -y, x),( y, -x),( -y, -x)
        ] {
            pts.push((cx+dx,cy+dy));
        }
        if d<=0 { d+=4*x+6; }
        else { d+=4*(x-y)+10; y-=1; }
        x+=1;
    }
    pts
}

// 34. DDA line
fn dda_line(x0:i32,y0:i32,x1:i32,y1:i32)->Vec<(i32,i32)>{
    let(dx,dy)=(x1-x0,y1-y0);
    let steps=dx.abs().max(dy.abs());
    (0..=steps).map(|i|{
        let x=x0+dx*i/steps; let y=y0+dy*i/steps;
        (x,y)
    }).collect()
}
// 35. Horner
fn horner(coeff:&[i64],x:i64)->i64{
    coeff.iter().rev().fold(0,|res,&c|res.wrapping_mul(x).wrapping_add(c))
}
// 36. Finite difference
fn finite_difference(seq:&[i64],ord:usize)->Vec<i64>{
    let mut d=seq.to_vec();
    for _ in 0..ord {
        if d.len()<2 {break}
        d=d.windows(2).map(|w|w[1]-w[0]).collect();
    }
    d
}

    // 37. Diophantine
fn solve_diophantine(a:i64,b:i64,c:i64)->Option<(i64,i64)>{
    let(g,x0,y0)=extended_gcd(a,b);
    if c%g!=0 {None} else { Some((x0*(c/g),y0*(c/g))) }
}

    // 38. Integer log10
fn integer_log10(mut n:u64)->u32{
    let mut l=0;
    while n>=10 { n/=10; l+=1; }
    l
}


    */

// 51. Euler’s Totient Function φ(n)
fn phi(mut n: u64) -> u64 {
    let mut result = n;
    if n % 2 == 0 {
        result = result / 2;
        while n % 2 == 0 {
            n /= 2;
        }
    }
    let mut p = 3;
    while p * p <= n {
        if n % p == 0 {
            result = result / p * (p - 1);
            while n % p == 0 {
                n /= p;
            }
        }
        p += 2;
    }
    if n > 1 {
        result = result / n * (n - 1);
    }
    result
}

// 52. Linear (Euler) sieve computing primes, φ and μ up to N
fn linear_sieve(n: usize) -> (Vec<usize>, Vec<u64>, Vec<i32>) {
    let mut is_comp = vec![false; n + 1];
    let mut primes = Vec::new();
    let mut phi = vec![0u64; n + 1];
    let mut mu = vec![0i32; n + 1];
    phi[1] = 1;
    mu[1] = 1;
    for i in 2..=n {
        if !is_comp[i] {
            primes.push(i);
            phi[i] = (i - 1) as u64;
            mu[i] = -1;
        }
        for &p in &primes {
            let ip = i * p;
            if ip > n {
                break;
            }
            is_comp[ip] = true;
            if i % p == 0 {
                phi[ip] = phi[i] * p as u64;
                mu[ip] = 0;
                break;
            } else {
                phi[ip] = phi[i] * (p as u64 - 1);
                mu[ip] = -mu[i];
            }
        }
    }
    (primes, phi, mu)
}

// 53. Sum‑of‑Divisors σ(n)
fn sigma(mut n: u64) -> u64 {
    let mut result: u64 = 1;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            let mut sum = 1u64;
            let mut power = 1u64;
            while n % p == 0 {
                n /= p;
                power *= p as u64;
                sum += power;
            }
            result *= sum;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        result *= 1 + n as u64;
    }
    result as u64
}

// 54. Divisor‑Count Function d(n)
fn divisor_count(mut n: u64) -> u64 {
    let mut count = 1;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            let mut exp = 0;
            while n % p == 0 {
                n /= p;
                exp += 1;
            }
            count *= exp + 1;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        count *= 2;
    }
    count
}

// 55. Möbius Function μ(n)
fn mobius(mut n: u64) -> i64 {
    let mut m = 1;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            n /= p;
            if n % p == 0 {
                return 0;
            }
            m = -m;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        m = -m;
    }
    m
}

// 56. Linear Sieve for μ(n) only
fn linear_mu(n: usize) -> Vec<i32> {
    let mut is_comp = vec![false; n + 1];
    let mut primes = Vec::new();
    let mut mu = vec![0i32; n + 1];
    mu[1] = 1;
    for i in 2..=n {
        if !is_comp[i] {
            primes.push(i);
            mu[i] = -1;
        }
        for &p in &primes {
            let ip = i * p;
            if ip > n {
                break;
            }
            is_comp[ip] = true;
            if i % p == 0 {
                mu[ip] = 0;
                break;
            } else {
                mu[ip] = -mu[i];
            }
        }
    }
    mu
}

// 57. Dirichlet Convolution (f * g)(n)
fn dirichlet_convolution<F, G>(n: u64, f: F, g: G) -> u64
where
    F: Fn(u64) -> u64,
    G: Fn(u64) -> u64,
{
    let mut sum = 0;
    let mut d = 1;
    while d * d <= n {
        if n % d == 0 {
            let d2 = n / d;
            if d == d2 {
                sum += f(d) * g(d2);
            } else {
                sum += f(d) * g(d2) + f(d2) * g(d);
            }
        }
        d += 1;
    }
    sum
}
/*
// Prime Counting (Legendre’s formula)
fn phi_legendre(x: u64, s: usize, primes: &[u64]) -> u64 {
    if s == 0 { return x; }
    if s == 1 { return x - x/2; }
    phi_legendre(x, s-1, primes) - phi_legendre(x / primes[s-1], s-1, primes)
}
*/

// Prime Counting (Naïve trial division)
fn pi_trial(n: u64) -> u64 {
    let mut cnt = 0;
    for i in 2..=n {
        if is_prime_small(i) {
            cnt += 1;
        }
    }
    cnt
}

// 61. Legendre symbol (a|p) via Euler’s criterion
fn legendre_symbol(a: i64, p: i64) -> i32 {
    let ls = mod_exp(a.rem_euclid(p) as u64, ((p - 1) / 2) as u64, p as u64);
    if ls == 1 {
        1
    } else if ls == p as u64 - 1 {
        -1
    } else {
        0
    }
}

// 62. Tonelli–Shanks: sqrt(a) mod p, p an odd prime
fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    if p == 2 {
        return Some(n);
    }
    // check solution exists
    if mod_exp(n, (p - 1) / 2, p) != 1 {
        return None;
    }
    // write p-1 = q * 2^s
    let mut q = p - 1;
    let mut s = 0;
    while q & 1 == 0 {
        q >>= 1;
        s += 1;
    }
    // find z a quadratic non-residue
    let mut z = 2;
    while mod_exp(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = mod_exp(z, q, p);
    let mut t = mod_exp(n, q, p);
    let mut r = mod_exp(n, (q + 1) / 2, p);
    while t != 1 {
        // find least i (0 < i < m) such that t^(2^i) == 1
        let mut t2i = t;
        let mut i = 0;
        while t2i != 1 {
            t2i = (t2i * t2i) % p;
            i += 1;
            if i == m {
                return None;
            }
        }
        let b = mod_exp(c, 1 << (m - i - 1), p);
        m = i;
        c = (b * b) % p;
        t = (t * c) % p;
        r = (r * b) % p;
    }
    Some(r)
}

fn cipolla(n: u64, p: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    if mod_exp(n, (p - 1) / 2, p) != 1 {
        return None;
    }
    // find a such that w = a^2 - n is non-residue
    let mut a = 0;
    let mut w = 0;
    for x in 1..p {
        w = (x * x + p - n % p) % p;
        if mod_exp(w, (p - 1) / 2, p) == p - 1 {
            a = x;
            break;
        }
    }
    // define multiplication in F_p^2
    #[derive(Copy, Clone)]
    struct Complex {
        x: u64,
        y: u64,
        w: u64,
        p: u64,
    }
    impl Complex {
        fn mul(self, other: Complex) -> Complex {
            let x = (self.x * other.x % self.p + self.y * other.y % self.p * self.w % self.p) % self.p;
            let y = (self.x * other.y % self.p + self.y * other.x % self.p) % self.p;
            Complex {
                x,
                y,
                w: self.w,
                p: self.p,
            }
        }
        fn pow(mut self, mut exp: u64) -> Complex {
            let mut res = Complex {
                x: 1,
                y: 0,
                w: self.w,
                p: self.p,
            };
            while exp > 0 {
                if exp & 1 == 1 {
                    res = res.mul(self);
                }
                self = self.mul(self);
                exp >>= 1;
            }
            res
        }
    }
    let comp = Complex { x: a, y: 1, w, p };
    let res = comp.pow((p + 1) / 2);
    Some(res.x)
}

// Lucas–Lehmer test for Mersenne primes M_p = 2^p - 1
fn lucas_lehmer(p: u64) -> bool {
    if p == 2 {
        return true;
    }
    let m = (1u64 << p) - 1;
    let mut s = 4u64;
    for _ in 0..(p - 2) {
        // use wrapping_mul so debug builds won't panic
        s = s.wrapping_mul(s).wrapping_sub(2) % m;
    }
    s == 0
}

// Lucas sequence U_n, V_n mod m (naïve O(n))
fn lucas_sequence(n: u64, P: i64, Q: i64, m: i64) -> (i64, i64) {
    if n == 0 {
        return (0, 2 % m);
    }
    if n == 1 {
        return (1 % m, P % m);
    }
    let mut U0 = 0i64;
    let mut V0 = 2i64 % m;
    let mut U1 = 1i64 % m;
    let mut V1 = P % m;
    for _ in 2..=n {
        let U2 = (P * U1 - Q * U0).rem_euclid(m);
        let V2 = (P * V1 - Q * V0).rem_euclid(m);
        U0 = U1;
        V0 = V1;
        U1 = U2;
        V1 = V2;
    }
    (U1, V1)
}

// Strong Lucas probable prime test
fn is_strong_lucas_prp(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    // find D,P,Q for Selfridge’s method
    let mut D = 5i64;
    let mut sign = 1;
    while legendre_symbol(D, n as i64) != -1 {
        D = -(D + 2 * sign);
        sign = -sign;
    }
    let P = 1i64;
    let Q = ((1 - D) / 4) as i64;
    // write n+1 = d*2^s
    let mut d = n + 1;
    let mut s = 0;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }
    // compute Lucas sequence
    let (mut U, mut V) = lucas_sequence(d, P, Q, n as i64);
    if U == 0 {
        return true;
    }
    for _ in 1..s {
        // U, V update for doubling
        let U2 = (U * V).rem_euclid(n as i64);
        let V2 = (V * V - 2 * mod_exp(Q.rem_euclid(n as i64) as u64, 1, n) as i64).rem_euclid(n as i64);
        U = U2;
        V = V2;
        if U == 0 {
            return true;
        }
    }
    false
}

// Baillie–PSW primality test
fn is_prime_miller(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &a in &[2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            return true;
        }
        let mut d = n - 1;
        let mut s = 0;
        while d & 1 == 0 {
            d >>= 1;
            s += 1;
        }
        let mut x = mod_exp(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = (x * x) % n;
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}
fn baillie_psw(n: u64) -> bool {
    is_prime_miller(n) && is_strong_lucas_prp(n)
}

// Newton’s Integer Square Root
fn newton_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

// Bareiss Algorithm for 3×3 Determinant
/// Standard direct formula for a 3×3 determinant.
fn det3_direct(mat: [[i64; 3]; 3]) -> i64 {
    let a = mat[0][0];
    let b = mat[0][1];
    let c = mat[0][2];
    let d = mat[1][0];
    let e = mat[1][1];
    let f = mat[1][2];
    let g = mat[2][0];
    let h = mat[2][1];
    let i = mat[2][2];
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Bareiss algorithm with zero‑pivot guard
fn det_bareiss_3x3(mat: [[i64; 3]; 3]) -> i64 {
    // Copy into a mutable 2D array
    let mut m2 = mat;
    let mut denom = 1i64;

    for k in 0..2 {
        let pivot = m2[k][k];
        // If either pivot (at k>0) or denominator is zero, fall back
        if pivot == 0 || denom == 0 {
            return det3_direct(mat);
        }
        for i in (k + 1)..3 {
            for j in (k + 1)..3 {
                // this division is now safe: denom != 0
                m2[i][j] = (m2[i][j] * pivot - m2[i][k] * m2[k][j]) / denom;
            }
        }
        denom = pivot;
    }

    // After two elimination steps, m2[2][2] is the determinant
    m2[2][2]
}

// Smith Normal Form for 2×2
fn smith_normal_form_2x2(mat: [[i64; 2]; 2]) -> (i64, i64) {
    let a = mat[0][0];
    let b = mat[0][1];
    let c = mat[1][0];
    let d = mat[1][1];
    let mut g = gcd(a.abs() as u64, b.abs() as u64);
    g = gcd(g, c.abs() as u64);
    g = gcd(g, d.abs() as u64);
    let det = a * d - b * c;
    (g as i64, (det.abs() as u64 / g) as i64)
}

// Hermite Normal Form for 2×2
fn hermite_normal_form_2x2(mat: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
    let m00 = mat[0][0];
    let m10 = mat[1][0];
    let m01 = mat[0][1];
    let m11 = mat[1][1];

    // If the entire first column is zero, g == 0 and we can't divide by it
    if m00 == 0 && m10 == 0 {
        // HNF of a matrix whose first column is zero:
        // you can choose to return `mat` unchanged, or
        // canonicalize second column into the diagonal:
        return [[0, m01], [0, m11]];
    }

    // Now gcd(m00, m10) is nonzero
    let (g, s, t) = extended_gcd(m00, m10);
    // g > 0 by construction, so these divides are safe
    let u = m00 / g;
    let v = m10 / g;
    let a = g;
    let b = ((s * m01 + t * m11) % a + a) % a;
    let d = -v * m01 + u * m11;
    [[a, b], [0, d]]
}

// LLL Reduction in 2D
fn lll_reduce_2d(mut b1: (i64, i64), mut b2: (i64, i64)) -> ((i64, i64), (i64, i64)) {
    loop {
        let dot = b2.0 * b1.0 + b2.1 * b1.1;
        let norm = b1.0 * b1.0 + b1.1 * b1.1;
        if norm == 0 {
            break;
        }
        let mu = ((2 * dot + norm) / (2 * norm)) as i64;
        let nb2 = (b2.0 - mu * b1.0, b2.1 - mu * b1.1);
        if nb2.0 * nb2.0 + nb2.1 * nb2.1 < norm {
            b2 = b1;
            b1 = nb2;
        } else {
            b2 = nb2;
            break;
        }
    }
    (b1, b2)
}

// Binary Long Division (u64 ÷ u64)
fn long_div(dividend: u64, divisor: u64) -> (u64, u64) {
    let mut q = 0u64;
    let mut r = 0u64;
    for i in (0..64).rev() {
        r = (r << 1) | ((dividend >> i) & 1);
        if r >= divisor {
            r -= divisor;
            q |= 1 << i;
        }
    }
    (q, r)
}

// Barrett Division for u64
fn barrett_div(n: u64, d: u64) -> u64 {
    // Compute mu = floor(2^64 / d) in u128
    let mu: u128 = (1u128 << 64) / (d as u128);

    // Multiply n·mu as u128, then shift down by 64 to approximate n/d
    let mut q = ((n as u128 * mu) >> 64) as u64;

    // Clamp to [0, d-1]
    if q >= d {
        q = d - 1;
    }

    q
}

// Sliding‑Window Modular Exponentiation (w=4)
fn mod_exp_sliding(base: u64, exp: u64, m: u64) -> u64 {
    let w = 4;
    let size = 1 << w;
    let mut pre = vec![0u64; size];
    pre[0] = 1 % m;
    pre[1] = base % m;
    for i in 2..size {
        pre[i] = (pre[i - 1] * pre[1]) % m;
    }
    let mut result = 1u64 % m;
    let mut bits = Vec::new();
    let mut e = exp;
    while e > 0 {
        bits.push((e & 1) as u8);
        e >>= 1;
    }
    bits.reverse();
    let mut i = 0;
    while i < bits.len() {
        if bits[i] == 0 {
            result = (result * result) % m;
            i += 1;
        } else {
            let mut l = 1;
            let maxl = (bits.len() - i).min(w);
            for ll in 2..=maxl {
                let mut val = 0;
                for j in 0..ll {
                    val = (val << 1) | bits[i + j] as usize;
                }
                if val & 1 == 0 {
                    break;
                }
                l = ll;
            }
            let mut window_val = 0;
            for j in 0..l {
                window_val = (window_val << 1) | bits[i + j] as usize;
            }
            for _ in 0..l {
                result = (result * result) % m;
            }
            result = (result * pre[window_val]) % m;
            i += l;
        }
    }
    result
}

// Montgomery Ladder Exponentiation
fn mod_exp_ladder(base: u64, exp: u64, m: u64) -> u64 {
    let mut r0 = 1 % m;
    let mut r1 = base % m;
    for i in (0..64).rev() {
        if ((exp >> i) & 1) == 0 {
            r1 = (r0 * r1) % m;
            r0 = (r0 * r0) % m;
        } else {
            r0 = (r0 * r1) % m;
            r1 = (r1 * r1) % m;
        }
    }
    r0
}

// Toom‑Cook 3‑Way Multiplication for 64‑bit
fn toom3_64(x: u64, y: u64) -> u64 {
    // use i128 everywhere so that 21*4=84‑bit shifts are legal
    let B: i128 = 1 << 21;
    let mask: i128 = B - 1;

    let x = x as i128;
    let y = y as i128;

    let a0 = x & mask;
    let a1 = (x >> 21) & mask;
    let a2 = x >> 42;
    let b0 = y & mask;
    let b1 = (y >> 21) & mask;
    let b2 = y >> 42;

    // point‑evaluations
    let p0 = a0 * b0;
    let p1 = (a0 + a1 + a2) * (b0 + b1 + b2);
    let pm1 = (a0 - a1 + a2) * (b0 - b1 + b2);
    let p2 = (a0 + 2 * a1 + 4 * a2) * (b0 + 2 * b1 + 4 * b2);
    let pinf = a2 * b2;

    // interpolation
    let r0 = p0;
    let r4 = pinf;
    let u1 = p1 - r0 - r4;
    let u2 = pm1 - r0 - r4;
    let r2 = (u1 + u2) / 2;
    let r1 = u1 - r2;
    let r3 = (p2 - r0 - 2 * r1 - 4 * r2 - 16 * r4) / 8;

    // recombine via shifts instead of overflowing multiplies:
    //
    //   r0
    // + r1 *  B      == r1 << 21
    // + r2 *  B^2    == r2 << (21*2)
    // + r3 *  B^3    == r3 << (21*3)
    // + r4 *  B^4    == r4 << (21*4)
    //
    let res = r0 + (r1 << 21) + (r2 << 42) + (r3 << 63) + (r4 << 84);

    // drop back to u64 (low 64 bits)
    res as u64
}

// Fast Walsh–Hadamard Transform (length=8)
fn fwht(a: &mut [u64; 8]) {
    let mut len = 1;
    while len < 8 {
        for i in (0..8).step_by(len * 2) {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u.wrapping_add(v);
                a[i + j + len] = u.wrapping_sub(v);
            }
        }
        len <<= 1;
    }
}

// Next lexicographic permutation
fn next_lexicographic_permutation(v: &mut [u64]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

// Next combination (k of n)
fn next_combination(comb: &mut [usize], n: usize) -> bool {
    let k = comb.len();
    for i in (0..k).rev() {
        if comb[i] < n - k + i {
            comb[i] += 1;
            for j in i + 1..k {
                comb[j] = comb[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

// Permutation ranking & unranking (n≤10)
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

fn perm_rank(v: &[u64]) -> u64 {
    let n = v.len();
    let mut rank = 0;
    let mut used = vec![false; n];
    for i in 0..n {
        let mut smaller = 0;
        for x in 0..(v[i] as usize) {
            if !used[x] {
                smaller += 1;
            }
        }
        rank += smaller as u64 * factorial((n - 1 - i) as u64);
        used[v[i] as usize] = true;
    }
    rank
}

fn perm_unrank(mut rank: u64, n: usize) -> Vec<u64> {
    let mut elems: Vec<u64> = (0..n as u64).collect();
    let mut res = Vec::with_capacity(n);
    for i in (0..n).rev() {
        let f = factorial(i as u64);
        let idx = (rank / f) as usize;
        rank %= f;
        res.push(elems.remove(idx));
    }
    res
}

// Combination ranking & unranking (n≤30,k≤15)
fn comb(n: usize, k: usize) -> u64 {
    let mut c = 1;
    for i in 1..=k {
        c = c * (n + 1 - i) as u64 / i as u64;
    }
    c
}
fn comb_rank(cmb: &[usize], n: usize) -> u64 {
    let k = cmb.len();
    let mut rank = 0;
    for i in 0..k {
        let start = if i == 0 { 0 } else { cmb[i - 1] + 1 };
        for j in start..cmb[i] {
            rank += comb(n - j - 1, k - i - 1);
        }
    }
    rank
}
fn comb_unrank(mut rank: u64, n: usize, k: usize) -> Vec<usize> {
    let mut res = Vec::with_capacity(k);
    let mut x = 0;
    for i in 0..k {
        for j in x..n {
            let c = comb(n - j - 1, k - i - 1);
            if rank < c {
                res.push(j);
                x = j + 1;
                break;
            }
            rank -= c;
        }
    }
    res
}

// Partition count p(n) via pentagonal theorem
fn partition_count(n: usize) -> u64 {
    let mut p = vec![0; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let mut k = 1;
        while {
            let g1 = k * (3 * k - 1) / 2;
            if g1 > i {
                false
            } else {
                let sign = if k % 2 == 0 { -1 } else { 1 };
                p[i] = (p[i] as i64 + sign * (p[i - g1] as i64)) as u64;
                true
            }
        } {
            k += 1;
        }
    }
    p[n]
}

// Enumerate partitions of n (simple recursive)
fn enum_partitions(n: u64) -> Vec<Vec<u64>> {
    fn helper(n: u64, max: u64, cur: &mut Vec<u64>, out: &mut Vec<Vec<u64>>) {
        if n == 0 {
            out.push(cur.clone());
            return;
        }
        // start k = min(max, n) so that n - k >= 0
        let mut k = max.min(n);
        while k > 0 {
            cur.push(k);
            helper(n - k, k, cur, out);
            cur.pop();
            k -= 1;
        }
    }
    let mut out = Vec::new();
    helper(n, n, &mut Vec::new(), &mut out);
    out
}

// Coin change: count ways (coins 1,5,10,25)
fn coin_change_count(n: usize) -> u64 {
    let coins = [1, 5, 10, 25];
    let mut dp = vec![0; n + 1];
    dp[0] = 1;
    for &c in &coins {
        for i in c..=n {
            dp[i] += dp[i - c];
        }
    }
    dp[n]
}

// Coin change: min coins
fn coin_change_min(n: usize) -> i64 {
    let coins = [1, 5, 10, 25];
    let mut dp = vec![i64::MAX; n + 1];
    dp[0] = 0;
    for &c in &coins {
        for i in c..=n {
            if dp[i - c] != i64::MAX {
                dp[i] = dp[i].min(dp[i - c] + 1);
            }
        }
    }
    if dp[n] == i64::MAX {
        -1
    } else {
        dp[n]
    }
}

// 0/1 Knapsack (n items)
fn knapsack(weights: &[usize], values: &[u64], cap: usize) -> u64 {
    let mut dp = vec![0; cap + 1];
    for i in 0..weights.len() {
        for w in (weights[i]..=cap).rev() {
            dp[w] = dp[w].max(dp[w - weights[i]] + values[i]);
        }
    }
    dp[cap]
}

// 91. Unbounded Knapsack
fn unbounded_knapsack(weights: &[usize], values: &[u64], cap: usize) -> u64 {
    let mut dp = vec![0u64; cap + 1];
    for w in 0..=cap {
        for i in 0..weights.len() {
            if weights[i] <= w {
                dp[w] = dp[w].max(dp[w - weights[i]] + values[i]);
            }
        }
    }
    dp[cap]
}

// 92. Longest Common Subsequence (length only)
fn lcs(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[m][n]
}

// 93. Longest Increasing Subsequence (O(n²))
fn lis_length(seq: &[u64]) -> usize {
    let n = seq.len();
    let mut dp = vec![1usize; n];
    let mut best = 0;
    for i in 0..n {
        for j in 0..i {
            if seq[j] < seq[i] {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
        best = best.max(dp[i]);
    }
    best
}

// 94. Levenshtein Edit Distance
fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

// 95. Damerau–Levenshtein Distance (with adjacent transpositions)
fn damerau_levenshtein(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + 1);
            }
        }
    }
    dp[m][n]
}

// 96. Matrix Chain Multiplication (min scalar multiplications)
fn matrix_chain(dims: &[usize]) -> usize {
    let n = dims.len() - 1;
    let mut dp = vec![vec![0usize; n]; n];
    for l in 2..=n {
        for i in 0..=n - l {
            let j = i + l - 1;
            dp[i][j] = usize::MAX;
            for k in i..j {
                let cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                dp[i][j] = dp[i][j].min(cost);
            }
        }
    }
    dp[0][n - 1]
}

// 97. Optimal Binary Search Tree (given key frequencies)
fn optimal_bst(freq: &[u64]) -> u64 {
    let n = freq.len();
    let mut dp = vec![vec![0u64; n]; n];
    let mut sum = vec![0u64; n + 1];
    for i in 1..=n {
        sum[i] = sum[i - 1] + freq[i - 1];
    }
    for l in 1..=n {
        for i in 0..=n - l {
            let j = i + l - 1;
            dp[i][j] = u64::MAX;
            let w = sum[j + 1] - sum[i];
            for r in i..=j {
                let c = w + (if r > i { dp[i][r - 1] } else { 0 }) + (if r < j { dp[r + 1][j] } else { 0 });
                dp[i][j] = dp[i][j].min(c);
            }
        }
    }
    dp[0][n - 1]
}

// 98. Dynamic Time Warping (absolute difference cost)
fn dtw(a: &[u64], b: &[u64]) -> u64 {
    let m = a.len();
    let n = b.len();
    let inf = u64::MAX / 4;
    let mut dp = vec![vec![inf; n + 1]; m + 1];
    dp[0][0] = 0;
    for i in 1..=m {
        dp[i][0] = inf;
    }
    for j in 1..=n {
        dp[0][j] = inf;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] > b[j - 1] {
                a[i - 1] - b[j - 1]
            } else {
                b[j - 1] - a[i - 1]
            };
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    dp[m][n]
}

// 99. Needleman–Wunsch Global Alignment (match=2, mismatch=−1, gap=−1)
fn needleman_wunsch(a: &[u8], b: &[u8]) -> i64 {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0i64; n + 1]; m + 1];
    let gap = -1;
    for i in 1..=m {
        dp[i][0] = dp[i - 1][0] + gap;
    }
    for j in 1..=n {
        dp[0][j] = dp[0][j - 1] + gap;
    }
    for i in 1..=m {
        for j in 1..=n {
            let score = if a[i - 1] == b[j - 1] { 2 } else { -1 };
            dp[i][j] = dp[i - 1][j - 1] + score;
            dp[i][j] = dp[i][j].max(dp[i - 1][j] + gap);
            dp[i][j] = dp[i][j].max(dp[i][j - 1] + gap);
        }
    }
    dp[m][n]
}

// 100. Smith–Waterman Local Alignment (match=2, mismatch=−1, gap=−1)
fn smith_waterman(a: &[u8], b: &[u8]) -> i64 {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0i64; n + 1]; m + 1];
    let mut best = 0;
    let gap = -1;
    for i in 1..=m {
        for j in 1..=n {
            let score = if a[i - 1] == b[j - 1] { 2 } else { -1 };
            dp[i][j] = (dp[i - 1][j - 1] + score).max(dp[i - 1][j] + gap).max(dp[i][j - 1] + gap).max(0);
            best = best.max(dp[i][j]);
        }
    }
    best
}

// 1. Stein’s Binary GCD Algorithm
fn stein_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            break;
        }
    }
    a << shift
}

// 2. GCD via only subtraction (no division/shifts)
fn sub_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    while a != b {
        if a > b {
            a -= b;
        } else {
            b -= a;
        }
    }
    a
}

// Binary‑Search Integer Division: ⌊a/b⌋
fn binary_div(a: u64, b: u64) -> u64 {
    if b == 0 {
        return 0;
    }

    let mut low: u64 = 0;
    let mut high: u64 = 1;

    while high.saturating_mul(b) <= a {
        high <<= 1;
    }
    while low < high {
        let mid = (low + high + 1) >> 1;
        if mid.saturating_mul(b) <= a {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

// 4. Integer Log via repeated multiplication: ⌊log_b(n)⌋
fn integer_log_mul(n: u64, b: u64) -> u32 {
    if b < 2 {
        return 0;
    }
    let mut p = b;
    let mut k = 0;
    while p <= n {
        p = p.saturating_mul(b);
        k += 1;
    }
    k
}

// 5. Integer Log via repeated division: ⌊log_b(n)⌋
fn integer_log_div(mut n: u64, b: u64) -> u32 {
    if b < 2 {
        return 0;
    }
    let mut k = 0;
    while n >= b {
        n /= b;
        k += 1;
    }
    k
}

// 6. Perfect Square Test
fn is_perfect_square(n: u64) -> bool {
    let mut lo = 0u64;
    let mut hi = (n >> 1).saturating_add(1);
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let sq = mid.saturating_mul(mid);
        if sq == n {
            return true;
        } else if sq < n {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    false
}

// 7. Perfect Power Test: n = a^k?
fn perfect_power(n: u64) -> Option<(u64, u32)> {
    if n < 2 {
        return None;
    }
    let max_k = 64 - n.leading_zeros();
    for k in 2..=max_k {
        let a = integer_nth_root(n, k);
        if a > 1 && a.pow(k) == n {
            return Some((a, k));
        }
    }
    None
}

// 8. Stern–Brocot Tree Navigation: path to num/den
fn stern_brocot_path(num: u64, den: u64) -> String {
    let mut path = String::new();
    let (mut ln, mut ld) = (0u64, 1u64);
    let (mut rn, mut rd) = (1u64, 0u64);
    while ln + rn != num || ld + rd != den {
        let mn = ln + rn;
        let md = ld + rd;
        if num * md > mn * den {
            // go right
            path.push('R');
            ln = mn;
            ld = md;
        } else {
            // go left
            path.push('L');
            rn = mn;
            rd = md;
        }
    }
    path
}

// 9. Continued Fraction Convergents
fn continued_fraction_convergents(mut num: u64, mut den: u64) -> Vec<(u64, u64)> {
    let mut coeffs = Vec::new();
    while den != 0 {
        let a = num / den;
        coeffs.push(a);
        let r = num % den;
        num = den;
        den = r;
    }
    let mut conv = Vec::new();
    let mut p0 = 1u64;
    let mut p1 = coeffs[0] as u64;
    let mut q0 = 0u64;
    let mut q1 = 1u64;
    conv.push((p1 as u64, q1 as u64));
    for &a in &coeffs[1..] {
        let p2 = a as u64 * p1 + p0;
        let q2 = a as u64 * q1 + q0;
        conv.push((p2 as u64, q2 as u64));
        p0 = p1;
        p1 = p2;
        q0 = q1;
        q1 = q2;
    }
    conv
}

// 10. Farey Sequence Generation (den ≤ n)
fn farey_sequence(n: u64) -> Vec<(u64, u64)> {
    let mut seq = vec![(0, 1), (1, n)];
    let mut a = 0u64;
    let mut b = 1u64;
    let mut c = 1u64;
    let mut d = n;
    while c <= n {
        let k = (n + b) / d;
        let e = k * c - a;
        let f = k * d - b;
        seq.push((c, d));
        a = c;
        b = d;
        c = e;
        d = f;
    }
    seq
}

// 11. Lucas Numbers: L₀=2, L₁=1, Lₙ=Lₙ₋₁+Lₙ₋₂
fn lucas(n: u32) -> u64 {
    match n {
        0 => 2,
        1 => 1,
        _ => {
            let mut a = 2u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let c = a + b;
                a = b;
                b = c;
            }
            b
        }
    }
}

// 12. Tribonacci Numbers: T₀=0, T₁=1, T₂=1, Tₙ=Tₙ₋₁+Tₙ₋₂+Tₙ₋₃
fn tribonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 | 2 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            let mut c = 1u64;
            for _ in 3..=n {
                let d = a + b + c;
                a = b;
                b = c;
                c = d;
            }
            c
        }
    }
}

// 13. Pell Numbers: P₀=0, P₁=1, Pₙ=2·Pₙ₋₁+Pₙ₋₂
fn pell(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let c = 2 * b + a;
                a = b;
                b = c;
            }
            b
        }
    }
}

// 14. Stirling Numbers of the First Kind s(n,k)
fn stirling1(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    let mut dp = vec![vec![0i64; k + 1]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        let m = i.min(k);
        for j in 1..=m {
            dp[i][j] = dp[i - 1][j - 1] - (i as i64 - 1) * dp[i - 1][j];
        }
    }
    dp[n][k]
}

// 15. Stirling Numbers of the Second Kind S(n,k)
fn stirling2(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let mut dp = vec![vec![0u64; k + 1]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        let m = i.min(k);
        for j in 1..=m {
            dp[i][j] = j as u64 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }
    dp[n][k]
}

// 16. Bell Numbers via Bell triangle
fn bell(n: usize) -> u64 {
    let mut bell = vec![vec![0u64; n + 1]; n + 1];
    bell[0][0] = 1;
    for i in 1..=n {
        bell[i][0] = bell[i - 1][i - 1];
        for j in 1..=i {
            bell[i][j] = bell[i][j - 1] + bell[i - 1][j - 1];
        }
    }
    bell[n][0]
}

// 17. Derangement Count !n, D(0)=1, D(1)=0, D(n)=(n-1)(D(n-1)+D(n-2))
fn derangement(n: u32) -> u64 {
    match n {
        0 => 1,
        1 => 0,
        _ => {
            let mut d0 = 1u64;
            let mut d1 = 0u64;
            for i in 2..=n {
                let d = (i as u64 - 1) * (d1 + d0);
                d0 = d1;
                d1 = d;
            }
            d1
        }
    }
}

// 18. Eulerian Numbers A(n,k): A(0,0)=1; A(n,k)=(n-k)*A(n-1,k-1)+(k+1)*A(n-1,k)
fn eulerian(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let mut dp = vec![vec![0u64; n]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        for j in 0..i {
            let a = if j > 0 { (i - j) as u64 * dp[i - 1][j - 1] } else { 0 };
            let b = if j < i - 1 { (j + 1) as u64 * dp[i - 1][j] } else { 0 };
            dp[i][j] = a + b;
        }
    }
    dp[n][k]
}

// 25. Eulerian Path/Circuit Check (undirected)
fn eulerian_path_circuit(adj: &Vec<Vec<bool>>) -> (bool, bool) {
    let n = adj.len();
    let mut odd = 0;
    for i in 0..n {
        let deg = adj[i].iter().filter(|&&b| b).count();
        if deg % 2 == 1 {
            odd += 1;
        }
    }
    let circuit = odd == 0;
    let path = odd == 0 || odd == 2;
    (circuit, path)
}

// 19. Narayana Numbers N(n,k) = (1/n)·C(n,k)·C(n,k-1)
fn narayana(n: u64, k: u64) -> u64 {
    if k < 1 || k > n {
        return 0;
    }
    let c1 = binomial(n, k);
    let c2 = binomial(n, k - 1);
    (c1 * c2 / n as u64) as u64
}

// 20. Motzkin Numbers via recurrence: M₀=1, M₁=1, Mₙ=((2n+1)Mₙ₋₁+(3n-3)Mₙ₋₂)/(n+2)
fn motzkin(n: usize) -> u64 {
    if n < 2 {
        return 1;
    }
    let mut m = vec![0u64; n + 1];
    m[0] = 1;
    m[1] = 1;
    for i in 2..=n {
        let num = (2 * (i as u64) + 1) * m[i - 1] + (3 * (i as u64) - 3) * m[i - 2];
        m[i] = num / (i as u64 + 2);
    }
    m[n]
}

// 21. Adjacency Matrix Powers (count paths of length k)
fn mat_mul(a: &Vec<Vec<u64>>, b: &Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let n = a.len();
    let mut c = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] = c[i][j].wrapping_add(a[i][k].wrapping_mul(b[k][j]));
            }
        }
    }
    c
}

#[allow(dead_code)]
fn mat_pow(adj: &Vec<Vec<u64>>, k: u32) -> Vec<Vec<u64>> {
    let n = adj.len();
    let mut res = vec![vec![0u64; n]; n];
    for i in 0..n {
        res[i][i] = 1;
    }
    let mut base = adj.clone();
    let mut exp = k;
    while exp > 0 {
        if exp & 1 == 1 {
            res = mat_mul(&res, &base);
        }
        base = mat_mul(&base, &base);
        exp >>= 1;
    }
    res
}

// 22. Perfect Matching Count (bitmask DP)
fn perfect_matchings(adj: &Vec<Vec<bool>>) -> u64 {
    let n = adj.len();
    let size = 1 << n;
    let mut dp = vec![0u64; size];
    dp[0] = 1;
    for mask in 0..size {
        let cnt = mask.count_ones();
        if cnt & 1 == 1 {
            continue;
        }
        if cnt == 0 {
            continue;
        }
        let i = mask.trailing_zeros() as usize;
        let mask_i = mask ^ (1 << i);
        for j in (i + 1)..n {
            if mask & (1 << j) != 0 && adj[i][j] {
                dp[mask] = dp[mask].wrapping_add(dp[mask_i ^ (1 << j)]);
            }
        }
    }
    dp[size - 1]
}

// 23. Chromatic Polynomial: brute count of k-colorings
fn chromatic_count(adj: &Vec<Vec<bool>>, k: u32) -> u64 {
    let n = adj.len();
    let mut count = 0;
    let mut colors = vec![0u32; n];
    fn dfs(i: usize, n: usize, k: u32, adj: &Vec<Vec<bool>>, colors: &mut Vec<u32>, cnt: &mut u64) {
        if i == n {
            *cnt += 1;
            return;
        }
        for c in 0..k {
            let mut ok = true;
            for j in 0..i {
                if adj[i][j] && colors[j] == c {
                    ok = false;
                    break;
                }
            }
            if ok {
                colors[i] = c;
                dfs(i + 1, n, k, adj, colors, cnt);
            }
        }
    }
    dfs(0, n, k, adj, &mut colors, &mut count);
    count
}

// 24. Spanning-Tree Count (Matrix-Tree Theorem via Bareiss)
fn bareiss_det(mut m: Vec<Vec<i64>>) -> i64 {
    let n = m.len();
    let mut denom = 1i64;
    for k in 0..n - 1 {
        if denom == 0 {
            return 0;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / denom;
            }
        }
        denom = m[k][k];
    }
    m[n - 1][n - 1]
}
fn spanning_tree_count(adj: &Vec<Vec<u64>>) -> u64 {
    let n = adj.len();
    let mut lap = vec![vec![0i64; n]; n];
    for i in 0..n {
        let mut deg = 0i64;
        for j in 0..n {
            if i != j && adj[i][j] > 0 {
                deg += 1;
                lap[i][j] = -(adj[i][j] as i64);
            }
        }
        lap[i][i] = deg;
    }
    // build minor by removing last row/col
    let mut minor = vec![vec![0i64; n - 1]; n - 1];
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            minor[i][j] = lap[i][j];
        }
    }
    bareiss_det(minor) as u64
}

// 26. Topological Sort (Kahn’s algorithm)
fn topo_sort(adj: &Vec<Vec<bool>>) -> Option<Vec<usize>> {
    let n = adj.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for v in 0..n {
            if adj[u][v] {
                indeg[v] += 1;
            }
        }
    }
    let mut q = Vec::new();
    for i in 0..n {
        if indeg[i] == 0 {
            q.push(i);
        }
    }
    let mut order = Vec::new();
    while let Some(u) = q.pop() {
        order.push(u);
        for v in 0..n {
            if adj[u][v] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    q.push(v);
                }
            }
        }
    }
    if order.len() == n {
        Some(order)
    } else {
        None
    }
}

// Strongly Connected Components (Tarjan)
fn tarjan_scc(adj: &Vec<Vec<bool>>) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index = vec![None; n];
    let mut low = vec![0; n];
    let mut onstack = vec![false; n];
    let mut stack = Vec::new();
    let mut idx = 0;
    let mut comps = Vec::new();
    fn dfs(
        u: usize,
        adj: &Vec<Vec<bool>>,
        index: &mut Vec<Option<usize>>,
        low: &mut Vec<usize>,
        stack: &mut Vec<usize>,
        onstack: &mut Vec<bool>,
        idx: &mut usize,
        comps: &mut Vec<Vec<usize>>,
    ) {
        index[u] = Some(*idx);
        low[u] = *idx;
        *idx += 1;
        stack.push(u);
        onstack[u] = true;
        for v in 0..adj.len() {
            if adj[u][v] {
                if index[v].is_none() {
                    dfs(v, adj, index, low, stack, onstack, idx, comps);
                    low[u] = low[u].min(low[v]);
                } else if onstack[v] {
                    low[u] = low[u].min(index[v].unwrap());
                }
            }
        }
        if low[u] == index[u].unwrap() {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                onstack[w] = false;
                comp.push(w);
                if w == u {
                    break;
                }
            }
            comps.push(comp);
        }
    }
    for u in 0..n {
        if index[u].is_none() {
            dfs(u, adj, &mut index, &mut low, &mut stack, &mut onstack, &mut idx, &mut comps);
        }
    }
    comps
}

// Bipartite Matching (DFS augmenting paths)
fn bipartite_match(adj: &Vec<Vec<bool>>) -> usize {
    let n = adj.len();
    let m = adj[0].len();
    let mut match_r = vec![None; m];
    fn dfs(u: usize, adj: &Vec<Vec<bool>>, seen: &mut Vec<bool>, match_r: &mut Vec<Option<usize>>) -> bool {
        for v in 0..adj[0].len() {
            if adj[u][v] && !seen[v] {
                seen[v] = true;
                if match_r[v].is_none() || dfs(match_r[v].unwrap(), adj, seen, match_r) {
                    match_r[v] = Some(u);
                    return true;
                }
            }
        }
        false
    }
    let mut result = 0;
    for u in 0..n {
        let mut seen = vec![false; m];
        if dfs(u, adj, &mut seen, &mut match_r) {
            result += 1;
        }
    }
    result
}

// Global Min‑Cut (Stoer‑Wagner)
fn stoer_wagner(mut w: Vec<Vec<u64>>) -> u64 {
    let mut n = w.len();
    let mut vertices: Vec<usize> = (0..n).collect();
    let mut best = u64::MAX;
    while n > 1 {
        let mut added = vec![false; vertices.len()];
        let mut weights = vec![0u64; vertices.len()];
        let mut prev = 0;
        for j in 0..n {
            let mut sel = None;
            for i in 0..n {
                if !added[i] && (sel.is_none() || weights[i] > weights[sel.unwrap()]) {
                    sel = Some(i);
                }
            }
            let sel = sel.unwrap();
            added[sel] = true;
            if j == n - 1 {
                best = best.min(weights[sel]);
                // merge sel and prev
                for i in 0..n {
                    w[prev][i] = w[prev][i].wrapping_add(w[sel][i]);
                    w[i][prev] = w[prev][i];
                }
                vertices.remove(sel);
                w.remove(sel);
                for row in w.iter_mut() {
                    row.remove(sel);
                }
                n -= 1;
                break;
            }
            prev = sel;
            for j in 0..n {
                if !added[j] {
                    weights[j] = weights[j].wrapping_add(w[sel][j]);
                }
            }
        }
    }
    best
}

// Graph Isomorphism (brute over permutations)
fn next_graph_permutation(v: &mut [usize]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}
fn is_isomorphic(a: &Vec<Vec<bool>>, b: &Vec<Vec<bool>>) -> bool {
    let n = a.len();
    let mut perm: Vec<usize> = (0..n).collect();
    loop {
        let mut ok = true;
        for i in 0..n {
            for j in 0..n {
                if a[i][j] != b[perm[i]][perm[j]] {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            return true;
        }
        if !next_graph_permutation(&mut perm) {
            break;
        }
    }
    false
}

// Pick’s Theorem Application ─────────────────────────────────────────

/// Given a simple lattice‐polygon (vertices in order), returns
/// (2×area, boundary_points, interior_points) via Pick’s theorem.
fn pick_theorem(poly: &[(i64, i64)]) -> (i64, i64, i64) {
    let n = poly.len();
    // 2 * signed area
    let mut area2 = 0i64;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        area2 += x1 * y2 - x2 * y1;
    }
    let area2 = area2.abs();
    // lattice points on boundary
    let mut b = 0i64;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        let dx = (x2 - x1).abs() as u64;
        let dy = (y2 - y1).abs() as u64;
        b += gcd(dx, dy) as i64;
    }
    // interior by Pick: area2 = 2I + B - 2  ⇒ I = (area2 - B + 2)/2
    let i = (area2 - b + 2) / 2;
    (area2, b, i)
}

// Manhattan Distance ─────────────────────────────────────────────────
fn manhattan(p: (i64, i64), q: (i64, i64)) -> u64 {
    let dx = (p.0 - q.0).abs() as u64;
    let dy = (p.1 - q.1).abs() as u64;
    dx + dy
}

fn chebyshev(p: (i64, i64), q: (i64, i64)) -> u64 {
    let dx = (p.0 - q.0).abs() as u64;
    let dy = (p.1 - q.1).abs() as u64;
    dx.max(dy)
}

// Integer Point‑in‑Polygon Test ──────────────────────────────────────
fn point_in_polygon(pt: (i64, i64), poly: &[(i64, i64)]) -> bool {
    let (x, y) = pt;
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[(i + 1) % n];
        if (yi > y) != (yj > y) {
            // compute intersection X‐coord of edge with horizontal line y
            let x_int = xi + (y - yi) * (xj - xi) / (yj - yi);
            if x < x_int {
                inside = !inside;
            }
        }
    }
    inside
}

// Convex Hull (Gift Wrapping) ─────────────────────────────────────────
fn orientation(a: (i64, i64), b: (i64, i64), c: (i64, i64)) -> i64 {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (cx, cy) = c;
    (bx - ax) as i64 * (cy - ay) as i64 - (by - ay) as i64 * (cx - ax) as i64
}

fn convex_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    // find leftmost lowest
    let mut left = 0;
    for i in 1..n {
        if points[i].0 < points[left].0 || (points[i].0 == points[left].0 && points[i].1 < points[left].1) {
            left = i;
        }
    }
    let mut hull = Vec::new();
    let mut p = left;
    loop {
        hull.push(points[p]);
        let mut q = (p + 1) % n;
        for r in 0..n {
            if orientation(points[p], points[q], points[r]) < 0 {
                q = r;
            }
        }
        p = q;
        if p == left {
            break;
        }
    }
    hull
}

// 43. Heap Sort
fn heap_sort(a: &mut [u64]) {
    let n = a.len();
    // build max-heap
    for start in (0..n / 2).rev() {
        sift_down(a, start, n);
    }
    // sort
    for end in (1..n).rev() {
        a.swap(0, end);
        sift_down(a, 0, end);
    }

    fn sift_down(a: &mut [u64], start: usize, end: usize) {
        let mut root = start;
        loop {
            let child = 2 * root + 1;
            if child >= end {
                break;
            }
            let mut swap = root;
            if a[swap] < a[child] {
                swap = child;
            }
            if child + 1 < end && a[swap] < a[child + 1] {
                swap = child + 1;
            }
            if swap == root {
                break;
            }
            a.swap(root, swap);
            root = swap;
        }
    }
}

// Line‑Segment Intersection
fn on_segment(p: (i64, i64), q: (i64, i64), r: (i64, i64)) -> bool {
    let (px, py) = p;
    let (qx, qy) = q;
    let (rx, ry) = r;
    px >= qx.min(rx) && px <= qx.max(rx) && py >= qy.min(ry) && py <= qy.max(ry)
}

fn segments_intersect(p1: (i64, i64), p2: (i64, i64), p3: (i64, i64), p4: (i64, i64)) -> bool {
    let o1 = orientation(p1, p2, p3);
    let o2 = orientation(p1, p2, p4);
    let o3 = orientation(p3, p4, p1);
    let o4 = orientation(p3, p4, p2);
    if (o1 > 0 && o2 < 0 || o1 < 0 && o2 > 0) && (o3 > 0 && o4 < 0 || o3 < 0 && o4 > 0) {
        return true;
    }
    // collinear cases
    if o1 == 0 && on_segment(p3, p1, p2) {
        return true;
    }
    if o2 == 0 && on_segment(p4, p1, p2) {
        return true;
    }
    if o3 == 0 && on_segment(p1, p3, p4) {
        return true;
    }
    if o4 == 0 && on_segment(p2, p3, p4) {
        return true;
    }
    false
}

// Flood‑Fill Algorithm
fn flood_fill(grid: &mut Vec<Vec<u8>>, sx: usize, sy: usize, new_color: u8) {
    let h = grid.len();
    let w = grid[0].len();
    let orig = grid[sy][sx];
    if orig == new_color {
        return;
    }
    let mut stack = vec![(sx, sy)];
    while let Some((x, y)) = stack.pop() {
        if grid[y][x] != orig {
            continue;
        }
        grid[y][x] = new_color;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
}

// Scan‑Line Polygon Fill
fn scanline_fill(poly: &[(i64, i64)], width: usize, height: usize) -> Vec<Vec<bool>> {
    let n = poly.len();
    let mut grid = vec![vec![false; width]; height];
    for y in 0..height {
        let yy = y as i64;
        let mut xs = Vec::new();
        for i in 0..n {
            let (xi, yi) = poly[i];
            let (xj, yj) = poly[(i + 1) % n];
            if (yi <= yy && yy < yj) || (yj <= yy && yy < yi) {
                let x_int = xi + (yy - yi) * (xj - xi) / (yj - yi);
                xs.push(x_int);
            }
        }
        xs.sort_unstable();
        let mut k = 0;
        while k + 1 < xs.len() {
            let x0 = xs[k].max(0) as usize;
            let x1 = xs[k + 1].min((width - 1) as i64) as usize;
            for x in x0..=x1 {
                grid[y][x] = true;
            }
            k += 2;
        }
    }
    grid
}

// Cohen–Sutherland Line Clipping ─────────────────────────────────────

const LEFT: i32 = 1;
const RIGHT: i32 = 2;
const BOTTOM: i32 = 4;
const TOP: i32 = 8;

fn region_code(x: i64, y: i64, xmin: i64, ymin: i64, xmax: i64, ymax: i64) -> i32 {
    let mut code = 0;
    if x < xmin {
        code |= LEFT;
    } else if x > xmax {
        code |= RIGHT;
    }
    if y < ymin {
        code |= BOTTOM;
    } else if y > ymax {
        code |= TOP;
    }
    code
}

fn cohen_sutherland_clip(
    mut x0: i64,
    mut y0: i64,
    mut x1: i64,
    mut y1: i64,
    xmin: i64,
    ymin: i64,
    xmax: i64,
    ymax: i64,
) -> Option<((i64, i64), (i64, i64))> {
    let mut code0 = region_code(x0, y0, xmin, ymin, xmax, ymax);
    let mut code1 = region_code(x1, y1, xmin, ymin, xmax, ymax);
    loop {
        if code0 == 0 && code1 == 0 {
            return Some(((x0, y0), (x1, y1)));
        }
        if (code0 & code1) != 0 {
            return None;
        }
        let code_out = if code0 != 0 { code0 } else { code1 };
        let (nx, ny) = if (code_out & TOP) != 0 {
            let x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
            (x, ymax)
        } else if (code_out & BOTTOM) != 0 {
            let x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
            (x, ymin)
        } else if (code_out & RIGHT) != 0 {
            let y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
            (xmax, y)
        } else {
            let y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
            (xmin, y)
        };
        if code_out == code0 {
            x0 = nx;
            y0 = ny;
            code0 = region_code(x0, y0, xmin, ymin, xmax, ymax);
        } else {
            x1 = nx;
            y1 = ny;
            code1 = region_code(x1, y1, xmin, ymin, xmax, ymax);
        }
    }
}

// ─── 40. Midpoint Circle Algorithm (Variant) ────────────────────────────────

fn midpoint_circle(cx: i64, cy: i64, r: i64) -> Vec<(i64, i64)> {
    let mut pts = Vec::new();
    let mut x = 0;
    let mut y = r;
    let mut d = 1 - r;
    while x <= y {
        pts.extend_from_slice(&[
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ]);
        if d < 0 {
            d += 2 * x + 3;
        } else {
            d += 2 * (x - y) + 5;
            y -= 1;
        }
        x += 1;
    }
    pts
}

// 41. Merge Sort
fn merge_sort(a: &mut [u64]) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    let mid = n / 2;
    merge_sort(&mut a[..mid]);
    merge_sort(&mut a[mid..]);
    let mut tmp = a.to_vec();
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < n {
        if a[i] <= a[j] {
            tmp[k] = a[i];
            i += 1;
        } else {
            tmp[k] = a[j];
            j += 1;
        }
        k += 1;
    }
    while i < mid {
        tmp[k] = a[i];
        i += 1;
        k += 1;
    }
    while j < n {
        tmp[k] = a[j];
        j += 1;
        k += 1;
    }
    a.copy_from_slice(&tmp);
}

// 42. Quick Sort (integer keys)
fn quick_sort(a: &mut [u64]) {
    if a.len() <= 1 {
        return;
    }
    let pivot = a[a.len() / 2];
    let (mut left, mut right) = (0, a.len() - 1);
    while left <= right {
        while a[left] < pivot {
            left += 1;
        }
        while a[right] > pivot {
            right = right.saturating_sub(1);
        }
        if left <= right {
            a.swap(left, right);
            left += 1;
            right = right.saturating_sub(1);
        }
    }
    if right > 0 {
        quick_sort(&mut a[..=right]);
    }
    quick_sort(&mut a[left..]);
}

// 44. Counting Sort
fn counting_sort(a: &[u64]) -> Vec<u64> {
    if a.is_empty() {
        return vec![];
    }
    let &max = a.iter().max().unwrap();
    let mut cnt = vec![0usize; (max as usize) + 1];
    for &x in a {
        cnt[x as usize] += 1;
    }
    let mut out = Vec::with_capacity(a.len());
    for (val, &c) in cnt.iter().enumerate() {
        for _ in 0..c {
            out.push(val as u64);
        }
    }
    out
}

// Radix Sort (LSD, base 256)fn radix_sort(a: &mut [u64]) {
fn radix_sort(a: &mut [u64]) {
    let n = a.len();
    let mut buf = vec![0u64; n];
    let base = 256;

    for shift in (0..64).step_by(8) {
        let mut cnt = vec![0usize; base];

        // count digits
        for &x in a.iter() {
            let d = ((x >> shift) & 0xFF) as usize;
            cnt[d] += 1;
        }

        // prefix‑sum
        for i in 1..base {
            cnt[i] += cnt[i - 1];
        }

        // scatter backwards
        for &x in a.iter().rev() {
            let d = ((x >> shift) & 0xFF) as usize;
            cnt[d] -= 1;
            buf[cnt[d]] = x;
        }

        // copy back
        a.copy_from_slice(&buf);
    }
}

// 46. Binary Indexed Tree (Fenwick Tree)
struct Fenwick {
    f: Vec<u64>,
}

impl Fenwick {
    fn from_vec(a: &[u64]) -> Self {
        let n = a.len();
        let mut f = vec![0; n];
        for i in 0..n {
            f[i] += a[i];
            let j = i | (i + 1);
            if j < n {
                f[j] += f[i];
            }
        }
        Fenwick { f }
    }
    /// sum of elements [0..=i]
    fn sum(&self, mut i: usize) -> u64 {
        let mut s = 0;
        loop {
            s += self.f[i];
            // compute the parent index
            let j = i & (i + 1);
            // if j == 0, we’re done
            if j == 0 {
                break;
            }
            // now safe to subtract
            i = j - 1;
        }
        s
    }
}

// Segment Tree (range sum)
struct SegTree {
    n: usize,
    t: Vec<u64>,
}
impl SegTree {
    fn from_vec(a: &[u64]) -> Self {
        let n = a.len();
        let mut t = vec![0; 2 * n];
        for i in 0..n {
            t[n + i] = a[i];
        }
        for i in (1..n).rev() {
            t[i] = t[2 * i] + t[2 * i + 1];
        }
        SegTree { n, t }
    }
    // sum over [l..=r]
    fn query(&self, mut l: usize, mut r: usize) -> u64 {
        let n = self.n;
        l += n;
        r += n;
        let mut s = 0;
        while l <= r {
            if l & 1 == 1 {
                s += self.t[l];
                l += 1;
            }
            if r & 1 == 0 {
                s += self.t[r];
                r -= 1;
            }
            l >>= 1;
            r >>= 1;
        }
        s
    }
}

// Quadratic Residue Test (Euler’s criterion)
fn is_quadratic_residue(a: u64, p: u64) -> bool {
    if a % p == 0 {
        return true;
    }
    mod_exp(a % p, (p - 1) / 2, p) == 1
}

// Carmichael Function λ(n)
fn carmichael(n: u64) -> u64 {
    let mut facs = trial_division_wheel(n);
    let mut counts = HashMap::new();
    for p in facs.drain(..) {
        *counts.entry(p).or_insert(0) += 1;
    }
    let mut res = 1;
    for (&p, &k) in &counts {
        let lam = if p == 2 && k >= 3 { 1 << (k - 2) } else { p.pow(k - 1) * (p - 1) };
        res = lcm(res, lam);
    }
    res
}

// Multiplicative Order of a mod n
fn multiplicative_order(a: u64, n: u64) -> Option<u64> {
    if gcd(a, n) != 1 {
        return None;
    }
    let mut cur = a % n;
    for k in 1..=n {
        if cur == 1 {
            return Some(k);
        }
        cur = cur.wrapping_mul(a) % n;
    }
    None
}

// Wilson’s Theorem Verification
fn is_wilson_prime(p: u64) -> bool {
    if p < 2 {
        return false;
    }
    let mut f = 1u64;
    for i in 1..p {
        f = f.wrapping_mul(i) % p;
    }
    f == p - 1
}

// Solovay–Strassen Primality Test
fn solovay_strassen(n: u64, k: u32) -> bool {
    if n < 2 {
        return false;
    }
    for _ in 0..k {
        let a = get_random_number() % (n - 2) + 2;
        let x = jacobi(a, n);
        let r = mod_exp(a, (n - 1) / 2, n) as i64;
        if x == 0 || (r - x) % (n as i64) != 0 {
            return false;
        }
    }
    true
}

// Fermat Factorization
fn fermat_factor(n: u64) -> Option<(u64, u64)> {
    if n < 2 {
        return None;
    }
    let mut x = integer_sqrt(n).saturating_add(1);
    loop {
        let t = x * x - n;
        let y = integer_sqrt(t);
        if y * y == t {
            return Some((x - y, x + y));
        }
        x += 1;
    }
}

// Trial Division with Wheel (2,3)
fn trial_division_wheel(mut n: u64) -> Vec<u64> {
    let mut f = Vec::new();
    for &p in &[2, 3] {
        while n % p == 0 {
            f.push(p);
            n /= p;
        }
    }
    let mut i = 5;
    while i * i <= n {
        for d in [i, i + 2].iter() {
            while n % *d == 0 {
                f.push(*d);
                n /= *d;
            }
        }
        i += 6;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

// Pocklington Primality Test (using small a values)
fn pocklington(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let facs = trial_division_wheel(n - 1);
    let mut pf = facs.clone();
    pf.sort();
    pf.dedup();
    for &a in &[2u64, 3u64] {
        if a >= n {
            continue;
        }
        if mod_exp(a, n - 1, n) != 1 {
            return false;
        }
        for &q in &pf {
            let g = gcd(mod_exp(a, (n - 1) / q, n).wrapping_sub(1), n);
            if g != 1 {
                return false;
            }
        }
        return true;
    }
    false
}

// Diffie-Hellman Key Exchange —
fn diffie_hellman() -> u64 {
    // fixed small prime and generator
    let p = 23;
    let g = 5;
    let a = (get_random_number() % (p - 2)) + 1;
    let b = (get_random_number() % (p - 2)) + 1;
    //let A = mod_exp(g, a, p);
    let B = mod_exp(g, b, p);
    mod_exp(B, a, p) // shared secret
}

// Shamir’s Secret Sharing (t-of-n) —
fn shamir_share(secret: u64, t: usize, n: usize, prime: u64) -> Vec<(u64, u64)> {
    let mut coeffs = Vec::with_capacity(t);
    coeffs.push(secret % prime);
    for _ in 1..t {
        coeffs.push(get_random_number() % prime);
    }
    let mut shares = Vec::with_capacity(n);
    for x in 1..=(n as u64) {
        let mut y = 0u64;
        for (i, &c) in coeffs.iter().enumerate() {
            y = (y + (c as u64) * mod_exp(x, i as u64, prime) as u64) % prime as u64;
        }
        shares.push((x, y as u64));
    }
    shares
}

// Linear Feedback Shift Register (16-bit taps) —
fn lfsr_next(state: &mut u16) -> u16 {
    // taps at bits 0,2,3,5 for a 16-bit register
    let bit = ((*state >> 0) ^ (*state >> 2) ^ (*state >> 3) ^ (*state >> 5)) & 1;
    *state = (*state >> 1) | (bit << 15);
    *state
}

// Blum Blum Shub Generator —
fn bbs_next(x: u64, n: u64) -> u64 {
    x.wrapping_mul(x) % n
}
// One-Time Pad (XOR) —
fn otp_encrypt(m: u64, k: u64) -> u64 {
    m ^ k
}

// Caesar Cipher (ASCII letters) —
fn caesar(ch: char, shift: u8) -> char {
    if ch.is_ascii_alphabetic() {
        let base = if ch.is_ascii_lowercase() { b'a' } else { b'A' } as u8;
        ((ch as u8 - base + shift % 26) % 26 + base) as char
    } else {
        ch
    }
}
// 65. Pollard’s p−1 factorization
fn pollards_p_minus_one(n: u64, bound: u64) -> Option<u64> {
    let mut a = 2u64;
    for p in 2..=bound {
        if is_prime_small(p) {
            a = mod_exp(a, p, n);
        }
    }
    let d = gcd(a.wrapping_sub(1), n);
    if d > 1 && d < n {
        Some(d)
    } else {
        None
    }
}

// Traveling Salesman (Held-Karp DP)
fn tsp_held_karp(dist: &Vec<Vec<u64>>) -> u64 {
    let n = dist.len();
    let size = 1 << n;
    let mut dp = vec![vec![u64::MAX / 2; n]; size];
    dp[1][0] = 0;
    for mask in 1..size {
        if mask & 1 == 0 {
            continue;
        }
        for u in 0..n {
            if mask & (1 << u) == 0 {
                continue;
            }
            let prev_mask = mask ^ (1 << u);
            if prev_mask == 0 {
                continue;
            }
            for v in 0..n {
                if prev_mask & (1 << v) != 0 {
                    dp[mask][u] = dp[mask][u].min(dp[prev_mask][v] + dist[v][u]);
                }
            }
        }
    }
    let full = size - 1;
    (0..n).map(|u| dp[full][u] + dist[u][0]).min().unwrap()
}

// Set Cover (Greedy Approximation)
fn set_cover_greedy(universe_size: usize, sets: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut covered = vec![false; universe_size];
    let mut result = Vec::new();
    while covered.iter().any(|&c| !c) {
        let (idx, _) = sets
            .iter()
            .enumerate()
            .max_by_key(|&(_, s)| s.iter().filter(|&&e| !covered[e]).count())
            .unwrap();
        result.push(idx);
        for &e in &sets[idx] {
            covered[e] = true;
        }
    }
    result
}

// Bin Packing (First Fit Decreasing)
fn bin_packing_ffd(sizes: &mut Vec<u64>, cap: u64) -> usize {
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let mut bins = Vec::new();
    for &s in sizes.iter() {
        if let Some(bin) = bins.iter_mut().find(|b| **b + s <= cap) {
            *bin += s;
        } else {
            bins.push(s);
        }
    }
    bins.len()
}

// Graph Coloring (Greedy)
fn greedy_coloring(adj: &Vec<Vec<bool>>) -> usize {
    let n = adj.len();
    let mut color = vec![None; n];
    for u in 0..n {
        let mut used = vec![false; n];
        for v in 0..n {
            if adj[u][v] {
                if let Some(c) = color[v] {
                    used[c] = true;
                }
            }
        }
        color[u] = Some(used.iter().position(|&u| !u).unwrap());
    }
    color.into_iter().map(|c| c.unwrap()).max().unwrap() + 1
}

// Maximum Independent Set (brute)
fn mis_bruteforce(adj: &Vec<Vec<bool>>) -> usize {
    let n = adj.len();
    let mut best = 0;
    for mask in 0..(1 << n) {
        let mut ok = true;
        let mut cnt = 0;
        for u in 0..n {
            if mask & (1 << u) != 0 {
                cnt += 1;
                for v in u + 1..n {
                    if mask & (1 << v) != 0 && adj[u][v] {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
        }
        if ok {
            best = best.max(cnt);
        }
    }
    best
}

// Vertex Cover (2-Approximation)
fn vertex_cover_approx(adj: &Vec<Vec<bool>>) -> Vec<(usize, usize)> {
    let n = adj.len();
    let mut used_edge = vec![vec![false; n]; n];
    let mut cover = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if adj[u][v] && !used_edge[u][v] {
                cover.push((u, v));
                for w in 0..n {
                    used_edge[u][w] = true;
                    used_edge[w][u] = true;
                    used_edge[v][w] = true;
                    used_edge[w][v] = true;
                }
            }
        }
    }
    cover
}

// Steiner Tree Approximation (MST on terminals)
fn steiner_approx(adj: &Vec<Vec<u64>>, terminals: &[usize]) -> u64 {
    let k = terminals.len();
    let mut in_mst = vec![false; k];
    let mut key = vec![u64::MAX; k];
    key[0] = 0;
    let mut total = 0;
    for _ in 0..k {
        // pick minimum‐key vertex
        let u = (0..k).filter(|&i| !in_mst[i]).min_by_key(|&i| key[i]).unwrap();
        in_mst[u] = true;
        total += key[u];
        // update neighbors
        for v in 0..k {
            if !in_mst[v] {
                let w = adj[terminals[u]][terminals[v]];
                if w < key[v] {
                    key[v] = w;
                }
            }
        }
    }
    total
}

// Job Scheduling (Shortest Processing Time first)
fn schedule_spt(jobs: &[(usize, u64)]) -> Vec<usize> {
    let mut arr = jobs.to_vec();
    arr.sort_unstable_by_key(|&(_, t)| t);
    arr.into_iter().map(|(i, _)| i).collect()
}

// Inclusion–Exclusion Principle for m sets
// Inclusion–Exclusion Principle for m sets
fn inc_ex_count(sets: &Vec<Vec<usize>>, universe: usize) -> usize {
    let m = sets.len();
    let mut total = 0;

    // make the range a `usize` range
    let max_mask = 1usize << m;
    for mask in 1usize..max_mask {
        // now count_ones() is on `usize`
        let bits = mask.count_ones() as usize;

        let mut cnt = 0;
        'x: for x in 0..universe {
            for i in 0..m {
                if (mask & (1 << i)) != 0 && !sets[i].contains(&x) {
                    continue 'x;
                }
            }
            cnt += 1;
        }

        // odd subsets add, even subtract
        if bits % 2 == 1 {
            total += cnt;
        } else {
            total -= cnt;
        }
    }

    total
}

// Burnside’s Lemma for necklace colorings
fn burnside_necklace(n: usize, k: u64) -> u64 {
    let n64 = n as u64;
    (0..n)
        .map(|r| {
            let g = gcd(r as u64, n64); // cast r and n to u64
            k.pow(g as u32) // cast gcd result to u32
        })
        .sum::<u64>()
        / n64
}

// factorial and binomial for rook & tableaux
fn fact(n: u64) -> u64 {
    (1..=n).map(|x| x as u64).product()
}
fn binom(n: usize, k: usize) -> u64 {
    (1..=k).fold(1u64, |acc, i| acc * (n + 1 - i) as u64 / i as u64)
}

// Rook polynomial coefficients for n×n board
fn rook_poly(n: usize) -> Vec<u64> {
    (0..=n).map(|r| binom(n, r).pow(2) * fact(r as u64)).collect()
}

// Permanent of a 0‑1 matrix via Ryser’s formula
// Permanent of a 0‑1 matrix via Ryser’s formula
fn permanent(a: &Vec<Vec<u64>>) -> u64 {
    let n = a.len();
    let mut res = 0i64;

    // build a usize‐typed range
    let max_mask = 1usize.checked_shl(n as u32).expect("n too large for bitmask");

    for mask in 1usize..max_mask {
        // now mask: usize, so count_ones() is defined
        let bits = mask.count_ones() as i64;

        let mut prod = 1i64;
        for j in 0..n {
            let mut sum = 0i64;
            for i in 0..n {
                // also make the 1<<i a usize‐shift
                if (mask & (1usize << i)) != 0 {
                    sum += a[i][j] as i64;
                }
            }
            prod *= sum;
        }

        if bits % 2 == 1 {
            res += prod;
        } else {
            res -= prod;
        }
    }

    res as u64
}

// Vandermonde Determinant
fn vandermonde(x: &Vec<i64>) -> i64 {
    let n = x.len();
    let mut prod = 1i64;
    for i in 0..n {
        for j in i + 1..n {
            prod *= (x[j] - x[i]) as i64;
        }
    }
    prod
}

// Young Tableaux count for r×c rectangle via hook‐length
fn young_tableaux(rc: (usize, usize)) -> u64 {
    let (r, c) = rc;
    let n = (r * c) as u64;
    let mut denom = 1u64;
    for i in 0..r {
        for j in 0..c {
            let hook = (r - i) + (c - j) - 1;
            denom *= hook as u64;
        }
    }
    fact(n) / denom
}

// Fibonacci via Matrix Exponentiation
fn fib_matrix(n: u64) -> u64 {
    fn mul(a: [[u64; 2]; 2], b: [[u64; 2]; 2]) -> [[u64; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    }
    let mut res = [[1, 0], [0, 1]];
    let mut base = [[1, 1], [1, 0]];
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            res = mul(res, base);
        }
        base = mul(base, base);
        exp >>= 1;
    }
    res[0][1]
}

// Frobenius Coin Problem (two denominations)
fn frobenius(a: u64, b: u64) -> u64 {
    if gcd(a, b) != 1 {
        return 0;
    }
    a * b - a - b
}

// Partition Function p(n) mod m via pentagonal recurrence
fn partition_mod(n: usize, m: u64) -> u64 {
    let mut p = vec![0u64; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let mut k = 1;
        while {
            let g1 = k * (3 * k - 1) / 2;
            let g2 = k * (3 * k + 1) / 2;
            if g1 > i {
                false
            } else {
                let sign = if k % 2 == 0 { m - 1 } else { 1 };
                p[i] = (p[i] + sign * p[i - g1] % m) % m;
                if g2 <= i {
                    p[i] = (p[i] + sign * p[i - g2] % m) % m;
                }
                true
            }
        } {
            k += 1;
        }
    }
    p[n]
}

// q‑Analog [n]_q = (q^n - 1)/(q - 1)
fn q_analog(n: u64, q: u64) -> u64 {
    if q == 1 {
        n
    } else {
        (q.pow(n as u32) - 1) / (q - 1)
    }
}

// Cyclotomic Polynomial Evaluation Φ_n(x)
// Cyclotomic Polynomial Evaluation Φ_n(x)
fn cyclotomic(n: u64, x: u64) -> u64 {
    if n == 1 {
        return x + 1;
    }

    let mut result = 1u64;
    for d in 1..=n {
        if n % d == 0 {
            let term = if x == 1 {
                // (1^d - 1)/(1 - 1) is d
                d
            } else {
                // Promote to 128 bits so pow and sub never overflow
                let xp: u128 = (x as u128).wrapping_pow(d as u32);
                let numerator = xp.saturating_sub(1); // always >= 0 in u128
                let denominator = (x - 1) as u128; // nonzero
                (numerator / denominator) as u64 // fits back in u64
            };
            result = result.saturating_mul(term);
        }
    }
    result
}

// Möbius Inversion: given g up to N, recover f
fn mobius_inversion(g: &[u64]) -> Vec<i64> {
    let n = g.len() - 1;
    let mut mu = vec![0i64; n + 1];
    for i in 1..=n {
        mu[i] = mobius(i as u64);
    }
    let mut f = vec![0i64; n + 1];
    for i in 1..=n {
        let mut sum = 0i64;
        for d in 1..=i {
            if i % d == 0 {
                sum += mu[d] * g[i / d] as i64;
            }
        }
        f[i] = sum;
    }
    f
}

// Ramsey Number Bounds for small (a,b)
fn ramsey_bounds(a: usize, b: usize) -> (usize, usize) {
    match (a, b) {
        (3, 3) => (6, 6),
        (3, 4) | (4, 3) => (9, 9),
        (4, 4) => (18, 25),
        (3, 5) | (5, 3) => (14, 25),
        _ => (0, 0),
    }
}

// Bellman–Ford Shortest Paths
fn bellman_ford(n: usize, edges: &[(usize, usize, i64)], src: usize) -> Option<Vec<i64>> {
    let mut dist = vec![i64::MAX / 2; n];
    dist[src] = 0;
    for _ in 0..n - 1 {
        for &(u, v, w) in edges {
            if dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
            }
        }
    }
    // check for negative cycle
    for &(u, v, w) in edges {
        if dist[u] + w < dist[v] {
            return None;
        }
    }
    Some(dist)
}

// Floyd–Warshall All-Pairs Shortest Paths
fn floyd_warshall(dist: &mut Vec<Vec<i64>>) {
    let n = dist.len();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let nd = dist[i][k].saturating_add(dist[k][j]);
                if nd < dist[i][j] {
                    dist[i][j] = nd;
                }
            }
        }
    }
}

// 71. Graph Diameter via BFS (unweighted)
fn graph_diameter(adj: &Vec<Vec<usize>>) -> usize {
    let n = adj.len();
    let mut maxd = 0;
    let mut dist = vec![usize::MAX; n];
    for s in 0..n {
        dist.fill(usize::MAX);
        let mut q = VecDeque::new();
        dist[s] = 0;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        for &d in &dist {
            if d != usize::MAX && d > maxd {
                maxd = d;
            }
        }
    }
    maxd
}

// 72. Articulation Points (Tarjan)
fn articulation_points(adj: &Vec<Vec<usize>>) -> Vec<usize> {
    let n = adj.len();
    let mut time = 0;
    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![usize::MAX; n];
    let mut ap = vec![false; n];

    fn dfs(
        u: usize,
        time: &mut usize,
        disc: &mut [usize],
        low: &mut [usize],
        parent: &mut [usize],
        ap: &mut [bool],
        adj: &Vec<Vec<usize>>,
    ) {
        *time += 1;
        disc[u] = *time;
        low[u] = *time;
        let mut children = 0;
        for &v in &adj[u] {
            if disc[v] == 0 {
                children += 1;
                parent[v] = u;
                dfs(v, time, disc, low, parent, ap, adj);
                low[u] = low[u].min(low[v]);
                if parent[u] == usize::MAX && children > 1 {
                    ap[u] = true;
                }
                if parent[u] != usize::MAX && low[v] >= disc[u] {
                    ap[u] = true;
                }
            } else if v != parent[u] {
                low[u] = low[u].min(disc[v]);
            }
        }
    }

    for u in 0..n {
        if disc[u] == 0 {
            dfs(u, &mut time, &mut disc, &mut low, &mut parent, &mut ap, adj);
        }
    }
    ap.iter().enumerate().filter(|&(_, &is_ap)| is_ap).map(|(i, _)| i).collect()
}

// 73. Bridge Finding (Tarjan)
fn find_bridges(adj: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let n = adj.len();
    let mut time = 0;
    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![usize::MAX; n];
    let mut bridges = Vec::new();

    fn dfs(
        u: usize,
        time: &mut usize,
        disc: &mut [usize],
        low: &mut [usize],
        parent: &mut [usize],
        adj: &Vec<Vec<usize>>,
        bridges: &mut Vec<(usize, usize)>,
    ) {
        *time += 1;
        disc[u] = *time;
        low[u] = *time;
        for &v in &adj[u] {
            if disc[v] == 0 {
                parent[v] = u;
                dfs(v, time, disc, low, parent, adj, bridges);
                low[u] = low[u].min(low[v]);
                if low[v] > disc[u] {
                    bridges.push((u, v));
                }
            } else if v != parent[u] {
                low[u] = low[u].min(disc[v]);
            }
        }
    }

    for u in 0..n {
        if disc[u] == 0 {
            dfs(u, &mut time, &mut disc, &mut low, &mut parent, adj, &mut bridges);
        }
    }
    bridges
}

// 76. Bron–Kerbosch Maximal Cliques
fn bron_kerbosch(r: Vec<usize>, mut p: Vec<usize>, mut x: Vec<usize>, adj: &Vec<Vec<bool>>, cliques: &mut Vec<Vec<usize>>) {
    // If no more candidates and no excluded, r is a maximal clique
    if p.is_empty() && x.is_empty() {
        cliques.push(r);
        return;
    }

    // We iterate over a *snapshot* of p, so we clone once here
    for &v in &p.clone() {
        // 1) R₂ = R ∪ {v}
        let mut r2 = r.clone();
        r2.push(v);

        // 2) P₂ = P ∩ N(v), X₂ = X ∩ N(v)
        let p2: Vec<usize> = p.iter().filter(|&&u| adj[v][u]).cloned().collect();
        let x2: Vec<usize> = x.iter().filter(|&&u| adj[v][u]).cloned().collect();

        // 3) Recurse
        bron_kerbosch(r2, p2, x2, adj, cliques);

        // 4) Move v from P to X for the next iteration
        p.retain(|&u| u != v);
        x.push(v);
    }
}

// 77. Greedy Dominating Set
fn greedy_dominating_set(adj: &Vec<Vec<bool>>) -> Vec<usize> {
    let n = adj.len();
    let mut dominated = vec![false; n];
    let mut ds = Vec::new();
    while dominated.iter().any(|&d| !d) {
        let mut best = 0;
        let mut best_u = 0;
        for u in 0..n {
            let cover = adj[u].iter().enumerate().filter(|&(v, &e)| e && !dominated[v]).count() + if !dominated[u] { 1 } else { 0 };
            if cover > best {
                best = cover;
                best_u = u;
            }
        }
        ds.push(best_u);
        dominated[best_u] = true;
        for v in 0..n {
            if adj[best_u][v] {
                dominated[v] = true;
            }
        }
    }
    ds
}

// 80. Closest Pair of Points (brute‑force)
fn closest_pair(points: &[(i64, i64)]) -> ((i64, i64), (i64, i64), u64) {
    let mut best = u64::MAX;
    let mut pair = (points[0], points[0]);
    let n = points.len();
    for i in 0..n {
        for j in i + 1..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dist2 = (dx * dx + dy * dy) as u64;
            if dist2 < best {
                best = dist2;
                pair = (points[i], points[j]);
            }
        }
    }
    (pair.0, pair.1, best)
}

// ─── 83. Graham Scan Convex Hull ────────────────────────────────────────────
#[allow(dead_code)]
fn graham_hull(pts: &Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    let mut p = pts.clone();
    p.sort_unstable();
    //let n = p.len();
    let mut lower = Vec::new();
    for &pt in &p {
        while lower.len() >= 2 && orientation(lower[lower.len() - 2], lower[lower.len() - 1], pt) <= 0 {
            lower.pop();
        }
        lower.push(pt);
    }
    let mut upper = Vec::new();
    for &pt in p.iter().rev() {
        while upper.len() >= 2 && orientation(upper[upper.len() - 2], upper[upper.len() - 1], pt) <= 0 {
            upper.pop();
        }
        upper.push(pt);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ─── 84. Rotating Calipers (Diameter) ──────────────────────────────────────

fn diameter(pts: &Vec<(i64, i64)>) -> u64 {
    let ch = graham_hull(pts);
    let m = ch.len();
    if m < 2 {
        return 0;
    }
    let mut j = 1;
    let mut best = 0;
    for i in 0..m {
        let ni = (i + 1) % m;
        while orientation(ch[i], ch[ni], ch[(j + 1) % m]).abs() > orientation(ch[i], ch[ni], ch[j]).abs() {
            j = (j + 1) % m;
        }
        best = best.max(dist2(ch[i], ch[j]));
        best = best.max(dist2(ch[ni], ch[j]));
    }
    best
}

// ─── 86. Convex Polygon Triangulation (fan) ─────────────────────────────────
#[allow(dead_code)]
fn triangulate_convex(poly: &Vec<(i64, i64)>) -> Vec<(usize, usize, usize)> {
    let n = poly.len();
    let mut tris = Vec::new();
    for i in 1..n - 1 {
        tris.push((0, i, i + 1));
    }
    tris
}

// ─── 87. Minkowski Sum (naïve + hull) ───────────────────────────────────────

fn minkowski_sum(a: &Vec<(i64, i64)>, b: &Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    let mut sum = Vec::new();
    for &pa in a {
        for &pb in b {
            sum.push((pa.0 + pb.0, pa.1 + pb.1));
        }
    }
    graham_hull(&sum)
}

// ─── 88. Visibility Graph (naïve) ───────────────────────────────────────────
#[allow(dead_code)]
fn visibility_graph(poly: &Vec<(i64, i64)>) -> Vec<Vec<bool>> {
    let n = poly.len();
    let mut g = vec![vec![false; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let mut ok = true;
            for k in 0..n {
                let k2 = (k + 1) % n;
                if segments_intersect(poly[i], poly[j], poly[k], poly[k2]) {
                    ok = false;
                    break;
                }
            }
            if ok {
                g[i][j] = true;
                g[j][i] = true;
            }
        }
    }
    g
}

// 93. Integer Matrix Multiplication (Strassen) for 2×2
#[allow(dead_code)]
fn strassen_2x2(a: [[i64; 2]; 2], b: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
    let m1 = (a[0][0] + a[1][1]) * (b[0][0] + b[1][1]);
    let m2 = (a[1][0] + a[1][1]) * b[0][0];
    let m3 = a[0][0] * (b[0][1] - b[1][1]);
    let m4 = a[1][1] * (b[1][0] - b[0][0]);
    let m5 = (a[0][0] + a[0][1]) * b[1][1];
    let m6 = (a[1][0] - a[0][0]) * (b[0][0] + b[0][1]);
    let m7 = (a[0][1] - a[1][1]) * (b[1][0] + b[1][1]);
    [[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]]
}

// 96. Eigenvalue Bounds (Gershgorin)
#[allow(dead_code)]
fn gershgorin_bounds(mat: &Vec<Vec<i64>>) -> Vec<(i64, i64)> {
    let n = mat.len();
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        let center = mat[i][i] as i64;
        let mut radius = 0i64;
        for j in 0..n {
            if j != i {
                radius += mat[i][j].abs() as i64;
            }
        }
        bounds.push((center - radius, center + radius));
    }
    bounds
}

// 97. Matrix Rank via Fraction‑Free Row Reduction
fn matrix_rank(mut a: Vec<Vec<i64>>) -> usize {
    let m = a.len();
    let n = a[0].len();
    let mut rank = 0;
    for c in 0..n {
        if rank == m {
            break;
        }
        // find a nonzero pivot in column c at or below row `rank`
        let mut piv = rank;
        while piv < m && a[piv][c] == 0 {
            piv += 1;
        }
        if piv == m {
            continue;
        }
        // swap pivot into place
        a.swap(rank, piv);

        // eliminate all other rows
        for i in 0..m {
            if i != rank && a[i][c] != 0 {
                let ai = a[i][c];
                let ar = a[rank][c];
                for j in c..n {
                    // use wrapping arithmetic to avoid panic
                    let prod1 = ai.wrapping_mul(a[rank][j]);
                    let prod2 = ar.wrapping_mul(a[i][j]);
                    a[i][j] = prod1.wrapping_sub(prod2);
                }
            }
        }

        rank += 1;
    }
    rank
}

// 98. Condition Number Estimation for 2×2 matrix (∞-norm)
fn condition_number_2x2(a: [[i64; 2]; 2]) -> u64 {
    let norm_a = ((a[0][0].abs() + a[0][1].abs()).max(a[1][0].abs() + a[1][1].abs())) as u64;
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det == 0 {
        return u64::MAX;
    }
    let norm_inv = ((a[1][1].abs() + a[0][1].abs()).max(a[0][0].abs() + a[1][0].abs())) as u64;
    norm_a * norm_inv / det.abs() as u64
}

// 99. Sparse Matrix-Vector Multiplication
#[allow(dead_code)]
fn spmv(rows: &Vec<Vec<(usize, i64)>>, x: &Vec<i64>) -> Vec<i64> {
    let m = rows.len();
    let mut y = vec![0i64; m];
    for i in 0..m {
        for &(j, v) in &rows[i] {
            y[i] += v * x[j];
        }
    }
    y
}

// 100. Singular Value Bounds via Frobenius and max-entry norms
fn singular_value_bounds(a: &Vec<Vec<i64>>) -> (u64, u64) {
    let mut frob2 = 0u64;
    let mut maxv = 0i64;
    for row in a {
        for &v in row {
            let av = v.abs() as u64;
            frob2 += av * av;
            maxv = maxv.max(v.abs());
        }
    }
    let upper = integer_sqrt(frob2);
    let lower = maxv as u64;
    (lower, upper)
}

// Brent’s Pollard Rho
fn pollard_rho_brent(n: u64) -> Option<u64> {
    if n < 2 {
        return None;
    }

    // seed values are all u64
    let mut y: u64 = get_random_number() % n;
    let c: u64 = get_random_number() % n;
    let m: u64 = (get_random_number() % (n - 1)) + 1;

    // counters / accumulators must be u64 as well
    let mut g: u64 = 1;
    let mut r: u64 = 1;
    let mut q: u64 = 1;

    let mut x: u64 = 0;
    let mut ys: u64 = 0;

    while g == 1 {
        x = y;
        // “tortoise” step
        for _ in 0..r {
            y = (y.wrapping_mul(y).wrapping_add(c)) % n;
        }
        let mut k: u64 = 0;
        // “hare” in blocks of size m
        while k < r && g == 1 {
            ys = y;
            for _ in 0..m {
                y = (y.wrapping_mul(y).wrapping_add(c)) % n;
                let diff = if x > y { x - y } else { y - x };
                // now q is u64, diff is u64 → wrapping_mul resolves!
                q = q.wrapping_mul(diff) % n;
            }
            g = gcd(q, n);
            k += m;
        }
        r <<= 1;
    }

    // if we hit a failure cycle, backtrack
    if g == n {
        loop {
            ys = (ys.wrapping_mul(ys).wrapping_add(c)) % n;
            let diff = if x > ys { x - ys } else { ys - x };
            g = gcd(diff, n);
            if g > 1 {
                break;
            }
        }
    }

    if g == n {
        None
    } else {
        Some(g)
    }
}

fn solve_quadratic_mod(a: u64, b: u64, c: u64, p: u64) -> Vec<u64> {
    // 1) Compute `2a mod p` as a u64
    let two_a = (2u128 * (a as u128) % p as u128) as u64;

    // 2) Find its modular inverse in the i64‐world
    let inv2a_i = mod_inv(two_a as i64, p as i64).expect("2*a should be invertible mod p");

    // 3) Convert that i64 back into [0..p) as u64
    let inv2a = (inv2a_i.rem_euclid(p as i64)) as u64;

    // 4) Compute discriminant d = b^2 − 4ac  (mod p)
    let b2 = (b as u128 * b as u128 % p as u128) as u64;
    let four_ac = ((4u128 * (a as u128) % p as u128) * (c as u128) % p as u128) as u64;
    let d = (b2 + p - (four_ac % p)) % p;

    // 5) Try to take a square root mod p
    if let Some(s) = tonelli_shanks(d, p) {
        // numerator = −b ± s  mod p
        let neg_b = (p - (b % p)) % p;
        let num1 = (neg_b + s) % p;
        let num2 = (neg_b + (p - s)) % p;

        // the two solutions:
        let x1 = num1.wrapping_mul(inv2a) % p;
        let x2 = num2.wrapping_mul(inv2a) % p;

        if x1 == x2 {
            vec![x1]
        } else {
            vec![x1, x2]
        }
    } else {
        // no solution
        vec![]
    }
}

// 3. Rabin Cryptosystem Core
fn rabin_encrypt(m: u64, n: u64) -> u64 {
    m.wrapping_mul(m) % n
}
fn rabin_decrypt(c: u64, p: u64, q: u64) -> Vec<u64> {
    let n = p * q;
    let r_p = tonelli_shanks(c % p, p).unwrap_or(0);
    let r_q = tonelli_shanks(c % q, q).unwrap_or(0);
    let inv_p = mod_inv(p as i64, q as i64).unwrap() as u64;
    let inv_q = mod_inv(q as i64, p as i64).unwrap() as u64;
    let mut res = Vec::new();
    for &s_p in &[r_p, (p - r_p) % p] {
        for &s_q in &[r_q, (q - r_q) % q] {
            let x = (s_p.wrapping_mul(q).wrapping_mul(inv_q) + s_q.wrapping_mul(p).wrapping_mul(inv_p)) % n;
            res.push(x);
        }
    }
    res
}

// 4. Merkle-Hellman Knapsack
#[allow(dead_code)]
fn mh_keygen(n: usize) -> (Vec<u64>, u64, u64) {
    let mut w = Vec::with_capacity(n);
    let mut sum = 0;
    for _ in 0..n {
        let x = sum + (get_random_number() % 10 + 1);
        w.push(x);
        sum += x;
    }
    let q = sum + (get_random_number() % 10 + 1);
    let mut r = get_random_number() % q;
    while gcd(r, q) != 1 {
        r = get_random_number() % q;
    }
    let beta = w.iter().map(|&wi| wi * r % q).collect();
    (beta, q, r)
}
fn mh_encrypt(pubkey: &Vec<u64>, msg: &[u8]) -> u64 {
    msg.iter().zip(pubkey).map(|(&b, &k)| (b as u64) * k).sum()
}

#[allow(dead_code)]
fn mh_decrypt(w: &Vec<u64>, q: u64, r: u64, c: u64) -> Vec<u8> {
    let inv_r = mod_inv(r as i64, q as i64).unwrap() as u64;
    let mut s = c.wrapping_mul(inv_r) % q;
    let mut m = Vec::new();
    for &wi in w.iter().rev() {
        if wi <= s {
            m.push(1);
            s -= wi;
        } else {
            m.push(0);
        }
    }
    m.reverse();
    m
}

// 5. Recurrence Relation Solver (companion matrix)
#[allow(dead_code)]
fn rec_nth(coefs: &[i64], init: &[i64], n: u64) -> i64 {
    let k = coefs.len();
    if n < k as u64 {
        return init[n as usize];
    }
    // build matrix
    let mut m = vec![vec![0i64; k]; k];
    for i in 0..k - 1 {
        m[i + 1][i] = 1;
    }
    for j in 0..k {
        m[0][j] = coefs[j];
    }
    let mp = mat_pow_i64(m, n - (k as u64 - 1));
    let mut res = 0i64;
    for i in 0..k {
        res += mp[0][i] as i64 * init[k - 1 - i] as i64;
    }
    res as i64
}

#[allow(dead_code)]
fn mat_mul_i64(a: Vec<Vec<i64>>, b: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let n = a.len();
    let mut c = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
fn mat_pow_i64(mut m: Vec<Vec<i64>>, mut e: u64) -> Vec<Vec<i64>> {
    let n = m.len();
    let mut res = vec![vec![0; n]; n];
    for i in 0..n {
        res[i][i] = 1;
    }
    while e > 0 {
        if e & 1 == 1 {
            res = mat_mul_i64(res, m.clone());
        }
        m = mat_mul_i64(m.clone(), m.clone());
        e >>= 1;
    }
    res
}

// 6. Linear Diophantine Solver ax+by+cz=d
fn solve_linear3(a: i64, b: i64, c: i64, d: i64) -> Option<(i64, i64, i64)> {
    // 1) gcd of a,b
    let (g, x0, y0) = extended_gcd(a, b);

    // 2) gcd of g,c
    let (g2, u, z0) = extended_gcd(g, c);

    // If g2 == 0, then a==b==c==0.
    // Equation is 0·x + 0·y + 0·z = d.
    if g2 == 0 {
        return if d == 0 {
            // any (x,y,z) works; choose (0,0,0)
            Some((0, 0, 0))
        } else {
            None
        };
    }

    // Now g2 != 0, safe to mod
    if d % g2 != 0 {
        return None;
    }

    // Scale the particular solution by d/g2
    let k = d / g2;
    let x00 = x0 * u * k;
    let y00 = y0 * u * k;
    let z00 = z0 * k;

    Some((x00, y00, z00))
}

// 7. kShortest Paths (Yen’s, simple BFS–Dijkstra)
#[allow(dead_code)]
fn k_shortest(src: usize, dst: usize, adj: &Vec<Vec<(usize, u64)>>, k: usize) -> Vec<u64> {
    let mut res = Vec::new();
    let mut pq = BinaryHeap::new();
    pq.push((0xFFFFFFFFFFFFFFFF, vec![src])); // max-heap: store (-cost, path)
    while let Some((nc, path)) = pq.pop() {
        let cost = !nc;
        let u = *path.last().unwrap();
        if u == dst {
            res.push(cost);
            if res.len() >= k {
                break;
            }
        }
        for &(v, w) in &adj[u] {
            if !path.contains(&v) {
                let mut p2 = path.clone();
                p2.push(v);
                pq.push((!(cost + w), p2));
            }
        }
    }
    res
}

// Continued Fraction Factorization (basic)
#[allow(dead_code)]
fn cf_factor(n: u64) -> Option<u64> {
    if n % 2 == 0 {
        return Some(2);
    }
    let a0 = integer_sqrt(n);
    if a0 * a0 == n {
        return Some(a0);
    }

    let mut p_prev = 0u64;
    let mut p = a0;
    let mut q = n - a0 * a0;
    let mut a = a0;
    let mut h2 = 0u64;
    let mut h1 = 1u64;
    let mut k2 = 1u64;
    let mut k1 = 0u64;

    for _ in 0..5000 {
        // 1) Next convergent h/k
        let h = a.wrapping_mul(h1).wrapping_add(h2);
        let k = a.wrapping_mul(k1).wrapping_add(k2);

        // 2) f = h² − n·k², skip on overflow
        let hh = match h.checked_mul(h) {
            Some(v) => v,
            None => continue,
        };
        let kk = match k.checked_mul(k) {
            Some(v) => v,
            None => continue,
        };
        let nk = match (n as u128).checked_mul(kk as u128) {
            Some(v) => v,
            None => continue,
        };

        let af = if (hh as u128) >= nk {
            (hh as u128 - nk) as u64
        } else {
            (nk - hh as u128) as u64
        };

        // 3) perfect square?
        let s = integer_sqrt(af);
        if s != 0 && s * s == af {
            let g = gcd(h + s, n);
            if g > 1 && g < n {
                return Some(g);
            }
        }

        // 4) advance CF terms—but in i128 to avoid underflow
        //    numerator = n - p_next^2
        let p_next = a.wrapping_mul(q).wrapping_sub(p_prev);
        let numerator = (n as i128) - (p_next as i128) * (p_next as i128);
        if numerator < 0 {
            // numerical glitch—skip this iteration
            p_prev = p;
            p = p_next;
            // keep q and a unchanged so we don’t break the CF stream
            h2 = h1;
            h1 = h;
            k2 = k1;
            k1 = k;
            continue;
        }
        // guard division by zero
        if q == 0 {
            break;
        }
        let q_next = (numerator / (q as i128)) as u64;
        let a_next = (a0 + p_next) / q_next;

        // rotate state
        p_prev = p;
        p = p_next;
        q = q_next;
        a = a_next;
        h2 = h1;
        h1 = h;
        k2 = k1;
        k1 = k;
    }
    None
}

// RSA Key Generation
fn rsa_keygen() -> (u64, u64, u64) {
    let p = next_prime(get_random_number() % 1000 + 100);
    let mut q;
    loop {
        q = next_prime(get_random_number() % 1000 + 100);
        if q != p {
            break;
        }
    }
    let n = p * q;
    let phi = (p - 1) * (q - 1);
    let e = 65537 % phi;
    let d = mod_inv(e as i64, phi as i64).unwrap() as u64;
    (n, e, d)
}

// ─── 17. ElGamal Key Generation ────────────────────────────────────────────
#[allow(dead_code)]
fn elgamal_keygen() -> (u64, u64, u64, u64) {
    let p = next_prime(get_random_number() % 1000 + 100);
    let g = primitive_root(p);
    let x = (get_random_number() % (p - 2)) + 1;
    let h = mod_exp(g, x, p);
    (p, g, h, x)
}

// ─── 19. DSA Core ─────────────────────────────────────────────────────────

fn dsa_sign(m: u64, p: u64, q: u64, g: u64, x: u64) -> (u64, u64) {
    let k = (get_random_number() % (q - 1)) + 1;
    let r = mod_exp(g, k, p) % q;
    let kinv = mod_inv(k as i64, q as i64).unwrap() as u64;
    let s = (kinv * ((m + x * r) % q)) % q;
    (r, s)
}
fn dsa_verify(m: u64, r: u64, s: u64, p: u64, q: u64, g: u64, y: u64) -> bool {
    if r == 0 || r >= q || s == 0 || s >= q {
        return false;
    }
    let w = mod_inv(s as i64, q as i64).unwrap() as u64;
    let u1 = (m * w) % q;
    let u2 = (r * w) % q;
    let v = (mod_exp(g, u1, p) * mod_exp(y, u2, p) % p) % q;
    v == r
}

// ─── 53. Generating Function Coefficients ─────────────────────────────────

fn gf_coeff(P: &[i64], Q: &[i64], N: usize) -> Vec<i64> {
    let d = Q.len() - 1;
    let mut a = vec![0i64; N + 1];
    for n in 0..=N {
        let mut sum = if n < P.len() { P[n] } else { 0 };
        for i in 1..=d {
            if n >= i {
                sum -= a[n - i] * Q[i];
            }
        }
        a[n] = sum / Q[0];
    }
    a
}

// ─── 81. Range Tree Queries (1D) ──────────────────────────────────────────

struct RangeTree1D {
    xs: Vec<i64>,
}

impl RangeTree1D {
    fn new(mut xs: Vec<i64>) -> Self {
        xs.sort_unstable();
        RangeTree1D { xs }
    }
    fn query(&self, l: i64, r: i64) -> usize {
        let left = self.xs.binary_search(&l).unwrap_or_else(|i| i);
        let right = self.xs.binary_search(&r).unwrap_or_else(|i| i);
        if right >= left {
            right - left
        } else {
            0
        }
    }
}

fn run_program(idx: u8) -> u64 {
    let gas_start = unsafe { gas() };

    match idx {
        0 => {
            let n = (get_random_number() % 30) as u32;
            call_log(3, None, &format!("tribonacci({}) = {}", n, tribonacci(n)));
        }

        1 => {
            // Narayana Numbers
            let n = (get_random_number() % 20) + 1;
            let k = (get_random_number() % n) + 1;
            call_log(3, None, &format!("narayana({}, {}) = {}", n, k, narayana(n, k)));
        }
        2 => {
            // Motzkin Numbers
            let n = (get_random_number() % 20) as usize;
            call_log(3, None, &format!("motzkin({}) = {}", n, motzkin(n)));
        }

        3 => {
            // Quadratic Residue Test
            let p = ((get_random_number() % 50) | 1) + 2;
            let a = get_random_number() % p;
            call_log(
                3,
                None,
                &format!("is_quadratic_residue({}, {}) = {}", a, p, is_quadratic_residue(a, p)),
            );
        }

        4 => {
            // Wilson's Theorem
            let p = ((get_random_number() % 50) | 1) + 2;
            call_log(3, None, &format!("is_wilson_prime({}) = {}", p, is_wilson_prime(p)));
        }
        5 => {
            // Solovay Strassen Test
            let n = (get_random_number() % 1000) + 3 | 1;
            call_log(3, None, &format!("solovay_strassen({}) = {}", n, solovay_strassen(n, 5)));
        }
        6 => {
            // Fermat Factorization ;
            let n = ((get_random_number() % 500) + 4) * 2 + 1;
            call_log(3, None, &format!("fermat_factor({}) = {:?}", n, fermat_factor(n)));
        }

        7 => {
            // Diffie-Hellman Shared Secret
            call_log(3, None, &format!("diffie hellman {}", diffie_hellman()));
        }

        8 => {
            // LFSR Sequence (16-bit)
            let mut state = (get_random_number() & 0xFFFF) as u16;
            // generate 8 bits
            let mut byte = 0u8;
            for i in 0..8 {
                let bit = (lfsr_next(&mut state) & 1) as u8;
                byte |= bit << i;
            }
            call_log(3, None, &format!("lfsr {:02X}", byte));
        }
        9 => {
            // Blum Blum Shub Output
            let p = 1009u64;
            let q = 1013u64;
            let n = p * q;
            let mut x = get_random_number() % n;
            while gcd(x, n) != 1 {
                x = get_random_number() % n;
            }
            x = bbs_next(x, n);
            call_log(3, None, &format!("bbs_next {}", x));
        }

        10 => {
            // Bin Packing FFD
            let mut items = (0..10).map(|_| get_random_number() % 50 + 1).collect::<Vec<_>>();
            let bins = bin_packing_ffd(&mut items, 100);
            call_log(3, None, &format!("bin_packing_ffd = {}", bins));
        }
        11 => {
            // Burnside’s Necklace
            let n = (get_random_number() % 10 + 1) as usize;
            let k = get_random_number() % 5 + 2;
            call_log(
                3,
                None,
                &format!("burnside_necklace distinct colorings = {}", burnside_necklace(n, k)),
            );
        }

        12 => {
            // Young Tableaux
            let r = (get_random_number() % 4 + 1) as usize;
            let c = (get_random_number() % 4 + 1) as usize;
            call_log(3, None, &format!("young_tableaux {}×{} → {}", r, c, young_tableaux((r, c))));
        }
        13 => {
            // Frobenius Coin
            let a = (get_random_number() % 20) + 2;
            let b = (get_random_number() % 20) + 2;
            call_log(3, None, &format!("frobenius({}, {}) = {}", a, b, frobenius(a, b)));
        }
        14 => {
            // Partition mod m
            let n = (get_random_number() % 50) as usize;
            let m = (get_random_number() % 100) + 1;
            call_log(3, None, &format!("p({}) mod {} = {}", n, m, partition_mod(n, m)));
        }
        15 => {
            // q-Analog
            let n = get_random_number() % 20;
            let q = (get_random_number() % 5) + 1;
            call_log(3, None, &format!("q_analog[{}]_{} = {}", n, q, q_analog(n, q)));
        }

        16 => {
            // Condition Number (2x2)
            let a = [
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
            ];
            let cond = condition_number_2x2(a);
            call_log(3, None, &format!("condition_number_2x2 A={:?} ={}", a, cond));
        }

        17 => {
            // Pollard-Rho Brent
            let n = (get_random_number() % 1000) | 1;
            call_log(3, None, &format!("pollard_rho_brent {:?}", pollard_rho_brent(n)));
        }
        18 => {
            // 1D Range Tree Query

            let xs: Vec<i64> = (0..20).map(|_| (get_random_number() % 100) as i64).collect();
            let tree = RangeTree1D::new(xs);
            let a = (get_random_number() % 100) as i64;
            let b = (get_random_number() % 100) as i64;
            let (l, r) = if a <= b { (a, b) } else { (b, a) };
            let cnt = tree.query(l, r);
            call_log(3, None, &format!("OneDRangeTreeQuery [{},{}] → {}", l, r, cnt));
        }

        19 => {
            // RSA KeyGen
            let (n, e, d) = rsa_keygen();
            call_log(3, None, &format!("rsa_keygen n={}, e={}, d={}", n, e, d));
        }

        20 => {
            // DSA Sign/Verify
            // fixed params
            let p = next_prime(500);
            let q = next_prime(100);
            let g = primitive_root(p);
            let x = (get_random_number() % (q - 1)) + 1;
            let y = mod_exp(g, x, p);

            let m = get_random_number() % q;
            let (r, s) = dsa_sign(m, p, q, g, x);
            let ok = dsa_verify(m, r, s, p, q, g, y);
            call_log(3, None, &format!("m={}, r={}, s={}, ok={}", m, r, s, ok));
        }
        21 => {
            //  GF Coeffs of 1/(1-x-x^2)
            let P = vec![1];
            let Q = vec![1, -1, -1];
            let n = (get_random_number() % 20) as usize;
            let coeffs = gf_coeff(&P, &Q, n);
            call_log(3, None, &format!("gf_coeff[{}] = {}", n, coeffs[n]));
        }
        22 => {
            // Legendre Symbol
            let p = ((get_random_number() % 1000) | 1) + 2;
            let a = (get_random_number() % p) as i64;
            call_log(
                3,
                None,
                &format!("legendre_symbol ( {}/{}) = {}", a, p, legendre_symbol(a, p as i64)),
            );
        }
        23 => {
            // Lucas–Lehmer Test
            let p = (get_random_number() % 50) + 2;
            call_log(3, None, &format!("lucas_lehmer M_{} is prime? {}", p, lucas_lehmer(p)));
        }
        24 => {
            // Lucas Sequence
            let n = get_random_number() % 20;
            let P = 1;
            let Q = 1;
            let m = (get_random_number() % 100) as i64 + 1;
            let (U, V) = lucas_sequence(n, P, Q, m);
            call_log(3, None, &format!("lucas_sequence U_{},V_{} mod {} = ({},{})", n, n, m, U, V));
        }

        25 => {
            // Baillie–PSW Primality Test
            let n = ((get_random_number() % 10_000) | 1) + 2;
            call_log(3, None, &format!("baillie_psw {} is prime? {}", n, baillie_psw(n)));
        }
        26 => {
            // Newton Integer √
            let n = get_random_number() % 1_000_000;
            call_log(3, None, &format!("newton_sqrt {} = {}", n, newton_sqrt(n)));
        }
        27 => {
            // Bareiss 3×3 Determinant
            let mut mat = [[0i64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    mat[i][j] = (get_random_number() % 101) as i64 - 50;
                }
            }
            call_log(3, None, &format!("det_bareiss_3x3 det = {}", det_bareiss_3x3(mat)));
        }
        28 => {
            // Smith Normal Form 2×2
            let mat = [
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
            ];
            let (d1, d2) = smith_normal_form_2x2(mat);
            call_log(3, None, &format!("smith_normal_form_2x2 diag({}, {})", d1, d2));
        }
        29 => {
            // Hermite Normal Form 2×2

            let mat = [
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
                [(get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50],
            ];
            let h = hermite_normal_form_2x2(mat);
            call_log(
                3,
                None,
                &format!("hermite_normal_form_2x2 H = [[{},{}],[{},{}]]", h[0][0], h[0][1], h[1][0], h[1][1]),
            );
        }
        30 => {
            // LLL Reduction in 2D
            let b1 = ((get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50);
            let b2 = ((get_random_number() % 101) as i64 - 50, (get_random_number() % 101) as i64 - 50);
            let (r1, r2) = lll_reduce_2d(b1, b2);
            call_log(3, None, &format!("lll_reduce_2d b1={:?}, b2={:?}", r1, r2));
        }

        31 => {
            // Montgomery Ladder ModExp
            let base = get_random_number() % 1_000;
            let exp = get_random_number() % 1_000;
            let m = (get_random_number() % 1_000) + 1;
            call_log(
                3,
                None,
                &format!("mod_exp_ladder({}, {}, {}) = {}", base, exp, m, mod_exp_ladder(base, exp, m)),
            );
        }

        32 => {
            // Stein’s Binary GCD
            let a = get_random_number() % 1_000_000;
            let b = get_random_number() % 1_000_000;
            call_log(3, None, &format!("stein_gcd({}, {}) = {}", a, b, stein_gcd(a, b)));
        }
        33 => {
            // Subtraction‑Only GCD
            let a = get_random_number() % 1_000_000;
            let b = get_random_number() % 1_000_000;
            call_log(3, None, &format!("sub_gcd({}, {}) = {}", a, b, sub_gcd(a, b)));
        }

        34 => {
            // Integer Log via Multiplication
            let n = (get_random_number() % 1_000_000) + 1;
            let b = (get_random_number() % 9) + 2;
            call_log(3, None, &format!("integer_log_mul({}, {}) = {}", n, b, integer_log_mul(n, b)));
        }
        35 => {
            // Integer Log via Division
            let n = (get_random_number() % 1_000_000) + 1;
            let b = (get_random_number() % 9) + 2;
            call_log(3, None, &format!("integer_log_div({}, {}) = {}", n, b, integer_log_div(n, b)));
        }
        36 => {
            // Perfect Square Test
            let n = get_random_number() % 1_000_000;
            call_log(3, None, &format!("is_perfect_square({}) = {}", n, is_perfect_square(n)));
        }

        37 => {
            // Coin Change Count
            let n = (get_random_number() % 100) as usize;
            call_log(3, None, &format!("coin_change_count({})={}", n, coin_change_count(n)));
        }
        38 => {
            // Coin Change Min
            let n = (get_random_number() % 100) as usize;
            call_log(3, None, &format!("coin_change_min({})={}", n, coin_change_min(n)));
        }

        39 => {
            // Modular Exponentiation & Inverse
            let base = (get_random_number() % 1000) + 1;
            let exp = get_random_number() % 1000;
            let m = (get_random_number() % 999) + 1;
            let me = mod_exp(base, exp, m);
            let inv = mod_inv(base as i64, m as i64);
            call_log(3, None, &format!("mod_exp({},{},{})={}, mod_inv={:?}", base, exp, m, me, inv));
        }
        40 => {
            // CRT2, Garner & Nth‑Root
            let a1 = (get_random_number() % 100) as i64;
            let n1 = ((get_random_number() % 98) + 2) as i64;
            let a2 = (get_random_number() % 100) as i64;
            let n2 = ((get_random_number() % 98) + 2) as i64;
            if gcd(n1 as u64, n2 as u64) == 1 {
                call_log(3, None, &format!("crt2 = {}", crt2(a1, n1, a2, n2)));
            }
            // Garner
            let mods = [2, 3, 5];
            let rems = [
                (get_random_number() % 2) as i64,
                (get_random_number() % 3) as i64,
                (get_random_number() % 5) as i64,
            ];
            call_log(3, None, &format!("garner = {}", garner(&rems, &mods)));
            // Nth‑root
            let n = get_random_number() % 1_000_000;
            let k = (get_random_number() % 4) + 2;
            call_log(
                3,
                None,
                &format!("nth_root({},{}) = {}", n, k, integer_nth_root(n, k.try_into().unwrap())),
            );
        }

        41 => {
            // Number Theoretic Transform
            let mut poly = [0u64; NTT_N];
            for i in 0..NTT_N {
                poly[i] = get_random_number() % MOD_NTT;
            }
            call_log(3, None, &format!("ntt({:?}) = {:?}", poly, ntt(&poly)));
        }
        42 => {
            // CORDIC Rotation
            let angle = (get_random_number() % 200_001) as i32 - 100_000;
            let (c, s) = cordic(angle);
            call_log(3, None, &format!("angle={} → cos≈{}, sin≈{}", angle, c, s));
        }

        43 => {
            // Pseudo-Random Generators
            let mut lcg = Lcg {
                state: get_random_number() as u32,
                a: 1664525,
                c: 1013904223,
            };
            call_log(3, None, &format!("lcg.next() = {}", lcg.next()));
            call_log(3, None, &format!("xorshift64 = {}", xorshift64(get_random_number() as u64)));
            let mut pcg = Pcg {
                state: get_random_number() as u64,
                inc: get_random_number() as u64,
            };
            call_log(3, None, &format!("pcg.next() = {}", pcg.next()));
            let mut mwc = Mwc {
                state: get_random_number() as u64,
                carry: get_random_number() as u64 & 0xFFFF_FFFF,
            };
            call_log(3, None, &format!("mwc.next() = {}", mwc.next()));
        }
        44 => {
            // CRC32, Adler-32, FNV-1a, Murmur, Jenkins
            let len = (get_random_number() % 32) as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(get_random_number() as u8);
            }
            call_log(
                3,
                None,
                &format!(
                    "crc32={:08x}, adler32={:08x}, fnv1a={:08x}, murmur3={:08x}, jenkins={:08x}",
                    crc32(&data),
                    adler32(&data),
                    fnv1a(&data),
                    murmur3_finalizer(get_random_number() as u32),
                    jenkins(&data)
                ),
            );
        }
        45 => {
            // Euler’s Totient φ(n)
            let n = (get_random_number() % 100_000) + 1;
            call_log(3, None, &format!("eulerTotient phi({}) = {}", n, phi(n)));
        }

        46 => {
            // Linear SieveMu
            let n = (get_random_number() % 1_000) as usize + 1;
            let mu = linear_mu(n);
            call_log(3, None, &format!("linear_mu n={}, [n]={}", n, mu[n]));
        }
        47 => {
            // Sum of Divisors
            let n = (get_random_number() % 100_000) + 1;
            call_log(3, None, &format!("SumOfDivisors sigma({}) = {}", n, sigma(n)));
        }
        48 => {
            // Divisor Count d(n)
            let n = (get_random_number() % 100_000) + 1;
            call_log(3, None, &format!("divisor_count({}) = {}", n, divisor_count(n)));
        }
        49 => {
            // Mobius
            let n = (get_random_number() % 100_000) + 1;
            call_log(3, None, &format!("mobius({}) = {}", n, mobius(n)));
        }
        50 => {
            // Dirichlet Convolution (1 * id)
            let n = (get_random_number() % 1_000) + 1;
            call_log(
                3,
                None,
                &format!("dirichlet_convolution (1 * id)({}) = {}", n, dirichlet_convolution(n, |_| 1, |d| d)),
            );
        }
        51 => {
            // Jacobi Symbol (a/n)
            let n = ((get_random_number() % 999) | 1) + 2; // odd ≥3
            let a = (get_random_number() % (n as u64)) as i64;
            call_log(
                2,
                None,
                &format!(
                    "jacobi( {}/{}) = {}",
                    a,
                    n,
                    jacobi(a.try_into().unwrap(), (n as i64).try_into().unwrap())
                ),
            );
        }
        52 => {
            // Cipolla’s Algorithm
            let p = ((get_random_number() % 1000) | 1) + 2;
            let n = get_random_number() % p;
            match cipolla(n, p) {
                Some(r) => call_log(3, None, &format!("{}", r)),
                None => call_log(3, None, &format!("none")),
            }
        }
        53..=u8::MAX => {
            call_log(2, None, &format!("not implemented {}", idx));
        }
    }

    let gas_end = unsafe { gas() };
    return gas_start - gas_end;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let args = if let Some(a) = parse_refine_args(start_address, length) {
        a
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    let out_ptr = unsafe { output_bytes_32.as_ptr() as u64 };
    let mut bytes_written = 0u64;
    let payload_ptr = args.wi_payload_start_address as *const u8;
    let payload_len = args.wi_payload_length as usize;
    let payload = unsafe { slice::from_raw_parts(payload_ptr, payload_len) };

    // run each (program_id, count) pair
    let pairs = payload_len / 2;
    call_log(2, None, &format!("algo refine PAYLOAD {} {}", payload_len, pairs));
    for i in 0..pairs {
        let program_id = payload[i * 2];
        let p_id = program_id % 170;
        let count = payload[i * 2 + 1] as u64;
        let iterations = count * count * count as u64;
        call_log(2, None, &format!("algo refine PROGRAM_ID {} ITERATIONS {}", program_id, iterations));
        let mut gas_used = 0 as u64;
        for _ in 0..iterations {
            gas_used += run_program(p_id);
        }
        // after the last run, grab your sum_bytes
        // only store up to 32 slots
        call_log(
            2,
            None,
            &format!("algo run_refine {} ITERATIONS {} gas_used {}", program_id, iterations, gas_used),
        )
    }

    return (args.wi_payload_start_address, payload_len as u64);
}

#[no_mangle]
static mut output_bytes_32: [u8; 32] = [0; 32];
static mut operand: [u8; 512] = [0; 512];

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    let (_timeslot, _service_index, number_of_operands) = if let Some(args) = parse_accumulate_args(start_address, length) {
        (args.t, args.s, args.number_of_operands)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };
    // fetch 36 byte output which will be (32 byte p_u_hash + 4 byte "a" from payload y)
    let operand_ptr = unsafe { operand.as_ptr() as u64 };
    let operand_len = unsafe { fetch(operand_ptr, 0, 512, 15, 0, 0) };
    let (payload_ptr, payload_len) = match parse_accumulate_operand_args(operand_ptr, operand_len) {
        Some(args) => (args.output_ptr, args.output_len),
        None => return (FIRST_READABLE_ADDRESS as u64, 0),
    };
    if payload_len < 2 {
        call_log(2, None, &format!("payload_len too small: {}", payload_len));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }
    let payload = unsafe { core::slice::from_raw_parts(payload_ptr as *const u8, payload_len as usize) };
    let pairs = payload_len / 2;
    call_log(
        2,
        None,
        &format!("algo {:x?} # of programs: {}", &payload[..payload_len as usize], pairs),
    );

    for j in 0..pairs {
        let x = (2 * j) as usize;
        let program_id = payload[x] % 53 as u8;
        let count = payload[x+1] as u64;
        let iterations = count * count as u64; // accumulate is square, refine is cube
        call_log(2, None, &format!("PROGRAM_ID {} ITERATIONS {}", program_id, iterations));
        let mut gas_used = 0 as u64;
        for _ in 0..iterations {
            gas_used += run_program(program_id);
        }
        call_log(
            2,
            None,
            &format!("algo run_accumulate {} ITERATIONS {} gas_used {}", program_id, iterations, gas_used),
        );

        call_log(2, None, &format!("algo {} p_id={} gas_used={}", j, program_id, gas_used));
    }

    (payload_ptr as u64, payload_len)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(start_address: u64, length: u64) -> (u64, u64) {
    // Note: This part executes only if there are deferred transfers AND this service is the receiver.
    let mut i: u64 = 0;

    loop {
        let (timeslot, service_index, sender, receiver, amount, memo, gas_limit) =
            if let Some(args) = parse_transfer_args(start_address, length, i) {
                (args.t, args.s, args.ts, args.td, args.ta, args.tm, args.tg)
            } else {
                break;
            };

        call_log(
            2,
            None,
            &format!(
                "ALGO on_transfer: timeslot={:?} service_index={:?} sender={:?} receiver={:?} amount={:?} memo={:?} gas_limit={:?}",
                timeslot, service_index, sender, receiver, amount, memo, gas_limit
            ),
        );

        let service_index_bytes = service_index.to_le_bytes();
        let service_index_ptr: u64 = service_index_bytes.as_ptr() as u64;
        let service_index_length: u64 = service_index_bytes.len() as u64;

        let memo_ptr: u64 = memo.as_ptr() as u64;
        let memo_length: u64 = memo.len() as u64;

        unsafe { write(service_index_ptr, service_index_length, memo_ptr, memo_length) };

        let gas_result = unsafe { gas() };
        write_result(gas_result, 4);
        call_log(2, None, &format!("ALGO on_transfer gas: got {:?} (recorded at key 4)", gas_result));

        i += 1;
    }

    return (FIRST_READABLE_ADDRESS as u64, 0);
}
