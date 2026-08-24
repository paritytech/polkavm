//! Arithmetic on the wide integers of the XReviveVec extension.
//!
//! A value is a little-endian slice of 64-bit limbs, two of them at 128 bits and sixteen at
//! 1024. Every operation takes operands of one length and writes a result of the same, so the
//! caller decides the width once, from `vtype`, and the arithmetic never has to know it.
//!
//! Division and remainder follow the same conventions as the scalar ones in [`crate::operation`],
//! which are RISC-V's: dividing by zero gives all ones, the remainder by zero is the dividend, and
//! the most negative value divided by minus one is itself. These are RISC-V instructions, so
//! matching that is what keeps the wide and narrow forms consistent.

/// The widest value, in limbs: 1024 bits.
pub const MAX_LIMBS: usize = 16;

#[inline]
fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&limb| limb == 0)
}

#[inline]
fn is_negative(a: &[u64]) -> bool {
    a.last().is_some_and(|&limb| limb >> 63 != 0)
}

/// Unsigned comparison, from the top limb down.
pub fn cmp_unsigned(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    debug_assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            core::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    core::cmp::Ordering::Equal
}

/// Signed comparison: the sign decides it unless both operands share one.
pub fn cmp_signed(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    match (is_negative(a), is_negative(b)) {
        (true, false) => core::cmp::Ordering::Less,
        (false, true) => core::cmp::Ordering::Greater,
        _ => cmp_unsigned(a, b),
    }
}

/// `dst = a + b`, wrapping.
pub fn add(dst: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert_eq!(dst.len(), a.len());
    let mut carry = 0u64;
    for i in 0..dst.len() {
        let (sum, c1) = a[i].overflowing_add(b[i]);
        let (sum, c2) = sum.overflowing_add(carry);
        dst[i] = sum;
        carry = u64::from(c1) + u64::from(c2);
    }
}

/// `dst = a - b`, wrapping.
pub fn sub(dst: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert_eq!(dst.len(), a.len());
    let mut borrow = 0u64;
    for i in 0..dst.len() {
        let (diff, b1) = a[i].overflowing_sub(b[i]);
        let (diff, b2) = diff.overflowing_sub(borrow);
        dst[i] = diff;
        borrow = u64::from(b1) + u64::from(b2);
    }
}

/// `dst = a * b`, truncated to the operand width.
///
/// Schoolbook, and the partial products that would land above the top limb are simply not
/// computed, which is what makes it a wrapping multiply.
pub fn mul(dst: &mut [u64], a: &[u64], b: &[u64]) {
    let n = dst.len();
    let mut out = [0u64; MAX_LIMBS];
    for i in 0..n {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..(n - i) {
            let at = i + j;
            let wide = u128::from(a[i]) * u128::from(b[j]) + u128::from(out[at]) + carry;
            out[at] = wide as u64;
            carry = wide >> 64;
        }
    }
    dst.copy_from_slice(&out[..n]);
}

#[inline]
fn bit(a: &[u64], index: usize) -> u64 {
    (a[index / 64] >> (index % 64)) & 1
}

#[inline]
fn set_bit(a: &mut [u64], index: usize) {
    a[index / 64] |= 1 << (index % 64);
}

#[inline]
fn shift_left_one(a: &mut [u64]) {
    let mut carry = 0u64;
    for limb in a.iter_mut() {
        let next = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next;
    }
}

/// Unsigned division, producing both results at once.
///
/// Shift-and-subtract, one bit per iteration, which is slow but short and obviously right; an
/// interpreter can afford 1024 iterations where it cannot afford a subtle bug.
pub fn div_rem_unsigned(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let n = a.len();
    quotient.fill(0);
    remainder.fill(0);

    if is_zero(b) {
        // As the scalar instructions: all ones, and the dividend as the remainder.
        quotient.fill(u64::MAX);
        remainder.copy_from_slice(a);
        return;
    }

    for i in (0..n * 64).rev() {
        shift_left_one(remainder);
        remainder[0] |= bit(a, i);
        if cmp_unsigned(remainder, b) != core::cmp::Ordering::Less {
            let mut tmp = [0u64; MAX_LIMBS];
            sub(&mut tmp[..n], remainder, b);
            remainder.copy_from_slice(&tmp[..n]);
            set_bit(quotient, i);
        }
    }
}

/// `dst = -a`, wrapping.
pub fn negate(dst: &mut [u64], a: &[u64]) {
    let n = dst.len();
    let mut zero = [0u64; MAX_LIMBS];
    zero[..n].fill(0);
    let z = zero;
    sub(dst, &z[..n], a);
}

/// Signed division, with the signs handled around the unsigned one.
pub fn div_rem_signed(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let n = a.len();
    if is_zero(b) {
        quotient.fill(u64::MAX); // -1
        remainder.copy_from_slice(a);
        return;
    }

    // The most negative value over minus one has no positive counterpart, so it wraps to itself
    // and leaves no remainder.
    let mut most_negative = [0u64; MAX_LIMBS];
    most_negative[n - 1] = 1 << 63;
    let mut minus_one = [0u64; MAX_LIMBS];
    minus_one[..n].fill(u64::MAX);
    if cmp_unsigned(a, &most_negative[..n]) == core::cmp::Ordering::Equal
        && cmp_unsigned(b, &minus_one[..n]) == core::cmp::Ordering::Equal
    {
        quotient.copy_from_slice(a);
        remainder.fill(0);
        return;
    }

    let negative_a = is_negative(a);
    let negative_b = is_negative(b);
    let mut abs_a = [0u64; MAX_LIMBS];
    let mut abs_b = [0u64; MAX_LIMBS];
    if negative_a {
        negate(&mut abs_a[..n], a);
    } else {
        abs_a[..n].copy_from_slice(a);
    }
    if negative_b {
        negate(&mut abs_b[..n], b);
    } else {
        abs_b[..n].copy_from_slice(b);
    }

    div_rem_unsigned(quotient, remainder, &abs_a[..n], &abs_b[..n]);

    // The quotient's sign is the operands' combined; the remainder takes the dividend's.
    if negative_a != negative_b {
        let mut tmp = [0u64; MAX_LIMBS];
        negate(&mut tmp[..n], quotient);
        quotient.copy_from_slice(&tmp[..n]);
    }
    if negative_a {
        let mut tmp = [0u64; MAX_LIMBS];
        negate(&mut tmp[..n], remainder);
        remainder.copy_from_slice(&tmp[..n]);
    }
}

/// `dst = base ** exponent`, truncated to the operand width, by square and multiply.
pub fn exp(dst: &mut [u64], base: &[u64], exponent: &[u64]) {
    let n = dst.len();
    let mut result = [0u64; MAX_LIMBS];
    result[0] = 1;
    let mut square = [0u64; MAX_LIMBS];
    square[..n].copy_from_slice(base);

    let top = n * 64 - exponent.iter().rev().map(|l| l.leading_zeros() as usize).take_while(|&z| z == 64).count() * 64;
    for i in 0..top {
        if bit(exponent, i) == 1 {
            let mut tmp = [0u64; MAX_LIMBS];
            mul(&mut tmp[..n], &result[..n], &square[..n]);
            result[..n].copy_from_slice(&tmp[..n]);
        }
        let mut tmp = [0u64; MAX_LIMBS];
        mul(&mut tmp[..n], &square[..n], &square[..n]);
        square[..n].copy_from_slice(&tmp[..n]);
    }
    dst.copy_from_slice(&result[..n]);
}

/// EVM's `signextend`: sign-extend `value` from the byte at `index`, counting from the least
/// significant. An index at or past the top byte leaves the value alone.
pub fn sign_extend(dst: &mut [u64], value: &[u64], index: &[u64]) {
    let n = dst.len();
    dst.copy_from_slice(value);

    // Any index that does not fit in a single limb is far past the top byte.
    let byte = if index[1..].iter().all(|&l| l == 0) { index[0] } else { u64::MAX };
    let top_byte = (n * 8 - 1) as u64;
    if byte >= top_byte {
        return;
    }

    let byte = byte as usize;
    let sign_bit = byte * 8 + 7;
    let negative = bit(value, sign_bit) == 1;
    for i in (sign_bit + 1)..(n * 64) {
        let limb = &mut dst[i / 64];
        let mask = 1u64 << (i % 64);
        if negative {
            *limb |= mask;
        } else {
            *limb &= !mask;
        }
    }
}

/// Twice the widest value, which a full multiply needs before it is reduced.
pub const MAX_LIMBS_WIDE: usize = MAX_LIMBS * 2;

/// `dst = a << amount`, wrapping. An amount at or past the width clears the value, which is
/// what the shift instructions do rather than leaving it undefined.
pub fn shift_left(dst: &mut [u64], a: &[u64], amount: u32) {
    let n = dst.len();
    let bits = n * 64;
    if amount as usize >= bits {
        dst.fill(0);
        return;
    }
    let (limbs, rest) = ((amount / 64) as usize, amount % 64);
    let mut out = [0u64; MAX_LIMBS];
    for i in (0..n).rev() {
        let mut value = 0u64;
        if i >= limbs {
            value = a[i - limbs] << rest;
            if rest > 0 && i > limbs {
                value |= a[i - limbs - 1] >> (64 - rest);
            }
        }
        out[i] = value;
    }
    dst.copy_from_slice(&out[..n]);
}

/// `dst = a >> amount`. `fill` is what shifts in at the top: zero for a logical shift, the sign
/// for an arithmetic one.
fn shift_right_with(dst: &mut [u64], a: &[u64], amount: u32, fill: u64) {
    let n = dst.len();
    let bits = n * 64;
    if amount as usize >= bits {
        dst.fill(fill);
        return;
    }
    let (limbs, rest) = ((amount / 64) as usize, amount % 64);
    let mut out = [0u64; MAX_LIMBS];
    for i in 0..n {
        let mut value = fill;
        if i + limbs < n {
            value = a[i + limbs] >> rest;
            if rest > 0 {
                let high = if i + limbs + 1 < n { a[i + limbs + 1] } else { fill };
                value |= high << (64 - rest);
            }
        }
        out[i] = value;
    }
    dst.copy_from_slice(&out[..n]);
}

/// `dst = a >> amount`, zero-filled.
pub fn shift_right_logical(dst: &mut [u64], a: &[u64], amount: u32) {
    shift_right_with(dst, a, amount, 0);
}

/// `dst = a >> amount`, sign-filled.
pub fn shift_right_arithmetic(dst: &mut [u64], a: &[u64], amount: u32) {
    let fill = if a.last().is_some_and(|&l| l >> 63 != 0) { u64::MAX } else { 0 };
    shift_right_with(dst, a, amount, fill);
}

/// Reverses the byte order of the whole value.
pub fn byte_swap(dst: &mut [u64], a: &[u64]) {
    let n = dst.len();
    for i in 0..n {
        dst[i] = a[n - 1 - i].swap_bytes();
    }
}

/// The full `2n`-limb product, which modular multiplication needs before it reduces.
fn mul_full(out: &mut [u64], a: &[u64], b: &[u64]) {
    let n = a.len();
    out[..n * 2].fill(0);
    for i in 0..n {
        if a[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..n {
            let at = i + j;
            let wide = u128::from(a[i]) * u128::from(b[j]) + u128::from(out[at]) + carry;
            out[at] = wide as u64;
            carry = wide >> 64;
        }
        out[i + n] = out[i + n].wrapping_add(carry as u64);
    }
}

/// Divides a `2n`-limb value by an `n`-limb one, keeping only the remainder.
fn rem_double(remainder: &mut [u64], a: &[u64], b: &[u64]) {
    let n = remainder.len();
    remainder.fill(0);
    for i in (0..n * 2 * 64).rev() {
        shift_left_one_slice(remainder);
        remainder[0] |= (a[i / 64] >> (i % 64)) & 1;
        if cmp_unsigned(remainder, b) != core::cmp::Ordering::Less {
            let mut tmp = [0u64; MAX_LIMBS];
            sub(&mut tmp[..n], remainder, b);
            remainder.copy_from_slice(&tmp[..n]);
        }
    }
}

#[inline]
fn shift_left_one_slice(a: &mut [u64]) {
    let mut carry = 0u64;
    for limb in a.iter_mut() {
        let next = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next;
    }
}

/// EVM's `addmod`: `(a + b) % m`, and zero when the modulus is zero.
///
/// The sum can exceed the width, so it is carried in one extra limb rather than wrapping.
pub fn add_mod(dst: &mut [u64], a: &[u64], b: &[u64], m: &[u64]) {
    let n = dst.len();
    if is_zero(m) {
        dst.fill(0);
        return;
    }
    let mut sum = [0u64; MAX_LIMBS_WIDE];
    let mut carry = 0u64;
    for i in 0..n {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(carry);
        sum[i] = s;
        carry = u64::from(c1) + u64::from(c2);
    }
    sum[n] = carry;
    rem_double(dst, &sum[..n * 2], m);
}

/// EVM's `mulmod`: `(a * b) % m`, and zero when the modulus is zero.
pub fn mul_mod(dst: &mut [u64], a: &[u64], b: &[u64], m: &[u64]) {
    let n = dst.len();
    if is_zero(m) {
        dst.fill(0);
        return;
    }
    let mut product = [0u64; MAX_LIMBS_WIDE];
    mul_full(&mut product, a, b);
    rem_double(dst, &product[..n * 2], m);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At 128 bits a value is a `u128`, so every operation can be checked against one.
    fn of(value: u128) -> [u64; 2] {
        [value as u64, (value >> 64) as u64]
    }
    fn to(limbs: &[u64]) -> u128 {
        u128::from(limbs[0]) | (u128::from(limbs[1]) << 64)
    }

    const CASES: &[u128] = &[
        0,
        1,
        2,
        7,
        u64::MAX as u128,
        (u64::MAX as u128) + 1,
        u128::MAX,
        u128::MAX - 1,
        1 << 127,
        (1 << 127) + 1,
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
    ];

    #[test]
    fn add_sub_mul_match_u128() {
        for &a in CASES {
            for &b in CASES {
                let (x, y) = (of(a), of(b));
                let mut r = [0u64; 2];
                add(&mut r, &x, &y);
                assert_eq!(to(&r), a.wrapping_add(b), "add {a:#x} {b:#x}");
                sub(&mut r, &x, &y);
                assert_eq!(to(&r), a.wrapping_sub(b), "sub {a:#x} {b:#x}");
                mul(&mut r, &x, &y);
                assert_eq!(to(&r), a.wrapping_mul(b), "mul {a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn compares_match_u128() {
        for &a in CASES {
            for &b in CASES {
                assert_eq!(cmp_unsigned(&of(a), &of(b)), a.cmp(&b), "u {a:#x} {b:#x}");
                let (sa, sb) = (a as i128, b as i128);
                assert_eq!(cmp_signed(&of(a), &of(b)), sa.cmp(&sb), "s {a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn division_matches_u128_and_the_scalar_conventions() {
        for &a in CASES {
            for &b in CASES {
                let (mut q, mut r) = ([0u64; 2], [0u64; 2]);
                div_rem_unsigned(&mut q, &mut r, &of(a), &of(b));
                if b == 0 {
                    // As the scalar instructions do.
                    assert_eq!(to(&q), u128::MAX, "divu by zero");
                    assert_eq!(to(&r), a, "remu by zero");
                } else {
                    assert_eq!(to(&q), a / b, "divu {a:#x} {b:#x}");
                    assert_eq!(to(&r), a % b, "remu {a:#x} {b:#x}");
                }

                div_rem_signed(&mut q, &mut r, &of(a), &of(b));
                let (sa, sb) = (a as i128, b as i128);
                if b == 0 {
                    assert_eq!(to(&q) as i128, -1, "div by zero");
                    assert_eq!(to(&r), a, "rem by zero");
                } else if sa == i128::MIN && sb == -1 {
                    assert_eq!(to(&q) as i128, i128::MIN, "MIN / -1");
                    assert_eq!(to(&r), 0, "MIN % -1");
                } else {
                    assert_eq!(to(&q) as i128, sa.wrapping_div(sb), "div {sa} {sb}");
                    assert_eq!(to(&r) as i128, sa.wrapping_rem(sb), "rem {sa} {sb}");
                }
            }
        }
    }

    #[test]
    fn exponentiation_matches_u128() {
        for &base in &[0u128, 1, 2, 3, 10, u64::MAX as u128, u128::MAX] {
            for e in [0u128, 1, 2, 3, 5, 17, 64, 127, 255] {
                let mut r = [0u64; 2];
                exp(&mut r, &of(base), &of(e));
                let mut want = 1u128;
                for _ in 0..e {
                    want = want.wrapping_mul(base);
                }
                assert_eq!(to(&r), want, "{base:#x} ** {e}");
            }
        }
    }

    #[test]
    fn sign_extend_matches_evm() {
        // Byte 0 of 0xff is -1 once extended; byte 0 of 0x7f stays positive.
        let mut r = [0u64; 2];
        sign_extend(&mut r, &of(0xff), &of(0));
        assert_eq!(to(&r), u128::MAX);
        sign_extend(&mut r, &of(0x7f), &of(0));
        assert_eq!(to(&r), 0x7f);
        // A larger index keeps more of the value.
        sign_extend(&mut r, &of(0xff80), &of(1));
        assert_eq!(to(&r), u128::MAX - 0x7f);
        // At or past the top byte the value is untouched, and so is a nonsense index.
        sign_extend(&mut r, &of(0x1234), &of(15));
        assert_eq!(to(&r), 0x1234);
        sign_extend(&mut r, &of(0x1234), &of(u128::MAX));
        assert_eq!(to(&r), 0x1234);
    }

    #[test]
    fn wider_widths_carry_across_limbs() {
        // 256 bits: the carry has to propagate through all four limbs.
        let a = [u64::MAX; 4];
        let one = [1u64, 0, 0, 0];
        let mut r = [0u64; 4];
        add(&mut r, &a, &one);
        assert_eq!(r, [0, 0, 0, 0], "all ones plus one wraps to zero");
        sub(&mut r, &[0; 4], &one);
        assert_eq!(r, [u64::MAX; 4], "zero minus one is all ones");

        // 1024 bits: a division that needs every limb.
        let mut big = [0u64; 16];
        big[15] = 1 << 63;
        let three = {
            let mut t = [0u64; 16];
            t[0] = 3;
            t
        };
        let (mut q, mut rem) = ([0u64; 16], [0u64; 16]);
        div_rem_unsigned(&mut q, &mut rem, &big, &three);
        let mut back = [0u64; 16];
        mul(&mut back, &q, &three);
        let mut sum = [0u64; 16];
        add(&mut sum, &back, &rem);
        assert_eq!(sum, big, "q * 3 + r reconstructs the dividend");
    }
}
