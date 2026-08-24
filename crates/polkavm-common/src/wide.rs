//! 256-bit integer arithmetic for the wide register file, and its 128-bit half.
//!
//! The operations follow EVM semantics rather than Rust's, because that is what the
//! instructions exist to implement: division and remainder by zero produce zero instead of
//! trapping, shift amounts of 256 or more clear the value, and everything else wraps.
//!
//! The 128-bit family at the end of the file is the same set of conventions one width down.
//! Half of a wide register fits in a machine integer, so those are free functions over `u128`
//! rather than methods on a limb array, and only the operations whose answer differs from
//! Rust's own need one at all.

/// A 256-bit integer, stored as four 64-bit limbs, least significant first.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Hash)]
#[repr(C)]
pub struct U256(pub [u64; 4]);

impl U256 {
    pub const ZERO: Self = Self([0; 4]);
    pub const ONE: Self = Self([1, 0, 0, 0]);

    /// The number of bytes one register holds.
    pub const BYTES: usize = 32;

    #[inline]
    pub const fn from_u64(value: u64) -> Self {
        Self([value, 0, 0, 0])
    }

    /// Sign extends an `i64` across the full width.
    #[inline]
    pub const fn from_i64(value: i64) -> Self {
        let fill = if value < 0 { u64::MAX } else { 0 };
        Self([value as u64, fill, fill, fill])
    }

    /// The least significant limb, which is what a narrowing conversion keeps.
    #[inline]
    pub const fn low_u64(self) -> u64 {
        self.0[0]
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    #[inline]
    const fn is_negative(self) -> bool {
        self.0[3] & (1 << 63) != 0
    }

    #[inline]
    pub fn from_le_bytes(bytes: [u8; 32]) -> Self {
        let mut limbs = [0; 4];
        let mut index = 0;
        while index < 4 {
            let mut limb = [0; 8];
            limb.copy_from_slice(&bytes[index * 8..index * 8 + 8]);
            limbs[index] = u64::from_le_bytes(limb);
            index += 1;
        }
        Self(limbs)
    }

    #[inline]
    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        let mut index = 0;
        while index < 4 {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&self.0[index].to_le_bytes());
            index += 1;
        }
        bytes
    }

    /// Reverses the byte order of the whole value.
    #[inline]
    pub fn swap_bytes(self) -> Self {
        Self([
            self.0[3].swap_bytes(),
            self.0[2].swap_bytes(),
            self.0[1].swap_bytes(),
            self.0[0].swap_bytes(),
        ])
    }

    #[inline]
    pub fn wrapping_add(self, other: Self) -> Self {
        let (value, _) = self.carrying_add(other);
        value
    }

    #[inline]
    fn carrying_add(self, other: Self) -> (Self, bool) {
        let mut limbs = [0; 4];
        let mut carry = false;
        let mut index = 0;
        while index < 4 {
            let (sum, carry_a) = self.0[index].overflowing_add(other.0[index]);
            let (sum, carry_b) = sum.overflowing_add(u64::from(carry));
            limbs[index] = sum;
            carry = carry_a | carry_b;
            index += 1;
        }
        (Self(limbs), carry)
    }

    #[inline]
    pub fn wrapping_sub(self, other: Self) -> Self {
        let mut limbs = [0; 4];
        let mut borrow = false;
        let mut index = 0;
        while index < 4 {
            let (difference, borrow_a) = self.0[index].overflowing_sub(other.0[index]);
            let (difference, borrow_b) = difference.overflowing_sub(u64::from(borrow));
            limbs[index] = difference;
            borrow = borrow_a | borrow_b;
            index += 1;
        }
        Self(limbs)
    }

    #[inline]
    pub fn wrapping_neg(self) -> Self {
        Self::ZERO.wrapping_sub(self)
    }

    #[inline]
    pub fn wrapping_mul(self, other: Self) -> Self {
        let full = self.widening_mul(other);
        Self([full[0], full[1], full[2], full[3]])
    }

    /// The full 512-bit product, least significant limb first.
    fn widening_mul(self, other: Self) -> [u64; 8] {
        let mut product = [0u64; 8];
        for (index_a, &limb_a) in self.0.iter().enumerate() {
            let mut carry = 0u128;
            for (index_b, &limb_b) in other.0.iter().enumerate() {
                let total = u128::from(limb_a) * u128::from(limb_b) + u128::from(product[index_a + index_b]) + carry;
                product[index_a + index_b] = total as u64;
                carry = total >> 64;
            }
            product[index_a + 4] = carry as u64;
        }
        product
    }

    #[inline]
    pub fn bitand(self, other: Self) -> Self {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    #[inline]
    pub fn bitor(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    #[inline]
    pub fn bitxor(self, other: Self) -> Self {
        Self([
            self.0[0] ^ other.0[0],
            self.0[1] ^ other.0[1],
            self.0[2] ^ other.0[2],
            self.0[3] ^ other.0[3],
        ])
    }

    #[inline]
    pub fn less_than(self, other: Self) -> bool {
        for index in (0..4).rev() {
            if self.0[index] != other.0[index] {
                return self.0[index] < other.0[index];
            }
        }
        false
    }

    /// Signed comparison, interpreting both operands as two's complement.
    #[inline]
    pub fn less_than_signed(self, other: Self) -> bool {
        match (self.is_negative(), other.is_negative()) {
            (true, false) => true,
            (false, true) => false,
            _ => self.less_than(other),
        }
    }

    /// Shifts left. A shift of 256 or more clears the value, as `SHL` does.
    pub fn shift_left(self, amount: u64) -> Self {
        if amount >= 256 {
            return Self::ZERO;
        }

        let limb_shift = (amount / 64) as usize;
        let bit_shift = amount % 64;
        let mut limbs = [0u64; 4];
        for index in (0..4).rev() {
            if index < limb_shift {
                continue;
            }
            let source = index - limb_shift;
            let mut value = self.0[source] << bit_shift;
            if bit_shift > 0 && source > 0 {
                value |= self.0[source - 1] >> (64 - bit_shift);
            }
            limbs[index] = value;
        }
        Self(limbs)
    }

    /// Shifts right, filling with zeroes. A shift of 256 or more clears the value.
    pub fn shift_right(self, amount: u64) -> Self {
        if amount >= 256 {
            return Self::ZERO;
        }

        let limb_shift = (amount / 64) as usize;
        let bit_shift = amount % 64;
        let mut limbs = [0u64; 4];
        for index in 0..4 {
            let source = index + limb_shift;
            if source >= 4 {
                break;
            }
            let mut value = self.0[source] >> bit_shift;
            if bit_shift > 0 && source + 1 < 4 {
                value |= self.0[source + 1] << (64 - bit_shift);
            }
            limbs[index] = value;
        }
        Self(limbs)
    }

    /// Shifts right, filling with the sign bit. A shift of 256 or more saturates to all sign
    /// bits, as `SAR` does.
    pub fn shift_right_signed(self, amount: u64) -> Self {
        let fill = if self.is_negative() { u64::MAX } else { 0 };
        if amount >= 256 {
            return Self([fill; 4]);
        }

        let limb_shift = (amount / 64) as usize;
        let bit_shift = amount % 64;
        let mut limbs = [fill; 4];
        for index in 0..4 {
            let source = index + limb_shift;
            if source >= 4 {
                break;
            }
            let mut value = self.0[source] >> bit_shift;
            if bit_shift > 0 {
                let high = if source + 1 < 4 { self.0[source + 1] } else { fill };
                value |= high << (64 - bit_shift);
            }
            limbs[index] = value;
        }
        Self(limbs)
    }

    /// The number of one bits.
    pub fn count_ones(self) -> u32 {
        self.0.iter().map(|limb| limb.count_ones()).sum()
    }

    /// The number of zero bits above the most significant one, or 256 if there is none.
    pub fn leading_zeros(self) -> u32 {
        let mut count = 0;
        for limb in self.0.iter().rev() {
            count += limb.leading_zeros();
            if *limb != 0 {
                break;
            }
        }
        count
    }

    /// The number of zero bits below the least significant one, or 256 if there is none.
    pub fn trailing_zeros(self) -> u32 {
        let mut count = 0;
        for limb in self.0.iter() {
            count += limb.trailing_zeros();
            if *limb != 0 {
                break;
            }
        }
        count
    }

    #[inline]
    fn bit(self, index: usize) -> bool {
        self.0[index / 64] & (1 << (index % 64)) != 0
    }

    #[inline]
    fn set_bit(&mut self, index: usize) {
        self.0[index / 64] |= 1 << (index % 64);
    }

    /// Unsigned division and remainder. Division by zero produces zero, as `DIV` does.
    pub fn div_rem(self, divisor: Self) -> (Self, Self) {
        if divisor.is_zero() {
            return (Self::ZERO, Self::ZERO);
        }

        if self.less_than(divisor) {
            return (Self::ZERO, self);
        }

        // Doubling the running remainder cannot carry out of the top here, unlike in
        // `mul_mod` where the numerator is twice as wide: after s steps the remainder is
        // below 2^s, so it only reaches 2^255 on the last one, and the loop ends there.
        let mut quotient = Self::ZERO;
        let mut remainder = Self::ZERO;
        for index in (0..256).rev() {
            remainder = remainder.shift_left(1);
            if self.bit(index) {
                remainder.0[0] |= 1;
            }
            if !remainder.less_than(divisor) {
                remainder = remainder.wrapping_sub(divisor);
                quotient.set_bit(index);
            }
        }

        (quotient, remainder)
    }

    /// Signed division, following `SDIV`: division by zero produces zero, and the one
    /// overflowing case wraps to itself.
    pub fn div_signed(self, divisor: Self) -> Self {
        if divisor.is_zero() {
            return Self::ZERO;
        }

        let negate = self.is_negative() != divisor.is_negative();
        let left = if self.is_negative() { self.wrapping_neg() } else { self };
        let right = if divisor.is_negative() { divisor.wrapping_neg() } else { divisor };
        let (quotient, _) = left.div_rem(right);
        if negate {
            quotient.wrapping_neg()
        } else {
            quotient
        }
    }

    /// Signed remainder, following `SMOD`: the result takes the sign of the dividend.
    pub fn rem_signed(self, divisor: Self) -> Self {
        if divisor.is_zero() {
            return Self::ZERO;
        }

        let negate = self.is_negative();
        let left = if self.is_negative() { self.wrapping_neg() } else { self };
        let right = if divisor.is_negative() { divisor.wrapping_neg() } else { divisor };
        let (_, remainder) = left.div_rem(right);
        if negate {
            remainder.wrapping_neg()
        } else {
            remainder
        }
    }

    /// `(self + other) % modulus`, computed without truncating the sum, as `ADDMOD` does.
    pub fn add_mod(self, other: Self, modulus: Self) -> Self {
        if modulus.is_zero() {
            return Self::ZERO;
        }

        let (_, left) = self.div_rem(modulus);
        let (_, right) = other.div_rem(modulus);
        let (sum, carry) = left.carrying_add(right);
        if carry || !sum.less_than(modulus) {
            sum.wrapping_sub(modulus)
        } else {
            sum
        }
    }

    /// `(self * other) % modulus`, computed on the full 512-bit product, as `MULMOD` does.
    pub fn mul_mod(self, other: Self, modulus: Self) -> Self {
        if modulus.is_zero() {
            return Self::ZERO;
        }

        let product = self.widening_mul(other);
        let mut remainder = Self::ZERO;
        for index in (0..512).rev() {
            // As in `div_rem`, the bit shifted off the top is part of the comparison.
            let carry = remainder.0[3] >> 63 != 0;
            remainder = remainder.shift_left(1);
            if product[index / 64] & (1 << (index % 64)) != 0 {
                remainder.0[0] |= 1;
            }
            if carry || !remainder.less_than(modulus) {
                remainder = remainder.wrapping_sub(modulus);
            }
        }
        remainder
    }

    /// `self` raised to `exponent`, wrapping, as `EXP` does.
    pub fn exp(self, exponent: Self) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        let mut remaining = exponent;
        while !remaining.is_zero() {
            if remaining.0[0] & 1 != 0 {
                result = result.wrapping_mul(base);
            }
            remaining = remaining.shift_right(1);
            if remaining.is_zero() {
                break;
            }
            base = base.wrapping_mul(base);
        }
        result
    }

    /// Sign extends `self` from the byte at index `byte`, counting from the least
    /// significant, as `SIGNEXTEND` does. A byte index of 31 or more leaves the value alone.
    pub fn sign_extend_byte(self, byte: Self) -> Self {
        if !byte.less_than(Self::from_u64(31)) {
            return self;
        }

        let index = byte.low_u64() as usize;
        let sign_bit = index * 8 + 7;
        let mask = Self::ONE.shift_left(sign_bit as u64 + 1).wrapping_sub(Self::ONE);
        if self.bit(sign_bit) {
            self.bitor(mask.bitxor(Self([u64::MAX; 4])))
        } else {
            self.bitand(mask)
        }
    }
}

/// The least significant 64 bits, which is what a narrowing conversion keeps.
#[inline]
pub const fn low_u64_128(value: u128) -> u64 {
    value as u64
}

/// Sign extends an `i64` across the full width.
#[inline]
pub const fn from_i64_128(value: i64) -> u128 {
    value as i128 as u128
}

/// Signed comparison, interpreting both operands as two's complement.
#[inline]
pub fn less_than_signed_128(value: u128, other: u128) -> bool {
    (value as i128) < (other as i128)
}

/// Unsigned division and remainder. Division by zero produces zero, as `DIV` does.
///
/// This is where the family parts ways with the scalar and vector instructions, which answer
/// a division by zero with all ones and a remainder by zero with the dividend.
#[inline]
pub fn div_rem_128(value: u128, divisor: u128) -> (u128, u128) {
    if divisor == 0 {
        return (0, 0);
    }

    (value / divisor, value % divisor)
}

/// Signed division, following `SDIV`: division by zero produces zero, and the one
/// overflowing case wraps to itself.
#[inline]
pub fn div_signed_128(value: u128, divisor: u128) -> u128 {
    if divisor == 0 {
        return 0;
    }

    (value as i128).wrapping_div(divisor as i128) as u128
}

/// Signed remainder, following `SMOD`: the result takes the sign of the dividend.
#[inline]
pub fn rem_signed_128(value: u128, divisor: u128) -> u128 {
    if divisor == 0 {
        return 0;
    }

    (value as i128).wrapping_rem(divisor as i128) as u128
}

/// Shifts left. A shift of 128 or more clears the value, as `SHL` does.
#[inline]
pub fn shift_left_128(value: u128, amount: u64) -> u128 {
    if amount >= 128 {
        return 0;
    }

    value << amount
}

/// Shifts right, filling with zeroes. A shift of 128 or more clears the value.
#[inline]
pub fn shift_right_128(value: u128, amount: u64) -> u128 {
    if amount >= 128 {
        return 0;
    }

    value >> amount
}

/// Shifts right, filling with the sign bit. A shift of 128 or more saturates to all sign
/// bits, as `SAR` does.
#[inline]
pub fn shift_right_signed_128(value: u128, amount: u64) -> u128 {
    let value = value as i128;
    let fill = if value < 0 { u128::MAX } else { 0 };
    if amount >= 128 {
        return fill;
    }

    (value >> amount) as u128
}

#[cfg(test)]
mod tests {
    use super::U256;

    fn from_parts(value: u128) -> U256 {
        U256([value as u64, (value >> 64) as u64, 0, 0])
    }

    #[test]
    fn add_and_sub_wrap() {
        let max = U256([u64::MAX; 4]);
        assert_eq!(max.wrapping_add(U256::ONE), U256::ZERO);
        assert_eq!(U256::ZERO.wrapping_sub(U256::ONE), max);
    }

    #[test]
    fn mul_matches_u128() {
        let a = from_parts(0x1234_5678_9abc_def0);
        let b = from_parts(0x0fed_cba9_8765_4321);
        assert_eq!(a.wrapping_mul(b), from_parts(0x1234_5678_9abc_def0u128 * 0x0fed_cba9_8765_4321u128));
    }

    #[test]
    fn div_rem_by_zero_is_zero() {
        let a = from_parts(1234);
        assert_eq!(a.div_rem(U256::ZERO), (U256::ZERO, U256::ZERO));
        assert_eq!(a.div_signed(U256::ZERO), U256::ZERO);
        assert_eq!(a.rem_signed(U256::ZERO), U256::ZERO);
    }

    #[test]
    fn div_rem_matches_u128() {
        for (a, b) in [(1000u128, 7u128), (u128::MAX, 3), (5, 9), (0, 11)] {
            let (quotient, remainder) = from_parts(a).div_rem(from_parts(b));
            assert_eq!(quotient, from_parts(a / b), "{a} / {b}");
            assert_eq!(remainder, from_parts(a % b), "{a} % {b}");
        }
    }

    #[test]
    fn signed_division_follows_the_dividend() {
        let minus_seven = from_parts(7).wrapping_neg();
        let two = from_parts(2);
        assert_eq!(minus_seven.div_signed(two), from_parts(3).wrapping_neg());
        assert_eq!(minus_seven.rem_signed(two), U256::ONE.wrapping_neg());
    }

    #[test]
    fn signed_division_overflow_wraps() {
        let minimum = U256([0, 0, 0, 1 << 63]);
        assert_eq!(minimum.div_signed(U256::ONE.wrapping_neg()), minimum);
    }

    #[test]
    fn shifts_past_the_width_clear() {
        let value = U256([u64::MAX; 4]);
        assert_eq!(value.shift_left(256), U256::ZERO);
        assert_eq!(value.shift_right(256), U256::ZERO);
        assert_eq!(value.shift_right_signed(256), value);
        assert_eq!(U256::ONE.shift_right_signed(256), U256::ZERO);
    }

    #[test]
    fn shifts_match_u128() {
        let value = from_parts(0x1234_5678_9abc_def0_1122_3344_5566_7788);
        for amount in [0, 1, 63, 64, 65, 127] {
            assert_eq!(
                value.shift_right(amount),
                from_parts(0x1234_5678_9abc_def0_1122_3344_5566_7788u128 >> amount)
            );
        }
        for amount in [0u64, 1, 63, 64] {
            let expected = 0x1234_5678_9abc_def0_1122_3344_5566_7788u128 << amount;
            assert_eq!(value.shift_left(amount).0[0..2], from_parts(expected).0[0..2]);
        }
    }

    #[test]
    fn mod_operations_use_the_full_product() {
        // 2^256 mod 7 is 2, which a truncating add would have lost.
        let max = U256([u64::MAX; 4]);
        assert_eq!(max.add_mod(U256::ONE, from_parts(7)), from_parts(2));
        assert_eq!(max.mul_mod(max, U256::ONE), U256::ZERO);
        assert_eq!(from_parts(5).mul_mod(from_parts(6), from_parts(7)), from_parts(30 % 7));
        assert_eq!(from_parts(5).add_mod(from_parts(6), U256::ZERO), U256::ZERO);
    }

    #[test]
    fn exp_wraps() {
        assert_eq!(from_parts(2).exp(from_parts(10)), from_parts(1024));
        assert_eq!(from_parts(3).exp(U256::ZERO), U256::ONE);
        assert_eq!(from_parts(2).exp(from_parts(256)), U256::ZERO);
    }

    #[test]
    fn sign_extend_byte_matches_evm() {
        let value = from_parts(0xff);
        assert_eq!(value.sign_extend_byte(U256::ZERO), U256([u64::MAX; 4]));
        assert_eq!(from_parts(0x7f).sign_extend_byte(U256::ZERO), from_parts(0x7f));
        assert_eq!(value.sign_extend_byte(U256::ONE), value);
        assert_eq!(value.sign_extend_byte(from_parts(31)), value);
        assert_eq!(value.sign_extend_byte(from_parts(1000)), value);
    }

    #[test]
    fn bit_counts_span_the_whole_width() {
        assert_eq!(U256::ZERO.count_ones(), 0);
        assert_eq!(U256([u64::MAX; 4]).count_ones(), 256);
        assert_eq!(U256([0, 0, 0, 1 << 63]).count_ones(), 1);

        assert_eq!(U256::ZERO.leading_zeros(), 256);
        assert_eq!(U256::ZERO.trailing_zeros(), 256);
        assert_eq!(U256::ONE.leading_zeros(), 255);
        assert_eq!(U256::ONE.trailing_zeros(), 0);
        assert_eq!(U256([0, 0, 0, 1 << 63]).leading_zeros(), 0);
        assert_eq!(U256([0, 0, 0, 1 << 63]).trailing_zeros(), 255);
        assert_eq!(U256([0, 1, 0, 0]).trailing_zeros(), 64);
        assert_eq!(U256([0, 1, 0, 0]).leading_zeros(), 191);
    }

    #[test]
    fn division_by_a_high_divisor() {
        // Divisors at or above 2^255 are where a wider numerator would overflow the running
        // remainder, so they are worth pinning down even though this one cannot.
        let high = U256([0, 0, 0, 1 << 63]);
        let (quotient, remainder) = U256([u64::MAX; 4]).div_rem(high);
        assert_eq!(quotient, U256::ONE);
        assert_eq!(remainder, U256([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1]));

        let divisor = U256([1, 0, 0, 1 << 63]);
        let (quotient, remainder) = U256([u64::MAX; 4]).div_rem(divisor);
        assert_eq!(quotient, U256::ONE);
        assert_eq!(remainder, U256([u64::MAX; 4]).wrapping_sub(divisor));
    }

    #[test]
    fn division_agrees_with_multiplication() {
        // `a == quotient * b + remainder` with `remainder < b` pins the result without a
        // second implementation to compare against. The values are the ones a 256-bit
        // division is most likely to get wrong: the limb boundaries and the top of the range.
        let one = U256::ONE;
        let interesting = [
            U256::ZERO,
            one,
            U256::from_u64(7),
            U256([u64::MAX; 4]),
            U256([u64::MAX; 4]).wrapping_sub(one),
            one.shift_left(255),
            one.shift_left(255).wrapping_add(one),
            one.shift_left(255).wrapping_sub(one),
            one.shift_left(64),
            one.shift_left(128),
            one.shift_left(192),
            U256([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 7, 1 << 63]),
        ];

        for a in interesting {
            for b in interesting {
                let (quotient, remainder) = a.div_rem(b);
                if b.is_zero() {
                    assert_eq!((quotient, remainder), (U256::ZERO, U256::ZERO));
                    continue;
                }

                assert!(remainder.less_than(b), "{a:?} / {b:?}");
                assert_eq!(quotient.wrapping_mul(b).wrapping_add(remainder), a, "{a:?} / {b:?}");
            }
        }
    }

    #[test]
    fn mul_mod_reduces_against_a_high_modulus() {
        // Reducing the 512-bit product walks twice as many steps as a division does, so the
        // running remainder does reach 2^255 and doubling it carries out of the top. Cross
        // checked against shift-and-add modular multiplication, which never holds more than
        // the modulus in range and so cannot make the same mistake.
        fn reference(a: U256, b: U256, modulus: U256) -> U256 {
            let mut result = U256::ZERO;
            let (_, mut base) = a.div_rem(modulus);
            let mut remaining = b;
            while !remaining.is_zero() {
                if remaining.0[0] & 1 != 0 {
                    result = result.add_mod(base, modulus);
                }
                base = base.add_mod(base, modulus);
                remaining = remaining.shift_right(1);
            }
            result
        }

        let one = U256::ONE;
        let moduli = [
            one.shift_left(255),
            one.shift_left(255).wrapping_add(one),
            U256([1, 2, 3, 1 << 63]),
            U256([u64::MAX; 4]),
            U256([u64::MAX; 4]).wrapping_sub(one),
        ];
        let operands = [
            U256([u64::MAX; 4]),
            one.shift_left(255),
            one.shift_left(255).wrapping_sub(one),
            U256([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 7, 11]),
            U256([0xdead_beef_cafe_babe, 3, 0, 1 << 62]),
            U256::from_u64(3),
        ];

        for modulus in moduli {
            for a in operands {
                for b in operands {
                    assert_eq!(a.mul_mod(b, modulus), reference(a, b, modulus), "{a:?} * {b:?} mod {modulus:?}");
                }
            }
        }
    }

    #[test]
    fn byte_swap_round_trips() {
        let value = U256([1, 2, 3, 4]);
        assert_eq!(value.swap_bytes().swap_bytes(), value);
        assert_eq!(U256::ONE.swap_bytes(), U256([0, 0, 0, 1 << 56]));
    }

    #[test]
    fn le_bytes_round_trip() {
        let value = U256([0x0123_4567_89ab_cdef, 2, 3, 4]);
        assert_eq!(U256::from_le_bytes(value.to_le_bytes()), value);
    }

    #[test]
    fn division_128_by_zero_is_zero() {
        // The 128-bit family answers a division by zero the way the 256-bit one does, which is
        // not what the scalar instructions of either width do.
        for value in [0, 1234, u128::MAX, i128::MIN as u128] {
            assert_eq!(super::div_rem_128(value, 0), (0, 0), "{value}");
            assert_eq!(super::div_signed_128(value, 0), 0, "{value}");
            assert_eq!(super::rem_signed_128(value, 0), 0, "{value}");
        }
    }

    #[test]
    fn division_128_matches_native_arithmetic() {
        // Everything but the zero divisor is what Rust computes, including the signed division
        // that has no representable result and wraps to itself.
        let values = [1u128, 7, 1234, 1 << 64, u128::MAX, i128::MAX as u128, i128::MIN as u128];
        for value in values {
            for divisor in values {
                assert_eq!(super::div_rem_128(value, divisor), (value / divisor, value % divisor));

                let (signed_value, signed_divisor) = (value as i128, divisor as i128);
                assert_eq!(
                    super::div_signed_128(value, divisor),
                    signed_value.wrapping_div(signed_divisor) as u128
                );
                assert_eq!(
                    super::rem_signed_128(value, divisor),
                    signed_value.wrapping_rem(signed_divisor) as u128
                );
            }
        }

        assert_eq!(super::div_signed_128(i128::MIN as u128, -1_i128 as u128), i128::MIN as u128);
        assert_eq!(super::rem_signed_128(i128::MIN as u128, -1_i128 as u128), 0);
    }

    #[test]
    fn shifts_128_past_the_width_clear() {
        // A native shift by the width or more is undefined, so the amount is checked rather
        // than handed to the machine, and the amount is the whole 64-bit register.
        let negative = 1_u128 << 127;
        for amount in [128, 129, 255, 256, 1 << 32, u64::MAX] {
            assert_eq!(super::shift_left_128(u128::MAX, amount), 0, "{amount}");
            assert_eq!(super::shift_right_128(u128::MAX, amount), 0, "{amount}");
            assert_eq!(super::shift_right_signed_128(negative, amount), u128::MAX, "{amount}");
            assert_eq!(super::shift_right_signed_128(1, amount), 0, "{amount}");
        }

        // One below the width is the last amount that still moves bits.
        assert_eq!(super::shift_left_128(1, 127), negative);
        assert_eq!(super::shift_right_128(negative, 127), 1);
        assert_eq!(super::shift_right_signed_128(negative, 127), u128::MAX);
        assert_eq!(super::shift_right_signed_128(negative, 1), negative | (negative >> 1));
    }

    #[test]
    fn conversions_128_narrow_and_widen() {
        assert_eq!(super::low_u64_128(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210), 0xfedc_ba98_7654_3210);
        assert_eq!(super::from_i64_128(-1), u128::MAX);
        assert_eq!(super::from_i64_128(i64::MIN), (i128::from(i64::MIN)) as u128);
        assert_eq!(super::from_i64_128(7), 7);
    }
}
