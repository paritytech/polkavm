//! 256-bit integer arithmetic for the wide register file.
//!
//! The operations follow EVM semantics rather than Rust's, because that is what the
//! instructions exist to implement: division and remainder by zero produce zero instead of
//! trapping, shift amounts of 256 or more clear the value, and everything else wraps.

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
            remainder = remainder.shift_left(1);
            if product[index / 64] & (1 << (index % 64)) != 0 {
                remainder.0[0] |= 1;
            }
            if !remainder.less_than(modulus) {
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
}
