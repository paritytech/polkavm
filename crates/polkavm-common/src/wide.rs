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

    /// The low four limbs of a wider value, which is as much of it as fits.
    #[inline]
    fn from_low_limbs(limbs: &[u64; 8]) -> Self {
        Self([limbs[0], limbs[1], limbs[2], limbs[3]])
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

    /// Unsigned division and remainder. Division by zero produces zero, as `DIV` does.
    pub fn div_rem(self, divisor: Self) -> (Self, Self) {
        if divisor.is_zero() {
            return (Self::ZERO, Self::ZERO);
        }

        if self.less_than(divisor) {
            return (Self::ZERO, self);
        }

        let numerator_length = significant_limbs(&self.0);
        let divisor_length = significant_limbs(&divisor.0);
        if divisor_length == 1 {
            if numerator_length == 1 {
                return (Self::from_u64(self.0[0] / divisor.0[0]), Self::from_u64(self.0[0] % divisor.0[0]));
            }

            let (quotient, remainder) = div_rem_by_limb(&self.0[..numerator_length], divisor.0[0]);
            return (Self::from_low_limbs(&quotient), Self::from_u64(remainder));
        }

        let (quotient, remainder) = long_division(&self.0[..numerator_length], &divisor.0[..divisor_length]);
        (Self::from_low_limbs(&quotient), remainder)
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
        let product_length = significant_limbs(&product);
        let modulus_length = significant_limbs(&modulus.0);
        if modulus_length == 1 {
            let (_, remainder) = div_rem_by_limb(&product[..product_length], modulus.0[0]);
            return Self::from_u64(remainder);
        }

        if less_than_limbs(&product[..product_length], &modulus.0[..modulus_length]) {
            return Self::from_low_limbs(&product);
        }

        if modulus_length == 4 {
            return rem_512_by_256(&product[..product_length], &modulus.0);
        }

        let (_, remainder) = long_division(&product[..product_length], &modulus.0[..modulus_length]);
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

/// How often the two paths that only a hard division reaches have run, so that the tests can
/// prove they cover both. Test builds only: a release build carries no trace of it.
#[cfg(test)]
mod path_counters {
    use core::sync::atomic::{AtomicUsize, Ordering};

    static QUOTIENT_CORRECTIONS: AtomicUsize = AtomicUsize::new(0);
    static ADD_BACKS: AtomicUsize = AtomicUsize::new(0);

    /// Records an estimate walked back by the second divisor limb.
    pub fn record_quotient_correction() {
        QUOTIENT_CORRECTIONS.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an estimate the correction missed, which the divisor had to be added back for.
    pub fn record_add_back() {
        ADD_BACKS.fetch_add(1, Ordering::Relaxed);
    }

    /// The totals so far, as (quotient corrections, add-backs).
    pub fn totals() -> (usize, usize) {
        (QUOTIENT_CORRECTIONS.load(Ordering::Relaxed), ADD_BACKS.load(Ordering::Relaxed))
    }
}

/// The number of limbs up to and including the most significant nonzero one, which is zero
/// for a zero value.
#[inline]
fn significant_limbs(limbs: &[u64]) -> usize {
    let mut length = limbs.len();
    while length > 0 && limbs[length - 1] == 0 {
        length -= 1;
    }
    length
}

/// Whether `left` is below `right`, both trimmed to their significant limbs.
#[inline]
fn less_than_limbs(left: &[u64], right: &[u64]) -> bool {
    if left.len() != right.len() {
        return left.len() < right.len();
    }

    for index in (0..left.len()).rev() {
        if left[index] != right[index] {
            return left[index] < right[index];
        }
    }
    false
}

/// `floor((2^19 - 3 * 2^8) / index)` for every index a normalized divisor's top nine bits can
/// be, which is where the reciprocal below starts from. Worked out at compile time, so that
/// the table cannot be mistyped.
static RECIPROCAL_SEEDS: [u16; 256] = {
    let mut seeds = [0u16; 256];
    let mut index = 0;
    while index < seeds.len() {
        seeds[index] = (((1 << 19) - 3 * (1 << 8)) / (index as u32 + 256)) as u16;
        index += 1;
    }
    seeds
};

/// The reciprocal a 2-by-1 division needs: `floor((2^128 - 1) / divisor) - 2^64`, which fits
/// in a limb exactly because the divisor is normalized. Computed by Moller and Granlund's
/// "Improved division by invariant integers", algorithm 3: a seed good to nine bits refined by
/// multiplication alone, which beats asking the hardware for a 128-bit division.
#[inline]
fn reciprocal(divisor: u64) -> u64 {
    debug_assert!(divisor >> 63 == 1);

    // The top bit being set is what makes the top nine bits land in the table's range.
    let seed = u64::from(RECIPROCAL_SEEDS[(divisor >> 55) as usize % RECIPROCAL_SEEDS.len()]);
    let upper = (divisor >> 24) + 1;
    let half = divisor.wrapping_add(1) >> 1;

    // Each step doubles the number of correct bits; the refinements are exact modulo 2^64 and
    // the intermediates are meant to wrap.
    let refined = (seed << 11) - ((seed * seed * upper) >> 40) - 1;
    let doubled = (refined << 13).wrapping_add(refined.wrapping_mul((1u64 << 60).wrapping_sub(refined * upper)) >> 47);
    let error = ((doubled >> 1) & (divisor & 1).wrapping_neg()).wrapping_sub(doubled.wrapping_mul(half));
    let widened = (multiply_high(doubled, error) >> 1).wrapping_add(doubled << 31);
    widened
        .wrapping_sub(multiply_add_high(widened, divisor, divisor))
        .wrapping_sub(divisor)
}

/// The high limb of a limb by limb product.
#[inline]
fn multiply_high(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) >> 64) as u64
}

/// The high limb of a limb by limb product with a limb added in.
#[inline]
fn multiply_add_high(left: u64, right: u64, addend: u64) -> u64 {
    ((u128::from(left) * u128::from(right) + u128::from(addend)) >> 64) as u64
}

/// `(high:low) / divisor` and its remainder by Moller and Granlund's "Improved division by
/// invariant integers", algorithm 4. The divisor must be normalized and `divisor_reciprocal`
/// its reciprocal, and `high` must be below it so that the quotient fits in a limb.
#[inline]
fn div_rem_128_by_64(high: u64, low: u64, divisor: u64, divisor_reciprocal: u64) -> (u64, u64) {
    debug_assert!(high < divisor);

    // The reciprocal turns the division into a multiplication that lands within one of the
    // answer; the two corrections below settle which side it landed on.
    let estimate = u128::from(divisor_reciprocal) * u128::from(high) + ((u128::from(high) << 64) | u128::from(low));
    let estimate_low = estimate as u64;
    let mut quotient = ((estimate >> 64) as u64).wrapping_add(1);
    let mut remainder = low.wrapping_sub(quotient.wrapping_mul(divisor));
    if remainder > estimate_low {
        quotient = quotient.wrapping_sub(1);
        remainder = remainder.wrapping_add(divisor);
    }
    if remainder >= divisor {
        quotient = quotient.wrapping_add(1);
        remainder -= divisor;
    }
    (quotient, remainder)
}

/// Shifts `value` up by `shift` bits into `destination`, the normalization that makes the
/// divisor's top bit set, and returns the bits that left the top limb.
#[inline]
fn normalize(value: &[u64], shift: u32, destination: &mut [u64]) -> u64 {
    debug_assert!(destination.len() >= value.len());

    let mut carry = 0;
    for (source, target) in value.iter().zip(destination.iter_mut()) {
        *target = (source << shift) | carry;
        carry = if shift > 0 { source >> (64 - shift) } else { 0 };
    }
    carry
}

/// Shifts the low `length` limbs of `value` back down by `shift`, undoing `normalize`. Only
/// a remainder is ever taken back down, and a remainder is below the divisor, so `length`
/// divisor limbs hold all of it.
#[inline]
fn denormalize(value: &[u64], shift: u32, length: usize) -> U256 {
    let mut limbs = [0u64; 4];
    for index in 0..length {
        limbs[index] = value[index] >> shift;
        if shift > 0 && index + 1 < length {
            limbs[index] |= value[index + 1] << (64 - shift);
        }
    }
    U256(limbs)
}

/// Divides up to eight limbs by a single nonzero limb. Each quotient limb lands at the
/// position of the numerator limb it came from.
#[inline]
fn div_rem_by_limb(numerator: &[u64], divisor: u64) -> ([u64; 8], u64) {
    debug_assert!(divisor != 0);
    debug_assert!(numerator.len() <= 8);

    let shift = divisor.leading_zeros();
    let normalized_divisor = divisor << shift;
    let divisor_reciprocal = reciprocal(normalized_divisor);

    // Normalizing the numerator would need a ninth limb; instead the bits leaving its top
    // seed the running remainder, which they may because they are below 2^shift and the
    // normalized divisor is not.
    let mut remainder = match numerator.last() {
        Some(&top) if shift > 0 => top >> (64 - shift),
        _ => 0,
    };
    let mut quotient = [0u64; 8];
    let digits = &mut quotient[..numerator.len()];
    for index in (0..digits.len()).rev() {
        let mut low = numerator[index] << shift;
        if shift > 0 && index > 0 {
            low |= numerator[index - 1] >> (64 - shift);
        }
        let (digit, next_remainder) = div_rem_128_by_64(remainder, low, normalized_divisor, divisor_reciprocal);
        digits[index] = digit;
        remainder = next_remainder;
    }
    (quotient, remainder >> shift)
}

/// Divides an up to eight limb numerator by a two to four limb divisor whose top limb is
/// nonzero, the numerator being at least the divisor: schoolbook long division over `u64`
/// digits, which is Knuth's algorithm D (TAOCP 4.3.1).
fn long_division(numerator: &[u64], divisor: &[u64]) -> ([u64; 8], U256) {
    let numerator_length = numerator.len();
    let divisor_length = divisor.len();
    debug_assert!((2..=4).contains(&divisor_length));
    debug_assert!(divisor[divisor_length - 1] != 0);
    debug_assert!((divisor_length..=8).contains(&numerator_length));

    let shift = divisor[divisor_length - 1].leading_zeros();

    // The shift is what it takes to set the divisor's top bit, so nothing leaves its top
    // limb; the numerator gets the extra limb the bits leaving its own top need.
    let mut normalized_divisor = [0u64; 4];
    normalize(divisor, shift, &mut normalized_divisor);
    let mut normalized_numerator = [0u64; 9];
    let shifted_out = normalize(numerator, shift, &mut normalized_numerator);
    normalized_numerator[numerator_length] = shifted_out;

    let divisor_reciprocal = reciprocal(normalized_divisor[divisor_length - 1]);
    let quotient = match divisor_length {
        2 => long_division_digits::<2>(&mut normalized_numerator, &normalized_divisor, numerator_length, divisor_reciprocal),
        3 => long_division_digits::<3>(&mut normalized_numerator, &normalized_divisor, numerator_length, divisor_reciprocal),
        _ => long_division_digits::<4>(&mut normalized_numerator, &normalized_divisor, numerator_length, divisor_reciprocal),
    };

    (quotient, denormalize(&normalized_numerator, shift, divisor_length))
}

/// The remainder of a 512-bit product by a four limb modulus, which is the shape `mul_mod`
/// has when the modulus fills its width: the same long division with no quotient to keep, and
/// one digit step per limb the product actually reaches into. The product is never shorter
/// than the modulus, so the saturation below never comes up; it is there for the same reason
/// as the one in `long_division_digits`.
fn rem_512_by_256(product: &[u64], modulus: &[u64; 4]) -> U256 {
    let product_length = product.len();
    debug_assert!(modulus[3] != 0);
    debug_assert!((4..=8).contains(&product_length));

    let shift = modulus[3].leading_zeros();
    let mut normalized_divisor = [0u64; 4];
    normalize(modulus, shift, &mut normalized_divisor);
    let mut normalized_numerator = [0u64; 9];
    let shifted_out = normalize(product, shift, &mut normalized_numerator);
    normalized_numerator[product_length] = shifted_out;

    let divisor_reciprocal = reciprocal(normalized_divisor[3]);
    for step in (0..=product_length.saturating_sub(4)).rev() {
        long_division_step(&mut normalized_numerator[step..=step + 4], &normalized_divisor, divisor_reciprocal);
    }

    denormalize(&normalized_numerator, shift, 4)
}

/// Every digit of the long division, over a divisor whose length is known at compile time so
/// that the digit steps come out as straight line code. The numerator is never shorter than
/// the divisor, so the saturation below never comes up; it is there because it is what tells
/// the compiler that the digit windows stay inside the numerator.
#[inline(always)]
fn long_division_digits<const DIVISOR_LENGTH: usize>(
    numerator: &mut [u64; 9],
    divisor: &[u64; 4],
    numerator_length: usize,
    divisor_reciprocal: u64,
) -> [u64; 8] {
    let mut quotient = [0u64; 8];
    for step in (0..=numerator_length.saturating_sub(DIVISOR_LENGTH)).rev() {
        quotient[step] = long_division_step(
            &mut numerator[step..=step + DIVISOR_LENGTH],
            &divisor[..DIVISOR_LENGTH],
            divisor_reciprocal,
        );
    }
    quotient
}

/// One quotient digit of the long division: estimate it from the divisor's top limb, walk
/// the estimate back against the second limb (twice at most, by Knuth's theorem 4.3.1B),
/// then subtract the divisor times the digit out of the running remainder. What the estimate
/// can still be over by is one, which shows up as a borrow out of the top and is undone by
/// adding the divisor back. `window` is the stretch of the running remainder the digit acts
/// on, one limb longer than the divisor; passing it as a slice rather than an index into the
/// whole numerator is what leaves the digit loops with no bound left to check.
#[inline(always)]
fn long_division_step(window: &mut [u64], divisor: &[u64], divisor_reciprocal: u64) -> u64 {
    let divisor_length = divisor.len();
    debug_assert!(window.len() == divisor_length + 1);

    let top = divisor[divisor_length - 1];
    let second = divisor[divisor_length - 2];
    let high = window[divisor_length];
    let low = window[divisor_length - 1];

    // A digit cannot exceed the largest one there is, which is where the estimate saturates
    // when the two leading limbs would divide out to more than that.
    let (mut estimate, mut estimate_remainder) = if high >= top {
        let leading = (u128::from(high) << 64) | u128::from(low);
        (u64::MAX, leading - u128::from(u64::MAX) * u128::from(top))
    } else {
        let (digit, remainder) = div_rem_128_by_64(high, low, top, divisor_reciprocal);
        (digit, u128::from(remainder))
    };

    while estimate_remainder <= u128::from(u64::MAX)
        && u128::from(estimate) * u128::from(second) > ((estimate_remainder << 64) | u128::from(window[divisor_length - 2]))
    {
        #[cfg(test)]
        path_counters::record_quotient_correction();

        estimate -= 1;
        estimate_remainder += u128::from(top);
    }

    let mut carry = 0u64;
    let mut borrow = false;
    for (limb, &divisor_limb) in window.iter_mut().zip(divisor) {
        let product = u128::from(estimate) * u128::from(divisor_limb) + u128::from(carry);
        carry = (product >> 64) as u64;
        let (difference, borrow_a) = limb.overflowing_sub(product as u64);
        let (difference, borrow_b) = difference.overflowing_sub(u64::from(borrow));
        *limb = difference;
        borrow = borrow_a | borrow_b;
    }
    let (difference, borrow_a) = window[divisor_length].overflowing_sub(carry);
    let (difference, borrow_b) = difference.overflowing_sub(u64::from(borrow));
    window[divisor_length] = difference;

    let mut digit = estimate;
    if borrow_a | borrow_b {
        #[cfg(test)]
        path_counters::record_add_back();

        digit -= 1;
        let mut carry = false;
        for (limb, &divisor_limb) in window.iter_mut().zip(divisor) {
            let (sum, carry_a) = limb.overflowing_add(divisor_limb);
            let (sum, carry_b) = sum.overflowing_add(u64::from(carry));
            *limb = sum;
            carry = carry_a | carry_b;
        }

        // This carry is the borrow from above coming back, so it is meant to wrap.
        window[divisor_length] = window[divisor_length].wrapping_add(u64::from(carry));
    }
    digit
}

#[cfg(test)]
mod tests {
    use super::{path_counters, reciprocal, U256};

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

    /// Enough biased random cases for the add-back path, which the operands have to conspire
    /// to reach, to appear tens of times.
    const BIASED_RANDOM_CASES: usize = 40_000;

    /// Fixed, so that a failure names a case that can be reproduced by rerunning the test.
    const BIASED_RANDOM_SEED: u64 = 0x5eed_5eed_5eed_5eed;

    /// The bit serial restoring division the kernels used before they went digit serial, kept
    /// as an oracle. It shares no estimate or correction machinery with the long division, so the
    /// two cannot be wrong in the same way, and it is what every result recorded on this
    /// branch so far was produced by.
    fn reference_div_rem(numerator: U256, divisor: U256) -> (U256, U256) {
        if divisor.is_zero() {
            return (U256::ZERO, U256::ZERO);
        }

        if numerator.less_than(divisor) {
            return (U256::ZERO, numerator);
        }

        let mut quotient = U256::ZERO;
        let mut remainder = U256::ZERO;
        for index in (0..256).rev() {
            remainder = remainder.shift_left(1);
            if numerator.0[index / 64] & (1 << (index % 64)) != 0 {
                remainder.0[0] |= 1;
            }
            if !remainder.less_than(divisor) {
                remainder = remainder.wrapping_sub(divisor);
                quotient.0[index / 64] |= 1 << (index % 64);
            }
        }
        (quotient, remainder)
    }

    /// The bit serial reduction `mul_mod` used before the rewrite, over the same untruncated
    /// product. The bit shifted off the top being part of the comparison is what made it
    /// correct for a numerator twice the width of the modulus.
    fn reference_mul_mod(left: U256, right: U256, modulus: U256) -> U256 {
        if modulus.is_zero() {
            return U256::ZERO;
        }

        let product = left.widening_mul(right);
        let mut remainder = U256::ZERO;
        for index in (0..512).rev() {
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

    /// `add_mod` over the reference division. The operation itself is untouched by the
    /// rewrite, so this is what says it still composes with division the way it used to.
    fn reference_add_mod(left: U256, right: U256, modulus: U256) -> U256 {
        if modulus.is_zero() {
            return U256::ZERO;
        }

        let (_, left) = reference_div_rem(left, modulus);
        let (_, right) = reference_div_rem(right, modulus);
        let (sum, carry) = left.carrying_add(right);
        if carry || !sum.less_than(modulus) {
            sum.wrapping_sub(modulus)
        } else {
            sum
        }
    }

    /// `div_signed` over the reference division, likewise untouched by the rewrite.
    fn reference_div_signed(numerator: U256, divisor: U256) -> U256 {
        if divisor.is_zero() {
            return U256::ZERO;
        }

        let negate = numerator.is_negative() != divisor.is_negative();
        let left = if numerator.is_negative() {
            numerator.wrapping_neg()
        } else {
            numerator
        };
        let right = if divisor.is_negative() { divisor.wrapping_neg() } else { divisor };
        let (quotient, _) = reference_div_rem(left, right);
        if negate {
            quotient.wrapping_neg()
        } else {
            quotient
        }
    }

    /// `rem_signed` over the reference division, likewise untouched by the rewrite.
    fn reference_rem_signed(numerator: U256, divisor: U256) -> U256 {
        if divisor.is_zero() {
            return U256::ZERO;
        }

        let negate = numerator.is_negative();
        let left = if numerator.is_negative() {
            numerator.wrapping_neg()
        } else {
            numerator
        };
        let right = if divisor.is_negative() { divisor.wrapping_neg() } else { divisor };
        let (_, remainder) = reference_div_rem(left, right);
        if negate {
            remainder.wrapping_neg()
        } else {
            remainder
        }
    }

    /// The values a 256-bit division is most likely to get wrong, and between them every
    /// divisor length the kernels have a separate path for: zero and one, the limb boundaries
    /// with their neighbours, the all-ones patterns, the powers of two, the signed extremes,
    /// and the modulus a curve25519 chain runs on.
    fn edge_values() -> [U256; 28] {
        let one = U256::ONE;
        [
            U256::ZERO,
            one,
            U256::from_u64(2),
            U256::from_u64(7),
            U256::from_u64(31),
            U256::from_u64(u64::MAX),
            one.shift_left(64),
            one.shift_left(64).wrapping_add(one),
            one.shift_left(65),
            one.shift_left(128).wrapping_sub(one),
            one.shift_left(128),
            one.shift_left(128).wrapping_add(one),
            one.shift_left(192).wrapping_sub(one),
            one.shift_left(192),
            one.shift_left(192).wrapping_add(one),
            one.shift_left(255).wrapping_sub(U256::from_u64(19)),
            one.shift_left(255).wrapping_sub(one),
            one.shift_left(255),
            one.shift_left(255).wrapping_add(one),
            U256([u64::MAX; 4]).wrapping_sub(one),
            U256([u64::MAX; 4]),
            U256([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 7, 1 << 63]),
            U256([0xdead_beef_cafe_babe, 3, 0, 1 << 62]),
            U256([u64::MAX, 0, u64::MAX, 0]),
            U256([0, u64::MAX, 0, u64::MAX]),
            U256([1, 2, 3, 1 << 63]),
            U256([0, u64::MAX - 1, 1 << 63, 0]),
            U256([u64::MAX, 1 << 63, 0, 0]),
        ]
    }

    /// A deterministic xorshift generator, which keeps the cross checks reproducible without
    /// the crate taking on a dependency for them.
    struct BiasedRng(u64);

    impl BiasedRng {
        fn next_u64(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            self.0 = value;
            value.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// A value whose limbs lean towards zero, the largest limb and their neighbours,
        /// because that is what it takes to make a quotient digit estimate wrong: under
        /// uniform limbs the add-back path comes up about once in 2^63 digit steps.
        fn next_u256(&mut self) -> U256 {
            let mut limbs = [0u64; 4];
            for limb in limbs.iter_mut() {
                *limb = match self.next_u64() % 5 {
                    0 => 0,
                    1 => u64::MAX,
                    2 => self.next_u64() % 16,
                    3 => u64::MAX - (self.next_u64() % 16),
                    _ => self.next_u64(),
                };
            }
            U256(limbs)
        }
    }

    #[test]
    fn div_rem_matches_the_reference_over_edge_values() {
        let values = edge_values();
        for numerator in values {
            for divisor in values {
                let (quotient, remainder) = numerator.div_rem(divisor);
                assert_eq!(
                    (quotient, remainder),
                    reference_div_rem(numerator, divisor),
                    "{numerator:?} / {divisor:?}"
                );

                if divisor.is_zero() {
                    continue;
                }

                assert!(remainder.less_than(divisor), "{numerator:?} / {divisor:?}");
                assert_eq!(
                    quotient.wrapping_mul(divisor).wrapping_add(remainder),
                    numerator,
                    "{numerator:?} / {divisor:?}"
                );
            }
        }
    }

    #[test]
    fn signed_division_matches_the_reference_over_edge_values() {
        let values = edge_values();
        for numerator in values {
            for divisor in values {
                assert_eq!(
                    numerator.div_signed(divisor),
                    reference_div_signed(numerator, divisor),
                    "{numerator:?} / {divisor:?}"
                );
                assert_eq!(
                    numerator.rem_signed(divisor),
                    reference_rem_signed(numerator, divisor),
                    "{numerator:?} % {divisor:?}"
                );
            }
        }
    }

    #[test]
    fn add_mod_matches_the_reference_over_edge_values() {
        let values = edge_values();
        for modulus in values {
            for left in values {
                for right in values {
                    let result = left.add_mod(right, modulus);
                    assert_eq!(
                        result,
                        reference_add_mod(left, right, modulus),
                        "{left:?} + {right:?} mod {modulus:?}"
                    );
                    assert!(
                        modulus.is_zero() || result.less_than(modulus),
                        "{left:?} + {right:?} mod {modulus:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mul_mod_matches_the_reference_over_edge_values() {
        let values = edge_values();
        for modulus in values {
            for left in values {
                for right in values {
                    let result = left.mul_mod(right, modulus);
                    assert_eq!(
                        result,
                        reference_mul_mod(left, right, modulus),
                        "{left:?} * {right:?} mod {modulus:?}"
                    );
                    assert!(
                        modulus.is_zero() || result.less_than(modulus),
                        "{left:?} * {right:?} mod {modulus:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn biased_random_values_match_the_reference() {
        let mut rng = BiasedRng(BIASED_RANDOM_SEED);
        for _ in 0..BIASED_RANDOM_CASES {
            let left = rng.next_u256();
            let right = rng.next_u256();
            let modulus = rng.next_u256();

            let (quotient, remainder) = left.div_rem(right);
            assert_eq!((quotient, remainder), reference_div_rem(left, right), "{left:?} / {right:?}");
            assert_eq!(left.div_signed(right), reference_div_signed(left, right), "{left:?} / {right:?}");
            assert_eq!(left.rem_signed(right), reference_rem_signed(left, right), "{left:?} % {right:?}");
            assert_eq!(
                left.add_mod(right, modulus),
                reference_add_mod(left, right, modulus),
                "{left:?} + {right:?} mod {modulus:?}"
            );
            assert_eq!(
                left.mul_mod(right, modulus),
                reference_mul_mod(left, right, modulus),
                "{left:?} * {right:?} mod {modulus:?}"
            );

            if !right.is_zero() {
                assert!(remainder.less_than(right), "{left:?} / {right:?}");
                assert_eq!(quotient.wrapping_mul(right).wrapping_add(remainder), left, "{left:?} / {right:?}");
            }
        }
    }

    #[test]
    fn single_limb_operands_divide_like_the_machine() {
        // Operands that fit one limb take a path that divides them directly, and what that
        // has to agree with is the machine's own division rather than any of the kernels.
        let mut rng = BiasedRng(BIASED_RANDOM_SEED);
        for _ in 0..BIASED_RANDOM_CASES {
            let numerator = rng.next_u64();
            let divisor = rng.next_u64() | 1;
            let (quotient, remainder) = U256::from_u64(numerator).div_rem(U256::from_u64(divisor));
            assert_eq!(quotient, U256::from_u64(numerator / divisor), "{numerator} / {divisor}");
            assert_eq!(remainder, U256::from_u64(numerator % divisor), "{numerator} % {divisor}");
        }

        // The boundary the path is chosen at, from either side: the largest single limb
        // numerator, and the smallest numerator that needs two.
        let largest_limb = U256::from_u64(u64::MAX);
        let two_limbs = U256::ONE.shift_left(64);
        assert_eq!(largest_limb.div_rem(largest_limb), (U256::ONE, U256::ZERO));
        assert_eq!(two_limbs.div_rem(largest_limb), (U256::ONE, U256::ONE));
        assert_eq!(largest_limb.div_rem(two_limbs), (U256::ZERO, largest_limb));
    }

    #[test]
    fn mul_mod_keeps_products_that_are_already_below_the_modulus() {
        // The reduction is skipped when the product is below the modulus, however wide the
        // modulus is, so the boundary is the product that equals it: one step either side of
        // that has to come out differently.
        let modulus = U256::ONE.shift_left(255).wrapping_sub(U256::from_u64(19));
        let largest_limb = U256::from_u64(u64::MAX);
        assert_eq!(largest_limb.mul_mod(largest_limb, modulus), U256([1, u64::MAX - 1, 0, 0]));
        assert_eq!(modulus.mul_mod(U256::ONE, modulus), U256::ZERO);
        assert_eq!(
            modulus.wrapping_sub(U256::ONE).mul_mod(U256::ONE, modulus),
            modulus.wrapping_sub(U256::ONE)
        );

        // Five limbs of product against a four limb modulus, which is the shape that says the
        // digit count follows the product rather than the width it could have filled.
        let wide = U256([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1]);
        assert_eq!(wide.mul_mod(largest_limb, modulus), reference_mul_mod(wide, largest_limb, modulus));
    }

    #[test]
    fn the_reciprocal_matches_the_exact_quotient() {
        // The reciprocal comes out of a fixed point iteration instead of a division now, so
        // what says the iteration lands exactly is the division it replaced, over the values
        // such an iteration is likeliest to slip on: every run of set bits, every pair of
        // runs, each of their neighbours, and biased random draws on top.
        fn exact(divisor: u64) -> u64 {
            (u128::MAX / u128::from(divisor)) as u64
        }

        fn run_of_bits(start: u32, end: u32) -> u64 {
            let width = end - start + 1;
            if width == 64 {
                u64::MAX
            } else {
                ((1 << width) - 1) << start
            }
        }

        fn check(value: u64) {
            // The top bit is set on every divisor a reciprocal is ever taken of.
            let divisor = value | (1 << 63);
            assert_eq!(reciprocal(divisor), exact(divisor), "{divisor:#018x}");
        }

        for start in 0..64 {
            for end in start..64 {
                let first = run_of_bits(start, end);
                check(first);
                check(first.wrapping_add(1));
                check(first.wrapping_sub(1));
                for second_start in (0..64).step_by(3) {
                    for second_end in (second_start..64).step_by(7) {
                        check(first | run_of_bits(second_start, second_end));
                    }
                }
            }
        }

        let mut rng = BiasedRng(BIASED_RANDOM_SEED);
        for _ in 0..BIASED_RANDOM_CASES {
            let value = rng.next_u64();
            check(value);
            check(value.wrapping_add(1));
            check(value.wrapping_sub(1));
        }
    }

    #[test]
    fn both_hard_division_paths_are_exercised() {
        // Neither path can be reached on demand from the outside: the correction is common
        // but the add-back needs the operands to conspire, so what proves the cross checks
        // above cover them is counting. Tests share the counters and run in parallel, which
        // can only add to the movement this one sees.
        let (corrections_before, add_backs_before) = path_counters::totals();

        let mut rng = BiasedRng(BIASED_RANDOM_SEED);
        for _ in 0..BIASED_RANDOM_CASES {
            let left = rng.next_u256();
            let right = rng.next_u256();
            let modulus = rng.next_u256();
            let _ = left.div_rem(right);
            let _ = left.mul_mod(right, modulus);
        }

        let (corrections, add_backs) = path_counters::totals();
        assert!(
            corrections > corrections_before,
            "no quotient correction in {BIASED_RANDOM_CASES} biased cases"
        );
        assert!(add_backs > add_backs_before, "no add-back in {BIASED_RANDOM_CASES} biased cases");
    }
}
