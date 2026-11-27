use crate::primes::square_start;

use self::patterns::{index_pattern, mask_pattern_set_u8};

pub mod patterns {

    /// Calculate index pattern for a given word width (BITS) and given skip
    /// factor. These are the relative addresses of the words we need to set
    /// bits on. If we have 8 bits, we have 8 addresses. This is a `const`
    /// function: it can be used in the dense resetter to produce compile-time
    /// constants.
    pub const fn index_pattern<const BITS: usize>(skip: usize) -> [usize; BITS] {
        let start = skip / 2;
        let mut pattern = [0; BITS];
        let mut i = 0;
        while i < BITS {
            let relative_index = start + i * skip;
            pattern[i] = relative_index / BITS;
            i += 1;
        }
        pattern
    }

    /// Calculate modulo pattern for a given word width (BITS) and given skip
    /// factor. These are the bit numbers within the words that we need to set.
    pub const fn modulo_pattern<const BITS: usize>(skip: usize) -> [usize; BITS] {
        let start = skip / 2;
        let mut pattern = [0; BITS];
        let mut i = 0;
        while i < BITS {
            let relative_index = start + i * skip;
            pattern[i] = relative_index % BITS;
            i += 1;
        }
        pattern
    }

    /// Produces a set of masks based on [`modulo_pattern`]. It's simply
    /// the bit address shifted left into the correct position. This produces
    /// a set of 8 single-bit masks for u8 words (bytes).
    pub const fn mask_pattern_set_u8(skip: usize) -> [u8; 8] {
        let mod_pattern = modulo_pattern::<8>(skip);
        let mut masks = [0; 8];
        let mut i = 0;
        while i < 8 {
            masks[i] = 1u8 << mod_pattern[i];
            i += 1;
        }
        masks
    }

    /// Calculate the equivalent (base) skip factor that will yield
    /// the same modulo pattern as a higher skip factor, for a given
    /// number of bits. They are periodic.
    pub const fn pattern_equivalent_skip(skip: usize, bits: usize) -> usize {
        let index = (skip - 3) / 2;
        let modulo = index % bits;
        (modulo * 2) + 3
    }
}

/// Reinterpret a slice of u64 as a slice of u8, with the correct length. You can't just use `transmute`
/// for this, because it doesn't calculate the correct length. Alignment should not be an issue, because
/// we're casting from a wider type to a narrower one.
fn reinterpret_slice_mut_u64_u8(words: &mut [u64]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 8) }
}

/// Specific implementation for the sparse resetter where we have words that
/// are relatively far apart. This is generic on `EQUIVALENT_SKIP`, which allows
/// the compiler to calculate the mask-patterns efficiently. However, we still
/// need to calculate the index-pattern dynamically, so this is less efficient,
/// but more general, than dense resetting. We're still resetting bits one
/// at a time.
pub struct ResetterSparseU8<const EQUIVALENT_SKIP: usize>();
impl<const EQUIVALENT_SKIP: usize> ResetterSparseU8<EQUIVALENT_SKIP> {
    const SINGLE_BIT_MASK_SET: [u8; 8] = mask_pattern_set_u8(EQUIVALENT_SKIP);

    #[inline(always)]
    pub fn reset_sparse(words: &mut [u64], skip: usize) {
        // calculate relative indices for the words we need to reset
        let relative_indices = index_pattern::<8>(skip);

        // cast our wide word vector to bytes
        let bytes: &mut [u8] = reinterpret_slice_mut_u64_u8(words);

        // determine the offset of the first skip-size chunk we need
        // to touch, and proceed from there.
        let square_start = square_start(skip);
        debug_assert!(
            square_start < bytes.len() * 8,
            "square_start should be within the bounds of our array; check caller"
        );
        let start_chunk_offset = square_start / 8 / skip * skip;
        debug_assert!(
            start_chunk_offset > skip / 2 / 8,
            "sparse resets are for larger skip factors; this starts too early: {}",
            start_chunk_offset
        );

        let mut chunks = bytes[start_chunk_offset..].chunks_exact_mut(skip);
        (&mut chunks).for_each(|chunk| {
            #[allow(clippy::needless_range_loop)]
            for i in 0..8 {
                let word_idx = relative_indices[i];
                // Safety: relative indices are all smaller than `skip` by construction
                unsafe {
                    *chunk.get_unchecked_mut(word_idx) |= Self::SINGLE_BIT_MASK_SET[i];
                }
            }
        });

        let remainder = chunks.into_remainder();
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            let word_idx = relative_indices[i];
            if word_idx < remainder.len() {
                // Safety: check above breaks the loop before we exceed remainder.len()
                unsafe {
                    *remainder.get_unchecked_mut(word_idx) |= Self::SINGLE_BIT_MASK_SET[i];
                }
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::unrolled::patterns::*;

    #[test]
    fn modulo_pattern_set_u8_correct() {
        assert_eq!(modulo_pattern::<8>(3), [1, 4, 7, 2, 5, 0, 3, 6]);
        assert_eq!(modulo_pattern::<8>(5), [2, 7, 4, 1, 6, 3, 0, 5]);
        assert_eq!(modulo_pattern::<8>(17), [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn mask_pattern_set_u8_correct() {
        let expected: Vec<u8> = [1, 4, 7, 2, 5, 0, 3, 6].iter().map(|b| 1 << b).collect();
        assert_eq!(mask_pattern_set_u8(3), expected[..]);
    }

    #[test]
    fn modulo_pattern_set_u32_correct() {
        assert_eq!(
            modulo_pattern::<32>(3),
            [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        );
        assert_eq!(
            modulo_pattern::<32>(5),
            [2, 7, 12, 17, 22, 27, 0, 5, 10, 15, 20, 25, 30, 3, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 31, 4, 9, 14, 19, 24, 29]
        );
        assert_eq!(
            modulo_pattern::<32>(65),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        );
    }

    #[test]
    fn index_pattern_set_u8_correct() {
        assert_eq!(index_pattern::<8>(3), [0, 0, 0, 1, 1, 2, 2, 2]);
        assert_eq!(index_pattern::<8>(5), [0, 0, 1, 2, 2, 3, 4, 4]);
        assert_eq!(index_pattern::<8>(17), [1, 3, 5, 7, 9, 11, 13, 15]);
        assert_eq!(index_pattern::<8>(51), [3, 9, 15, 22, 28, 35, 41, 47,]);
    }

    #[test]
    fn index_pattern_set_u32_correct() {
        assert_eq!(
            index_pattern::<32>(3),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,]
        );
        assert_eq!(
            index_pattern::<32>(5),
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,]
        );
        assert_eq!(
            index_pattern::<32>(51),
            [0, 2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 18, 19, 21, 23, 24, 26, 27, 29, 31, 32, 34, 35, 37, 39, 40, 42, 43, 45, 47, 48, 50,]
        )
    }
}
