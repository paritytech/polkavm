use picosimd::amd64::sse::si128;

const _: () = {
    if core::mem::align_of::<u128>() != 16 {
        panic!("incorrect u128 alignment");
    }
};

static LOOKUP_ARRAY_SHR: [u128; 128] = {
    let mut output = [u128::MAX; 128];
    let mut n = 0;
    while n < 128 {
        output[n] >>= n;
        n += 1;
    }
    output
};

static LOOKUP_ARRAY_SHL: [u128; 128] = {
    let mut output = [u128::MAX; 128];
    let mut n = 0;
    while n < 128 {
        output[n] <<= n;
        n += 1;
    }
    output
};

static LOOKUP_ARRAY_BIT: [u128; 128] = {
    let mut output = [1; 128];
    let mut n = 0;
    while n < 128 {
        output[n] <<= n;
        n += 1;
    }
    output
};

#[inline(always)]
fn lookup_shr(count: usize) -> si128 {
    assert!(count < 128);
    unsafe {
        si128::load_aligned(LOOKUP_ARRAY_SHR.as_ptr().cast::<u8>().add(count * core::mem::size_of::<si128>()))
    }
}

#[inline(always)]
fn lookup_shl(count: usize) -> si128 {
    assert!(count < 128);
    unsafe {
        si128::load_aligned(LOOKUP_ARRAY_SHL.as_ptr().cast::<u8>().add(count * core::mem::size_of::<si128>()))
    }
}

#[inline(always)]
fn lookup_bit(position: usize) -> si128 {
    assert!(position < 128);
    unsafe {
        si128::load_aligned(LOOKUP_ARRAY_BIT.as_ptr().cast::<u8>().add(position * core::mem::size_of::<si128>()))
    }
}

impl RawMask for si128 {
    #[inline(always)]
    fn zero() -> Self {
        unsafe {
            si128::zero()
        }
    }

    #[inline(always)]
    fn full() -> Self {
        unsafe {
            si128::negative_one()
        }
    }

    #[inline(always)]
    fn bit(position: usize) -> Self {
        lookup_bit(position)
    }

    #[inline(always)]
    fn mask_lo(offset: usize, length: usize) -> Self {
        lookup_shl(offset) & lookup_shr(Self::alignment() - (offset + length))
    }

    #[inline(always)]
    fn mask_hi(length: usize) -> Self {
        lookup_shr(Self::alignment() - length)
    }

    #[inline(always)]
    fn bitandnot_assign(&mut self, rhs: Self) {
        unsafe {
            *self = self.and_not(rhs);
        }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            Self::full().and_not(self)
        }
    }

    #[inline(always)]
    fn is_equal(self, rhs: Self) -> bool {
        unsafe {
            self.is_equal_slow(rhs)
        }
    }

    #[inline(always)]
    fn is_zero(self) -> bool {
        unsafe {
            self.is_equal_slow(si128::zero())
        }
    }

    #[cfg(test)]
    fn trailing_zeros(self) -> u32 {
        let mut output = 0;
        for x in unsafe { self.as_i64x2().to_array() } {
            let x = x.trailing_zeros();
            output += x;

            if x != 64 {
                break;
            }
        }

        output
    }

    #[cfg(test)]
    fn leading_zeros(self) -> u32 {
        let mut output = 0;
        for x in unsafe { self.as_i64x2().to_array().into_iter().rev() } {
            let x = x.leading_zeros();
            output += x;

            if x != 64 {
                break;
            }
        }

        output
    }
}

#[test]
fn test_bit_set_si128() {
    let _ = env_logger::try_init();

    let mut set = BitSet::<si128, {256 / bits::<si128>()}>::new();
    set.insert_range(0, 1);
    assert!(set.contains_range(0, 1));
    assert!(!set.contains_range(0, 2));
    assert!(!set.contains_range(1, 2));
    set.insert_range(2, 3);
    set.insert_range(8, 9);

    set.clear();
    set.insert_one(0);
    assert_eq!(set.first_non_zero(), Some(0));
    assert_eq!(set.last_non_zero(), Some(0));
    assert!(set.contains_range(0, 1));
    assert!(!set.contains_range(0, 2));
    assert!(!set.contains_range(1, 2));

    set.clear();
    set.insert_one(1);
    assert_eq!(set.first_non_zero(), Some(1));
    assert_eq!(set.last_non_zero(), Some(1));
    assert!(!set.contains_range(0, 1));
    assert!(!set.contains_range(0, 2));
    assert!(set.contains_range(1, 2));
    assert!(!set.contains_range(1, 3));
    assert!(!set.contains_range(2, 3));

    set.clear();
    set.insert_one(65);
    assert_eq!(set.first_non_zero(), Some(65));
    assert_eq!(set.last_non_zero(), Some(65));
    assert!(set.contains_range(65, 66));
    assert!(!set.contains_range(64, 65));
    assert!(!set.contains_range(64, 66));
    assert!(!set.contains_range(64, 67));
    assert!(!set.contains_range(66, 67));

    set.clear();
    set.insert_range(50, 51);
    assert!(set.contains_range(50, 51));
    assert!(!set.contains_range(50, 52));
    assert!(!set.contains_range(49, 51));
    assert!(!set.contains_range(49, 52));
    assert!(!set.contains_range(0, 64));
    assert!(!set.contains_range(0, 256));

    set.clear();
    assert!(set.contains_range(0, 0));
    assert!(set.contains_range(123, 123));
    assert!(!set.contains_range(0, 64));
    assert!(!set.contains_range(0, 128));
    assert!(!set.contains_range(64, 128));

    set.insert_range(64, 128);
    assert!(set.contains_range(64, 128));
    assert!(set.contains_range(65, 127));
    assert!(set.contains_range(64, 65));
    assert!(!set.contains_range(63, 65));
    assert!(set.contains_range(127, 128));
    assert!(!set.contains_range(127, 129));
    assert_eq!(set.first_non_zero(), Some(64));
    assert_eq!(set.last_non_zero(), Some(127));

    set.remove_range(64, 128);
    assert!(!set.contains_range(64, 128));

    set.insert_range(32, 160);
    assert_eq!(set.first_non_zero(), Some(32));
    assert_eq!(set.last_non_zero(), Some(159));
    assert!(set.contains_range(64, 128));
    assert!(set.contains_range(63, 128));
    assert!(set.contains_range(64, 129));
    assert!(set.contains_range(63, 129));
    assert!(set.contains_range(65, 128));
    assert!(set.contains_range(64, 127));
    assert!(set.contains_range(65, 127));
    assert!(set.contains_range(100, 101));
    assert!(set.contains_range(32, 160));
    assert!(!set.contains_range(31, 160));
    assert!(!set.contains_range(32, 161));

    set.remove_range(50, 51);
    assert!(set.contains_range(51, 160));
    assert!(set.contains_range(51, 159));
    assert!(!set.contains_range(50, 160));
    assert!(!set.contains_range(50, 159));
    assert!(set.contains_range(32, 50));
    assert!(!set.contains_range(32, 51));
    assert!(set.contains_range(33, 50));
    assert!(!set.contains_range(33, 51));
}
