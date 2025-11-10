use core::marker::PhantomData;
use polkavm_common::cast::cast;

#[inline(always)]
const fn bits<T>() -> usize {
    core::mem::size_of::<T>() * 8
}

trait RawMask:
    Copy + Clone + Sized + core::ops::BitAnd<Output = Self> + core::ops::BitOr<Output = Self> + core::ops::BitAndAssign + core::ops::BitOrAssign
{
    #[inline(always)]
    fn alignment() -> usize {
        bits::<Self>()
    }

    fn zero() -> Self;
    fn full() -> Self;
    fn bit(position: usize) -> Self;
    fn mask_lo(offset: usize, length: usize) -> Self;
    fn mask_hi(length: usize) -> Self;
    fn bitandnot_assign(&mut self, rhs: Self);
    fn not(self) -> Self;
    fn is_equal(self, rhs: Self) -> bool;
    fn is_zero(self) -> bool;

    #[cfg(test)]
    fn trailing_zeros(self) -> u32;

    #[cfg(test)]
    fn leading_zeros(self) -> u32;
}

impl RawMask for u64 {
    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn full() -> Self {
        Self::MAX
    }

    #[inline(always)]
    fn bit(position: usize) -> Self {
        1 << position
    }

    #[inline(always)]
    fn mask_lo(offset: usize, length: usize) -> Self {
        (Self::MAX >> (Self::alignment() - length)) << offset
    }

    #[inline(always)]
    fn mask_hi(length: usize) -> Self {
        Self::MAX >> (Self::alignment() - length)
    }

    #[inline(always)]
    fn bitandnot_assign(&mut self, rhs: Self) {
        *self &= !rhs;
    }

    #[inline(always)]
    fn not(self) -> Self {
        !self
    }

    #[inline(always)]
    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    #[inline(always)]
    fn is_zero(self) -> bool {
        self == 0
    }

    #[cfg(test)]
    fn trailing_zeros(self) -> u32 {
        u64::trailing_zeros(self)
    }

    #[cfg(test)]
    fn leading_zeros(self) -> u32 {
        u64::leading_zeros(self)
    }
}

#[derive(Debug)]
struct AlignedRange<M>
where
    M: RawMask,
{
    mask_ty: PhantomData<M>,
    aligned_start_group: usize,
    aligned_start: usize,
    aligned_end_group: usize,
    aligned_end: usize,
    unaligned_lo_start: usize,
    unaligned_lo_end: usize,
    unaligned_lo_group: usize,
    unaligned_lo_length: usize,
    unaligned_lo_offset: usize,
    unaligned_hi_start: usize,
    unaligned_hi_end: usize,
    unaligned_hi_group: usize,
    unaligned_hi_length: usize,
}

impl<M> AlignedRange<M>
where
    M: RawMask,
{
    #[inline(always)]
    fn unaligned_mask_lo(&self) -> M {
        if self.unaligned_lo_length == 0 {
            M::zero()
        } else {
            M::mask_lo(self.unaligned_lo_offset, self.unaligned_lo_length)
        }
    }

    #[inline(always)]
    fn unaligned_mask_hi(&self) -> M {
        if self.unaligned_hi_length == 0 {
            M::zero()
        } else {
            M::mask_hi(self.unaligned_hi_length)
        }
    }

    #[inline(always)]
    fn has_unaligned_lo(&self) -> bool {
        self.unaligned_lo_length > 0
    }

    #[inline(always)]
    fn has_unaligned_hi(&self) -> bool {
        self.unaligned_hi_length > 0
    }
}

macro_rules! kani_assert {
    ($cond:expr) => {{
        #[cfg(any(kani, test))]
        assert!($cond);

        #[cfg(not(any(kani, test)))]
        if !$cond {
            // SAFETY: The condition was proven by Kani to always hold.
            unsafe {
                core::hint::unreachable_unchecked();
            }
        }
    }};
}

macro_rules! kani_assert_eq {
    ($lhs:expr, $rhs:expr) => {{
        #[cfg(any(kani, test))]
        assert_eq!($lhs, $rhs);

        #[cfg(not(any(kani, test)))]
        if $lhs != $rhs {
            // SAFETY: The condition was proven by Kani to always hold.
            unsafe {
                core::hint::unreachable_unchecked();
            }
        }
    }};
}

#[inline(always)]
fn align_range<M>(start: usize, end: usize) -> AlignedRange<M>
where
    M: RawMask,
{
    #[inline(always)]
    fn opt(cond: bool, value: usize) -> usize {
        let mask = if cond { !0 } else { 0 };
        mask & value
    }

    let original_start = start;
    let start = start.min(end);

    let aligned_end_group = end / M::alignment();
    let aligned_start_group = start.div_ceil(M::alignment());
    let has_aligned = aligned_end_group > aligned_start_group;
    let aligned_end_group = opt(has_aligned, aligned_end_group);
    let aligned_start_group = opt(has_aligned, aligned_start_group);
    let aligned_end = aligned_end_group * M::alignment();
    let aligned_start = aligned_start_group * M::alignment();

    let has_unaligned_lo = (start % M::alignment()) != 0 || !has_aligned;
    let unaligned_lo_start = opt(has_unaligned_lo, start);
    let unaligned_lo_group = unaligned_lo_start / M::alignment();
    let unaligned_lo_group_start = unaligned_lo_group * M::alignment();
    let unaligned_lo_end = opt(has_unaligned_lo, (unaligned_lo_group_start.saturating_add(M::alignment())).min(end));
    let unaligned_lo_length = unaligned_lo_end - unaligned_lo_start;
    let unaligned_lo_offset = unaligned_lo_start - unaligned_lo_group_start;

    let has_unaligned_hi = (end % M::alignment()) != 0 && end > unaligned_lo_end;
    let unaligned_hi_end = opt(has_unaligned_hi, end);
    let unaligned_hi_group = unaligned_hi_end / M::alignment();
    let unaligned_hi_start = unaligned_hi_group * M::alignment();
    let unaligned_hi_length = unaligned_hi_end - unaligned_hi_start;

    kani_assert!(unaligned_lo_start <= unaligned_lo_end);
    kani_assert!(unaligned_hi_start <= unaligned_hi_end);
    kani_assert_eq!(unaligned_lo_end - unaligned_lo_start, unaligned_lo_length);
    kani_assert_eq!(unaligned_hi_end - unaligned_hi_start, unaligned_hi_length);
    kani_assert_eq!(aligned_start % M::alignment(), 0);
    kani_assert_eq!(aligned_end % M::alignment(), 0);
    kani_assert_eq!(unaligned_hi_start % M::alignment(), 0);
    kani_assert!(aligned_end >= aligned_start);
    kani_assert!(unaligned_lo_offset < M::alignment());
    kani_assert!(unaligned_hi_length == 0 || unaligned_hi_start >= unaligned_lo_end);
    kani_assert!(unaligned_hi_length == 0 || unaligned_hi_start >= aligned_end);
    kani_assert!(aligned_start == aligned_end || unaligned_lo_end <= aligned_start);

    kani_assert!(aligned_start_group < usize::MAX / M::alignment());
    kani_assert!(aligned_end_group <= usize::MAX / M::alignment());
    kani_assert!(unaligned_lo_group <= usize::MAX / M::alignment());
    kani_assert!(unaligned_hi_group <= usize::MAX / M::alignment());

    kani_assert!(start <= original_start);
    kani_assert!(start <= end);
    kani_assert!(aligned_start_group <= aligned_end_group);
    kani_assert!(aligned_start_group <= (start / M::alignment() + 1));
    kani_assert!(aligned_end_group <= end / M::alignment());
    kani_assert!(unaligned_lo_group <= start / M::alignment());
    kani_assert!(unaligned_hi_group <= end / M::alignment());

    AlignedRange {
        mask_ty: PhantomData,
        aligned_start_group,
        aligned_start,
        aligned_end_group,
        aligned_end,

        unaligned_lo_start,
        unaligned_lo_end,
        unaligned_lo_group,
        unaligned_lo_length,
        unaligned_lo_offset,

        unaligned_hi_start,
        unaligned_hi_end,
        unaligned_hi_group,
        unaligned_hi_length,
    }
}

#[cfg(kani)]
#[kani::proof]
fn proof_align_range() {
    let start: usize = kani::any();
    let end: usize = kani::any();
    align_range::<u64>(start, end);
}

#[test]
fn test_align_range() {
    let r = align_range::<u64>(0, 64);
    assert_eq!(r.aligned_start_group, 0);
    assert_eq!(r.aligned_end_group, 1);
    assert_eq!(r.aligned_start, 0);
    assert_eq!(r.aligned_end, 64);
    assert_eq!(r.unaligned_lo_start, r.unaligned_lo_end);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_lo_offset, 0);
    assert_eq!(r.unaligned_lo_length, 0);
    assert_eq!(r.unaligned_hi_start, r.unaligned_hi_end);
    assert_eq!(r.unaligned_hi_group, 0);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0);
    assert_eq!(r.unaligned_mask_hi(), 0);

    let r = align_range::<u64>(1, 64);
    assert_eq!(r.aligned_start_group, r.aligned_end_group);
    assert_eq!(r.aligned_start, r.aligned_end);
    assert_eq!(r.unaligned_lo_start, 1);
    assert_eq!(r.unaligned_lo_end, 64);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_lo_offset, 1);
    assert_eq!(r.unaligned_lo_length, 63);
    assert_eq!(r.unaligned_hi_start, r.unaligned_hi_end);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0xffffffff_fffffffe);
    assert_eq!(r.unaligned_mask_hi(), 0);

    let r = align_range::<u64>(1, 128);
    assert_eq!(r.aligned_start_group, 1);
    assert_eq!(r.aligned_end_group, 2);
    assert_eq!(r.aligned_start, 64);
    assert_eq!(r.aligned_end, 128);
    assert_eq!(r.unaligned_lo_start, 1);
    assert_eq!(r.unaligned_lo_end, 64);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_lo_offset, 1);
    assert_eq!(r.unaligned_lo_length, 63);
    assert_eq!(r.unaligned_hi_start, r.unaligned_hi_end);
    assert_eq!(r.unaligned_hi_group, 0);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0xffffffff_fffffffe);
    assert_eq!(r.unaligned_mask_hi(), 0);

    let r = align_range::<u64>(0, 63);
    assert_eq!(r.aligned_start_group, 0);
    assert_eq!(r.aligned_end_group, 0);
    assert_eq!(r.aligned_start, 0);
    assert_eq!(r.aligned_end, 0);
    assert_eq!(r.unaligned_lo_start, 0);
    assert_eq!(r.unaligned_lo_end, 63);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_lo_offset, 0);
    assert_eq!(r.unaligned_lo_length, 63);
    assert_eq!(r.unaligned_hi_start, r.unaligned_hi_end);
    assert_eq!(r.unaligned_hi_group, 0);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0x7fffffff_ffffffff);
    assert_eq!(r.unaligned_mask_hi(), 0);

    let r = align_range::<u64>(1, 129);
    assert_eq!(r.aligned_start_group, 1);
    assert_eq!(r.aligned_end_group, 2);
    assert_eq!(r.aligned_start, 64);
    assert_eq!(r.aligned_end, 128);
    assert_eq!(r.unaligned_lo_start, 1);
    assert_eq!(r.unaligned_lo_end, 64);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_lo_offset, 1);
    assert_eq!(r.unaligned_lo_length, 63);
    assert_eq!(r.unaligned_hi_start, 128);
    assert_eq!(r.unaligned_hi_end, 129);
    assert_eq!(r.unaligned_hi_group, 2);
    assert_eq!(r.unaligned_hi_length, 1);
    assert_eq!(r.unaligned_mask_lo(), 0xffffffff_fffffffe);
    assert_eq!(r.unaligned_mask_hi(), 0x00000000_00000001);

    let r = align_range::<u64>(33, 33);
    assert_eq!(r.aligned_start_group, r.aligned_end_group);
    assert_eq!(r.aligned_start, r.aligned_end);
    assert_eq!(r.unaligned_lo_group, 0);
    assert_eq!(r.unaligned_hi_group, 0);
    assert_eq!(r.unaligned_lo_length, 0);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0);
    assert_eq!(r.unaligned_mask_hi(), 0);

    let r = align_range::<u64>(66, 70);
    assert_eq!(r.aligned_start_group, r.aligned_end_group);
    assert_eq!(r.aligned_start, r.aligned_end);
    assert_eq!(r.unaligned_lo_group, 1);
    assert_eq!(r.unaligned_lo_offset, 2);
    assert_eq!(r.unaligned_lo_length, 4);
    assert_eq!(r.unaligned_hi_length, 0);
    assert_eq!(r.unaligned_mask_lo(), 0b111100, "unexpected mask: 0b{:b}", r.unaligned_mask_lo());
    assert_eq!(r.unaligned_mask_hi(), 0);

    align_range::<u64>(4, 2);
}

#[derive(Clone)]
#[repr(align(32))]
struct AlignedArray<M, const N: usize>([M; N]);

impl<M, const N: usize> core::ops::Deref for AlignedArray<M, N> {
    type Target = [M; N];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M, const N: usize> core::ops::DerefMut for AlignedArray<M, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
struct BitSet<M, const N: usize>
where
    M: RawMask,
{
    data: Box<AlignedArray<M, N>>,
}

impl<M, const N: usize> BitSet<M, N>
where
    M: RawMask,
{
    fn new() -> Self {
        let byte_length = const { core::mem::size_of::<M>().checked_mul(N).unwrap() };

        // SAFETY: We properly initialize everything we allocate before calling `assume_init`.
        let data = unsafe {
            let mut data: Box<core::mem::MaybeUninit<AlignedArray<M, N>>> = Box::new_uninit();
            core::ptr::write_bytes(data.as_mut_ptr().cast::<u8>(), 0, byte_length);
            data.assume_init()
        };

        Self { data }
    }

    fn clear(&mut self) {
        self.data.fill(M::zero());
    }

    #[inline(always)]
    fn insert_range(&mut self, start: usize, end: usize) {
        let r = align_range::<M>(start, end);
        assert!(
            (r.unaligned_lo_group < self.data.len())
                & (r.unaligned_hi_group < self.data.len())
                & (r.aligned_end_group <= self.data.len())
                & (r.aligned_start_group < self.data.len())
                & (r.aligned_start_group <= r.aligned_end_group)
        );

        self.data[r.aligned_start_group..r.aligned_end_group].fill(M::full());
        self.data[r.unaligned_lo_group] |= r.unaligned_mask_lo();
        self.data[r.unaligned_hi_group] |= r.unaligned_mask_hi();
    }

    #[inline(always)]
    fn insert_one(&mut self, index: usize) {
        let group = index / M::alignment();
        let offset = index % M::alignment();
        self.data[group] |= M::bit(offset);
    }

    #[inline(always)]
    fn remove_one(&mut self, index: usize) {
        let group = index / M::alignment();
        let offset = index % M::alignment();
        self.data[group].bitandnot_assign(M::bit(offset));
    }

    #[inline(always)]
    fn contains_one(&self, index: usize) -> bool {
        let group = index / M::alignment();
        let offset = index % M::alignment();
        let mask = M::bit(offset);
        !(self.data[group] & mask).is_zero()
    }

    #[inline(always)]
    fn remove_range(&mut self, start: usize, end: usize) {
        let r = align_range::<M>(start, end);
        self.data[r.aligned_start_group..r.aligned_end_group].fill(M::zero());
        self.data[r.unaligned_lo_group].bitandnot_assign(r.unaligned_mask_lo());
        self.data[r.unaligned_hi_group].bitandnot_assign(r.unaligned_mask_hi());
    }

    #[inline(always)]
    fn contains_range(&self, start: usize, end: usize) -> bool {
        if start == end {
            return true;
        }

        let r = align_range::<M>(start, end);
        if r.aligned_end_group > self.data.len() {
            return false;
        }

        let mut result = M::full();
        for &value in &self.data[r.aligned_start_group..r.aligned_end_group] {
            result &= value;
        }

        result &= r.unaligned_mask_lo().not() | (self.data[r.unaligned_lo_group] & r.unaligned_mask_lo());
        result &= r.unaligned_mask_hi().not() | (self.data[r.unaligned_hi_group] & r.unaligned_mask_hi());
        result.is_equal(M::full())
    }

    #[cfg(test)]
    fn first_non_zero(&self) -> Option<usize> {
        let index = self.data.iter().position(|&value| !value.is_zero())?;
        Some(index * M::alignment() + cast(self.data[index].trailing_zeros()).to_usize())
    }

    #[cfg(test)]
    fn last_non_zero(&self) -> Option<usize> {
        let index = self.data.iter().rev().position(|&value| !value.is_zero())?;
        let index = self.data.len() - index - 1;
        Some(index * M::alignment() + (M::alignment() - 1 - cast(self.data[index].leading_zeros()).to_usize()))
    }
}

#[test]
fn test_bit_set_u64() {
    let _ = env_logger::try_init();

    let mut set = BitSet::<u64, { 256 / bits::<u64>() }>::new();
    set.insert_range(0, 1);
    assert!(set.contains_range(0, 1));
    assert!(!set.contains_range(0, 2));
    assert!(!set.contains_range(1, 2));
    assert_eq!(set.data[0], (1 << 0), "unexpected: 0b{:b}", set.data[0]);
    set.insert_range(2, 3);
    assert_eq!(set.data[0], (1 << 0) | (1 << 2), "unexpected: 0b{:b}", set.data[0]);
    set.insert_range(8, 9);
    assert_eq!(set.data[0], (1 << 0) | (1 << 2) | (1 << 8), "unexpected: 0b{:b}", set.data[0]);

    {
        #[allow(clippy::undocumented_unsafe_blocks)]
        let raw_slice = unsafe { core::slice::from_raw_parts(set.data.as_ptr().cast::<u8>(), 256 / 8) };
        assert_eq!(raw_slice[0], 0b00000101);
        assert_eq!(raw_slice[1], 0b00000001);
    }

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
    assert_eq!(&set.data[..], [0x00040000_00000000, 0, 0, 0]);
    assert!(set.contains_range(50, 51));
    assert!(!set.contains_range(50, 52));
    assert!(!set.contains_range(49, 51));
    assert!(!set.contains_range(49, 52));
    assert!(!set.contains_range(0, 64));
    assert!(!set.contains_range(0, 256));

    set.clear();
    assert_eq!(&set.data[..], [0, 0, 0, 0]);
    assert!(set.contains_range(0, 0));
    assert!(set.contains_range(123, 123));
    assert!(!set.contains_range(0, 64));
    assert!(!set.contains_range(0, 128));
    assert!(!set.contains_range(64, 128));

    set.insert_range(64, 128);
    assert_eq!(&set.data[..], [0, u64::MAX, 0, 0]);
    assert!(set.contains_range(64, 128));
    assert!(set.contains_range(65, 127));
    assert!(set.contains_range(64, 65));
    assert!(!set.contains_range(63, 65));
    assert!(set.contains_range(127, 128));
    assert!(!set.contains_range(127, 129));
    assert_eq!(set.first_non_zero(), Some(64));
    assert_eq!(set.last_non_zero(), Some(127));

    set.remove_range(64, 128);
    assert_eq!(&set.data[..], [0, 0, 0, 0]);
    assert!(!set.contains_range(64, 128));

    set.insert_range(32, 160);
    assert_eq!(set.first_non_zero(), Some(32));
    assert_eq!(set.last_non_zero(), Some(159));
    assert_eq!(&set.data[..], [0xffffffff_00000000, u64::MAX, 0x00000000_ffffffff, 0]);
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
    assert_eq!(&set.data[..], [0xfffbffff_00000000, u64::MAX, 0x00000000_ffffffff, 0]);
    assert!(set.contains_range(51, 160));
    assert!(set.contains_range(51, 159));
    assert!(!set.contains_range(50, 160));
    assert!(!set.contains_range(50, 159));
    assert!(set.contains_range(32, 50));
    assert!(!set.contains_range(32, 51));
    assert!(set.contains_range(33, 50));
    assert!(!set.contains_range(33, 51));
}

impl<M, const N: usize> Default for BitSet<M, N>
where
    M: RawMask,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::undocumented_unsafe_blocks)]
mod sse {
    use super::RawMask;
    #[cfg(test)]
    use super::{bits, BitSet};

    include!("page_set_sse.rs");
}

#[cfg(target_arch = "x86_64")]
type PageSetGroup = picosimd::amd64::sse::si128;

#[cfg(not(target_arch = "x86_64"))]
type PageSetGroup = u64;

const PAGE_COUNT: usize = 2usize.pow(32) / 4096; // 1048576
const PAGES_PER_GROUP: usize = bits::<PageSetGroup>();
const GROUP_COUNT: usize = PAGE_COUNT / PAGES_PER_GROUP; // 16386 for 64, 8192 for 128, 4096 for 256

#[derive(Clone, Default)]
pub struct PageSet {
    /// A page bitset, with one bit per every page in a partial group.
    pages: BitSet<PageSetGroup, { PAGE_COUNT / bits::<PageSetGroup>() }>,
    /// A group bitset, with one bit for every partially-filled group.
    groups_partial: BitSet<PageSetGroup, { GROUP_COUNT / bits::<PageSetGroup>() }>,
    /// A group bitset, with one bit for every filled group.
    groups_filled: BitSet<PageSetGroup, { GROUP_COUNT / bits::<PageSetGroup>() }>,
}

impl PageSet {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn insert(&mut self, (min, max): (u32, u32)) {
        let min = cast(min).to_usize();
        let max = cast(max).to_usize();
        self.insert_exclusive((min, max + 1));
    }

    #[inline(never)]
    pub fn insert_exclusive(&mut self, (start, end): (usize, usize)) {
        let r = align_range::<PageSetGroup>(start, end);
        self.groups_filled.insert_range(r.aligned_start_group, r.aligned_end_group);

        if r.has_unaligned_lo() {
            if !self.groups_partial.contains_one(r.unaligned_lo_group) {
                self.pages.data[r.unaligned_lo_group] = <PageSetGroup as RawMask>::zero();
            }

            self.pages.insert_range(r.unaligned_lo_start, r.unaligned_lo_end);
            if RawMask::is_equal(self.pages.data[r.unaligned_lo_group], PageSetGroup::full()) {
                self.groups_partial.remove_one(r.unaligned_lo_group);
                self.groups_filled.insert_one(r.unaligned_lo_group);
            } else {
                self.groups_partial.insert_one(r.unaligned_lo_group);
            }
        }

        if r.has_unaligned_hi() {
            if !self.groups_partial.contains_one(r.unaligned_hi_group) {
                self.pages.data[r.unaligned_hi_group] = <PageSetGroup as RawMask>::zero();
            }

            self.pages.insert_range(r.unaligned_hi_start, r.unaligned_hi_end);
            if RawMask::is_equal(self.pages.data[r.unaligned_hi_group], PageSetGroup::full()) {
                self.groups_partial.remove_one(r.unaligned_hi_group);
                self.groups_filled.insert_one(r.unaligned_hi_group);
            } else {
                self.groups_partial.insert_one(r.unaligned_hi_group);
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, (min, max): (u32, u32)) {
        let min = cast(min).to_usize();
        let max = cast(max).to_usize();
        self.remove_exclusive((min, max + 1));
    }

    #[inline(never)]
    pub fn remove_exclusive(&mut self, (start, end): (usize, usize)) {
        let r = align_range::<PageSetGroup>(start, end);

        self.groups_filled.remove_range(r.aligned_start_group, r.aligned_end_group);
        self.groups_partial.remove_range(r.aligned_start_group, r.aligned_end_group);
        self.pages.remove_range(r.unaligned_lo_start, r.unaligned_lo_end);
        self.pages.remove_range(r.unaligned_hi_start, r.unaligned_hi_end);
        if r.has_unaligned_lo() {
            if self.groups_filled.contains_one(r.unaligned_lo_group) {
                self.groups_filled.remove_one(r.unaligned_lo_group);
                self.groups_partial.insert_one(r.unaligned_lo_group);
                self.pages
                    .insert_range(r.unaligned_lo_group * PAGES_PER_GROUP, r.unaligned_lo_start);
                self.pages
                    .insert_range(r.unaligned_lo_end, (r.unaligned_lo_group + 1) * PAGES_PER_GROUP);
            } else if RawMask::is_zero(self.pages.data[r.unaligned_lo_group]) {
                self.groups_partial.remove_one(r.unaligned_lo_group);
            }
        }

        if r.has_unaligned_hi() {
            if self.groups_filled.contains_one(r.unaligned_hi_group) {
                self.groups_filled.remove_one(r.unaligned_hi_group);
                self.groups_partial.insert_one(r.unaligned_hi_group);
                self.pages
                    .insert_range(r.unaligned_hi_end, (r.unaligned_hi_group + 1) * PAGES_PER_GROUP);
            } else if RawMask::is_zero(self.pages.data[r.unaligned_hi_group]) {
                self.groups_partial.remove_one(r.unaligned_hi_group);
            }
        }
    }

    #[inline]
    pub fn contains(&self, (min, max): (u32, u32)) -> bool {
        let min = cast(min).to_usize();
        let max = cast(max).to_usize();
        self.contains_exclusive((min, max + 1))
    }

    fn contains_partial_all(&self, group: usize, mask: PageSetGroup) -> bool {
        self.groups_filled.contains_one(group)
            || (self.groups_partial.contains_one(group) && RawMask::is_equal(self.pages.data[group] & mask, mask))
    }

    fn contains_partial_any(&self, group: usize, mask: PageSetGroup) -> bool {
        self.groups_filled.contains_one(group)
            || (self.groups_partial.contains_one(group) && !RawMask::is_zero(self.pages.data[group] & mask))
    }

    #[allow(dead_code)]
    pub fn contains_one(&self, entry: u32) -> bool {
        // TODO: Add a more efficient implementation.
        self.contains((entry, entry))
    }

    #[inline(never)]
    pub fn contains_exclusive(&self, (start, end): (usize, usize)) -> bool {
        let r = align_range::<PageSetGroup>(start, end);

        if !self.groups_filled.contains_range(r.aligned_start_group, r.aligned_end_group) {
            return false;
        }

        if r.has_unaligned_lo() && !self.contains_partial_all(r.unaligned_lo_group, r.unaligned_mask_lo()) {
            return false;
        }

        if r.has_unaligned_hi() && !self.contains_partial_all(r.unaligned_hi_group, r.unaligned_mask_hi()) {
            return false;
        }

        true
    }

    #[inline(always)]
    pub fn is_whole_region_empty(&self, (min, max): (u32, u32)) -> bool {
        let min = cast(min).to_usize();
        let max = cast(max).to_usize();
        self.is_whole_region_empty_exclusive((min, max + 1))
    }

    #[inline(never)]
    pub fn is_whole_region_empty_exclusive(&self, (start, end): (usize, usize)) -> bool {
        let r = align_range::<PageSetGroup>(start, end);

        if r.aligned_start != r.aligned_end && self.groups_filled.contains_range(r.aligned_start_group, r.aligned_end_group) {
            return false;
        }

        if r.has_unaligned_lo() && self.contains_partial_any(r.unaligned_lo_group, r.unaligned_mask_lo()) {
            return false;
        }

        if r.has_unaligned_hi() && self.contains_partial_any(r.unaligned_hi_group, r.unaligned_mask_hi()) {
            return false;
        }

        true
    }

    pub fn clear(&mut self) {
        self.groups_filled.clear();
        self.groups_partial.clear();
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<(u32, u32)> {
        // This is horribly inefficient, but it's only for tests so that's fine.

        let mut all = Vec::new();
        if let Some(start) = self.groups_filled.first_non_zero() {
            for group_index in start..=self.groups_filled.last_non_zero().unwrap() {
                if self.groups_filled.contains_one(group_index) {
                    for page_index in group_index * PageSetGroup::alignment()..(group_index + 1) * PageSetGroup::alignment() {
                        let page_index = cast(page_index).assert_always_fits_in_u32();
                        all.push(page_index);
                    }
                }
            }
        }

        if let Some(start) = self.groups_partial.first_non_zero() {
            for group_index in start..=self.groups_partial.last_non_zero().unwrap() {
                if self.groups_partial.contains_one(group_index) {
                    for page_index in group_index * PageSetGroup::alignment()..(group_index + 1) * PageSetGroup::alignment() {
                        if self.pages.contains_one(page_index) {
                            let page_index = cast(page_index).assert_always_fits_in_u32();
                            all.push(page_index);
                        }
                    }
                }
            }
        }

        all.sort_unstable();
        all.dedup();

        if all.is_empty() {
            return Vec::new();
        }

        let mut first = all[0];
        let mut last = all[0];
        let mut out = Vec::new();
        for index in all.into_iter().skip(1) {
            if last + 1 != index {
                out.push((first, last));
                first = index;
            }

            last = index;
        }

        out.push((first, last));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::PageSet;
    use alloc::vec;

    #[test]
    fn test_page_set_basic() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((1, 5));
        assert!(set.contains((1, 5)));
        assert!(set.contains((1, 1)));
        assert!(set.contains((5, 5)));
        assert!(set.contains((2, 4)));
        assert!(!set.contains((0, 1)));
        assert!(!set.contains((0, 2)));
        assert!(!set.contains((4, 6)));
        assert!(!set.contains((5, 6)));

        assert!(set.is_whole_region_empty((0, 0)));
        assert!(!set.is_whole_region_empty((0, 1)));
        assert!(!set.is_whole_region_empty((1, 1)));
        assert!(!set.is_whole_region_empty((1, 5)));
        assert!(!set.is_whole_region_empty((5, 5)));
        assert!(!set.is_whole_region_empty((5, 6)));
        assert!(set.is_whole_region_empty((6, 6)));

        {
            // Insert duplicate.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((3, 6));
            assert_eq!(set.to_vec(), vec![(3, 6)]);
        }

        {
            // Insert into middle, no-op.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((4, 5));
            assert_eq!(set.to_vec(), vec![(3, 6)]);
        }

        {
            // Insert bigger on both sides.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((2, 7));
            assert_eq!(set.to_vec(), vec![(2, 7)]);
        }

        {
            // Insert adjacent on the left.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((2, 2));
            assert_eq!(set.to_vec(), vec![(2, 6)]);
        }

        {
            // Insert adjacent on the left, 1 overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((2, 3));
            assert_eq!(set.to_vec(), vec![(2, 6)]);
        }

        {
            // Insert adjacent on the left, 2 overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((2, 4));
            assert_eq!(set.to_vec(), vec![(2, 6)]);
        }

        {
            // Insert adjacent on the left, whole overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((2, 6));
            assert_eq!(set.to_vec(), vec![(2, 6)]);
        }

        {
            // Insert adjacent on the right.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((7, 7));
            assert_eq!(set.to_vec(), vec![(3, 7)]);
        }

        {
            // Insert adjacent on the right, one overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((6, 7));
            assert_eq!(set.to_vec(), vec![(3, 7)]);
        }

        {
            // Insert adjacent on the right, two overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((5, 7));
            assert_eq!(set.to_vec(), vec![(3, 7)]);
        }

        {
            // Insert adjacent on the right, whole overlap.
            let mut set = PageSet::new();
            set.insert((3, 6));
            set.insert((3, 7));
            assert_eq!(set.to_vec(), vec![(3, 7)]);
        }

        {
            // Insert disjoint on the right.
            let mut set = PageSet::new();
            set.insert((1, 3));
            set.insert((5, 7));
            assert_eq!(set.to_vec(), vec![(1, 3), (5, 7)]);
        }

        {
            // Insert disjoint on the left.
            let mut set = PageSet::new();
            set.insert((5, 7));
            set.insert((1, 3));
            assert_eq!(set.to_vec(), vec![(1, 3), (5, 7)]);
        }

        {
            // Join disjoint in the middle.
            let mut set = PageSet::new();
            set.insert((0, 2));
            set.insert((6, 8));
            set.insert((4, 4));
            assert_eq!(set.to_vec(), vec![(0, 2), (4, 4), (6, 8)]);
            assert!(set.is_whole_region_empty((3, 3)));
            assert!(set.is_whole_region_empty((5, 5)));
            assert!(set.is_whole_region_empty((9, 9)));
            assert!(!set.is_whole_region_empty((3, 4)));
            assert!(!set.is_whole_region_empty((3, 5)));
            assert!(!set.is_whole_region_empty((3, 5)));
            assert!(!set.is_whole_region_empty((3, 6)));
            assert!(!set.is_whole_region_empty((3, 7)));
            assert!(!set.is_whole_region_empty((3, 8)));
            assert!(!set.is_whole_region_empty((3, 9)));
        }

        {
            // Join in the middle, merge all.
            let mut set = PageSet::new();
            set.insert((0, 2));
            set.insert((6, 8));
            set.insert((3, 5));
            assert_eq!(set.to_vec(), vec![(0, 8)]);
        }

        {
            // Join in the middle, merge all, one overlap.
            let mut set = PageSet::new();
            set.insert((0, 2));
            set.insert((6, 8));
            set.insert((2, 6));
            assert_eq!(set.to_vec(), vec![(0, 8)]);
        }

        {
            // Join in the middle, merge all, two overlap.
            let mut set = PageSet::new();
            set.insert((0, 2));
            set.insert((6, 8));
            set.insert((1, 7));
            assert_eq!(set.to_vec(), vec![(0, 8)]);
        }

        {
            // Join in the middle, merge all, whole overlap.
            let mut set = PageSet::new();
            set.insert((0, 2));
            set.insert((6, 8));
            set.insert((0, 8));
            assert_eq!(set.to_vec(), vec![(0, 8)]);
        }

        {
            // Join in the middle, merge all, extend.
            let mut set = PageSet::new();
            set.insert((1, 3));
            set.insert((5, 7));
            set.insert((0, 8));
            assert_eq!(set.to_vec(), vec![(0, 8)]);
        }

        {
            let mut set = PageSet::new();
            set.insert((0, 100));
            assert_eq!(set.to_vec(), vec![(0, 100)]);
            set.insert((120, 130));
            set.insert((140, 140));
            set.insert((150, 150));
            set.insert((160, 160));
            set.insert((170, 180));
            set.insert((200, 300));

            {
                let mut set = set.clone();
                set.insert((100, 200));
                assert_eq!(set.to_vec(), vec![(0, 300)]);
            }

            {
                let mut set = set.clone();
                set.insert((101, 199));
                assert_eq!(set.to_vec(), vec![(0, 300)]);
            }

            {
                let mut set = set.clone();
                set.insert((102, 198));
                assert_eq!(set.to_vec(), vec![(0, 100), (102, 198), (200, 300)]);
            }
        }
    }

    #[test]
    fn test_page_set_remove() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((20, 30));

        // Remove nonexisting on the left.
        set.remove((10, 19));
        assert_eq!(set.to_vec(), vec![(20, 30)]);

        // Remove nonexisting on the right.
        set.remove((31, 40));
        assert_eq!(set.to_vec(), vec![(20, 30)]);

        {
            let mut set = set.clone();
            set.remove((10, 20));
            assert_eq!(set.to_vec(), vec![(21, 30)]);
        }

        {
            let mut set = set.clone();
            set.remove((10, 21));
            assert_eq!(set.to_vec(), vec![(22, 30)]);
        }

        {
            let mut set = set.clone();
            set.remove((10, 29));
            assert_eq!(set.to_vec(), vec![(30, 30)]);
        }

        {
            let mut set = set.clone();
            set.remove((10, 30));
            assert_eq!(set.to_vec(), vec![]);
        }

        {
            let mut set = set.clone();
            set.remove((10, 40));
            assert_eq!(set.to_vec(), vec![]);
        }

        {
            let mut set = set.clone();
            set.remove((30, 40));
            assert_eq!(set.to_vec(), vec![(20, 29)]);
        }

        {
            let mut set = set.clone();
            set.remove((29, 40));
            assert_eq!(set.to_vec(), vec![(20, 28)]);
        }

        {
            let mut set = set.clone();
            set.remove((21, 40));
            assert_eq!(set.to_vec(), vec![(20, 20)]);
        }

        {
            let mut set = set.clone();
            set.remove((20, 40));
            assert_eq!(set.to_vec(), vec![]);
        }

        {
            let mut set = set.clone();
            set.remove((10, 40));
            assert_eq!(set.to_vec(), vec![]);
        }
    }

    #[test]
    fn disjoint_removal() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((55, 221));
        set.remove((117, 131));
        set.remove((65, 131));
        assert_eq!(set.to_vec(), vec![(55, 64), (132, 221)]);
        assert!(!set.contains((85, 88)));
        assert!(set.contains((55, 64)));
        assert!(!set.contains((54, 64)));
        assert!(!set.contains((55, 65)));
        assert!(set.contains((132, 221)));
        assert!(!set.contains((131, 221)));
        assert!(!set.contains((132, 222)));
    }

    #[test]
    fn remove_in_the_middle_1() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((117, 221));
        set.remove((137, 137));
        assert_eq!(set.to_vec(), vec![(117, 136), (138, 221)]);
        assert!(set.contains((181, 181)));
    }

    #[test]
    fn remove_in_the_middle_2() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((65, 221));
        set.remove((85, 147));
        assert_eq!(set.to_vec(), vec![(65, 84), (148, 221)]);
        assert!(!set.contains((131, 131)));
        assert!(set.contains((150, 151)));
    }

    #[test]
    fn insert_low() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((158, 255));
        set.insert((0, 158));
        assert_eq!(set.to_vec(), vec![(0, 255)]);
        assert!(set.contains((0, 255)));
    }

    #[test]
    fn remove_twice() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((255, 255));
        assert_eq!(set.to_vec(), vec![(255, 255)]);
        set.remove((121, 255));
        assert_eq!(set.to_vec(), vec![]);
        set.remove((121, 221));
        assert_eq!(set.to_vec(), vec![]);
        assert!(!set.contains((255, 255)));
    }

    #[test]
    fn insert_remove_insert() {
        let _ = env_logger::try_init();

        let mut set = PageSet::new();
        set.insert((38, 103));
        assert_eq!(set.to_vec(), vec![(38, 103)]);
        set.remove((64, 141));
        assert_eq!(set.to_vec(), vec![(38, 63)]);
        set.insert((85, 121));
        assert_eq!(set.to_vec(), vec![(38, 63), (85, 121)]);
        assert!(!set.contains((65, 85)));
    }
}
