use crate::misc::{EncodeFlags, FixupKind, InstructionT, Label, MAXIMUM_INSTRUCTION_SIZE};
use alloc::vec::Vec;

/// An assembly failure. The first failure is retained until the assembler is cleared.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AssemblerError {
    CodeSizeLimit { limit: usize },
    AllocationFailed,
    InvalidEncoding,
    InvalidFixup,
    FixupOutOfRange,
    TooManyLabels,
}

impl core::fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CodeSizeLimit { limit } => write!(f, "assembled code exceeds the {limit}-byte limit"),
            Self::AllocationFailed => f.write_str("assembler allocation failed"),
            Self::InvalidEncoding => f.write_str("invalid instruction encoding"),
            Self::InvalidFixup => f.write_str("invalid instruction fixup"),
            Self::FixupOutOfRange => f.write_str("instruction fixup is out of range or misaligned"),
            Self::TooManyLabels => f.write_str("assembler label count exceeds the representable range"),
        }
    }
}

#[derive(Copy, Clone)]
struct Fixup {
    target_label: Label,
    instruction_offset: usize,
    instruction_length: u8,
    kind: FixupKind,
}

pub struct Assembler {
    origin: u64,
    code: Vec<u8>,
    labels: Vec<isize>,
    fixups: Vec<Fixup>,
    guaranteed_capacity: usize,
    code_size_limit: usize,
    error: Option<AssemblerError>,
}

#[allow(clippy::derivable_impls)]
impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(transparent)]
pub struct AssembledCode<'a>(&'a mut Assembler);

impl<'a> core::ops::Deref for AssembledCode<'a> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.code
    }
}

impl<'a> From<AssembledCode<'a>> for Vec<u8> {
    fn from(code: AssembledCode<'a>) -> Vec<u8> {
        core::mem::take(&mut code.0.code)
    }
}

impl<'a> Drop for AssembledCode<'a> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

/// # Safety
///
/// `VALUE` must be non-zero, and `Self::Next::VALUE` must be `VALUE - 1` if `Self::Next` is `NonZero`.
pub unsafe trait NonZero {
    const VALUE: usize;
    type Next;
}

pub struct U0;

macro_rules! impl_non_zero {
    ($(($name:ident = $value:expr, $next:ident))*) => {
        $(
            pub struct $name;

            const _: () = {
                assert!($value != 0);
            };

            /// SAFETY: `VALUE` is non-zero.
            unsafe impl NonZero for $name {
                const VALUE: usize = $value;
                type Next = $next;
            }
        )*
    }
}

impl_non_zero! {
    (U1 = 1, U0)
    (U2 = 2, U1)
    (U3 = 3, U2)
    (U4 = 4, U3)
    (U5 = 5, U4)
    (U6 = 6, U5)
}

#[repr(transparent)]
pub struct ReservedAssembler<'a, R>(&'a mut Assembler, core::marker::PhantomData<R>);

impl<'a> ReservedAssembler<'a, U0> {
    #[allow(clippy::unused_self)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn assert_reserved_exactly_as_needed(self) {}
}

impl<'a, R> ReservedAssembler<'a, R> {
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn push<T>(self, instruction: T) -> ReservedAssembler<'a, R::Next>
    where
        R: NonZero,
        T: InstructionT,
    {
        // SAFETY: `R: NonZero` guarantees space unless reservation failed, in which case emission is a no-op.
        unsafe {
            self.0.push_unchecked(instruction);
        }

        ReservedAssembler(self.0, core::marker::PhantomData)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn push_if<T>(self, condition: bool, instruction: T) -> ReservedAssembler<'a, R::Next>
    where
        R: NonZero,
        T: InstructionT,
    {
        if condition {
            // SAFETY: `R: NonZero` guarantees space unless reservation failed, in which case emission is a no-op.
            unsafe {
                self.0.push_unchecked(instruction);
            }
        }

        ReservedAssembler(self.0, core::marker::PhantomData)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn push_none(self) -> ReservedAssembler<'a, R::Next>
    where
        R: NonZero,
    {
        ReservedAssembler(self.0, core::marker::PhantomData)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn get_label_origin_offset(&self, label: Label) -> Option<isize> {
        self.0.get_label_origin_offset(label)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Assembler {
    pub const fn new() -> Self {
        Assembler {
            origin: 0,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            guaranteed_capacity: 0,
            code_size_limit: isize::MAX as usize,
            error: None,
        }
    }

    /// Limit emitted bytes, independently of spare capacity used by instruction encoding.
    /// Lowering the limit below the current length records a sticky error.
    pub fn set_code_size_limit(&mut self, limit: usize) {
        self.code_size_limit = limit.min(isize::MAX as usize);
        self.check_code_size(self.code.len());
    }

    pub fn error(&self) -> Option<&AssemblerError> {
        self.error.as_ref()
    }

    #[inline]
    fn fail(&mut self, error: AssemblerError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    #[inline]
    fn check_code_size(&mut self, size: usize) -> bool {
        if self.error.is_some() {
            return false;
        }
        if size > self.code_size_limit {
            self.fail(AssemblerError::CodeSizeLimit {
                limit: self.code_size_limit,
            });
            return false;
        }
        true
    }

    #[inline]
    fn check_code_growth(&mut self, additional: usize) -> bool {
        match self.code.len().checked_add(additional) {
            Some(size) => self.check_code_size(size),
            None => {
                self.fail(AssemblerError::CodeSizeLimit {
                    limit: self.code_size_limit,
                });
                false
            }
        }
    }

    pub fn origin(&self) -> u64 {
        self.origin
    }

    pub fn set_origin(&mut self, origin: u64) {
        self.origin = origin;
    }

    pub fn current_address(&self) -> u64 {
        self.origin + self.code.len() as u64
    }

    pub fn forward_declare_label(&mut self) -> Label {
        if self.error.is_some() {
            return Label::from_raw(0);
        }
        if self.labels.len() >= u32::MAX as usize {
            self.fail(AssemblerError::TooManyLabels);
            return Label::from_raw(0);
        }
        self.reserve_labels(1);
        if self.error.is_some() {
            return Label::from_raw(0);
        }
        let label = Label::from_raw(self.labels.len() as u32);
        self.labels.push(isize::MAX);
        label
    }

    pub fn create_label(&mut self) -> Label {
        let label = self.forward_declare_label();
        self.define_label(label);
        label
    }

    pub fn define_label(&mut self, label: Label) -> &mut Self {
        if self.error.is_some() {
            return self;
        }
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {}:", self.origin + self.code.len() as u64, label);

        assert_eq!(
            self.labels[label.raw() as usize],
            isize::MAX,
            "tried to redefine an already defined label"
        );
        self.labels[label.raw() as usize] = self.code.len() as isize;
        self
    }

    /// Define all remaining undefined labels to point at `offset`.
    /// This is used to make undefined branch targets jump to a trap handler
    /// instead of leaving them unresolved (which on AArch64 would cause
    /// infinite self-branch loops).
    pub fn define_all_undefined_labels(&mut self, offset: usize) {
        if !self.check_code_size(offset) {
            return;
        }
        for label_offset in self.labels.iter_mut() {
            if *label_offset == isize::MAX {
                *label_offset = offset as isize;
            }
        }
    }

    pub fn push_with_label<T>(&mut self, label: Label, instruction: T) -> &mut Self
    where
        T: InstructionT,
    {
        self.define_label(label);
        self.push(instruction)
    }

    #[inline]
    pub fn get_label_origin_offset(&self, label: Label) -> Option<isize> {
        let offset = *self.labels.get(label.raw() as usize)?;
        if offset == isize::MAX {
            None
        } else {
            Some(offset)
        }
    }

    pub fn get_label_origin_offset_or_panic(&self, label: Label) -> isize {
        self.get_label_origin_offset(label)
            .expect("tried to fetch a label offset for a label that was not defined")
    }

    pub fn set_label_origin_offset(&mut self, label: Label, offset: isize) {
        if self.error.is_some() {
            return;
        }
        self.labels[label.raw() as usize] = offset;
    }

    #[inline(always)]
    fn add_fixup(&mut self, instruction_offset: usize, instruction_length: usize, target_label: Label, kind: FixupKind) {
        if (target_label.raw() as usize) >= self.labels.len()
            || (!kind.is_aarch64() && (kind.offset() == 0 || !matches!(kind.length(), 1 | 4)))
            || kind.offset() as usize + kind.length() as usize > instruction_length
        {
            self.fail(AssemblerError::InvalidFixup);
            return;
        }
        self.reserve_fixups(1);
        if self.error.is_some() {
            return;
        }
        self.fixups.push(Fixup {
            target_label,
            instruction_offset,
            instruction_length: instruction_length as u8,
            kind,
        });
    }

    #[inline(always)]
    pub fn reserve<T>(&mut self) -> ReservedAssembler<T>
    where
        T: NonZero,
    {
        self.reserve_instructions(T::VALUE);
        ReservedAssembler(self, core::marker::PhantomData)
    }

    #[inline(always)]
    fn reserve_instructions(&mut self, count: usize) {
        if self.error.is_some() {
            return;
        }
        // Encoding writes a full InstBuf, even when the instruction is shorter.
        // This is capacity only: the byte limit is checked against actual emission.
        let Some(bytes) = count.checked_mul(MAXIMUM_INSTRUCTION_SIZE) else {
            self.fail(AssemblerError::AllocationFailed);
            return;
        };
        if self.code.try_reserve(bytes).is_err() {
            self.fail(AssemblerError::AllocationFailed);
            return;
        }
        self.guaranteed_capacity = count;
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn push<T>(&mut self, instruction: T) -> &mut Self
    where
        T: InstructionT,
    {
        if self.guaranteed_capacity == 0 {
            self.reserve_instructions(1);
        }

        // SAFETY: Space was reserved, or a sticky allocation error prevents the write.
        unsafe { self.push_unchecked(instruction) }
    }

    // SAFETY: Unless the assembler has failed, the buffer must have space for one full InstBuf.
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn push_unchecked<T>(&mut self, instruction: T) -> &mut Self
    where
        T: InstructionT,
    {
        if self.error.is_some() {
            return self;
        }
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {}", self.origin + self.code.len() as u64, instruction);

        debug_assert!(self.guaranteed_capacity > 0);
        let instruction_offset = self.code.len();

        let bytes = instruction.encode(EncodeFlags::default());
        let bytes_len = bytes.len();
        if !bytes.is_valid() {
            self.fail(AssemblerError::InvalidEncoding);
            return self;
        }
        if !self.check_code_growth(bytes_len) {
            return self;
        }

        // SAFETY: The caller reserved space for at least one instruction.
        unsafe {
            bytes.encode_into_vec_unsafe(&mut self.code);
        }
        self.guaranteed_capacity -= 1;

        if let Some((label, fixup)) = instruction.fixup(EncodeFlags::default()) {
            self.add_fixup(instruction_offset, bytes_len, label, fixup);
        }

        self
    }

    pub fn push_raw(&mut self, bytes: &[u8]) -> &mut Self {
        if !self.check_code_growth(bytes.len()) {
            return self;
        }
        if self.code.try_reserve(bytes.len()).is_err() {
            self.fail(AssemblerError::AllocationFailed);
            return self;
        }
        self.guaranteed_capacity = 0;
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {:x?}", self.origin + self.code.len() as u64, bytes);
        self.code.extend_from_slice(bytes);
        self
    }

    pub fn finalize(&mut self) -> Result<AssembledCode<'_>, AssemblerError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        for index in 0..self.fixups.len() {
            if let Err(error) = self.apply_fixup(self.fixups[index]) {
                self.fail(error);
                return Err(error);
            }
        }
        self.fixups.clear();
        Ok(AssembledCode(self))
    }

    fn apply_fixup(&mut self, fixup: Fixup) -> Result<(), AssemblerError> {
        let target_absolute = *self
            .labels
            .get(fixup.target_label.raw() as usize)
            .ok_or(AssemblerError::InvalidFixup)?;
        let p = fixup.instruction_offset;
        let end = p
            .checked_add(fixup.instruction_length as usize)
            .ok_or(AssemblerError::InvalidFixup)?;
        if end > self.code.len() {
            return Err(AssemblerError::InvalidFixup);
        }
        if target_absolute == isize::MAX {
            log::trace!("Undefined label found: {}", fixup.target_label);
            return Ok(());
        }

        if fixup.kind.is_aarch64() {
            let existing = u32::from_le_bytes(self.code[p..p + 4].try_into().unwrap());
            if Self::is_aarch64_adrp(existing) {
                if fixup.instruction_length != 8 {
                    return Err(AssemblerError::InvalidFixup);
                }
                let add = u32::from_le_bytes(self.code[p + 4..p + 8].try_into().unwrap());
                let (adrp, add) = Self::patch_aarch64_adrp_pair(existing, add, p, target_absolute)?;
                self.code[p..p + 4].copy_from_slice(&adrp.to_le_bytes());
                self.code[p + 4..p + 8].copy_from_slice(&add.to_le_bytes());
            } else {
                let offset = target_absolute.checked_sub(p as isize).ok_or(AssemblerError::FixupOutOfRange)?;
                let patched = Self::patch_aarch64_branch(existing, offset)?;
                self.code[p..p + 4].copy_from_slice(&patched.to_le_bytes());
            }
        } else {
            let offset = target_absolute.checked_sub(end as isize).ok_or(AssemblerError::FixupOutOfRange)?;
            let fixup_offset = fixup.kind.offset() as usize;
            let p = p + fixup_offset;
            match fixup.kind.length() {
                1 => self.code[p] = i8::try_from(offset).map_err(|_| AssemblerError::FixupOutOfRange)? as u8,
                4 => {
                    self.code[p..p + 4].copy_from_slice(&i32::try_from(offset).map_err(|_| AssemblerError::FixupOutOfRange)?.to_le_bytes())
                }
                _ => return Err(AssemblerError::InvalidFixup),
            }
            let opcode = fixup.kind.0.to_le_bytes();
            self.code[fixup.instruction_offset..p].copy_from_slice(&opcode[..fixup_offset]);
        }
        Ok(())
    }

    #[inline]
    fn is_aarch64_adrp(instruction: u32) -> bool {
        instruction & 0x9f000000 == 0x90000000
    }

    fn patch_aarch64_adrp_pair(adrp: u32, add: u32, instruction_offset: usize, target_offset: isize) -> Result<(u32, u32), AssemblerError> {
        if add & 0xffc00000 != 0x91000000 {
            return Err(AssemblerError::InvalidFixup);
        }
        let instruction_page = (instruction_offset as isize) >> 12;
        let target_page = target_offset >> 12;
        let page_offset = target_page - instruction_page;
        if !(-(1 << 20)..(1 << 20)).contains(&page_offset) {
            return Err(AssemblerError::FixupOutOfRange);
        }
        let encoded_page_offset = page_offset as u32;
        let immlo = (encoded_page_offset & 0x3) << 29;
        let immhi = ((encoded_page_offset >> 2) & 0x7ffff) << 5;
        let patched_adrp = (adrp & !(0x3 << 29) & !(0x7ffff << 5)) | immlo | immhi;
        let page_byte_offset = (target_offset as u32) & 0xfff;
        let patched_add = (add & !(0xfff << 10)) | (page_byte_offset << 10);
        Ok((patched_adrp, patched_add))
    }

    /// Patch an AArch64 instruction word with a PC-relative byte offset.
    /// Detects the instruction type from its encoding and places the offset
    /// in the correct bit fields.
    fn patch_aarch64_branch(instruction: u32, byte_offset: isize) -> Result<u32, AssemblerError> {
        let op0 = (instruction >> 24) & 0xff;
        // ADR addresses bytes; unlike branches its displacement need not be aligned.
        if (op0 & 0x9f) == 0b00010000 {
            if !(-(1 << 20)..(1 << 20)).contains(&byte_offset) {
                return Err(AssemblerError::FixupOutOfRange);
            }
            let offset = byte_offset as u32;
            return Ok((instruction & !(0x3 << 29) & !(0x7ffff << 5)) | ((offset & 0x3) << 29) | (((offset >> 2) & 0x7ffff) << 5));
        }
        if byte_offset & 3 != 0 {
            return Err(AssemblerError::FixupOutOfRange);
        }
        let offset = byte_offset >> 2;
        let (bits, shift) = match op0 {
            x if (x >> 2) == 0b000101 || (x >> 2) == 0b100101 => (26, 0),
            0b01010100 => (19, 5),
            x if (x & 0x7e) == 0b0110100 => (19, 5),
            _ => return Err(AssemblerError::InvalidFixup),
        };
        if !(-(1isize << (bits - 1))..(1isize << (bits - 1))).contains(&offset) {
            return Err(AssemblerError::FixupOutOfRange);
        }
        let mask = (1u32 << bits) - 1;
        Ok((instruction & !(mask << shift)) | (((offset as u32) & mask) << shift))
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn code_mut(&mut self) -> &mut [u8] {
        &mut self.code
    }

    pub fn truncate(&mut self, length: usize) {
        self.code.truncate(length);
        while let Some(fixup) = self.fixups.last() {
            if fixup.instruction_offset >= length {
                self.fixups.pop();
            } else {
                break;
            }
        }
    }

    pub fn spare_capacity(&self) -> usize {
        self.code.capacity() - self.code.len()
    }

    pub fn resize(&mut self, size: usize, fill_with: u8) {
        if !self.check_code_size(size) {
            return;
        }
        if size <= self.code.len() {
            self.truncate(size);
            return;
        }
        if self.code.try_reserve(size - self.code.len()).is_err() {
            self.fail(AssemblerError::AllocationFailed);
            return;
        }
        self.guaranteed_capacity = 0;
        self.code.resize(size, fill_with);
    }

    pub fn reserve_code(&mut self, length: usize) {
        if self.check_code_growth(length) && self.code.try_reserve(length).is_err() {
            self.fail(AssemblerError::AllocationFailed);
        }
    }

    pub fn reserve_labels(&mut self, length: usize) {
        if self.error.is_none() && self.labels.try_reserve(length).is_err() {
            self.fail(AssemblerError::AllocationFailed);
        }
    }

    pub fn reserve_fixups(&mut self, length: usize) {
        if self.error.is_none() && self.fixups.try_reserve(length).is_err() {
            self.fail(AssemblerError::AllocationFailed);
        }
    }

    pub fn clear(&mut self) {
        self.origin = 0;
        self.code.clear();
        self.labels.clear();
        self.fixups.clear();
        self.guaranteed_capacity = 0;
        self.error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{aarch64, amd64::inst as amd64};

    #[test]
    fn instruction_limit_is_exact_and_failure_is_sticky() {
        let mut asm = Assembler::new();
        asm.set_code_size_limit(1);
        asm.push(amd64::nop());
        assert_eq!(&*asm.finalize().unwrap(), &[0x90]);

        asm.push(amd64::nop()).push(amd64::nop());
        assert_eq!(asm.len(), 1);
        assert_eq!(asm.error(), Some(&AssemblerError::CodeSizeLimit { limit: 1 }));
        asm.truncate(0);
        asm.set_code_size_limit(100);
        asm.push_raw(&[0xcc]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 1 })));
        assert!(asm.is_empty());

        asm.clear();
        asm.set_code_size_limit(1);
        asm.push(amd64::nop());
        asm.clear();
        asm.push_raw(&[0x90, 0x90]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 1 })));
    }

    #[test]
    fn reserved_emission_checks_actual_bytes() {
        let mut asm = Assembler::new();
        asm.set_code_size_limit(1);
        asm.reserve::<U3>()
            .push_if(false, amd64::nop())
            .push_none()
            .push(amd64::nop())
            .assert_reserved_exactly_as_needed();
        assert_eq!(&*asm.finalize().unwrap(), &[0x90]);

        asm.reserve::<U3>()
            .push(amd64::nop())
            .push(amd64::nop())
            .push(amd64::nop())
            .assert_reserved_exactly_as_needed();
        assert_eq!(asm.code_mut(), &[0x90]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 1 })));
    }

    #[test]
    fn raw_and_resize_enforce_the_limit_before_writing() {
        let mut asm = Assembler::new();
        asm.set_code_size_limit(3);
        asm.push_raw(&[1, 2]);
        asm.resize(3, 3);
        assert_eq!(&*asm.finalize().unwrap(), &[1, 2, 3]);

        asm.push_raw(&[1, 2]);
        asm.push_raw(&[3, 4]);
        assert_eq!(asm.code_mut(), &[1, 2]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 3 })));

        asm.clear();
        asm.push_raw(&[1, 2]);
        asm.resize(4, 3);
        assert_eq!(asm.code_mut(), &[1, 2]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 3 })));
        asm.clear();
        asm.push_raw(&[1, 2]);
        asm.set_code_size_limit(1);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 1 })));
    }

    #[test]
    fn capacity_is_not_reused_after_raw_growth_or_code_extraction() {
        let mut asm = Assembler::new();
        asm.reserve::<U6>();
        asm.push_raw(&[0xcc; 96]);
        asm.push(amd64::nop());
        let code: Vec<u8> = asm.finalize().unwrap().into();
        assert_eq!(&code[..96], &[0xcc; 96]);
        assert_eq!(code[96], 0x90);

        asm.reserve::<U6>().push(amd64::nop());
        let code: Vec<u8> = asm.finalize().unwrap().into();
        assert_eq!(code, [0x90]);
        asm.push(amd64::nop());
        assert_eq!(&*asm.finalize().unwrap(), &[0x90]);

        asm.reserve::<U6>();
        asm.resize(96, 0xcc);
        asm.push(amd64::nop());
        assert_eq!(&asm.finalize().unwrap()[95..], &[0xcc, 0x90]);
    }

    #[test]
    fn reservations_fail_without_emission_or_panics() {
        let mut asm = Assembler::new();
        asm.set_code_size_limit(1);
        asm.reserve_code(1);
        asm.push(amd64::nop());
        assert_eq!(&*asm.finalize().unwrap(), &[0x90]);
        asm.reserve_code(2);
        assert!(matches!(asm.finalize(), Err(AssemblerError::CodeSizeLimit { limit: 1 })));
        assert!(asm.is_empty());

        asm.clear();
        asm.reserve_labels(usize::MAX);
        assert!(matches!(asm.finalize(), Err(AssemblerError::AllocationFailed)));
        asm.clear();
        asm.reserve_fixups(usize::MAX);
        assert!(matches!(asm.finalize(), Err(AssemblerError::AllocationFailed)));

        struct HugeReservation;
        // SAFETY: The value is nonzero and Next is not NonZero.
        unsafe impl NonZero for HugeReservation {
            const VALUE: usize = usize::MAX;
            type Next = U0;
        }
        asm.clear();
        asm.reserve::<HugeReservation>().push(amd64::nop());
        assert!(matches!(asm.finalize(), Err(AssemblerError::AllocationFailed)));
        assert!(asm.is_empty());
    }

    #[test]
    fn rel8_fixups_reject_both_out_of_range_directions() {
        for offset in [-129isize, -128, 127, 128, isize::MIN] {
            let mut asm = Assembler::new();
            let label = asm.forward_declare_label();
            asm.push(amd64::jmp_label8(label));
            let target = offset.checked_add(2).unwrap();
            asm.set_label_origin_offset(label, target);
            if (-128..=127).contains(&offset) {
                assert_eq!(&*asm.finalize().unwrap(), &[0xeb, offset as u8]);
            } else {
                assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
                asm.set_label_origin_offset(label, 0);
                assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
            }
        }
    }

    #[test]
    fn rel32_fixups_reject_both_out_of_range_directions() {
        for offset in [i32::MIN as i64 - 1, i32::MIN as i64, i32::MAX as i64, i32::MAX as i64 + 1] {
            let Ok(target) = isize::try_from(offset + 5) else {
                continue;
            };
            let mut asm = Assembler::new();
            let label = asm.forward_declare_label();
            asm.push(amd64::jmp_label32(label));
            asm.set_label_origin_offset(label, target);
            if let Ok(offset) = i32::try_from(offset) {
                let code = asm.finalize().unwrap();
                assert_eq!(code[0], 0xe9);
                assert_eq!(&code[1..], &offset.to_le_bytes());
            } else {
                assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
            }
        }
    }

    #[test]
    fn aarch64_branch_fixup_ranges_and_alignment() {
        use aarch64::*;
        fn check(make: fn(Label) -> crate::Instruction<AArch64Inst>, bits: u32) {
            let bound = 1isize << (bits + 1);
            for offset in [-bound - 4, -bound, bound - 4, bound, 2, isize::MIN] {
                let mut asm = Assembler::new();
                let label = asm.forward_declare_label();
                asm.push(make(label));
                asm.set_label_origin_offset(label, offset);
                if offset >= -bound && offset < bound && offset & 3 == 0 {
                    let code = asm.finalize().unwrap();
                    let word = u32::from_le_bytes(code[..].try_into().unwrap());
                    let shift = if bits == 26 { 0 } else { 5 };
                    let decoded = (((word >> shift) << (32 - bits)) as i32) >> (32 - bits);
                    assert_eq!((decoded as isize) << 2, offset);
                } else {
                    assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
                }
            }
        }
        check(b_label, 26);
        check(bl_label, 26);
        check(|label| b_cond_label(Condition::EQ, label), 19);
        check(|label| cbz_label(RegSize::X64, x0, label), 19);
        check(|label| cbnz_label(RegSize::W32, x0, label), 19);
    }

    #[test]
    fn aarch64_adr_allows_byte_addresses_but_not_out_of_range_targets() {
        for offset in [-(1isize << 20) - 1, -(1 << 20), 3, (1 << 20) - 1, 1 << 20] {
            let mut asm = Assembler::new();
            let label = asm.forward_declare_label();
            asm.push(aarch64::adr_label(aarch64::x0, label));
            asm.set_label_origin_offset(label, offset);
            if (-(1 << 20)..(1 << 20)).contains(&offset) {
                let code = asm.finalize().unwrap();
                let word = u32::from_le_bytes(code[..].try_into().unwrap());
                let immediate = ((word >> 29) & 3) | (((word >> 5) & 0x7ffff) << 2);
                assert_eq!(((immediate << 11) as i32 >> 11) as isize, offset);
            } else {
                assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
            }
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn aarch64_adrp_fixups_use_signed_pages_and_low_bytes() {
        for target in [-(1isize << 32) - 1, -(1 << 32), 0x1234, (1 << 32) - 1, 1 << 32, isize::MIN] {
            let mut asm = Assembler::new();
            let label = asm.forward_declare_label();
            asm.push(aarch64::adrp_add_label(aarch64::x0, label));
            asm.set_label_origin_offset(label, target);
            if (-(1 << 32)..(1 << 32)).contains(&target) {
                let code = asm.finalize().unwrap();
                let adrp = u32::from_le_bytes(code[..4].try_into().unwrap());
                let add = u32::from_le_bytes(code[4..].try_into().unwrap());
                let immediate = ((adrp >> 29) & 3) | (((adrp >> 5) & 0x7ffff) << 2);
                let page = ((immediate << 11) as i32 >> 11) as isize;
                assert_eq!((page << 12) + ((add >> 10) & 0xfff) as isize, target);
            } else {
                assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
            }
        }
    }

    #[test]
    fn truncated_or_invalid_fixup_encodings_cannot_finalize() {
        let mut asm = Assembler::new();
        let label = asm.create_label();
        asm.push(amd64::jmp_label32(label));
        asm.resize(2, 0);
        assert!(matches!(asm.finalize(), Err(AssemblerError::InvalidFixup)));
        asm.clear();
        let label = asm.create_label();
        asm.push(aarch64::b_label(label));
        asm.code_mut().copy_from_slice(&[0; 4]);
        assert!(matches!(asm.finalize(), Err(AssemblerError::InvalidFixup)));
    }

    #[test]
    fn invalid_instruction_length_is_rejected_before_the_unsafe_write() {
        let mut instruction = aarch64::nop();
        for _ in 0..13 {
            instruction.bytes.append(0);
        }
        let mut asm = Assembler::new();
        asm.push(instruction);
        assert!(asm.is_empty());
        assert!(matches!(asm.finalize(), Err(AssemblerError::InvalidEncoding)));
    }

    #[test]
    fn fixup_address_arithmetic_cannot_wrap() {
        let mut asm = Assembler::new();
        let label = asm.forward_declare_label();
        asm.push(amd64::nop()).push(amd64::jmp_label32(label));
        asm.set_label_origin_offset(label, isize::MIN);
        assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));

        asm.clear();
        let label = asm.forward_declare_label();
        asm.push(aarch64::nop()).push(aarch64::b_label(label));
        asm.set_label_origin_offset(label, isize::MIN);
        assert!(matches!(asm.finalize(), Err(AssemblerError::FixupOutOfRange)));
    }
}
