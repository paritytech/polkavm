use crate::misc::{EncodeFlags, FixupKind, InstBuf, InstructionT, Label};
use alloc::vec::Vec;

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
        // SAFETY: `R: NonZero`, so we still have space in the buffer.
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
            // SAFETY: `R: NonZero`, so we still have space in the buffer.
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
        let label = self.labels.len() as u32;
        self.labels.push(isize::MAX);
        Label::from_raw(label)
    }

    pub fn create_label(&mut self) -> Label {
        let label = self.labels.len() as u32;
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {}:", self.origin + self.code.len() as u64, Label::from_raw(label));

        self.labels.push(self.code.len() as isize);
        Label::from_raw(label)
    }

    pub fn define_label(&mut self, label: Label) -> &mut Self {
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
        let offset = self.labels[label.raw() as usize];
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
        self.labels[label.raw() as usize] = offset;
    }

    #[inline(always)]
    fn add_fixup(&mut self, instruction_offset: usize, instruction_length: usize, target_label: Label, kind: FixupKind) {
        debug_assert!((target_label.raw() as usize) < self.labels.len());
        debug_assert!(
            kind.is_aarch64() || (kind.offset() as usize) < instruction_length,
            "instruction is {} bytes long and yet its target fixup starts at {}",
            instruction_length,
            kind.offset()
        );
        debug_assert!(kind.is_aarch64() || (kind.length() as usize) < instruction_length);
        debug_assert!((kind.offset() as usize + kind.length() as usize) <= instruction_length);
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
        InstBuf::reserve(&mut self.code, T::VALUE);

        self.guaranteed_capacity = T::VALUE;
        ReservedAssembler(self, core::marker::PhantomData)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    pub fn push<T>(&mut self, instruction: T) -> &mut Self
    where
        T: InstructionT,
    {
        if self.guaranteed_capacity == 0 {
            InstBuf::reserve_const::<1>(&mut self.code);
            self.guaranteed_capacity = 1;
        }

        // SAFETY: We've reserved space for at least one instruction.
        unsafe { self.push_unchecked(instruction) }
    }

    // SAFETY: The buffer *must* have space for at least one instruction.
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn push_unchecked<T>(&mut self, instruction: T) -> &mut Self
    where
        T: InstructionT,
    {
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {}", self.origin + self.code.len() as u64, instruction);

        debug_assert!(self.guaranteed_capacity > 0);
        let instruction_offset = self.code.len();

        let bytes = instruction.encode(EncodeFlags::default());
        let bytes_len = bytes.len();

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
        #[cfg(debug_assertions)]
        log::trace!("{:08x}: {:x?}", self.origin + self.code.len() as u64, bytes);
        self.code.extend_from_slice(bytes);
        self
    }

    pub fn finalize(&mut self) -> AssembledCode {
        for fixup in self.fixups.drain(..) {
            let target_absolute = self.labels[fixup.target_label.raw() as usize];
            if target_absolute == isize::MAX {
                log::trace!("Undefined label found: {}", fixup.target_label);
                continue;
            }

            if fixup.kind.is_aarch64() {
                // AArch64 fixup: PC-relative offset from the instruction itself.
                let offset = target_absolute - fixup.instruction_offset as isize;
                let p = fixup.instruction_offset;
                let existing = u32::from_le_bytes([self.code[p], self.code[p + 1], self.code[p + 2], self.code[p + 3]]);
                if Self::is_aarch64_adrp(existing) {
                    let add = u32::from_le_bytes([self.code[p + 4], self.code[p + 5], self.code[p + 6], self.code[p + 7]]);
                    let (patched_adrp, patched_add) = Self::patch_aarch64_adrp_pair(existing, add, p, target_absolute);
                    self.code[p..p + 4].copy_from_slice(&patched_adrp.to_le_bytes());
                    self.code[p + 4..p + 8].copy_from_slice(&patched_add.to_le_bytes());
                } else {
                    let patched = Self::patch_aarch64_branch(existing, offset);
                    self.code[p..p + 4].copy_from_slice(&patched.to_le_bytes());
                }
            } else {
                // x86-style fixup: offset is from end of instruction.
                let origin = fixup.instruction_offset + fixup.instruction_length as usize;
                let opcode = (fixup.kind.0 << 8) >> 8;
                let fixup_offset = fixup.kind.offset();
                let fixup_length = fixup.kind.length();

                if fixup_offset >= 1 {
                    self.code[fixup.instruction_offset] = opcode as u8;
                    if fixup_offset >= 2 {
                        self.code[fixup.instruction_offset + 1] = (opcode >> 8) as u8;
                        if fixup_offset >= 3 {
                            self.code[fixup.instruction_offset + 2] = (opcode >> 16) as u8;
                        }
                    }
                }

                let offset = target_absolute - origin as isize;
                let p = fixup.instruction_offset + fixup_offset as usize;
                if fixup_length == 1 {
                    if offset > i8::MAX as isize || offset < i8::MIN as isize {
                        panic!("out of range jump");
                    }
                    self.code[p] = offset as i8 as u8;
                } else if fixup_length == 4 {
                    if offset > i32::MAX as isize || offset < i32::MIN as isize {
                        panic!("out of range jump");
                    }
                    self.code[p..p + 4].copy_from_slice(&(offset as i32).to_le_bytes());
                } else {
                    unreachable!()
                }
            }
        }

        AssembledCode(self)
    }

    #[inline]
    fn is_aarch64_adrp(instruction: u32) -> bool {
        instruction & 0x9f000000 == 0x90000000
    }

    fn patch_aarch64_adrp_pair(adrp: u32, add: u32, instruction_offset: usize, target_offset: isize) -> (u32, u32) {
        debug_assert!(Self::is_aarch64_adrp(adrp));
        debug_assert_eq!(add & 0xff000000, 0x91000000);

        let instruction_page = (instruction_offset as isize) & !0xfff;
        let target_page = target_offset & !0xfff;
        let page_offset = (target_page - instruction_page) >> 12;
        assert!(page_offset >= -(1 << 20) && page_offset < (1 << 20), "ADRP: out of range");

        let encoded_page_offset = page_offset as u32;
        let immlo = (encoded_page_offset & 0x3) << 29;
        let immhi = ((encoded_page_offset >> 2) & 0x7ffff) << 5;
        let patched_adrp = (adrp & !(0x3 << 29) & !(0x7ffff << 5)) | immlo | immhi;

        let page_byte_offset = (target_offset as u32) & 0xfff;
        let patched_add = (add & !(0xfff << 10)) | (page_byte_offset << 10);
        (patched_adrp, patched_add)
    }

    /// Patch an AArch64 instruction word with a PC-relative byte offset.
    /// Detects the instruction type from its encoding and places the offset
    /// in the correct bit fields.
    fn patch_aarch64_branch(instruction: u32, byte_offset: isize) -> u32 {
        debug_assert!(
            byte_offset & 3 == 0,
            "AArch64 branch offset must be 4-byte aligned: {}",
            byte_offset
        );
        let inst_offset = byte_offset >> 2; // Convert to instruction offset

        // Detect instruction type from top bits:
        let op0 = (instruction >> 24) & 0xFF;

        match op0 {
            // B imm26: 000101xx
            x if (x >> 2) == 0b000101 => {
                // Unconditional branch: imm26 in bits [25:0]
                assert!(inst_offset >= -(1 << 25) && inst_offset < (1 << 25), "B: out of range");
                let imm26 = (inst_offset as u32) & 0x03FFFFFF;
                (instruction & !0x03FFFFFF) | imm26
            }
            // BL imm26: 100101xx
            x if (x >> 2) == 0b100101 => {
                assert!(inst_offset >= -(1 << 25) && inst_offset < (1 << 25), "BL: out of range");
                let imm26 = (inst_offset as u32) & 0x03FFFFFF;
                (instruction & !0x03FFFFFF) | imm26
            }
            // B.cond imm19: 01010100
            0b01010100 => {
                assert!(inst_offset >= -(1 << 18) && inst_offset < (1 << 18), "B.cond: out of range");
                let imm19 = (inst_offset as u32) & 0x7FFFF;
                (instruction & !(0x7FFFF << 5)) | (imm19 << 5)
            }
            // CBZ/CBNZ imm19: x0110100 / x0110101
            x if (x & 0x7E) == 0b0110100 => {
                assert!(inst_offset >= -(1 << 18) && inst_offset < (1 << 18), "CBZ/CBNZ: out of range");
                let imm19 = (inst_offset as u32) & 0x7FFFF;
                (instruction & !(0x7FFFF << 5)) | (imm19 << 5)
            }
            // ADR: immlo in bits [30:29], immhi in bits [23:5]
            x if (x & 0x9F) == 0b00010000 => {
                // ADR uses byte offset directly, not instruction offset
                let off = byte_offset;
                assert!(off >= -(1 << 20) && off < (1 << 20), "ADR: out of range");
                let off = off as u32;
                let immlo = (off & 0x3) << 29;
                let immhi = ((off >> 2) & 0x7FFFF) << 5;
                (instruction & !(0x3 << 29) & !(0x7FFFF << 5)) | immlo | immhi
            }
            _ => panic!("unknown AArch64 instruction type for fixup: 0x{:08x}", instruction),
        }
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
        self.code.resize(size, fill_with)
    }

    pub fn reserve_code(&mut self, length: usize) {
        self.code.reserve(length);
    }

    pub fn reserve_labels(&mut self, length: usize) {
        self.labels.reserve(length);
    }

    pub fn reserve_fixups(&mut self, length: usize) {
        self.fixups.reserve(length);
    }

    pub fn clear(&mut self) {
        self.origin = 0;
        self.code.clear();
        self.labels.clear();
        self.fixups.clear();
    }
}
