//! The vector register file and the operations that run on it.
//!
//! The interpreter and the recompiler both execute the wide and vector instructions, and
//! they must agree bit for bit. Everything that reads or writes the register file therefore
//! lives here, once: the interpreter calls these methods directly, and the recompiler calls
//! them through a native helper that receives a [`WideOperation`] packed at translation
//! time. Memory is the one thing that stays outside, because each executor has its own way
//! of reaching guest memory; a memory operation is answered with a [`UnitStrideCopy`]
//! describing the bytes to move.

use crate::cast::cast;
use crate::program::{Reg, VecReg, WideReg, VECTOR_LENGTH_WORDS};
use crate::vector::{VectorArithmetic, VectorConfig, VectorOperand, VectorOperation};
use crate::wide::U256;

/// How many 64-bit words one vector register holds.
pub const VECTOR_WORDS_PER_REGISTER: usize = VECTOR_LENGTH_WORDS;

/// How many 64-bit words one wide register holds, which is a pair of vector registers.
pub const WIDE_WORDS_PER_REGISTER: usize = VECTOR_WORDS_PER_REGISTER * 2;

/// How many bytes one vector register holds.
pub const VECTOR_BYTES_PER_REGISTER: usize = VECTOR_WORDS_PER_REGISTER * 8;

/// How many bytes one wide register holds.
pub const WIDE_BYTES_PER_REGISTER: usize = WIDE_WORDS_PER_REGISTER * 8;

/// How many 64-bit words the whole register file holds.
pub const VECTOR_FILE_WORDS: usize = VecReg::ALL.len() * VECTOR_WORDS_PER_REGISTER;

/// The value an element holds once truncated to its width.
pub const fn element_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits).wrapping_sub(1)
    }
}

/// The element sign extended from its width to the full word.
pub const fn sign_extend_element(value: u64, bits: u32) -> i64 {
    if bits >= 64 {
        return value as i64;
    }

    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

const fn truncate_u128(value: u128) -> u64 {
    value as u64
}

const fn truncate_i128(value: i128) -> u64 {
    value as u64
}

/// The vector register file, its configuration, and every operation on them.
///
/// The layout is one flat array of words: a vector register is two of them, and a wide
/// register is the four words of the vector register pair it names, so the two files are
/// one. The layout is fixed so that the recompiler can address the words from generated
/// code.
#[repr(C)]
pub struct VectorState {
    words: [u64; VECTOR_FILE_WORDS],
    config: VectorConfig,
}

impl Default for VectorState {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorState {
    /// A fresh file, all zeroes, with the empty configuration.
    pub const fn new() -> Self {
        Self {
            words: [0; VECTOR_FILE_WORDS],
            config: VectorConfig::new(0, 0),
        }
    }

    /// The configuration the next vector instruction runs under.
    pub fn config(&self) -> VectorConfig {
        self.config
    }

    /// Applies a configuration, capping the element count at what the configuration holds.
    ///
    /// The configuration instructions on real hardware cannot produce a count above `VLMAX`,
    /// and the linker refuses a static configuration that carries one, but an instruction in
    /// a handcrafted blob still can: its immediate holds the count in sixteen bits. Without
    /// the cap, every element loop would run tens of thousands of clamped iterations against
    /// the flat price of one instruction.
    pub fn set_config(&mut self, config: VectorConfig) {
        let capped = core::cmp::min(config.vl(), config.max_element_count());
        self.config = VectorConfig::new(config.vtype(), capped);
    }

    /// Applies a dynamic configuration request and returns the element count settled on.
    ///
    /// The requested length is capped by what the configuration holds, and the length that
    /// was settled on is what the configuration instruction writes back.
    pub fn configure_dynamic(&mut self, vtype: u32, requested: u64) -> u64 {
        let length = core::cmp::min(requested, u64::from(VectorConfig::new(vtype, 0).max_element_count()));
        self.config = VectorConfig::new(vtype, cast(length).truncate_to_u32());
        length
    }

    pub fn wide_reg(&self, reg: WideReg) -> U256 {
        let base = reg.to_usize() * WIDE_WORDS_PER_REGISTER;
        U256([self.words[base], self.words[base + 1], self.words[base + 2], self.words[base + 3]])
    }

    pub fn set_wide_reg(&mut self, dst: WideReg, value: U256) {
        let base = dst.to_usize() * WIDE_WORDS_PER_REGISTER;
        self.words[base] = value.0[0];
        self.words[base + 1] = value.0[1];
        self.words[base + 2] = value.0[2];
        self.words[base + 3] = value.0[3];
    }

    pub fn vector_reg(&self, reg: VecReg) -> [u64; VECTOR_WORDS_PER_REGISTER] {
        let base = reg.to_usize() * VECTOR_WORDS_PER_REGISTER;
        [self.words[base], self.words[base + 1]]
    }

    pub fn set_vector_reg(&mut self, dst: VecReg, value: [u64; VECTOR_WORDS_PER_REGISTER]) {
        let base = dst.to_usize() * VECTOR_WORDS_PER_REGISTER;
        self.words[base] = value[0];
        self.words[base + 1] = value[1];
    }

    /// One element of a register group, as the current configuration divides it up.
    ///
    /// Elements never straddle a word because their width divides it, so this is a shift
    /// rather than an unaligned read. A group that runs past the end of the file cannot be
    /// produced by a well formed program, and reads zero rather than trapping.
    pub fn vector_element(&self, group: VecReg, index: u32, bits: u32) -> u64 {
        let byte = index * (bits / 8);
        let word = group.to_usize() * VECTOR_WORDS_PER_REGISTER + cast(byte / 8).to_usize();
        let Some(&word) = self.words.get(word) else {
            return 0;
        };

        let value = word >> ((byte % 8) * 8);
        if bits == 64 {
            value
        } else {
            value & ((1 << bits) - 1)
        }
    }

    /// The counterpart of [`Self::vector_element`], which the same bounds apply to.
    pub fn set_vector_element(&mut self, group: VecReg, index: u32, bits: u32, value: u64) {
        let byte = index.wrapping_mul(bits / 8);
        let word = group
            .to_usize()
            .wrapping_mul(VECTOR_WORDS_PER_REGISTER)
            .wrapping_add(cast(byte / 8).to_usize());
        let Some(word) = self.words.get_mut(word) else {
            return;
        };

        let shift = byte.wrapping_rem(8).wrapping_mul(8);
        let mask = element_mask(bits);
        *word = (*word & !(mask << shift)) | ((value & mask) << shift);
    }

    /// One byte of the register file, counted from the start of register zero.
    pub fn vector_byte(&self, index: usize) -> u8 {
        let word = self.words[index.wrapping_div(8)];
        cast(word >> (index.wrapping_rem(8).wrapping_mul(8))).truncate_to_u8()
    }

    pub fn set_vector_byte(&mut self, index: usize, value: u8) {
        let shift = index.wrapping_rem(8).wrapping_mul(8);
        let word = &mut self.words[index.wrapping_div(8)];
        *word = (*word & !(0xff_u64 << shift)) | (u64::from(value) << shift);
    }

    /// Writes one bit per active element, clearing what is above them.
    ///
    /// The specification lets a mask destination leave those at anything, and clearing them
    /// keeps the machine deterministic.
    fn write_mask(&mut self, dst: VecReg, callback: impl Fn(&Self, u32) -> bool) {
        let mut mask = [0; VECTOR_WORDS_PER_REGISTER];
        for index in 0..self.config.vl() {
            if callback(self, index) {
                let index = cast(index).to_usize();
                mask[index.wrapping_div(64)] |= 1_u64 << index.wrapping_rem(64);
            }
        }

        self.set_vector_reg(dst, mask);
    }

    /// An element-wise comparison, whose result is one bit per element rather than a value.
    pub fn compare(&mut self, dst: VecReg, src1: VecReg, src2: VecReg, equal: bool) {
        let bits = self.config.element_bits();
        self.write_mask(dst, |this, index| {
            (this.vector_element(src1, index, bits) == this.vector_element(src2, index, bits)) == equal
        });
    }

    /// A comparison against one value repeated across every element.
    ///
    /// The immediate stands for a value in a register, so it reaches the element width sign
    /// extended and is then compared at that width.
    pub fn compare_immediate(&mut self, dst: VecReg, src: VecReg, immediate: i32, equal: bool) {
        let bits = self.config.element_bits();
        let value = cast(cast(immediate).to_i64_sign_extend()).bitwise_as_u64() & element_mask(bits);
        self.write_mask(dst, |this, index| (this.vector_element(src, index, bits) == value) == equal);
    }

    /// A bitwise operation over the active bits of two mask registers.
    ///
    /// A mask holds one bit per element, so the operation reaches the low `vl` bits of the
    /// register. What is above them is left alone, which the specification permits.
    pub fn mask_operation(&mut self, dst: VecReg, src1: VecReg, src2: VecReg, callback: impl Fn(u64, u64) -> u64) {
        let (first, second, mut result) = (self.vector_reg(src1), self.vector_reg(src2), self.vector_reg(dst));
        let mut counted = 0;
        for index in 0..VECTOR_WORDS_PER_REGISTER {
            let bits = self.config.vl().saturating_sub(counted).min(64);
            let mask = element_mask(bits);
            result[index] = (result[index] & !mask) | (callback(first[index], second[index]) & mask);
            counted = counted.wrapping_add(64);
        }

        self.set_vector_reg(dst, result);
    }

    /// How many of the active bits of a mask register are set.
    ///
    /// Only the bits standing for an active element are counted; what is above `vl` is not
    /// part of the mask. When `masked` is set, only the elements the mask in `v0` selects
    /// take part.
    pub fn count_mask(&self, src: VecReg, masked: bool) -> u64 {
        let selector = if masked {
            self.vector_reg(VecReg::V0)
        } else {
            [u64::MAX; VECTOR_WORDS_PER_REGISTER]
        };
        let length = self.config.vl();
        let value = self.vector_reg(src);
        let mut count = 0;
        let mut counted = 0;
        for index in 0..VECTOR_WORDS_PER_REGISTER {
            let bits = length.saturating_sub(counted).min(64);
            let mask = element_mask(bits);
            count += u64::from((value[index] & selector[index] & mask).count_ones());
            counted = counted.wrapping_add(64);
        }

        count
    }

    /// The index of the first selected element, or minus one when there is none.
    pub fn first_mask(&self, src: VecReg, masked: bool) -> u64 {
        let selector = self.vector_reg(VecReg::V0);
        let value = self.vector_reg(src);
        let mut found = -1_i64;
        for index in 0..cast(self.config.vl()).to_usize() {
            let word = index.wrapping_div(64);
            let bit = index.wrapping_rem(64);
            if masked && selector[word] >> bit & 1 == 0 {
                continue;
            }

            if value[word] >> bit & 1 != 0 {
                let Ok(index) = i64::try_from(index) else {
                    unreachable!("ICE: an element index that does not fit in a register")
                };
                found = index;
                break;
            }
        }

        cast(found).bitwise_as_u64()
    }

    /// Writes the same value to every active element of a register group.
    pub fn splat(&mut self, dst: VecReg, value: u64) {
        let bits = self.config.element_bits();
        for index in 0..self.config.vl() {
            self.set_vector_element(dst, index, bits, value);
        }
    }

    /// Writes a value to the first element, leaving everything else in the group alone.
    pub fn insert(&mut self, dst: VecReg, value: u64) {
        let bits = self.config.element_bits();
        if self.config.vl() > 0 {
            self.set_vector_element(dst, 0, bits, value);
        }
    }

    /// The first element, sign extended to the full register whatever the element width is.
    pub fn extract(&self, src: VecReg) -> u64 {
        let bits = self.config.element_bits();
        cast(sign_extend_element(self.vector_element(src, 0, bits), bits)).bitwise_as_u64()
    }

    /// Writes each active element's own index into it.
    pub fn element_index(&mut self, dst: VecReg) {
        let bits = self.config.element_bits();
        for index in 0..self.config.vl() {
            self.set_vector_element(dst, index, bits, u64::from(index));
        }
    }

    /// The bytes a unit-stride access moves, which is one element per active element.
    ///
    /// The element width is the instruction's rather than the configuration's, so the two
    /// can differ; the count is still `vl`. It is capped at what the register file holds so
    /// that a program whose configuration does not match its instruction cannot reach past
    /// the end of it.
    pub fn unit_stride_length(&self, register: VecReg, element_bytes: u32) -> usize {
        let requested = self.config.vl().saturating_mul(element_bytes);
        let available = (VecReg::ALL.len().wrapping_sub(register.to_usize())).wrapping_mul(VECTOR_BYTES_PER_REGISTER);
        core::cmp::min(cast(requested).to_usize(), available)
    }

    /// Runs one element-wise operation across the active elements.
    ///
    /// Elements are held zero extended, so the signed operations sign extend from the
    /// element width first and the result is truncated back to it on the way out.
    pub fn arithmetic(&mut self, packed: u32, register_value: impl Fn(Reg) -> u64) {
        let instruction = VectorArithmetic::from_packed(packed);
        let (Some(dst), Some(src)) = (VecReg::from_raw(instruction.dst), VecReg::from_raw(instruction.src)) else {
            unreachable!("ICE: a vector register field wider than the file")
        };

        let bits = self.config.element_bits();
        let selector = if instruction.operation == VectorOperation::Merge {
            self.vector_reg(VecReg::V0)
        } else {
            [0; VECTOR_WORDS_PER_REGISTER]
        };
        let operand = match instruction.operand {
            VectorOperand::Vector(reg) => {
                let Some(reg) = VecReg::from_raw(reg) else {
                    unreachable!("ICE: a vector register field wider than the file")
                };
                Err(reg)
            }
            VectorOperand::Register(reg) => {
                let Some(reg) = Reg::from_raw(reg) else {
                    unreachable!("ICE: a general purpose register field wider than the file")
                };
                Ok(register_value(reg))
            }
            VectorOperand::Immediate(value) => Ok(cast(cast(value).to_i64_sign_extend()).bitwise_as_u64()),
        };

        // The slides are the two operations that do not line their elements up, so they
        // read their source at an offset rather than at the index being written.
        if matches!(instruction.operation, VectorOperation::SlideUp | VectorOperation::SlideDown) {
            // The immediate form's offset is a count rather than a value, so the five bits
            // it was encoded in are read unsigned.
            let offset = match operand {
                Ok(value) => value & 0b11111,
                Err(_) => unreachable!("ICE: a slide by a vector register"),
            };
            let offset = cast(offset).truncate_to_u32();
            let length = self.config.vl();
            let up = instruction.operation == VectorOperation::SlideUp;
            for index in 0..length {
                let value = if up {
                    if index < offset {
                        continue;
                    }
                    self.vector_element(src, index.wrapping_sub(offset), bits)
                } else {
                    let source = index.wrapping_add(offset);
                    if source >= self.config.max_element_count() {
                        0
                    } else {
                        self.vector_element(src, source, bits)
                    }
                };

                self.set_vector_element(dst, index, bits, value);
            }

            return;
        }

        for index in 0..self.config.vl() {
            let first = self.vector_element(src, index, bits);
            let second = match operand {
                Ok(value) => value & element_mask(bits),
                Err(reg) => self.vector_element(reg, index, bits),
            };

            let result = match instruction.operation {
                VectorOperation::Add => first.wrapping_add(second),
                VectorOperation::Subtract => first.wrapping_sub(second),
                VectorOperation::SubtractFrom => second.wrapping_sub(first),
                VectorOperation::MinimumUnsigned => core::cmp::min(first, second),
                VectorOperation::MaximumUnsigned => core::cmp::max(first, second),
                VectorOperation::MinimumSigned => {
                    cast(core::cmp::min(sign_extend_element(first, bits), sign_extend_element(second, bits))).bitwise_as_u64()
                }
                VectorOperation::MaximumSigned => {
                    cast(core::cmp::max(sign_extend_element(first, bits), sign_extend_element(second, bits))).bitwise_as_u64()
                }
                VectorOperation::And => first & second,
                VectorOperation::Or => first | second,
                VectorOperation::Xor => first ^ second,
                // Only as many bits of the shift amount as the element width can use.
                VectorOperation::ShiftLeft => first.wrapping_shl(cast(second).truncate_to_u32()),
                VectorOperation::ShiftRight => (first & element_mask(bits)).wrapping_shr(cast(second).truncate_to_u32() % bits),
                VectorOperation::ShiftRightSigned => {
                    cast(sign_extend_element(first, bits).wrapping_shr(cast(second).truncate_to_u32() % bits)).bitwise_as_u64()
                }
                VectorOperation::Multiply => first.wrapping_mul(second),
                VectorOperation::MultiplyHighUnsigned => truncate_u128(u128::from(first).wrapping_mul(u128::from(second)) >> bits),
                VectorOperation::MultiplyHighSigned => {
                    let first = i128::from(sign_extend_element(first, bits));
                    let second = i128::from(sign_extend_element(second, bits));
                    truncate_i128(first.wrapping_mul(second) >> bits)
                }
                // Division by zero has the result the scalar instructions give it.
                VectorOperation::DivideUnsigned => {
                    if second == 0 {
                        u64::MAX
                    } else {
                        first.wrapping_div(second)
                    }
                }
                VectorOperation::RemainderUnsigned => {
                    if second == 0 {
                        first
                    } else {
                        first.wrapping_rem(second)
                    }
                }
                VectorOperation::DivideSigned => {
                    let (first, second) = (sign_extend_element(first, bits), sign_extend_element(second, bits));
                    if second == 0 {
                        u64::MAX
                    } else {
                        cast(first.wrapping_div(second)).bitwise_as_u64()
                    }
                }
                VectorOperation::RemainderSigned => {
                    let (first, second) = (sign_extend_element(first, bits), sign_extend_element(second, bits));
                    if second == 0 {
                        cast(first).bitwise_as_u64()
                    } else {
                        cast(first.wrapping_rem(second)).bitwise_as_u64()
                    }
                }
                // The accumulating forms read the destination as a third operand.
                VectorOperation::MultiplyAdd => self.vector_element(dst, index, bits).wrapping_add(first.wrapping_mul(second)),
                VectorOperation::MultiplySubtract => self.vector_element(dst, index, bits).wrapping_sub(first.wrapping_mul(second)),
                VectorOperation::MultiplyAddToSource => self.vector_element(dst, index, bits).wrapping_mul(second).wrapping_add(first),
                VectorOperation::MultiplySubtractFromSource => {
                    first.wrapping_sub(self.vector_element(dst, index, bits).wrapping_mul(second))
                }
                VectorOperation::SlideUp | VectorOperation::SlideDown => {
                    unreachable!("ICE: a slide reached the element-wise loop")
                }
                VectorOperation::Merge => {
                    let index = cast(index).to_usize();
                    let selected = selector[index.wrapping_div(64)] >> index.wrapping_rem(64) & 1 != 0;
                    if selected {
                        second
                    } else {
                        first
                    }
                }
            };

            self.set_vector_element(dst, index, bits, result);
        }
    }
}

/// The operations a [`WideOperation`] can carry, which are all of the wide and vector
/// instructions.
///
/// A memory access appears here even though the helper cannot reach guest memory: it is
/// answered with the bytes to move rather than moved, and the recompiled code moves them.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum WideOperationKind {
    WideAdd = 0,
    WideSubtract = 1,
    WideMultiply = 2,
    WideAnd = 3,
    WideOr = 4,
    WideXor = 5,
    WideDivideUnsigned = 6,
    WideDivideSigned = 7,
    WideRemainderUnsigned = 8,
    WideRemainderSigned = 9,
    WideExponent = 10,
    WideSignExtendByte = 11,
    WideAddModulo = 12,
    WideMultiplyModulo = 13,
    WideSetEqual = 14,
    WideSetNotEqual = 15,
    WideSetLessThanUnsigned = 16,
    WideSetLessThanSigned = 17,
    WideShiftLeft = 18,
    WideShiftRight = 19,
    WideShiftRightSigned = 20,
    WideShiftLeftImmediate = 21,
    WideShiftRightImmediate = 22,
    WideShiftRightSignedImmediate = 23,
    WideReverseBytes = 24,
    WideToRegister = 25,
    WideFromRegisterUnsigned = 26,
    WideFromRegisterSigned = 27,
    WideCountSetBits = 28,
    WideCountLeadingZeroBits = 29,
    WideCountTrailingZeroBits = 30,
    VectorArithmetic = 31,
    VectorConfig = 32,
    VectorConfigDynamic = 33,
    VectorConfigDynamicDiscard = 34,
    VectorSetEqual = 35,
    VectorSetNotEqual = 36,
    VectorSetEqualImmediate = 37,
    VectorSetNotEqualImmediate = 38,
    VectorMaskAnd = 39,
    VectorMaskAndNot = 40,
    VectorMaskOr = 41,
    VectorMaskXor = 42,
    VectorMaskNand = 43,
    VectorMaskNor = 44,
    VectorMaskOrNot = 45,
    VectorMaskXnor = 46,
    VectorCountMask = 47,
    VectorCountMaskMasked = 48,
    VectorFirstMask = 49,
    VectorFirstMaskMasked = 50,
    VectorExtract = 51,
    VectorSplat = 52,
    VectorSplatImmediate = 53,
    VectorInsert = 54,
    VectorInsertImmediate = 55,
    VectorElementIndex = 56,
    VectorLoadElements = 57,
    VectorStoreElements = 58,
    WideMove = 59,
    WideLoadImmediateUnsigned = 60,
    WideLoadImmediateSigned = 61,
    WideLoad = 62,
    WideStore = 63,
    WideLoadAbsolute = 64,
    VectorMove = 65,
    VectorLoad = 66,
    VectorStore = 67,
}

impl WideOperationKind {
    fn from_u8(value: u8) -> Option<Self> {
        use WideOperationKind::*;
        Some(match value {
            0 => WideAdd,
            1 => WideSubtract,
            2 => WideMultiply,
            3 => WideAnd,
            4 => WideOr,
            5 => WideXor,
            6 => WideDivideUnsigned,
            7 => WideDivideSigned,
            8 => WideRemainderUnsigned,
            9 => WideRemainderSigned,
            10 => WideExponent,
            11 => WideSignExtendByte,
            12 => WideAddModulo,
            13 => WideMultiplyModulo,
            14 => WideSetEqual,
            15 => WideSetNotEqual,
            16 => WideSetLessThanUnsigned,
            17 => WideSetLessThanSigned,
            18 => WideShiftLeft,
            19 => WideShiftRight,
            20 => WideShiftRightSigned,
            21 => WideShiftLeftImmediate,
            22 => WideShiftRightImmediate,
            23 => WideShiftRightSignedImmediate,
            24 => WideReverseBytes,
            25 => WideToRegister,
            26 => WideFromRegisterUnsigned,
            27 => WideFromRegisterSigned,
            28 => WideCountSetBits,
            29 => WideCountLeadingZeroBits,
            30 => WideCountTrailingZeroBits,
            31 => VectorArithmetic,
            32 => VectorConfig,
            33 => VectorConfigDynamic,
            34 => VectorConfigDynamicDiscard,
            35 => VectorSetEqual,
            36 => VectorSetNotEqual,
            37 => VectorSetEqualImmediate,
            38 => VectorSetNotEqualImmediate,
            39 => VectorMaskAnd,
            40 => VectorMaskAndNot,
            41 => VectorMaskOr,
            42 => VectorMaskXor,
            43 => VectorMaskNand,
            44 => VectorMaskNor,
            45 => VectorMaskOrNot,
            46 => VectorMaskXnor,
            47 => VectorCountMask,
            48 => VectorCountMaskMasked,
            49 => VectorFirstMask,
            50 => VectorFirstMaskMasked,
            51 => VectorExtract,
            52 => VectorSplat,
            53 => VectorSplatImmediate,
            54 => VectorInsert,
            55 => VectorInsertImmediate,
            56 => VectorElementIndex,
            57 => VectorLoadElements,
            58 => VectorStoreElements,
            59 => WideMove,
            60 => WideLoadImmediateUnsigned,
            61 => WideLoadImmediateSigned,
            62 => WideLoad,
            63 => WideStore,
            64 => WideLoadAbsolute,
            65 => VectorMove,
            66 => VectorLoad,
            67 => VectorStore,
            _ => return None,
        })
    }
}

/// One wide or vector instruction, packed into the value a recompiled call site carries.
///
/// The interpreter holds an instruction's operands in its handler arguments; recompiled
/// code has nowhere to hold them, so a call site packs them into one immediate and the
/// native helper unpacks them here. The three register fields mean different things per
/// operation; [`VectorState::dispatch`] is the one place that reads them.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct WideOperation {
    pub kind: WideOperationKind,
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub immediate: i32,
}

impl WideOperation {
    pub const fn to_packed(self) -> u64 {
        (self.kind as u64)
            | ((self.a as u64) << 8)
            | ((self.b as u64) << 16)
            | ((self.c as u64) << 24)
            | ((self.immediate as u32 as u64) << 32)
    }

    pub fn from_packed(packed: u64) -> Option<Self> {
        Some(Self {
            kind: WideOperationKind::from_u8(cast(packed).truncate_to_u8())?,
            a: cast(packed >> 8).truncate_to_u8(),
            b: cast(packed >> 16).truncate_to_u8(),
            c: cast(packed >> 24).truncate_to_u8(),
            immediate: cast(cast(packed >> 32).truncate_to_u32()).bitwise_as_i32(),
        })
    }
}

/// A unit-stride access [`VectorState::dispatch`] answers a memory operation with.
///
/// The bytes between the register file and guest memory are contiguous on both sides, so
/// the executor moves them however it reaches guest memory. The length is already capped
/// at what the register file holds, and can be zero.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct UnitStrideCopy {
    /// The byte offset of the first byte inside the register file.
    pub file_offset: usize,
    /// The guest address of the first byte on the memory side.
    pub guest_address: u32,
    /// How many bytes move.
    pub length: usize,
    /// Whether the bytes move from guest memory into the register file.
    pub into_file: bool,
}

/// The address a wide or vector memory access reaches, which is the base register plus a
/// sign extended offset, truncated to the address space.
fn address_from_fields(operation: WideOperation, get_register: &impl Fn(Reg) -> u64) -> u32 {
    let base = get_register(register_from_field(operation.c));
    cast(base.wrapping_add(cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64())).truncate_to_u32()
}

/// Keeps a copy from running past the end of the address space.
///
/// The interpreter answers an access whose range crosses the top of the address space with
/// a trap before it moves anything. The recompiled copy moves bytes one at a time and would
/// wrap into the bottom of the address space instead, so the whole access is redirected to
/// address zero, which is never mapped: the copy faults on its first byte and traps the
/// same way, with nothing moved.
fn guarded_copy(copy: UnitStrideCopy) -> UnitStrideCopy {
    let end = u64::from(copy.guest_address).wrapping_add(copy.length as u64);
    if end > (1 << 32) {
        UnitStrideCopy {
            guest_address: 0,
            length: 1,
            ..copy
        }
    } else {
        copy
    }
}

fn wide_from_field(field: u8) -> WideReg {
    WideReg::ALL[usize::from(field) % WideReg::ALL.len()]
}

fn vector_from_field(field: u8) -> VecReg {
    VecReg::ALL[usize::from(field) % VecReg::ALL.len()]
}

fn register_from_field(field: u8) -> Reg {
    Reg::ALL[usize::from(field) % Reg::ALL.len()]
}

impl VectorState {
    /// Runs one packed operation, reading and writing general purpose registers through the
    /// callbacks.
    ///
    /// Memory operations do not touch memory here; they answer with the copy to perform.
    /// The register fields were packed from decoded instructions, so an out-of-range field
    /// cannot occur through the recompiler; they are clamped rather than trusted anyway.
    pub fn dispatch(
        &mut self,
        operation: WideOperation,
        get_register: impl Fn(Reg) -> u64,
        mut set_register: impl FnMut(Reg, u64),
    ) -> Option<UnitStrideCopy> {
        use WideOperationKind::*;

        let wide3 = |this: &mut Self, callback: fn(U256, U256) -> U256| {
            let value = callback(
                this.wide_reg(wide_from_field(operation.b)),
                this.wide_reg(wide_from_field(operation.c)),
            );
            this.set_wide_reg(wide_from_field(operation.a), value);
        };
        let wide_compare = |this: &mut Self, set_register: &mut dyn FnMut(Reg, u64), callback: fn(U256, U256) -> bool| {
            let value = callback(
                this.wide_reg(wide_from_field(operation.b)),
                this.wide_reg(wide_from_field(operation.c)),
            );
            set_register(register_from_field(operation.a), u64::from(value));
        };
        let wide_shift = |this: &mut Self, amount: u64, callback: fn(U256, u64) -> U256| {
            let value = callback(this.wide_reg(wide_from_field(operation.b)), amount);
            this.set_wide_reg(wide_from_field(operation.a), value);
        };
        // The immediate stands for a general purpose register the caller would have loaded,
        // so it is sign extended to the register width the shift would have read.
        let immediate_amount = cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64();

        match operation.kind {
            WideAdd => wide3(self, U256::wrapping_add),
            WideSubtract => wide3(self, U256::wrapping_sub),
            WideMultiply => wide3(self, U256::wrapping_mul),
            WideAnd => wide3(self, U256::bitand),
            WideOr => wide3(self, U256::bitor),
            WideXor => wide3(self, U256::bitxor),
            WideDivideUnsigned => wide3(self, |a, b| a.div_rem(b).0),
            WideDivideSigned => wide3(self, U256::div_signed),
            WideRemainderUnsigned => wide3(self, |a, b| a.div_rem(b).1),
            WideRemainderSigned => wide3(self, U256::rem_signed),
            WideExponent => wide3(self, U256::exp),
            WideSignExtendByte => wide3(self, U256::sign_extend_byte),
            WideAddModulo | WideMultiplyModulo => {
                let first = self.wide_reg(wide_from_field(operation.b));
                let second = self.wide_reg(wide_from_field(operation.c));
                let modulus = self.wide_reg(wide_from_field(cast(cast(operation.immediate).bitwise_as_u32()).truncate_to_u8()));
                let value = if operation.kind == WideAddModulo {
                    first.add_mod(second, modulus)
                } else {
                    first.mul_mod(second, modulus)
                };
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideSetEqual => wide_compare(self, &mut set_register, |a, b| a == b),
            WideSetNotEqual => wide_compare(self, &mut set_register, |a, b| a != b),
            WideSetLessThanUnsigned => wide_compare(self, &mut set_register, U256::less_than),
            WideSetLessThanSigned => wide_compare(self, &mut set_register, U256::less_than_signed),
            WideShiftLeft => wide_shift(self, get_register(register_from_field(operation.c)), U256::shift_left),
            WideShiftRight => wide_shift(self, get_register(register_from_field(operation.c)), U256::shift_right),
            WideShiftRightSigned => wide_shift(self, get_register(register_from_field(operation.c)), U256::shift_right_signed),
            WideShiftLeftImmediate => wide_shift(self, immediate_amount, U256::shift_left),
            WideShiftRightImmediate => wide_shift(self, immediate_amount, U256::shift_right),
            WideShiftRightSignedImmediate => wide_shift(self, immediate_amount, U256::shift_right_signed),
            WideReverseBytes => {
                let value = self.wide_reg(wide_from_field(operation.b)).swap_bytes();
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideToRegister => {
                let value = self.wide_reg(wide_from_field(operation.b)).low_u64();
                set_register(register_from_field(operation.a), value);
            }
            WideFromRegisterUnsigned => {
                let value = U256::from_u64(get_register(register_from_field(operation.c)));
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideFromRegisterSigned => {
                let value = U256::from_i64(cast(get_register(register_from_field(operation.c))).bitwise_as_i64());
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideCountSetBits => {
                let value = u64::from(self.wide_reg(wide_from_field(operation.b)).count_ones());
                set_register(register_from_field(operation.a), value);
            }
            WideCountLeadingZeroBits => {
                let value = u64::from(self.wide_reg(wide_from_field(operation.b)).leading_zeros());
                set_register(register_from_field(operation.a), value);
            }
            WideCountTrailingZeroBits => {
                let value = u64::from(self.wide_reg(wide_from_field(operation.b)).trailing_zeros());
                set_register(register_from_field(operation.a), value);
            }
            VectorArithmetic => self.arithmetic(cast(operation.immediate).bitwise_as_u32(), get_register),
            VectorConfig => self.set_config(crate::vector::VectorConfig::from_packed(cast(operation.immediate).bitwise_as_u32())),
            VectorConfigDynamic | VectorConfigDynamicDiscard => {
                let requested = get_register(register_from_field(operation.c));
                let length = self.configure_dynamic(cast(operation.immediate).bitwise_as_u32(), requested);
                if operation.kind == VectorConfigDynamic {
                    set_register(register_from_field(operation.a), length);
                }
            }
            VectorSetEqual => self.compare(
                vector_from_field(operation.a),
                vector_from_field(operation.b),
                vector_from_field(operation.c),
                true,
            ),
            VectorSetNotEqual => self.compare(
                vector_from_field(operation.a),
                vector_from_field(operation.b),
                vector_from_field(operation.c),
                false,
            ),
            VectorSetEqualImmediate => self.compare_immediate(
                vector_from_field(operation.a),
                vector_from_field(operation.b),
                operation.immediate,
                true,
            ),
            VectorSetNotEqualImmediate => self.compare_immediate(
                vector_from_field(operation.a),
                vector_from_field(operation.b),
                operation.immediate,
                false,
            ),
            VectorMaskAnd => self.mask_from_fields(operation, |a, b| a & b),
            VectorMaskAndNot => self.mask_from_fields(operation, |a, b| a & !b),
            VectorMaskOr => self.mask_from_fields(operation, |a, b| a | b),
            VectorMaskXor => self.mask_from_fields(operation, |a, b| a ^ b),
            VectorMaskNand => self.mask_from_fields(operation, |a, b| !(a & b)),
            VectorMaskNor => self.mask_from_fields(operation, |a, b| !(a | b)),
            VectorMaskOrNot => self.mask_from_fields(operation, |a, b| a | !b),
            VectorMaskXnor => self.mask_from_fields(operation, |a, b| !(a ^ b)),
            VectorCountMask => {
                let value = self.count_mask(vector_from_field(operation.b), false);
                set_register(register_from_field(operation.a), value);
            }
            VectorCountMaskMasked => {
                let value = self.count_mask(vector_from_field(operation.b), true);
                set_register(register_from_field(operation.a), value);
            }
            VectorFirstMask => {
                let value = self.first_mask(vector_from_field(operation.b), false);
                set_register(register_from_field(operation.a), value);
            }
            VectorFirstMaskMasked => {
                let value = self.first_mask(vector_from_field(operation.b), true);
                set_register(register_from_field(operation.a), value);
            }
            VectorExtract => {
                let value = self.extract(vector_from_field(operation.b));
                set_register(register_from_field(operation.a), value);
            }
            VectorSplat => {
                let value = get_register(register_from_field(operation.c));
                self.splat(vector_from_field(operation.a), value);
            }
            VectorSplatImmediate => {
                // The immediate stands for a value in a register, so it reaches the element
                // width sign extended, as the register form's contents would be.
                let value = cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64();
                self.splat(vector_from_field(operation.a), value);
            }
            VectorInsert => {
                let value = get_register(register_from_field(operation.c));
                self.insert(vector_from_field(operation.a), value);
            }
            VectorInsertImmediate => {
                let value = cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64();
                self.insert(vector_from_field(operation.a), value);
            }
            VectorElementIndex => self.element_index(vector_from_field(operation.a)),
            VectorLoadElements | VectorStoreElements => {
                let register = vector_from_field(operation.a);
                return Some(guarded_copy(UnitStrideCopy {
                    file_offset: register.to_usize().wrapping_mul(VECTOR_BYTES_PER_REGISTER),
                    guest_address: address_from_fields(operation, &get_register),
                    length: self.unit_stride_length(register, u32::from(operation.b)),
                    into_file: operation.kind == VectorLoadElements,
                }));
            }
            WideMove => {
                let value = self.wide_reg(wide_from_field(operation.b));
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            // The immediate stands in for a general purpose register the caller would have
            // loaded it into, so it is widened the same way: sign extended to the register
            // width first, then taken as unsigned or signed.
            WideLoadImmediateUnsigned => {
                let value = U256::from_u64(cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64());
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideLoadImmediateSigned => {
                let value = U256::from_i64(cast(operation.immediate).to_i64_sign_extend());
                self.set_wide_reg(wide_from_field(operation.a), value);
            }
            WideLoad | WideStore | WideLoadAbsolute => {
                let register = wide_from_field(operation.a);
                let guest_address = if operation.kind == WideLoadAbsolute {
                    cast(cast(cast(operation.immediate).to_i64_sign_extend()).bitwise_as_u64()).truncate_to_u32()
                } else {
                    address_from_fields(operation, &get_register)
                };
                return Some(guarded_copy(UnitStrideCopy {
                    file_offset: register.to_usize().wrapping_mul(WIDE_BYTES_PER_REGISTER),
                    guest_address,
                    length: WIDE_BYTES_PER_REGISTER,
                    into_file: operation.kind != WideStore,
                }));
            }
            VectorMove => {
                let value = self.vector_reg(vector_from_field(operation.b));
                self.set_vector_reg(vector_from_field(operation.a), value);
            }
            VectorLoad | VectorStore => {
                let register = vector_from_field(operation.a);
                return Some(guarded_copy(UnitStrideCopy {
                    file_offset: register.to_usize().wrapping_mul(VECTOR_BYTES_PER_REGISTER),
                    guest_address: address_from_fields(operation, &get_register),
                    length: VECTOR_BYTES_PER_REGISTER,
                    into_file: operation.kind == VectorLoad,
                }));
            }
        }

        None
    }

    fn mask_from_fields(&mut self, operation: WideOperation, callback: impl Fn(u64, u64) -> u64) {
        self.mask_operation(
            vector_from_field(operation.a),
            vector_from_field(operation.b),
            vector_from_field(operation.c),
            callback,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_operations_round_trip_through_packing() {
        let operation = WideOperation {
            kind: WideOperationKind::WideShiftLeft,
            a: 3,
            b: 15,
            c: 12,
            immediate: -1,
        };
        assert_eq!(WideOperation::from_packed(operation.to_packed()), Some(operation));

        let operation = WideOperation {
            kind: WideOperationKind::VectorStoreElements,
            a: 31,
            b: 8,
            c: 0,
            immediate: i32::MIN,
        };
        assert_eq!(WideOperation::from_packed(operation.to_packed()), Some(operation));
    }

    #[test]
    fn dispatch_runs_a_wide_addition() {
        let mut state = VectorState::new();
        state.set_wide_reg(WideReg::W1, U256([u64::MAX; 4]));
        state.set_wide_reg(WideReg::W2, U256::from_u64(3));

        let operation = WideOperation {
            kind: WideOperationKind::WideAdd,
            a: 0,
            b: 1,
            c: 2,
            immediate: 0,
        };
        let copy = state.dispatch(operation, |_| unreachable!(), |_, _| unreachable!());
        assert_eq!(copy, None);
        assert_eq!(state.wide_reg(WideReg::W0), U256::from_u64(2));
    }

    #[test]
    fn set_config_caps_the_element_count_at_what_the_configuration_holds() {
        let mut state = VectorState::new();
        state.set_config(VectorConfig::new(0b011_001, u16::MAX.into()));
        assert_eq!(state.config().vl(), 4);

        state.set_config(VectorConfig::new(0b011_001, 3));
        assert_eq!(state.config().vl(), 3);
    }

    #[test]
    fn dispatch_answers_a_unit_stride_store_with_the_copy() {
        let mut state = VectorState::new();
        state.set_config(VectorConfig::new(0b011 << 3, 2));

        let operation = WideOperation {
            kind: WideOperationKind::VectorStoreElements,
            a: 4,
            b: 8,
            c: 2,
            immediate: -16,
        };
        let copy = state.dispatch(operation, |_| 0x1_0000, |_, _| unreachable!());
        assert_eq!(
            copy,
            Some(UnitStrideCopy {
                file_offset: 4 * VECTOR_BYTES_PER_REGISTER,
                guest_address: 0xfff0,
                length: 16,
                into_file: false,
            })
        );
    }
}
