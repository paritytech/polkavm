//! The state the vector extensions keep alongside the register file.
//!
//! A vector instruction reads its element width and its element count from `vtype` and `vl`
//! rather than from its own encoding, so both have to be modelled. PolkaVM fixes the vector
//! register width, which means the element count a given configuration produces is a
//! constant: a program is translated once and every vector instruction in it already knows
//! how many elements it touches.

use crate::program::VECTOR_LENGTH_BITS;

/// The configuration a vector instruction runs under.
///
/// This is the `vtype` register the vector extensions specify, in the same encoding, paired
/// with the `vl` that was in effect when it was written. Both are set together by the
/// configuration instructions and read by everything else.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Hash)]
#[repr(C)]
pub struct VectorConfig {
    vtype: u32,
    vl: u32,
}

impl VectorConfig {
    /// The `vtype` bit that says the configuration is not supported.
    ///
    /// It is the top bit of the register on real hardware. Here the field is narrow, so it
    /// sits directly above the ones the encoding uses.
    pub const INVALID: u32 = 1 << 8;

    /// Builds a configuration out of the `vtype` encoding and an element count.
    pub const fn new(vtype: u32, vl: u32) -> Self {
        Self { vtype, vl }
    }

    /// Both fields in the single immediate the configuration instruction carries.
    pub const fn to_packed(self) -> u32 {
        (self.vtype & 0xffff) | (self.vl << 16)
    }

    /// The inverse of [`Self::to_packed`].
    pub const fn from_packed(packed: u32) -> Self {
        Self {
            vtype: packed & 0xffff,
            vl: packed >> 16,
        }
    }

    /// The `vtype` encoding.
    pub const fn vtype(self) -> u32 {
        self.vtype
    }

    /// The number of elements the next vector instruction operates on.
    pub const fn vl(self) -> u32 {
        self.vl
    }

    /// The width of one element in bits, from the `vsew` field.
    pub const fn element_bits(self) -> u32 {
        8 << ((self.vtype >> 3) & 0b111)
    }

    /// The number of registers one operand occupies, from the `vlmul` field.
    ///
    /// Fractional settings occupy one register and are reported as such; they differ from
    /// `vlmul = 1` only in how many elements fit, which `vl` already carries.
    pub const fn registers_per_group(self) -> u32 {
        match self.vtype & 0b111 {
            0b000 => 1,
            0b001 => 2,
            0b010 => 4,
            0b011 => 8,
            _ => 1,
        }
    }

    /// Whether the configuration is one this machine implements.
    pub const fn is_valid(self) -> bool {
        if self.vtype & Self::INVALID != 0 {
            return false;
        }

        // `vsew` above 64 bits and the reserved `vlmul` encoding have no meaning here.
        let sew = (self.vtype >> 3) & 0b111;
        let lmul = self.vtype & 0b111;
        sew <= 0b011 && lmul != 0b100 && self.vl <= self.max_element_count()
    }

    /// The largest element count this configuration can hold, the `VLMAX` of the extensions.
    pub const fn max_element_count(self) -> u32 {
        let bits = VECTOR_LENGTH_BITS * self.registers_per_group();
        let bits = match self.vtype & 0b111 {
            0b101 => bits / 8,
            0b110 => bits / 4,
            0b111 => bits / 2,
            _ => bits,
        };
        bits / self.element_bits()
    }
}

/// An element-wise operation on the vector registers.
///
/// The vector extensions spell these out per operand shape, so each appears three times in
/// the instruction encoding: once against another register group, once against a general
/// purpose register and once against an immediate. Only the shape differs, so the operation
/// is named once here and the shape travels with it.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum VectorOperation {
    Add,
    Subtract,
    SubtractFrom,
    MinimumUnsigned,
    MinimumSigned,
    MaximumUnsigned,
    MaximumSigned,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
    ShiftRightSigned,
    Multiply,
    MultiplyHighSigned,
    MultiplyHighUnsigned,
    DivideUnsigned,
    DivideSigned,
    RemainderUnsigned,
    RemainderSigned,
    MultiplyAdd,
    MultiplyAddToSource,
    MultiplySubtract,
    MultiplySubtractFromSource,
    /// Each element taken from one operand or the other, as the mask in `v0` selects.
    Merge,
    /// The elements moved to higher positions, leaving the ones below the offset alone.
    SlideUp,
    /// The elements moved to lower positions, with zeroes shifted in at the top.
    SlideDown,
}

impl VectorOperation {
    /// The operation an instruction's `funct3` and `funct6` fields name, if it is one this
    /// machine implements.
    pub const fn decode(funct3: u32, funct6: u32) -> Option<Self> {
        use VectorOperation::*;
        Some(match (funct3, funct6) {
            (0b000, 0b000000) => Add,
            (0b100, 0b000000) => Add,
            (0b011, 0b000000) => Add,
            (0b000, 0b000010) => Subtract,
            (0b100, 0b000010) => Subtract,
            (0b100, 0b000011) => SubtractFrom,
            (0b011, 0b000011) => SubtractFrom,
            (0b000, 0b000100) => MinimumUnsigned,
            (0b100, 0b000100) => MinimumUnsigned,
            (0b000, 0b000101) => MinimumSigned,
            (0b100, 0b000101) => MinimumSigned,
            (0b000, 0b000110) => MaximumUnsigned,
            (0b100, 0b000110) => MaximumUnsigned,
            (0b000, 0b000111) => MaximumSigned,
            (0b100, 0b000111) => MaximumSigned,
            (0b000, 0b001001) => And,
            (0b100, 0b001001) => And,
            (0b011, 0b001001) => And,
            (0b000, 0b001010) => Or,
            (0b100, 0b001010) => Or,
            (0b011, 0b001010) => Or,
            (0b000, 0b001011) => Xor,
            (0b100, 0b001011) => Xor,
            (0b011, 0b001011) => Xor,
            (0b000, 0b100101) => ShiftLeft,
            (0b100, 0b100101) => ShiftLeft,
            (0b011, 0b100101) => ShiftLeft,
            (0b000, 0b101000) => ShiftRight,
            (0b100, 0b101000) => ShiftRight,
            (0b011, 0b101000) => ShiftRight,
            (0b000, 0b101001) => ShiftRightSigned,
            (0b100, 0b101001) => ShiftRightSigned,
            (0b011, 0b101001) => ShiftRightSigned,
            (0b010, 0b100101) => Multiply,
            (0b110, 0b100101) => Multiply,
            (0b010, 0b100111) => MultiplyHighSigned,
            (0b110, 0b100111) => MultiplyHighSigned,
            (0b010, 0b100100) => MultiplyHighUnsigned,
            (0b110, 0b100100) => MultiplyHighUnsigned,
            (0b010, 0b100000) => DivideUnsigned,
            (0b110, 0b100000) => DivideUnsigned,
            (0b010, 0b100001) => DivideSigned,
            (0b110, 0b100001) => DivideSigned,
            (0b010, 0b100010) => RemainderUnsigned,
            (0b110, 0b100010) => RemainderUnsigned,
            (0b010, 0b100011) => RemainderSigned,
            (0b110, 0b100011) => RemainderSigned,
            (0b010, 0b101101) => MultiplyAdd,
            (0b110, 0b101101) => MultiplyAdd,
            (0b010, 0b101001) => MultiplyAddToSource,
            (0b110, 0b101001) => MultiplyAddToSource,
            (0b010, 0b101111) => MultiplySubtract,
            (0b110, 0b101111) => MultiplySubtract,
            (0b010, 0b101011) => MultiplySubtractFromSource,
            (0b110, 0b101011) => MultiplySubtractFromSource,
            (0b011, 0b001110) => SlideUp,
            (0b100, 0b001110) => SlideUp,
            (0b011, 0b001111) => SlideDown,
            (0b100, 0b001111) => SlideDown,
            _ => return None,
        })
    }

    const fn to_index(self) -> u32 {
        use VectorOperation::*;
        match self {
            Add => 0,
            Subtract => 1,
            SubtractFrom => 2,
            MinimumUnsigned => 3,
            MinimumSigned => 4,
            MaximumUnsigned => 5,
            MaximumSigned => 6,
            And => 7,
            Or => 8,
            Xor => 9,
            ShiftLeft => 10,
            ShiftRight => 11,
            ShiftRightSigned => 12,
            Multiply => 13,
            MultiplyHighSigned => 14,
            MultiplyHighUnsigned => 15,
            DivideUnsigned => 16,
            DivideSigned => 17,
            RemainderUnsigned => 18,
            RemainderSigned => 19,
            MultiplyAdd => 20,
            MultiplyAddToSource => 21,
            MultiplySubtract => 22,
            MultiplySubtractFromSource => 23,
            Merge => 24,
            SlideUp => 25,
            SlideDown => 26,
        }
    }

    const fn from_index(index: u32) -> Option<Self> {
        use VectorOperation::*;
        Some(match index {
            0 => Add,
            1 => Subtract,
            2 => SubtractFrom,
            3 => MinimumUnsigned,
            4 => MinimumSigned,
            5 => MaximumUnsigned,
            6 => MaximumSigned,
            7 => And,
            8 => Or,
            9 => Xor,
            10 => ShiftLeft,
            11 => ShiftRight,
            12 => ShiftRightSigned,
            13 => Multiply,
            14 => MultiplyHighSigned,
            15 => MultiplyHighUnsigned,
            16 => DivideUnsigned,
            17 => DivideSigned,
            18 => RemainderUnsigned,
            19 => RemainderSigned,
            20 => MultiplyAdd,
            21 => MultiplyAddToSource,
            22 => MultiplySubtract,
            23 => MultiplySubtractFromSource,
            24 => Merge,
            25 => SlideUp,
            26 => SlideDown,
            _ => return None,
        })
    }

    /// A short spelling, for the disassembly.
    pub const fn name(self) -> &'static str {
        use VectorOperation::*;
        match self {
            Add => "+",
            Subtract => "-",
            SubtractFrom => "-<",
            MinimumUnsigned => "minu",
            MinimumSigned => "min",
            MaximumUnsigned => "maxu",
            MaximumSigned => "max",
            And => "&",
            Or => "|",
            Xor => "^",
            ShiftLeft => "<<",
            ShiftRight => ">>",
            ShiftRightSigned => ">>a",
            Multiply => "*",
            MultiplyHighSigned => "*h",
            MultiplyHighUnsigned => "*hu",
            DivideUnsigned => "/u",
            DivideSigned => "/",
            RemainderUnsigned => "%u",
            RemainderSigned => "%",
            MultiplyAdd => "+=*",
            MultiplyAddToSource => "*+",
            MultiplySubtract => "-=*",
            MultiplySubtractFromSource => "*-",
            Merge => "merge",
            SlideUp => "slideup",
            SlideDown => "slidedown",
        }
    }
}

/// Where the second operand of a [`VectorOperation`] comes from.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum VectorOperand {
    /// Another register group, element by element.
    Vector(u32),
    /// One general purpose register, the same value for every element.
    Register(u32),
    /// One immediate, the same value for every element. Five bits, signed.
    Immediate(i32),
}

/// One element-wise instruction, in the single immediate it is encoded in.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct VectorArithmetic {
    pub operation: VectorOperation,
    pub dst: u32,
    /// The operand the vector extensions call `vs2`, which is always a register group.
    pub src: u32,
    /// The operand they call `vs1`, `rs1` or the immediate.
    pub operand: VectorOperand,
}

impl VectorArithmetic {
    pub const fn to_packed(self) -> u32 {
        let (shape, operand) = match self.operand {
            VectorOperand::Vector(reg) => (0, reg),
            VectorOperand::Register(reg) => (1, reg),
            VectorOperand::Immediate(value) => (2, (value as u32) & 0b11111),
        };

        self.operation.to_index() | (shape << 6) | (self.dst << 8) | (self.src << 13) | (operand << 18)
    }

    pub const fn from_packed(packed: u32) -> Self {
        let Some(operation) = VectorOperation::from_index(packed & 0b111111) else {
            unreachable!()
        };

        let operand = (packed >> 18) & 0b11111;
        Self {
            operation,
            dst: (packed >> 8) & 0b11111,
            src: (packed >> 13) & 0b11111,
            operand: match (packed >> 6) & 0b11 {
                0 => VectorOperand::Vector(operand),
                1 => VectorOperand::Register(operand),
                // Five bits, signed, as the encoding it came from has it.
                _ => VectorOperand::Immediate(((operand << 27) as i32) >> 27),
            },
        }
    }
}

impl core::fmt::Display for VectorArithmetic {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(fmt, "v{} = v{} {} ", self.dst, self.src, self.operation.name())?;
        match self.operand {
            VectorOperand::Vector(reg) => write!(fmt, "v{reg}"),
            VectorOperand::Register(reg) => write!(fmt, "r{reg}"),
            VectorOperand::Immediate(value) => write!(fmt, "{value}"),
        }
    }
}
