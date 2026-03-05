// AArch64 (ARM64) instruction encoding for PolkaVM JIT backend.
//
// AArch64 instructions are fixed 32-bit wide, little-endian.
// This module provides register definitions and instruction encoding functions
// that return `Instruction<T>` for use with the generic assembler framework.

use crate::misc::{FixupKind, InstBuf, Instruction, Label};

// ── Register definitions ──────────────────────────────────────────────────────

/// AArch64 general-purpose registers (64-bit X-regs / 32-bit W-regs share the same index).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum Reg {
    x0 = 0, x1 = 1, x2 = 2, x3 = 3,
    x4 = 4, x5 = 5, x6 = 6, x7 = 7,
    x8 = 8, x9 = 9, x10 = 10, x11 = 11,
    x12 = 12, x13 = 13, x14 = 14, x15 = 15,
    x16 = 16, x17 = 17, x18 = 18, x19 = 19,
    x20 = 20, x21 = 21, x22 = 22, x23 = 23,
    x24 = 24, x25 = 25, x26 = 26, x27 = 27,
    x28 = 28,
    /// Frame pointer
    fp = 29,
    /// Link register
    lr = 30,
    /// Stack pointer (or zero register in certain instruction encodings)
    sp = 31,
}

/// A register index that can also be `xzr` (zero register).
/// In AArch64, register 31 is `sp` for some instructions and `xzr` for others.
/// We use `Reg::sp` when we mean the stack pointer, and `ZR` constant when we mean zero register.
pub const ZR: Reg = Reg::sp; // Same encoding, different semantic

#[allow(unused)]
pub use Reg::*;

impl Reg {
    #[inline(always)]
    pub const fn index(self) -> u32 {
        self as u32
    }
}

impl core::fmt::Display for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self {
            Reg::fp => write!(f, "x29"),
            Reg::lr => write!(f, "x30"),
            Reg::sp => write!(f, "sp"),
            other => write!(f, "x{}", other as u8),
        }
    }
}

/// Register width for AArch64 instructions.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RegSize {
    /// 32-bit (W-register)
    W32,
    /// 64-bit (X-register)
    X64,
}

impl RegSize {
    #[inline(always)]
    const fn sf(self) -> u32 {
        match self {
            RegSize::W32 => 0,
            RegSize::X64 => 1,
        }
    }
}

/// Condition codes for conditional branches and conditional select.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Condition {
    /// Equal (Z=1)
    EQ = 0b0000,
    /// Not equal (Z=0)
    NE = 0b0001,
    /// Carry set / unsigned higher or same (C=1)
    HS = 0b0010,
    /// Carry clear / unsigned lower (C=0)
    LO = 0b0011,
    /// Minus / negative (N=1)
    MI = 0b0100,
    /// Plus / positive or zero (N=0)
    PL = 0b0101,
    /// Overflow (V=1)
    VS = 0b0110,
    /// No overflow (V=0)
    VC = 0b0111,
    /// Unsigned higher (C=1 && Z=0)
    HI = 0b1000,
    /// Unsigned lower or same (C=0 || Z=1)
    LS = 0b1001,
    /// Signed greater or equal (N=V)
    GE = 0b1010,
    /// Signed less than (N!=V)
    LT = 0b1011,
    /// Signed greater than (Z=0 && N=V)
    GT = 0b1100,
    /// Signed less or equal (Z=1 || N!=V)
    LE = 0b1101,
    /// Always (unconditional)
    AL = 0b1110,
}

impl Condition {
    #[inline(always)]
    pub const fn invert(self) -> Self {
        // Invert the lowest bit to get the opposite condition.
        // SAFETY: All condition code pairs differ only in bit 0.
        unsafe { core::mem::transmute(self as u8 ^ 1) }
    }

    #[inline(always)]
    pub const fn code(self) -> u32 {
        self as u32
    }
}

impl core::fmt::Display for Condition {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let name = match self {
            Condition::EQ => "eq", Condition::NE => "ne",
            Condition::HS => "hs", Condition::LO => "lo",
            Condition::MI => "mi", Condition::PL => "pl",
            Condition::VS => "vs", Condition::VC => "vc",
            Condition::HI => "hi", Condition::LS => "ls",
            Condition::GE => "ge", Condition::LT => "lt",
            Condition::GT => "gt", Condition::LE => "le",
            Condition::AL => "al",
        };
        f.write_str(name)
    }
}

// ── Instruction encoding helpers ──────────────────────────────────────────────

/// Encode a 32-bit AArch64 instruction into an `InstBuf`.
#[inline(always)]
fn encode32(bits: u32) -> InstBuf {
    InstBuf::from_array(bits.to_le_bytes())
}

// ── Fixup kinds for AArch64 branches ──────────────────────────────────────────

/// AArch64 branch fixups encode the branch offset into specific bit fields of
/// the 32-bit instruction word.
///
/// We reuse `FixupKind` with a convention:
/// - offset = 0 (the fixup starts at byte 0 of the instruction)
/// - length = 4 (the instruction is 4 bytes)
///
/// We store the instruction template in the opcode field. During finalize,
/// we need custom patching logic for AArch64 which is handled by the assembler's
/// finalize_aarch64 method.

/// Create a FixupKind for AArch64 branch fixups.
/// The instruction template (with offset bits zeroed) is stored in the code bytes.
/// During finalize, the assembler detects the instruction type and patches the offset
/// into the appropriate bit fields.
#[inline(always)]
const fn fixup_aarch64() -> FixupKind {
    FixupKind::new_aarch64()
}

// ── Branch type markers for fixup patching ────────────────────────────────────

/// Identifies how to patch the branch offset into the instruction word.
#[derive(Copy, Clone, Debug)]
pub enum BranchFixupKind {
    /// B imm26: offset in bits [25:0], shifted right 2. Range: +/-128MB
    B26,
    /// B.cond imm19: offset in bits [23:5], shifted right 2. Range: +/-1MB
    BCond19,
    /// CBZ/CBNZ imm19: offset in bits [23:5], shifted right 2. Range: +/-1MB
    CB19,
    /// TBZ/TBNZ imm14: offset in bits [18:5], shifted right 2. Range: +/-32KB
    TB14,
    /// ADR imm21: immhi in bits [23:5], immlo in bits [30:29]
    Adr21,
}

// ── Data processing (register) ────────────────────────────────────────────────

// For display purposes we define a wrapper type that implements Display.
#[derive(Copy, Clone, Debug)]
pub struct AArch64Inst {
    mnemonic: &'static str,
    bits: u32,
}

impl core::fmt::Display for AArch64Inst {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{} ; 0x{:08x}", self.mnemonic, self.bits)
    }
}

#[inline(always)]
fn inst(mnemonic: &'static str, bits: u32) -> Instruction<AArch64Inst> {
    Instruction {
        instruction: AArch64Inst { mnemonic, bits },
        bytes: encode32(bits),
        fixup: None,
    }
}

#[inline(always)]
fn inst_fixup(mnemonic: &'static str, bits: u32, label: Label, kind: FixupKind) -> Instruction<AArch64Inst> {
    Instruction {
        instruction: AArch64Inst { mnemonic, bits },
        bytes: encode32(bits),
        fixup: Some((label, kind)),
    }
}

// ── Arithmetic instructions ───────────────────────────────────────────────────

/// ADD Rd, Rn, Rm (shifted register)
#[inline(always)]
pub fn add(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0001011_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("add", bits)
}

/// ADD Rd, Rn, #imm12 (immediate, optionally shifted left 12)
#[inline(always)]
pub fn add_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32) -> Instruction<AArch64Inst> {
    debug_assert!(imm12 < 4096, "add_imm: immediate out of range");
    // sf=size | 00 | 100010 | sh=0 | imm12 | Rn | Rd
    let bits = (size.sf() << 31) | (0b0010001 << 24) | (0 << 22) | (imm12 << 10) | (rn.index() << 5) | rd.index();
    inst("add", bits)
}

/// ADDS (setting flags) Rd, Rn, Rm — used for CMN
#[inline(always)]
pub fn adds(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0101011_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("adds", bits)
}

/// SUB Rd, Rn, Rm
#[inline(always)]
pub fn sub(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b1001011_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("sub", bits)
}

/// SUB Rd, Rn, #imm12
#[inline(always)]
pub fn sub_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32) -> Instruction<AArch64Inst> {
    debug_assert!(imm12 < 4096, "sub_imm: immediate out of range");
    // sf=size | 10 | 100010 | sh=0 | imm12 | Rn | Rd
    let bits = (size.sf() << 31) | (0b1010001 << 24) | (0 << 22) | (imm12 << 10) | (rn.index() << 5) | rd.index();
    inst("sub", bits)
}

/// SUB Rd, Rn, #imm12, LSL #12  (subtracts imm12 << 12)
#[inline(always)]
pub fn sub_imm_lsl12(size: RegSize, rd: Reg, rn: Reg, imm12: u32) -> Instruction<AArch64Inst> {
    debug_assert!(imm12 < 4096, "sub_imm_lsl12: immediate out of range");
    // sf=size | 10 | 100010 | sh=1 | imm12 | Rn | Rd
    let bits = (size.sf() << 31) | (0b1010001 << 24) | (1 << 22) | (imm12 << 10) | (rn.index() << 5) | rd.index();
    inst("sub", bits)
}

/// SUBS Rd, Rn, Rm (sets flags)
#[inline(always)]
pub fn subs(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b1101011_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("subs", bits)
}

/// SUBS Rd, Rn, #imm12 (sets flags) — CMP when Rd=xzr
#[inline(always)]
pub fn subs_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32) -> Instruction<AArch64Inst> {
    debug_assert!(imm12 < 4096, "subs_imm: immediate out of range");
    // sf=size | 11 | 100010 | sh=0 | imm12 | Rn | Rd
    let bits = (size.sf() << 31) | (0b1110001 << 24) | (0 << 22) | (imm12 << 10) | (rn.index() << 5) | rd.index();
    inst("subs", bits)
}

/// CMP Rn, Rm  (alias for SUBS xzr, Rn, Rm)
#[inline(always)]
pub fn cmp(size: RegSize, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    subs(size, ZR, rn, rm)
}

/// CMP Rn, #imm12  (alias for SUBS xzr, Rn, #imm12)
#[inline(always)]
pub fn cmp_imm(size: RegSize, rn: Reg, imm12: u32) -> Instruction<AArch64Inst> {
    subs_imm(size, ZR, rn, imm12)
}

/// CMN Rn, Rm  (alias for ADDS xzr, Rn, Rm)
#[inline(always)]
pub fn cmn(size: RegSize, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    adds(size, ZR, rn, rm)
}

/// NEG Rd, Rm  (alias for SUB Rd, xzr, Rm)
#[inline(always)]
pub fn neg(size: RegSize, rd: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    sub(size, rd, ZR, rm)
}

/// MUL Rd, Rn, Rm (alias for MADD Rd, Rn, Rm, xzr)
#[inline(always)]
pub fn mul(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    madd(size, rd, rn, rm, ZR)
}

/// MADD Rd, Rn, Rm, Ra  (Rd = Ra + Rn*Rm)
#[inline(always)]
pub fn madd(size: RegSize, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11011_000 << 21) | (rm.index() << 16) | (0 << 15) | (ra.index() << 10) | (rn.index() << 5) | rd.index();
    inst("madd", bits)
}

/// MSUB Rd, Rn, Rm, Ra  (Rd = Ra - Rn*Rm)
#[inline(always)]
pub fn msub(size: RegSize, rd: Reg, rn: Reg, rm: Reg, ra: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11011_000 << 21) | (rm.index() << 16) | (1 << 15) | (ra.index() << 10) | (rn.index() << 5) | rd.index();
    inst("msub", bits)
}

/// MNEG Rd, Rn, Rm  (alias for MSUB Rd, Rn, Rm, xzr)
#[inline(always)]
pub fn mneg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    msub(size, rd, rn, rm, ZR)
}

/// SMULH Xd, Xn, Xm — Signed multiply high (64×64→upper 64 bits)
#[inline(always)]
pub fn smulh(rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b10011011_010 << 21) | (rm.index() << 16) | (0b011111 << 10) | (rn.index() << 5) | rd.index();
    inst("smulh", bits)
}

/// UMULH Xd, Xn, Xm — Unsigned multiply high (64×64→upper 64 bits)
#[inline(always)]
pub fn umulh(rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b10011011_110 << 21) | (rm.index() << 16) | (0b011111 << 10) | (rn.index() << 5) | rd.index();
    inst("umulh", bits)
}

/// SMULL Xd, Wn, Wm — Signed multiply long (32×32→64)
#[inline(always)]
pub fn smull(rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    // SMADDL Xd, Wn, Wm, XZR
    let bits = (0b10011011_001 << 21) | (rm.index() << 16) | (0 << 15) | (ZR.index() << 10) | (rn.index() << 5) | rd.index();
    inst("smull", bits)
}

/// UMULL Xd, Wn, Wm — Unsigned multiply long (32×32→64)
#[inline(always)]
pub fn umull(rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    // UMADDL Xd, Wn, Wm, XZR
    let bits = (0b10011011_101 << 21) | (rm.index() << 16) | (0 << 15) | (ZR.index() << 10) | (rn.index() << 5) | rd.index();
    inst("umull", bits)
}

/// SDIV Rd, Rn, Rm
#[inline(always)]
pub fn sdiv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b000011 << 10) | (rn.index() << 5) | rd.index();
    inst("sdiv", bits)
}

/// UDIV Rd, Rn, Rm
#[inline(always)]
pub fn udiv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b000010 << 10) | (rn.index() << 5) | rd.index();
    inst("udiv", bits)
}

// ── Bitwise / logical instructions ────────────────────────────────────────────

/// AND Rd, Rn, Rm (shifted register)
#[inline(always)]
pub fn and(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0001010_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("and", bits)
}

/// ANDS Rd, Rn, Rm (sets flags) — TST when Rd=xzr
#[inline(always)]
pub fn ands(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b1101010_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("ands", bits)
}

/// TST Rn, Rm  (alias for ANDS xzr, Rn, Rm)
#[inline(always)]
pub fn tst(size: RegSize, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    ands(size, ZR, rn, rm)
}

/// ORR Rd, Rn, Rm
#[inline(always)]
pub fn orr(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0101010_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("orr", bits)
}

/// ORN Rd, Rn, Rm  (Rd = Rn | ~Rm)
#[inline(always)]
pub fn orn(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0101010_00_1 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("orn", bits)
}

/// EOR Rd, Rn, Rm
#[inline(always)]
pub fn eor(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b1001010_00_0 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("eor", bits)
}

/// EON Rd, Rn, Rm  (Rd = Rn ^ ~Rm)
#[inline(always)]
pub fn eon(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b1001010_00_1 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("eon", bits)
}

/// BIC Rd, Rn, Rm  (Rd = Rn & ~Rm)
#[inline(always)]
pub fn bic(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b0001010_00_1 << 21) | (rm.index() << 16) | (0b000000 << 10) | (rn.index() << 5) | rd.index();
    inst("bic", bits)
}

/// MVN Rd, Rm  (alias for ORN Rd, xzr, Rm)
#[inline(always)]
pub fn mvn(size: RegSize, rd: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    orn(size, rd, ZR, rm)
}

/// MOV Rd, Rm  (alias for ORR Rd, xzr, Rm)
#[inline(always)]
pub fn mov_reg(size: RegSize, rd: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    orr(size, rd, ZR, rm)
}

// ── Logical immediate ─────────────────────────────────────────────────────────

/// Encode a bitmask immediate for logical operations.
/// Returns (N, immr, imms) or None if the value cannot be encoded.
pub fn encode_logical_imm(size: RegSize, value: u64) -> Option<(u32, u32, u32)> {
    let value = match size {
        RegSize::W32 => {
            let v = value as u32;
            if v == 0 || v == !0 { return None; }
            (v as u64) | ((v as u64) << 32)
        }
        RegSize::X64 => {
            if value == 0 || value == !0u64 { return None; }
            value
        }
    };

    // Try each possible element size: 2, 4, 8, 16, 32, 64
    let mut element_size = 2u32;
    while element_size <= 64 {
        let mask = !0u64 >> (64 - element_size);
        let element = value & mask;

        // Check if the pattern repeats across the entire 64-bit value
        let mut valid = true;
        let mut shift = element_size;
        while shift < 64 {
            if ((value >> shift) & mask) != element {
                valid = false;
                break;
            }
            shift += element_size;
        }

        if valid {
            // The element must be a contiguous run of 1s, possibly rotated
            let ones = element.count_ones();
            if ones == 0 || ones == element_size { element_size *= 2; continue; }

            // Check contiguity within element_size bits:
            // Rotate away trailing ones to get all ones at the top,
            // then check the result has exactly (element_size - ones) trailing zeros
            // followed by `ones` ones and nothing else.
            let trailing_ones = (element & mask).trailing_ones() as u32;
            let rotated_elem = if trailing_ones > 0 && trailing_ones < element_size {
                ((element >> trailing_ones) | (element << (element_size - trailing_ones))) & mask
            } else {
                element & mask
            };
            // After rotating away trailing ones, should have zeros at bottom, then ones at top
            let tz = rotated_elem.trailing_zeros() as u32;
            if tz + ones != element_size {
                // Not a contiguous run of ones
                element_size *= 2;
                continue;
            }

            let immr = if trailing_ones > 0 {
                // Wrapping case: ones span the bottom and top of the element.
                // immr = number of top-portion ones
                (ones - trailing_ones) % element_size
            } else {
                // Non-wrapping case: contiguous block of ones in the middle/top.
                // The ones start at trailing_zeros; immr = element_size - start.
                (element_size - element.trailing_zeros() as u32) % element_size
            };
            let imms = {
                let len_encoding = match element_size {
                    2 =>  0b111100,
                    4 =>  0b111000,
                    8 =>  0b110000,
                    16 => 0b100000,
                    32 => 0b000000,
                    64 => 0b000000,
                    _ => unreachable!(),
                };
                len_encoding | (ones - 1)
            };

            let n = if element_size == 64 { 1u32 } else { 0u32 };

            return Some((n, immr, imms));
        }

        element_size *= 2;
    }

    None
}

/// AND Rd, Rn, #imm (logical immediate)
#[inline(always)]
pub fn and_imm(size: RegSize, rd: Reg, rn: Reg, n: u32, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_100100 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("and", bits)
}

/// ORR Rd, Rn, #imm (logical immediate)
#[inline(always)]
pub fn orr_imm(size: RegSize, rd: Reg, rn: Reg, n: u32, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b01_100100 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("orr", bits)
}

/// EOR Rd, Rn, #imm (logical immediate)
#[inline(always)]
pub fn eor_imm(size: RegSize, rd: Reg, rn: Reg, n: u32, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_100100 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("eor", bits)
}

/// ANDS Rd, Rn, #imm (logical immediate, sets flags) — TST when Rd=xzr
#[inline(always)]
pub fn ands_imm(size: RegSize, rd: Reg, rn: Reg, n: u32, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b11_100100 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("ands", bits)
}

/// TST Rn, #imm  (alias for ANDS xzr, Rn, #imm)
#[inline(always)]
pub fn tst_imm(size: RegSize, rn: Reg, n: u32, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    ands_imm(size, ZR, rn, n, immr, imms)
}

// ── Shift instructions ────────────────────────────────────────────────────────

/// LSL Rd, Rn, Rm  (alias for LSLV)
#[inline(always)]
pub fn lsl(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b001000 << 10) | (rn.index() << 5) | rd.index();
    inst("lsl", bits)
}

/// LSR Rd, Rn, Rm  (alias for LSRV)
#[inline(always)]
pub fn lsr(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b001001 << 10) | (rn.index() << 5) | rd.index();
    inst("lsr", bits)
}

/// ASR Rd, Rn, Rm  (alias for ASRV)
#[inline(always)]
pub fn asr(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b001010 << 10) | (rn.index() << 5) | rd.index();
    inst("asr", bits)
}

/// ROR Rd, Rn, Rm  (alias for RORV)
#[inline(always)]
pub fn ror(size: RegSize, rd: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010110 << 21) | (rm.index() << 16) | (0b001011 << 10) | (rn.index() << 5) | rd.index();
    inst("ror", bits)
}

/// LSL Rd, Rn, #shift  (alias for UBFM)
#[inline(always)]
pub fn lsl_imm(size: RegSize, rd: Reg, rn: Reg, shift: u32) -> Instruction<AArch64Inst> {
    let max = match size { RegSize::W32 => 31, RegSize::X64 => 63 };
    let immr = (max + 1 - shift) & max;
    let imms = max - shift;
    ubfm(size, rd, rn, immr, imms)
}

/// LSR Rd, Rn, #shift  (alias for UBFM)
#[inline(always)]
pub fn lsr_imm(size: RegSize, rd: Reg, rn: Reg, shift: u32) -> Instruction<AArch64Inst> {
    let imms = match size { RegSize::W32 => 31, RegSize::X64 => 63 };
    ubfm(size, rd, rn, shift, imms)
}

/// ASR Rd, Rn, #shift  (alias for SBFM)
#[inline(always)]
pub fn asr_imm(size: RegSize, rd: Reg, rn: Reg, shift: u32) -> Instruction<AArch64Inst> {
    let imms = match size { RegSize::W32 => 31, RegSize::X64 => 63 };
    sbfm(size, rd, rn, shift, imms)
}

/// ROR Rd, Rs, #shift  (alias for EXTR Rd, Rs, Rs, #shift)
#[inline(always)]
pub fn ror_imm(size: RegSize, rd: Reg, rs: Reg, shift: u32) -> Instruction<AArch64Inst> {
    extr(size, rd, rs, rs, shift)
}

// ── Bitfield instructions ─────────────────────────────────────────────────────

/// UBFM Rd, Rn, #immr, #imms
#[inline(always)]
pub fn ubfm(size: RegSize, rd: Reg, rn: Reg, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let n = size.sf(); // N = sf for 64-bit
    let bits = (size.sf() << 31) | (0b10_100110 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("ubfm", bits)
}

/// SBFM Rd, Rn, #immr, #imms
#[inline(always)]
pub fn sbfm(size: RegSize, rd: Reg, rn: Reg, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let n = size.sf();
    let bits = (size.sf() << 31) | (0b00_100110 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("sbfm", bits)
}

/// BFM Rd, Rn, #immr, #imms (bitfield move — insert bits)
#[inline(always)]
pub fn bfm(size: RegSize, rd: Reg, rn: Reg, immr: u32, imms: u32) -> Instruction<AArch64Inst> {
    let n = size.sf();
    let bits = (size.sf() << 31) | (0b01_100110 << 23) | (n << 22) | (immr << 16) | (imms << 10) | (rn.index() << 5) | rd.index();
    inst("bfm", bits)
}

/// EXTR Rd, Rn, Rm, #lsb  (extract from pair of registers)
#[inline(always)]
pub fn extr(size: RegSize, rd: Reg, rn: Reg, rm: Reg, lsb: u32) -> Instruction<AArch64Inst> {
    let n = size.sf();
    let bits = (size.sf() << 31) | (0b00_100111 << 23) | (n << 22) | (0 << 21) | (rm.index() << 16) | (lsb << 10) | (rn.index() << 5) | rd.index();
    inst("extr", bits)
}

/// SXTB Rd, Rn  (sign-extend byte, alias for SBFM Rd, Rn, #0, #7)
#[inline(always)]
pub fn sxtb(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    sbfm(size, rd, rn, 0, 7)
}

/// SXTH Rd, Rn  (sign-extend halfword, alias for SBFM Rd, Rn, #0, #15)
#[inline(always)]
pub fn sxth(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    sbfm(size, rd, rn, 0, 15)
}

/// SXTW Xd, Wn  (sign-extend word, alias for SBFM Xd, Xn, #0, #31)
#[inline(always)]
pub fn sxtw(rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    sbfm(RegSize::X64, rd, rn, 0, 31)
}

/// UXTB Wd, Wn  (zero-extend byte, alias for UBFM Wd, Wn, #0, #7)
#[inline(always)]
pub fn uxtb(rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    ubfm(RegSize::W32, rd, rn, 0, 7)
}

/// UXTH Wd, Wn  (zero-extend halfword, alias for UBFM Wd, Wn, #0, #15)
#[inline(always)]
pub fn uxth(rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    ubfm(RegSize::W32, rd, rn, 0, 15)
}

// ── Data processing (misc) ────────────────────────────────────────────────────

/// REV Rd, Rn  (byte-reverse)
#[inline(always)]
pub fn rev(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    let opc = match size { RegSize::W32 => 0b10, RegSize::X64 => 0b11 };
    let bits = (size.sf() << 31) | (0b10_11010110_00000_0000 << 12) | (opc << 10) | (rn.index() << 5) | rd.index();
    inst("rev", bits)
}

/// CLZ Rd, Rn  (count leading zeros)
#[inline(always)]
pub fn clz(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_11010110_00000_00010_0 << 10) | (rn.index() << 5) | rd.index();
    inst("clz", bits)
}

/// CLS Rd, Rn  (count leading sign bits)
#[inline(always)]
pub fn cls(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_11010110_00000_00010_1 << 10) | (rn.index() << 5) | rd.index();
    inst("cls", bits)
}

/// RBIT Rd, Rn  (reverse bits)
#[inline(always)]
pub fn rbit(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_11010110_00000_00000_0 << 10) | (rn.index() << 5) | rd.index();
    inst("rbit", bits)
}

/// CTZ via RBIT + CLZ
/// AArch64 doesn't have native CTZ; use RBIT + CLZ.
/// We provide a helper that emits two instructions, but for the single-instruction
/// pattern, callers should emit rbit() then clz() separately.

// ── Move instructions ─────────────────────────────────────────────────────────

/// MOVZ Rd, #imm16, LSL #shift  (move wide with zero)
#[inline(always)]
pub fn movz(size: RegSize, rd: Reg, imm16: u16, shift: u32) -> Instruction<AArch64Inst> {
    debug_assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = shift / 16;
    let bits = (size.sf() << 31) | (0b10_100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rd.index();
    inst("movz", bits)
}

/// MOVK Rd, #imm16, LSL #shift  (move wide with keep)
#[inline(always)]
pub fn movk(size: RegSize, rd: Reg, imm16: u16, shift: u32) -> Instruction<AArch64Inst> {
    debug_assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = shift / 16;
    let bits = (size.sf() << 31) | (0b11_100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rd.index();
    inst("movk", bits)
}

/// MOVN Rd, #imm16, LSL #shift  (move wide with NOT)
#[inline(always)]
pub fn movn(size: RegSize, rd: Reg, imm16: u16, shift: u32) -> Instruction<AArch64Inst> {
    debug_assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
    let hw = shift / 16;
    let bits = (size.sf() << 31) | (0b00_100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rd.index();
    inst("movn", bits)
}

// ── Conditional select ────────────────────────────────────────────────────────

/// CSEL Rd, Rn, Rm, cond  (Rd = cond ? Rn : Rm)
#[inline(always)]
pub fn csel(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010100 << 21) | (rm.index() << 16) | (cond.code() << 12) | (0b00 << 10) | (rn.index() << 5) | rd.index();
    inst("csel", bits)
}

/// CSINC Rd, Rn, Rm, cond  (Rd = cond ? Rn : Rm+1)
#[inline(always)]
pub fn csinc(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b00_11010100 << 21) | (rm.index() << 16) | (cond.code() << 12) | (0b01 << 10) | (rn.index() << 5) | rd.index();
    inst("csinc", bits)
}

/// CSINV Rd, Rn, Rm, cond  (Rd = cond ? Rn : ~Rm)
#[inline(always)]
pub fn csinv(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_11010100 << 21) | (rm.index() << 16) | (cond.code() << 12) | (0b00 << 10) | (rn.index() << 5) | rd.index();
    inst("csinv", bits)
}

/// CSNEG Rd, Rn, Rm, cond  (Rd = cond ? Rn : -Rm)
#[inline(always)]
pub fn csneg(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    let bits = (size.sf() << 31) | (0b10_11010100 << 21) | (rm.index() << 16) | (cond.code() << 12) | (0b01 << 10) | (rn.index() << 5) | rd.index();
    inst("csneg", bits)
}

/// CSET Rd, cond  (alias for CSINC Rd, xzr, xzr, invert(cond))
/// Rd = 1 if cond, 0 otherwise
#[inline(always)]
pub fn cset(size: RegSize, rd: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    csinc(size, rd, ZR, ZR, cond.invert())
}

/// CSETM Rd, cond  (alias for CSINV Rd, xzr, xzr, invert(cond))
/// Rd = -1 if cond, 0 otherwise
#[inline(always)]
pub fn csetm(size: RegSize, rd: Reg, cond: Condition) -> Instruction<AArch64Inst> {
    csinv(size, rd, ZR, ZR, cond.invert())
}

// ── Memory instructions ───────────────────────────────────────────────────────

/// Load/store size encoding
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MemSize {
    B8 = 0,
    B16 = 1,
    B32 = 2,
    B64 = 3,
}

/// LDR Xt, [Xn, #imm]  (unsigned offset, scaled)
#[inline(always)]
pub fn ldr_imm(size: MemSize, rt: Reg, rn: Reg, offset: u32) -> Instruction<AArch64Inst> {
    let scale = size as u32;
    let scaled_offset = offset >> scale;
    debug_assert!(offset == scaled_offset << scale, "ldr_imm: unaligned offset");
    debug_assert!(scaled_offset < 4096, "ldr_imm: offset out of range");
    let bits = (size as u32) << 30 | (0b11_1001_01 << 22) | (scaled_offset << 10) | (rn.index() << 5) | rt.index();
    inst("ldr", bits)
}

/// STR Xt, [Xn, #imm]  (unsigned offset, scaled)
#[inline(always)]
pub fn str_imm(size: MemSize, rt: Reg, rn: Reg, offset: u32) -> Instruction<AArch64Inst> {
    let scale = size as u32;
    let scaled_offset = offset >> scale;
    debug_assert!(offset == scaled_offset << scale, "str_imm: unaligned offset");
    debug_assert!(scaled_offset < 4096, "str_imm: offset out of range");
    let bits = (size as u32) << 30 | (0b11_1001_00 << 22) | (scaled_offset << 10) | (rn.index() << 5) | rt.index();
    inst("str", bits)
}

/// LDR Xt, [Xn, Xm]  (register offset)
#[inline(always)]
pub fn ldr_reg(size: MemSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    // option = 011 (LSL), S = 0 (no scaling)
    let bits = (size as u32) << 30 | (0b11_1000_01_1 << 21) | (rm.index() << 16) | (0b011_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldr", bits)
}

/// STR Xt, [Xn, Xm]  (register offset)
#[inline(always)]
pub fn str_reg(size: MemSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size as u32) << 30 | (0b11_1000_00_1 << 21) | (rm.index() << 16) | (0b011_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("str", bits)
}

/// LDR Xt, [Xn, Wm, UXTW]  (32-bit register offset, zero-extended)
#[inline(always)]
pub fn ldr_reg_uxtw(size: MemSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    // option = 010 (UXTW), S = 0
    let bits = (size as u32) << 30 | (0b11_1000_01_1 << 21) | (rm.index() << 16) | (0b010_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldr", bits)
}

/// STR Xt, [Xn, Wm, UXTW]  (32-bit register offset, zero-extended)
#[inline(always)]
pub fn str_reg_uxtw(size: MemSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (size as u32) << 30 | (0b11_1000_00_1 << 21) | (rm.index() << 16) | (0b010_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("str", bits)
}

/// LDUR Xt, [Xn, #simm9]  (unscaled offset, signed 9-bit)
#[inline(always)]
pub fn ldur(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255, "ldur: offset out of range");
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_01_0 << 21) | (imm9 << 12) | (0b00 << 10) | (rn.index() << 5) | rt.index();
    inst("ldur", bits)
}

/// STUR Xt, [Xn, #simm9]  (unscaled offset, signed 9-bit)
#[inline(always)]
pub fn stur(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255, "stur: offset out of range");
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_00_0 << 21) | (imm9 << 12) | (0b00 << 10) | (rn.index() << 5) | rt.index();
    inst("stur", bits)
}

/// Sign-extending loads

/// LDRSB Rt, [Xn, Xm]  (sign-extend byte to 32 or 64 bit)
#[inline(always)]
pub fn ldrsb_reg(size: RegSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    // opc: 11=LDRSB to W, 10=LDRSB to X
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b00_111000 << 24) | (opc << 22) | (1 << 21) | (rm.index() << 16) | (0b011_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsb", bits)
}

/// LDRSH Rt, [Xn, Xm]
#[inline(always)]
pub fn ldrsh_reg(size: RegSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b01_111000 << 24) | (opc << 22) | (1 << 21) | (rm.index() << 16) | (0b011_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsh", bits)
}

/// LDRSW Xt, [Xn, Xm]
#[inline(always)]
pub fn ldrsw_reg(rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b10_111000_10_1 << 21) | (rm.index() << 16) | (0b011_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsw", bits)
}

/// LDRSB Rt, [Xn, Wm, UXTW]  (sign-extend byte, 32-bit offset zero-extended)
#[inline(always)]
pub fn ldrsb_reg_uxtw(size: RegSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b00_111000 << 24) | (opc << 22) | (1 << 21) | (rm.index() << 16) | (0b010_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsb", bits)
}

/// LDRSH Rt, [Xn, Wm, UXTW]
#[inline(always)]
pub fn ldrsh_reg_uxtw(size: RegSize, rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b01_111000 << 24) | (opc << 22) | (1 << 21) | (rm.index() << 16) | (0b010_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsh", bits)
}

/// LDRSW Xt, [Xn, Wm, UXTW]
#[inline(always)]
pub fn ldrsw_reg_uxtw(rt: Reg, rn: Reg, rm: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b10_111000_10_1 << 21) | (rm.index() << 16) | (0b010_0_10 << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsw", bits)
}

/// LDRSB Rt, [Xn, #imm12] (unsigned offset, scaled)
#[inline(always)]
pub fn ldrsb_imm(size: RegSize, rt: Reg, rn: Reg, offset: u32) -> Instruction<AArch64Inst> {
    debug_assert!(offset < 4096, "ldrsb_imm: offset out of range");
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b00_111001 << 24) | (opc << 22) | (offset << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsb", bits)
}

/// LDRSH Rt, [Xn, #imm12] (scaled by 2)
#[inline(always)]
pub fn ldrsh_imm(size: RegSize, rt: Reg, rn: Reg, offset: u32) -> Instruction<AArch64Inst> {
    let scaled = offset >> 1;
    debug_assert!(offset == scaled << 1 && scaled < 4096, "ldrsh_imm: offset out of range");
    let opc = match size { RegSize::W32 => 0b11, RegSize::X64 => 0b10 };
    let bits = (0b01_111001 << 24) | (opc << 22) | (scaled << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsh", bits)
}

/// LDRSW Xt, [Xn, #imm12] (scaled by 4)
#[inline(always)]
pub fn ldrsw_imm(rt: Reg, rn: Reg, offset: u32) -> Instruction<AArch64Inst> {
    let scaled = offset >> 2;
    debug_assert!(offset == scaled << 2 && scaled < 4096, "ldrsw_imm: offset out of range");
    let bits = (0b10_111001_10 << 22) | (scaled << 10) | (rn.index() << 5) | rt.index();
    inst("ldrsw", bits)
}

/// LDP Xt1, Xt2, [Xn, #imm7]  (load pair, signed offset scaled by 8/4)
#[inline(always)]
pub fn ldp(size: MemSize, rt1: Reg, rt2: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let (opc, scale) = match size { MemSize::B32 => (0b00u32, 4), MemSize::B64 => (0b10, 8), _ => panic!("ldp: unsupported size") };
    let scaled = offset / scale;
    debug_assert!(offset == scaled * scale);
    let imm7 = (scaled as u32) & 0x7F;
    let bits = (opc << 30) | (0b10_1_0_010_1 << 22) | (imm7 << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt1.index();
    inst("ldp", bits)
}

/// STP Xt1, Xt2, [Xn, #imm7]  (store pair, signed offset scaled by 8/4)
#[inline(always)]
pub fn stp(size: MemSize, rt1: Reg, rt2: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let (opc, scale) = match size { MemSize::B32 => (0b00u32, 4), MemSize::B64 => (0b10, 8), _ => panic!("stp: unsupported size") };
    let scaled = offset / scale;
    debug_assert!(offset == scaled * scale);
    let imm7 = (scaled as u32) & 0x7F;
    let bits = (opc << 30) | (0b10_1_0_010_0 << 22) | (imm7 << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt1.index();
    inst("stp", bits)
}

/// STP (pre-index): STP Xt1, Xt2, [Xn, #imm7]!
#[inline(always)]
pub fn stp_pre(size: MemSize, rt1: Reg, rt2: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let (opc, scale) = match size { MemSize::B32 => (0b00u32, 4), MemSize::B64 => (0b10, 8), _ => panic!("stp_pre: unsupported size") };
    let scaled = offset / scale;
    debug_assert!(offset == scaled * scale);
    let imm7 = (scaled as u32) & 0x7F;
    let bits = (opc << 30) | (0b10_1_0_011_0 << 22) | (imm7 << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt1.index();
    inst("stp", bits)
}

/// LDP (post-index): LDP Xt1, Xt2, [Xn], #imm7
#[inline(always)]
pub fn ldp_post(size: MemSize, rt1: Reg, rt2: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let (opc, scale) = match size { MemSize::B32 => (0b00u32, 4), MemSize::B64 => (0b10, 8), _ => panic!("ldp_post: unsupported size") };
    let scaled = offset / scale;
    debug_assert!(offset == scaled * scale);
    let imm7 = (scaled as u32) & 0x7F;
    let bits = (opc << 30) | (0b10_1_0_001_1 << 22) | (imm7 << 15) | (rt2.index() << 10) | (rn.index() << 5) | rt1.index();
    inst("ldp", bits)
}

/// LDR (pre-index): LDR Xt, [Xn, #simm9]!
#[inline(always)]
pub fn ldr_pre(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255);
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_01_0 << 21) | (imm9 << 12) | (0b11 << 10) | (rn.index() << 5) | rt.index();
    inst("ldr", bits)
}

/// STR (pre-index): STR Xt, [Xn, #simm9]!
#[inline(always)]
pub fn str_pre(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255);
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_00_0 << 21) | (imm9 << 12) | (0b11 << 10) | (rn.index() << 5) | rt.index();
    inst("str", bits)
}

/// LDR (post-index): LDR Xt, [Xn], #simm9
#[inline(always)]
pub fn ldr_post(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255);
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_01_0 << 21) | (imm9 << 12) | (0b01 << 10) | (rn.index() << 5) | rt.index();
    inst("ldr", bits)
}

/// STR (post-index): STR Xt, [Xn], #simm9
#[inline(always)]
pub fn str_post(size: MemSize, rt: Reg, rn: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset >= -256 && offset <= 255);
    let imm9 = (offset as u32) & 0x1FF;
    let bits = (size as u32) << 30 | (0b11_1000_00_0 << 21) | (imm9 << 12) | (0b01 << 10) | (rn.index() << 5) | rt.index();
    inst("str", bits)
}

// ── Branch instructions ───────────────────────────────────────────────────────

/// B offset  (unconditional branch, PC-relative, +/- 128MB)
/// offset is in bytes, must be 4-byte aligned.
#[inline(always)]
pub fn b(offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset & 3 == 0, "b: offset not 4-byte aligned");
    let imm26 = ((offset >> 2) as u32) & 0x03FFFFFF;
    let bits = (0b000101 << 26) | imm26;
    inst("b", bits)
}

/// B label (forward reference)
#[inline(always)]
pub fn b_label(label: Label) -> Instruction<AArch64Inst> {
    let template = 0b000101 << 26;
    inst_fixup("b", template, label, fixup_aarch64())
}

/// BL offset  (branch with link, PC-relative)
#[inline(always)]
pub fn bl(offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset & 3 == 0, "bl: offset not 4-byte aligned");
    let imm26 = ((offset >> 2) as u32) & 0x03FFFFFF;
    let bits = (0b100101 << 26) | imm26;
    inst("bl", bits)
}

/// BL label (forward reference)
#[inline(always)]
pub fn bl_label(label: Label) -> Instruction<AArch64Inst> {
    let template = 0b100101 << 26;
    inst_fixup("bl", template, label, fixup_aarch64())
}

/// B.cond offset  (conditional branch)
#[inline(always)]
pub fn b_cond(cond: Condition, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset & 3 == 0, "b_cond: offset not 4-byte aligned");
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    let bits = (0b01010100 << 24) | (imm19 << 5) | cond.code();
    inst("b.cond", bits)
}

/// B.cond label (forward reference)
#[inline(always)]
pub fn b_cond_label(cond: Condition, label: Label) -> Instruction<AArch64Inst> {
    let template = (0b01010100 << 24) | cond.code();
    inst_fixup("b.cond", template, label, fixup_aarch64())
}

/// BR Xn  (branch to register)
#[inline(always)]
pub fn br(rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b1101011_0000_11111_000000 << 10) | (rn.index() << 5) | 0b00000;
    inst("br", bits)
}

/// BLR Xn  (branch with link to register)
#[inline(always)]
pub fn blr(rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b1101011_0001_11111_000000 << 10) | (rn.index() << 5) | 0b00000;
    inst("blr", bits)
}

/// RET Xn  (return, default Xn=x30/lr)
#[inline(always)]
pub fn ret(rn: Reg) -> Instruction<AArch64Inst> {
    let bits = (0b1101011_0010_11111_000000 << 10) | (rn.index() << 5) | 0b00000;
    inst("ret", bits)
}

/// RET (to LR)
#[inline(always)]
pub fn ret_lr() -> Instruction<AArch64Inst> {
    ret(Reg::lr)
}

/// CBZ Rt, offset  (compare and branch if zero)
#[inline(always)]
pub fn cbz(size: RegSize, rt: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset & 3 == 0, "cbz: offset not 4-byte aligned");
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    let bits = (size.sf() << 31) | (0b0110100 << 24) | (imm19 << 5) | rt.index();
    inst("cbz", bits)
}

/// CBZ label (forward reference)
#[inline(always)]
pub fn cbz_label(size: RegSize, rt: Reg, label: Label) -> Instruction<AArch64Inst> {
    let template = (size.sf() << 31) | (0b0110100 << 24) | rt.index();
    inst_fixup("cbz", template, label, fixup_aarch64())
}

/// CBNZ Rt, offset  (compare and branch if not zero)
#[inline(always)]
pub fn cbnz(size: RegSize, rt: Reg, offset: i32) -> Instruction<AArch64Inst> {
    debug_assert!(offset & 3 == 0, "cbnz: offset not 4-byte aligned");
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    let bits = (size.sf() << 31) | (0b0110101 << 24) | (imm19 << 5) | rt.index();
    inst("cbnz", bits)
}

/// CBNZ label (forward reference)
#[inline(always)]
pub fn cbnz_label(size: RegSize, rt: Reg, label: Label) -> Instruction<AArch64Inst> {
    let template = (size.sf() << 31) | (0b0110101 << 24) | rt.index();
    inst_fixup("cbnz", template, label, fixup_aarch64())
}

// ── System instructions ───────────────────────────────────────────────────────

/// BRK #imm16  (breakpoint / trap)
#[inline(always)]
pub fn brk(imm16: u16) -> Instruction<AArch64Inst> {
    let bits = (0b11010100_001 << 21) | ((imm16 as u32) << 5) | 0b00000;
    inst("brk", bits)
}

/// UDF — permanently undefined instruction (generates SIGILL)
#[inline(always)]
pub fn udf(imm16: u16) -> Instruction<AArch64Inst> {
    let bits = (imm16 as u32) & 0xFFFF;
    inst("udf", bits)
}

/// NOP
#[inline(always)]
pub fn nop() -> Instruction<AArch64Inst> {
    inst("nop", 0xD503201F)
}

/// ADR Xd, label  (PC-relative address, +/- 1MB)
#[inline(always)]
pub fn adr(rd: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let immlo = ((offset as u32) & 0x3) << 29;
    let immhi = (((offset >> 2) as u32) & 0x7FFFF) << 5;
    let bits = immlo | (0b10000 << 24) | immhi | rd.index();
    inst("adr", bits)
}

/// ADR Xd, label (forward reference)
#[inline(always)]
pub fn adr_label(rd: Reg, label: Label) -> Instruction<AArch64Inst> {
    let template = (0b10000 << 24) | rd.index();
    inst_fixup("adr", template, label, fixup_aarch64())
}

/// ADRP Xd, label  (PC-relative page address)
#[inline(always)]
pub fn adrp(rd: Reg, offset: i32) -> Instruction<AArch64Inst> {
    let page_offset = offset >> 12;
    let immlo = ((page_offset as u32) & 0x3) << 29;
    let immhi = (((page_offset >> 2) as u32) & 0x7FFFF) << 5;
    let bits = (1 << 31) | immlo | (0b10000 << 24) | immhi | rd.index();
    inst("adrp", bits)
}

/// MOV Xd, Xn  (alias for ADD Xd, Xn, #0 — used for sp moves)
/// Use this when one of the operands is sp, since ORR-based mov doesn't work with sp.
#[inline(always)]
pub fn mov_sp(size: RegSize, rd: Reg, rn: Reg) -> Instruction<AArch64Inst> {
    add_imm(size, rd, rn, 0)
}

// ── Immediate loading helpers ─────────────────────────────────────────────────

/// Load a 32-bit immediate into a W-register using MOVZ + MOVK.
/// Returns 1 or 2 instructions worth of bytes packed, but since we need
/// to call `push` for each instruction, we provide separate helpers.
///
/// For the compiler, use `load_imm32` and `load_imm64` which handle the
/// multi-instruction sequence.

/// Count of MOVZ/MOVK instructions needed for a 64-bit immediate.
pub fn count_imm64_instructions(value: u64) -> u32 {
    if value == 0 { return 1; }
    let not_value = !value;

    // Check if it can be a single MOVZ
    let chunks = [
        (value & 0xFFFF) as u16,
        ((value >> 16) & 0xFFFF) as u16,
        ((value >> 32) & 0xFFFF) as u16,
        ((value >> 48) & 0xFFFF) as u16,
    ];

    let nonzero_chunks = chunks.iter().filter(|&&c| c != 0).count() as u32;
    let not_chunks = [
        (not_value & 0xFFFF) as u16,
        ((not_value >> 16) & 0xFFFF) as u16,
        ((not_value >> 32) & 0xFFFF) as u16,
        ((not_value >> 48) & 0xFFFF) as u16,
    ];
    let not_nonzero_chunks = not_chunks.iter().filter(|&&c| c != 0).count() as u32;

    // Use MOVZ + MOVK for non-zero chunks, or MOVN + MOVK for inverted
    core::cmp::min(nonzero_chunks, not_nonzero_chunks + 1).max(1)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn encode_to_u32(inst: Instruction<AArch64Inst>) -> u32 {
        let bytes = inst.bytes.to_vec();
        assert_eq!(bytes.len(), 4);
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[test]
    fn test_nop() {
        assert_eq!(encode_to_u32(nop()), 0xD503201F);
    }

    #[test]
    fn test_brk() {
        // BRK #0 = 0xD4200000
        assert_eq!(encode_to_u32(brk(0)), 0xD4200000);
        // BRK #1 = 0xD4200020
        assert_eq!(encode_to_u32(brk(1)), 0xD4200020);
    }

    #[test]
    fn test_ret() {
        // RET (x30) = 0xD65F03C0
        assert_eq!(encode_to_u32(ret_lr()), 0xD65F03C0);
    }

    #[test]
    fn test_add_reg() {
        // ADD X0, X1, X2 = 0x8B020020
        assert_eq!(encode_to_u32(add(RegSize::X64, x0, x1, x2)), 0x8B020020);
        // ADD W0, W1, W2 = 0x0B020020
        assert_eq!(encode_to_u32(add(RegSize::W32, x0, x1, x2)), 0x0B020020);
    }

    #[test]
    fn test_sub_reg() {
        // SUB X0, X1, X2 = 0xCB020020
        assert_eq!(encode_to_u32(sub(RegSize::X64, x0, x1, x2)), 0xCB020020);
    }

    #[test]
    fn test_mov_reg() {
        // MOV X0, X1 (= ORR X0, XZR, X1) = 0xAA0103E0
        assert_eq!(encode_to_u32(mov_reg(RegSize::X64, x0, x1)), 0xAA0103E0);
    }

    #[test]
    fn test_movz() {
        // MOVZ X0, #0x1234 = 0xD2824680
        assert_eq!(encode_to_u32(movz(RegSize::X64, x0, 0x1234, 0)), 0xD2824680);
        // MOVZ X0, #0x5678, LSL #16 = 0xD2AACF00
        assert_eq!(encode_to_u32(movz(RegSize::X64, x0, 0x5678, 16)), 0xD2AACF00);
    }

    #[test]
    fn test_cmp() {
        // CMP X1, X2 (= SUBS XZR, X1, X2) = 0xEB02003F
        assert_eq!(encode_to_u32(cmp(RegSize::X64, x1, x2)), 0xEB02003F);
    }

    #[test]
    fn test_b_cond() {
        // B.EQ #8 (offset=8, imm19=2) = 0x54000040
        assert_eq!(encode_to_u32(b_cond(Condition::EQ, 8)), 0x54000040);
    }

    #[test]
    fn test_br() {
        // BR X16 = 0xD61F0200
        assert_eq!(encode_to_u32(br(x16)), 0xD61F0200);
    }

    #[test]
    fn test_blr() {
        // BLR X16 = 0xD63F0200
        assert_eq!(encode_to_u32(blr(x16)), 0xD63F0200);
    }

    #[test]
    fn test_sdiv() {
        // SDIV W0, W1, W2 = 0x1AC20C20
        assert_eq!(encode_to_u32(sdiv(RegSize::W32, x0, x1, x2)), 0x1AC20C20);
    }

    #[test]
    fn test_udiv() {
        // UDIV W0, W1, W2 = 0x1AC20820
        assert_eq!(encode_to_u32(udiv(RegSize::W32, x0, x1, x2)), 0x1AC20820);
    }

    #[test]
    fn test_mul() {
        // MUL W0, W1, W2 (= MADD W0, W1, W2, WZR) = 0x1B027C20
        assert_eq!(encode_to_u32(mul(RegSize::W32, x0, x1, x2)), 0x1B027C20);
    }

    #[test]
    fn test_and() {
        // AND X0, X1, X2 = 0x8A020020
        assert_eq!(encode_to_u32(and(RegSize::X64, x0, x1, x2)), 0x8A020020);
    }

    #[test]
    fn test_orr() {
        // ORR X0, X1, X2 = 0xAA020020
        assert_eq!(encode_to_u32(orr(RegSize::X64, x0, x1, x2)), 0xAA020020);
    }

    #[test]
    fn test_eor() {
        // EOR X0, X1, X2 = 0xCA020020
        assert_eq!(encode_to_u32(eor(RegSize::X64, x0, x1, x2)), 0xCA020020);
    }

    #[test]
    fn test_lsl_lsr_asr() {
        // LSL X0, X1, X2 = 0x9AC22020
        assert_eq!(encode_to_u32(lsl(RegSize::X64, x0, x1, x2)), 0x9AC22020);
        // LSR X0, X1, X2 = 0x9AC22420
        assert_eq!(encode_to_u32(lsr(RegSize::X64, x0, x1, x2)), 0x9AC22420);
        // ASR X0, X1, X2 = 0x9AC22820
        assert_eq!(encode_to_u32(asr(RegSize::X64, x0, x1, x2)), 0x9AC22820);
    }

    #[test]
    fn test_cbz_cbnz() {
        // CBZ X0, #12 = 0xB4000060
        assert_eq!(encode_to_u32(cbz(RegSize::X64, x0, 12)), 0xB4000060);
        // CBNZ W0, #8 = 0x35000040
        assert_eq!(encode_to_u32(cbnz(RegSize::W32, x0, 8)), 0x35000040);
    }

    #[test]
    fn test_clz() {
        // CLZ X0, X1 = 0xDAC01020
        assert_eq!(encode_to_u32(clz(RegSize::X64, x0, x1)), 0xDAC01020);
        // CLZ W0, W1 = 0x5AC01020
        assert_eq!(encode_to_u32(clz(RegSize::W32, x0, x1)), 0x5AC01020);
    }

    #[test]
    fn test_rbit() {
        // RBIT X0, X1 = 0xDAC00020
        assert_eq!(encode_to_u32(rbit(RegSize::X64, x0, x1)), 0xDAC00020);
    }

    #[test]
    fn test_rev() {
        // REV X0, X1 = 0xDAC00C20
        assert_eq!(encode_to_u32(rev(RegSize::X64, x0, x1)), 0xDAC00C20);
        // REV W0, W1 = 0x5AC00820
        assert_eq!(encode_to_u32(rev(RegSize::W32, x0, x1)), 0x5AC00820);
    }

    #[test]
    fn test_csel() {
        // CSEL X0, X1, X2, EQ = 0x9A820020
        assert_eq!(encode_to_u32(csel(RegSize::X64, x0, x1, x2, Condition::EQ)), 0x9A820020);
    }

    #[test]
    fn test_cset() {
        // CSET X0, EQ (= CSINC X0, XZR, XZR, NE) = 0x9A9F17E0
        assert_eq!(encode_to_u32(cset(RegSize::X64, x0, Condition::EQ)), 0x9A9F17E0);
    }

    #[test]
    fn test_ldr_str_imm() {
        // LDR X0, [X1, #8] = 0xF9400420
        assert_eq!(encode_to_u32(ldr_imm(MemSize::B64, x0, x1, 8)), 0xF9400420);
        // STR X0, [X1, #16] = 0xF9000820
        assert_eq!(encode_to_u32(str_imm(MemSize::B64, x0, x1, 16)), 0xF9000820);
        // LDR W0, [X1, #4] = 0xB9400420
        assert_eq!(encode_to_u32(ldr_imm(MemSize::B32, x0, x1, 4)), 0xB9400420);
        // LDRB W0, [X1, #0] = 0x39400020
        assert_eq!(encode_to_u32(ldr_imm(MemSize::B8, x0, x1, 0)), 0x39400020);
    }

    #[test]
    fn test_stp_ldp() {
        // STP X29, X30, [SP, #-16]! (pre-index)
        let encoded = encode_to_u32(stp_pre(MemSize::B64, fp, lr, sp, -16));
        assert_eq!(encoded, 0xA9BF7BFD);
    }

    #[test]
    fn test_add_imm() {
        // ADD X0, X1, #42 = 0x9100A820
        assert_eq!(encode_to_u32(add_imm(RegSize::X64, x0, x1, 42)), 0x9100A820);
    }

    #[test]
    fn test_sub_imm() {
        // SUB SP, SP, #16 = 0xD10043FF
        assert_eq!(encode_to_u32(sub_imm(RegSize::X64, sp, sp, 16)), 0xD10043FF);
    }

    #[test]
    fn test_mvn() {
        // MVN X0, X1 (= ORN X0, XZR, X1) = 0xAA2103E0
        assert_eq!(encode_to_u32(mvn(RegSize::X64, x0, x1)), 0xAA2103E0);
    }

    #[test]
    fn test_bic() {
        // BIC X0, X1, X2 = 0x8A220020
        assert_eq!(encode_to_u32(bic(RegSize::X64, x0, x1, x2)), 0x8A220020);
    }

    #[test]
    fn test_condition_invert() {
        assert_eq!(Condition::EQ.invert() as u8, Condition::NE as u8);
        assert_eq!(Condition::LT.invert() as u8, Condition::GE as u8);
        assert_eq!(Condition::HI.invert() as u8, Condition::LS as u8);
    }

    #[test]
    fn test_logical_imm_encoding() {
        // 0xFF = 8 consecutive 1s in a 64-bit pattern
        let enc = encode_logical_imm(RegSize::X64, 0xFF);
        assert!(enc.is_some(), "should encode 0xFF");

        // 0 and all-1s should not be encodable
        assert!(encode_logical_imm(RegSize::X64, 0).is_none());
        assert!(encode_logical_imm(RegSize::X64, !0u64).is_none());
        assert!(encode_logical_imm(RegSize::W32, 0).is_none());
        assert!(encode_logical_imm(RegSize::W32, 0xFFFFFFFF).is_none());
    }
}
