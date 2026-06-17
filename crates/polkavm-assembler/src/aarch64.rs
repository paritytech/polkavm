//! AArch64 (A64) instruction encoder; ARM counterpart to [`crate::amd64`].
//! Each builder emits one little-endian `u32`; branches carry a fixup for the shared assembler to patch.

#![allow(non_camel_case_types)]

use crate::misc::{FixupKind, InstBuf, Instruction, Label};

/// A GPR (`x0`..`x30`) or the zero register (`xzr`, encoding `31`). `sp` (also `31`) is never emitted here.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Reg {
    X0 = 0,
    X1 = 1,
    X2 = 2,
    X3 = 3,
    X4 = 4,
    X5 = 5,
    X6 = 6,
    X7 = 7,
    X8 = 8,
    X9 = 9,
    X10 = 10,
    X11 = 11,
    X12 = 12,
    X13 = 13,
    X14 = 14,
    X15 = 15,
    X16 = 16,
    X17 = 17,
    X18 = 18,
    X19 = 19,
    X20 = 20,
    X21 = 21,
    X22 = 22,
    X23 = 23,
    X24 = 24,
    X25 = 25,
    X26 = 26,
    X27 = 27,
    X28 = 28,
    X29 = 29,
    X30 = 30,
    XZR = 31,
}

impl Reg {
    #[inline]
    pub const fn idx(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn equals(self, other: Self) -> bool {
        (self as u32) == (other as u32)
    }
}

impl core::fmt::Display for Reg {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        if matches!(self, Reg::XZR) {
            fmt.write_str("xzr")
        } else {
            fmt.write_fmt(core::format_args!("x{}", self.idx()))
        }
    }
}

/// Operation width: `W` is 32-bit, `X` is 64-bit. Selects the `sf` bit.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RegSize {
    W = 0,
    X = 1,
}

impl RegSize {
    #[inline]
    pub const fn sf(self) -> u32 {
        self as u32
    }
}

/// Memory access width for loads and stores. The value is the A64 `size` field.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Size {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3,
}

impl Size {
    #[inline]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Condition codes for `B.cond` and conditional selects.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Cond {
    Eq = 0,
    Ne = 1,
    Cs = 2, // unsigned >=  (HS)
    Cc = 3, // unsigned <   (LO)
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8, // unsigned >
    Ls = 9, // unsigned <=
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
    Al = 14,
}

impl Cond {
    #[inline]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    /// The inverted condition (e.g. used to branch *over* a long unconditional jump).
    #[inline]
    pub const fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Cs => Cond::Cc,
            Cond::Cc => Cond::Cs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Al,
        }
    }
}

impl core::fmt::Display for Cond {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        let name = match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Cs => "cs",
            Cond::Cc => "cc",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
        };
        fmt.write_str(name)
    }
}

/// Logical shift amount for shifted-register data-processing instructions.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

impl Shift {
    #[inline]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Generates an instruction type with `encode`/`fixup`/`Display` and a constructor; `encode` yields the 32-bit word.
macro_rules! a64_inst {
    ($(
        $name:ident ( $($an:ident : $at:ty),* $(,)? ) => $enc:expr $(, fixup = $fix:expr)? ;
    )+) => {
        pub(crate) mod types {
            use super::*;
            $(
                #[derive(Copy, Clone, PartialEq, Eq, Debug)]
                pub struct $name { $(pub $an: $at),* }

                impl core::fmt::Display for $name {
                    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
                        // Trace-only; the Debug form is good enough for disassembly logging.
                        fmt.write_fmt(core::format_args!("{:?}", self))
                    }
                }

                impl $name {
                    #[inline]
                    pub fn encode(self) -> InstBuf {
                        #[allow(unused_variables)]
                        let $name { $($an),* } = self;
                        let word: u32 = $enc;
                        let mut buf = InstBuf::new();
                        buf.append_packed_bytes(word, 32);
                        buf
                    }

                    #[inline]
                    #[allow(unused_variables)]
                    pub(crate) fn fixup(self) -> Option<(Label, FixupKind)> {
                        let $name { $($an),* } = self;
                        a64_inst!(@fixup $($fix)?)
                    }
                }
            )+
        }

        $(
            #[inline]
            pub fn $name($($an: $at),*) -> Instruction<types::$name> {
                let instruction = types::$name { $($an),* };
                Instruction {
                    instruction,
                    bytes: instruction.encode(),
                    fixup: instruction.fixup(),
                }
            }
        )+
    };

    (@fixup) => { None };
    (@fixup $e:expr) => { $e };
}

a64_inst! {
    // ---- Move wide immediate (MOVN/MOVZ/MOVK) ----
    // sf | opc(2) | 100101 | hw(2) | imm16 | Rd
    movn(size: RegSize, rd: Reg, imm16: u16, hw: u32) =>
        (size.sf() << 31) | (0b100101 << 23) | (hw << 21) | ((u32::from(imm16)) << 5) | rd.idx();
    movz(size: RegSize, rd: Reg, imm16: u16, hw: u32) =>
        (size.sf() << 31) | (0b10 << 29) | (0b100101 << 23) | (hw << 21) | ((u32::from(imm16)) << 5) | rd.idx();
    movk(size: RegSize, rd: Reg, imm16: u16, hw: u32) =>
        (size.sf() << 31) | (0b11 << 29) | (0b100101 << 23) | (hw << 21) | ((u32::from(imm16)) << 5) | rd.idx();

    // ---- Add/Subtract (immediate) ----
    // sf | op | S | 100010 | sh | imm12 | Rn | Rd
    add_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32, shift12: bool) =>
        (size.sf() << 31) | (0b100010 << 23) | ((u32::from(shift12)) << 22) | ((imm12 & 0xfff) << 10) | (rn.idx() << 5) | rd.idx();
    sub_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32, shift12: bool) =>
        (size.sf() << 31) | (1 << 30) | (0b100010 << 23) | ((u32::from(shift12)) << 22) | ((imm12 & 0xfff) << 10) | (rn.idx() << 5) | rd.idx();
    subs_imm(size: RegSize, rd: Reg, rn: Reg, imm12: u32, shift12: bool) =>
        (size.sf() << 31) | (1 << 30) | (1 << 29) | (0b100010 << 23) | ((u32::from(shift12)) << 22) | ((imm12 & 0xfff) << 10) | (rn.idx() << 5) | rd.idx();

    // ---- Add/Subtract (shifted register), shift amount 0 ----
    // sf | op | S | 01011 | shift(2) | 0 | Rm | imm6 | Rn | Rd
    add_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b01011 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    sub_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (1 << 30) | (0b01011 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    subs_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (1 << 30) | (1 << 29) | (0b01011 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();

    // ---- Logical (shifted register), shift amount 0, N=0 ----
    // sf | opc(2) | 01010 | shift(2) | N | Rm | imm6 | Rn | Rd
    and_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b01010 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    orr_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b01 << 29) | (0b01010 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    eor_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b10 << 29) | (0b01010 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();

    // ---- Load/Store register (unsigned immediate offset), offset scaled by access size ----
    // size(2) | 111 | 0 | 01 | opc(2) | imm12 | Rn | Rt
    str_imm(size: Size, rt: Reg, rn: Reg, imm12_scaled: u32) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 24) | ((imm12_scaled & 0xfff) << 10) | (rn.idx() << 5) | rt.idx();
    ldr_imm(size: Size, rt: Reg, rn: Reg, imm12_scaled: u32) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 24) | (0b01 << 22) | ((imm12_scaled & 0xfff) << 10) | (rn.idx() << 5) | rt.idx();
    // Sign-extending load into a 64-bit register (opc = 10).
    ldrs64_imm(size: Size, rt: Reg, rn: Reg, imm12_scaled: u32) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 24) | (0b10 << 22) | ((imm12_scaled & 0xfff) << 10) | (rn.idx() << 5) | rt.idx();

    // ---- Load/Store register (unscaled signed 9-bit immediate): LDUR/STUR ----
    // size(2) | 111 | 0 | 00 | opc(2) | 0 | imm9 | 00 | Rn | Rt
    stur(size: Size, rt: Reg, rn: Reg, imm9: i32) =>
        (size.raw() << 30) | (0b111 << 27) | (((imm9 as u32) & 0x1ff) << 12) | (rn.idx() << 5) | rt.idx();
    ldur(size: Size, rt: Reg, rn: Reg, imm9: i32) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 22) | (((imm9 as u32) & 0x1ff) << 12) | (rn.idx() << 5) | rt.idx();

    // ---- Unconditional branch (immediate) ----
    // op | 00101 | imm26
    b(label: Label) => 0b000101 << 26, fixup = Some((label, FixupKind::aarch64_branch(0, 26)));
    bl(label: Label) => 0b100101 << 26, fixup = Some((label, FixupKind::aarch64_branch(0, 26)));

    // ---- Conditional branch (immediate) ----
    // 0101010 0 | imm19 | 0 | cond
    b_cond(cond: Cond, label: Label) => (0b01010100 << 24) | cond.raw(), fixup = Some((label, FixupKind::aarch64_branch(5, 19)));

    // ---- Compare and branch ----
    // sf | 011010 | op | imm19 | Rt
    cbz(size: RegSize, rt: Reg, label: Label) =>
        (size.sf() << 31) | (0b011010 << 25) | rt.idx(), fixup = Some((label, FixupKind::aarch64_branch(5, 19)));
    cbnz(size: RegSize, rt: Reg, label: Label) =>
        (size.sf() << 31) | (0b011010 << 25) | (1 << 24) | rt.idx(), fixup = Some((label, FixupKind::aarch64_branch(5, 19)));

    // ---- PC-relative address ----
    // 0 immlo(2) 10000 immhi(19) Rd ; 21-bit signed byte offset relative to this instruction.
    adr(rd: Reg, label: Label) => 0x1000_0000 | rd.idx(), fixup = Some((label, FixupKind::aarch64_adr()));

    // ---- Unconditional branch (register) ----
    br(rn: Reg) => 0xD61F0000 | (rn.idx() << 5);
    blr(rn: Reg) => 0xD63F0000 | (rn.idx() << 5);
    ret(rn: Reg) => 0xD65F0000 | (rn.idx() << 5);

    // ---- Logical (shifted register), inverted forms (N=1), shift amount 0 ----
    bic_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b01010 << 24) | (1 << 21) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    orn_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b01 << 29) | (0b01010 << 24) | (1 << 21) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    eon_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b10 << 29) | (0b01010 << 24) | (1 << 21) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();
    ands_reg(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b11 << 29) | (0b01010 << 24) | (rm.idx() << 16) | (rn.idx() << 5) | rd.idx();

    // ---- Multiply (3-source) ----
    // sf 00 11011 000 Rm o0 Ra Rn Rd  (MADD o0=0, MSUB o0=1)
    madd(size: RegSize, rd: Reg, rn: Reg, rm: Reg, ra: Reg) =>
        (size.sf() << 31) | (0b0011011 << 24) | (rm.idx() << 16) | (ra.idx() << 10) | (rn.idx() << 5) | rd.idx();
    msub(size: RegSize, rd: Reg, rn: Reg, rm: Reg, ra: Reg) =>
        (size.sf() << 31) | (0b0011011 << 24) | (rm.idx() << 16) | (1 << 15) | (ra.idx() << 10) | (rn.idx() << 5) | rd.idx();
    // SMULH / UMULH (64-bit high half), Ra = xzr
    smulh(rd: Reg, rn: Reg, rm: Reg) =>
        0x9B40_0000 | (rm.idx() << 16) | (0b11111 << 10) | (rn.idx() << 5) | rd.idx();
    umulh(rd: Reg, rn: Reg, rm: Reg) =>
        0x9BC0_0000 | (rm.idx() << 16) | (0b11111 << 10) | (rn.idx() << 5) | rd.idx();

    // ---- Data-processing (2-source): divide and variable shift ----
    // sf 0 0 11010110 Rm opcode(6) Rn Rd
    udiv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b000010 << 10) | (rn.idx() << 5) | rd.idx();
    sdiv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b000011 << 10) | (rn.idx() << 5) | rd.idx();
    lslv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b001000 << 10) | (rn.idx() << 5) | rd.idx();
    lsrv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b001001 << 10) | (rn.idx() << 5) | rd.idx();
    asrv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b001010 << 10) | (rn.idx() << 5) | rd.idx();
    rorv(size: RegSize, rd: Reg, rn: Reg, rm: Reg) =>
        (size.sf() << 31) | (0b0011010110 << 21) | (rm.idx() << 16) | (0b001011 << 10) | (rn.idx() << 5) | rd.idx();

    // ---- Data-processing (1-source): bit counting / reversal ----
    // Base opcodes (32-bit); the `sf` bit selects the 64-bit form.
    rbit(size: RegSize, rd: Reg, rn: Reg) =>
        0x5AC0_0000 | (size.sf() << 31) | (rn.idx() << 5) | rd.idx();
    clz(size: RegSize, rd: Reg, rn: Reg) =>
        0x5AC0_1000 | (size.sf() << 31) | (rn.idx() << 5) | rd.idx();
    rev16(size: RegSize, rd: Reg, rn: Reg) =>
        0x5AC0_0400 | (size.sf() << 31) | (rn.idx() << 5) | rd.idx();
    // REV: 32-bit form reverses the 4 bytes of a word; 64-bit form (opc=0b11) reverses all 8 bytes.
    rev32_word(rd: Reg, rn: Reg) => 0x5AC0_0800 | (rn.idx() << 5) | rd.idx();
    rev64(rd: Reg, rn: Reg) => 0xDAC0_0C00 | (rn.idx() << 5) | rd.idx();

    // ---- Bitfield move (SBFM / UBFM): used for shifts-by-immediate and sign/zero extension ----
    // sf opc(2) 100110 N immr(6) imms(6) Rn Rd ; for 64-bit N=1, for 32-bit N=0
    sbfm(size: RegSize, rd: Reg, rn: Reg, immr: u32, imms: u32) =>
        (size.sf() << 31) | (0b100110 << 23) | (size.sf() << 22) | ((immr & 0x3f) << 16) | ((imms & 0x3f) << 10) | (rn.idx() << 5) | rd.idx();
    ubfm(size: RegSize, rd: Reg, rn: Reg, immr: u32, imms: u32) =>
        (size.sf() << 31) | (0b10 << 29) | (0b100110 << 23) | (size.sf() << 22) | ((immr & 0x3f) << 16) | ((imms & 0x3f) << 10) | (rn.idx() << 5) | rd.idx();

    // ---- Conditional select ----
    // sf 0 0 11010100 Rm cond op2(2) Rn Rd
    csel(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Cond) =>
        (size.sf() << 31) | (0b0011010100 << 21) | (rm.idx() << 16) | (cond.raw() << 12) | (rn.idx() << 5) | rd.idx();
    csinc(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Cond) =>
        (size.sf() << 31) | (0b0011010100 << 21) | (rm.idx() << 16) | (cond.raw() << 12) | (0b01 << 10) | (rn.idx() << 5) | rd.idx();
    // CSINV: rd = cond ? rn : ~rm
    csinv(size: RegSize, rd: Reg, rn: Reg, rm: Reg, cond: Cond) =>
        0x5A80_0000 | (size.sf() << 31) | (rm.idx() << 16) | (cond.raw() << 12) | (rn.idx() << 5) | rd.idx();

    // ---- Load/Store register (register offset), option = LSL/UXTX (#0) ----
    // size 111 0 00 opc 1 Rm option(3)=011 S=0 10 Rn Rt
    str_reg(size: Size, rt: Reg, rn: Reg, rm: Reg) =>
        (size.raw() << 30) | (0b111 << 27) | (1 << 21) | (rm.idx() << 16) | (0b011 << 13) | (0b10 << 10) | (rn.idx() << 5) | rt.idx();
    ldr_reg(size: Size, rt: Reg, rn: Reg, rm: Reg) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 22) | (1 << 21) | (rm.idx() << 16) | (0b011 << 13) | (0b10 << 10) | (rn.idx() << 5) | rt.idx();
    // Sign-extending load (unsigned immediate offset) to a 32-bit register (opc = 11).
    ldrs32_imm(size: Size, rt: Reg, rn: Reg, imm12_scaled: u32) =>
        (size.raw() << 30) | (0b111 << 27) | (0b01 << 24) | (0b11 << 22) | ((imm12_scaled & 0xfff) << 10) | (rn.idx() << 5) | rt.idx();
    // Sign-extending load (register offset) into a 64-bit register (opc = 10).
    ldrs64_reg(size: Size, rt: Reg, rn: Reg, rm: Reg) =>
        (size.raw() << 30) | (0b111 << 27) | (0b10 << 22) | (1 << 21) | (rm.idx() << 16) | (0b011 << 13) | (0b10 << 10) | (rn.idx() << 5) | rt.idx();

    // ---- NEON: used for popcount (cnt). SIMD registers are passed as raw indices 0..31. ----
    fmov_gpr_to_s(sd: u32, wn: Reg) => 0x1E27_0000 | (wn.idx() << 5) | sd; // 32-bit GPR -> SIMD
    fmov_gpr_to_d(dd: u32, xn: Reg) => 0x9E67_0000 | (xn.idx() << 5) | dd; // 64-bit GPR -> SIMD
    fmov_s_to_gpr(wd: Reg, sn: u32) => 0x1E26_0000 | (sn << 5) | wd.idx(); // SIMD -> 32-bit GPR
    cnt_8b(vd: u32, vn: u32) => 0x0E20_5800 | (vn << 5) | vd; // per-byte popcount of Vn.8B
    addv_8b(bd: u32, vn: u32) => 0x0E31_B800 | (vn << 5) | bd; // sum the 8 bytes of Vn.8B

    // ---- Exceptions / hint ----
    // UDF #imm16 — permanently undefined; raises an Undefined Instruction exception (SIGILL).
    udf(imm16: u16) => u32::from(imm16);
    // BRK #imm16 — software breakpoint (SIGTRAP).
    brk(imm16: u16) => 0xD4200000 | ((u32::from(imm16)) << 5);
    nop() => 0xD503201F;
}

/// `mov rd, rm` — encoded as `orr rd, xzr, rm`.
#[inline]
pub fn mov_reg(size: RegSize, rd: Reg, rm: Reg) -> Instruction<types::orr_reg> {
    orr_reg(size, rd, Reg::XZR, rm)
}

/// `cmp rn, rm` — encoded as `subs xzr, rn, rm`.
#[inline]
pub fn cmp_reg(size: RegSize, rn: Reg, rm: Reg) -> Instruction<types::subs_reg> {
    subs_reg(size, Reg::XZR, rn, rm)
}

/// `cmp rn, #imm12` — encoded as `subs xzr, rn, #imm12`.
#[inline]
pub fn cmp_imm(size: RegSize, rn: Reg, imm12: u32, shift12: bool) -> Instruction<types::subs_imm> {
    subs_imm(size, Reg::XZR, rn, imm12, shift12)
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn word<T: core::fmt::Display>(inst: Instruction<T>) -> u32 {
        let bytes = inst.bytes.to_vec();
        assert_eq!(bytes.len(), 4, "every A64 instruction must be 4 bytes");
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[test]
    fn encodings_match_known_good() {
        // Cross-checked against `llvm-mc -arch=aarch64 --show-encoding`.
        assert_eq!(word(movz(RegSize::X, Reg::X0, 0x1234, 0)), 0xD282_4680);
        assert_eq!(word(movk(RegSize::X, Reg::X0, 0xABCD, 1)), 0xF2B5_79A0);
        assert_eq!(word(movz(RegSize::W, Reg::X1, 0, 0)), 0x5280_0001);
        assert_eq!(word(add_imm(RegSize::X, Reg::X0, Reg::X1, 0x10, false)), 0x9100_4020);
        assert_eq!(word(sub_imm(RegSize::X, Reg::X2, Reg::X3, 1, false)), 0xD100_0462);
        assert_eq!(word(add_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x8B02_0020);
        assert_eq!(word(add_reg(RegSize::W, Reg::X0, Reg::X1, Reg::X2)), 0x0B02_0020);
        assert_eq!(word(sub_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0xCB02_0020);
        assert_eq!(word(orr_reg(RegSize::X, Reg::X0, Reg::XZR, Reg::X1)), 0xAA01_03E0); // mov x0, x1
        assert_eq!(word(and_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x8A02_0020);
        assert_eq!(word(eor_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0xCA02_0020);
        assert_eq!(word(subs_reg(RegSize::X, Reg::XZR, Reg::X1, Reg::X2)), 0xEB02_003F); // cmp x1, x2
        assert_eq!(word(ldr_imm(Size::U64, Reg::X0, Reg::X1, 1)), 0xF940_0420); // ldr x0,[x1,#8]
        assert_eq!(word(str_imm(Size::U32, Reg::X0, Reg::X1, 2)), 0xB900_0820); // str w0,[x1,#8]
        assert_eq!(word(ldur(Size::U64, Reg::X0, Reg::X1, -8)), 0xF85F_8020);
        assert_eq!(word(br(Reg::X8)), 0xD61F_0100);
        assert_eq!(word(ret(Reg::X30)), 0xD65F_03C0);
        assert_eq!(word(udf(0)), 0x0000_0000);
        assert_eq!(word(brk(0)), 0xD420_0000);
        assert_eq!(word(nop()), 0xD503_201F);
    }

    #[test]
    fn extended_encodings_match_known_good() {
        // mul/div/shift/bit/select/extend, cross-checked against `llvm-mc -arch=aarch64`.
        assert_eq!(word(madd(RegSize::X, Reg::X0, Reg::X1, Reg::X2, Reg::XZR)), 0x9B02_7C20); // mul x0,x1,x2
        assert_eq!(word(udiv(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x9AC2_0820);
        assert_eq!(word(sdiv(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x9AC2_0C20);
        assert_eq!(word(lslv(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x9AC2_2020);
        assert_eq!(word(asrv(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x9AC2_2820);
        assert_eq!(word(smulh(Reg::X0, Reg::X1, Reg::X2)), 0x9B42_7C20);
        assert_eq!(word(umulh(Reg::X0, Reg::X1, Reg::X2)), 0x9BC2_7C20);
        assert_eq!(word(clz(RegSize::X, Reg::X0, Reg::X1)), 0xDAC0_1020);
        assert_eq!(word(rbit(RegSize::X, Reg::X0, Reg::X1)), 0xDAC0_0020);
        assert_eq!(word(csel(RegSize::X, Reg::X0, Reg::X1, Reg::X2, Cond::Eq)), 0x9A82_0020);
        assert_eq!(word(csinc(RegSize::X, Reg::X0, Reg::X1, Reg::X2, Cond::Ne)), 0x9A82_1420);
        assert_eq!(word(sbfm(RegSize::X, Reg::X0, Reg::X1, 0, 7)), 0x9340_1C20); // sxtb x0,x1
        assert_eq!(word(ubfm(RegSize::X, Reg::X0, Reg::X1, 1, 63)), 0xD341_FC20); // lsr x0,x1,#1
        assert_eq!(word(bic_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0x8A22_0020);
        assert_eq!(word(orn_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0xAA22_0020);
        assert_eq!(word(ands_reg(RegSize::X, Reg::X0, Reg::X1, Reg::X2)), 0xEA02_0020);
        assert_eq!(word(ldr_reg(Size::U64, Reg::X0, Reg::X1, Reg::X2)), 0xF862_6820);
        assert_eq!(word(str_reg(Size::U64, Reg::X0, Reg::X1, Reg::X2)), 0xF822_6820);
        assert_eq!(word(fmov_gpr_to_s(0, Reg::X1)), 0x1E27_0020);
        assert_eq!(word(fmov_gpr_to_d(0, Reg::X1)), 0x9E67_0020);
        assert_eq!(word(fmov_s_to_gpr(Reg::X0, 1)), 0x1E26_0020);
        assert_eq!(word(cnt_8b(0, 1)), 0x0E20_5820);
        assert_eq!(word(addv_8b(0, 1)), 0x0E31_B820);
    }

    #[test]
    fn branch_fixups_are_patched() {
        let mut asm = crate::Assembler::new();
        // Forward conditional branch over one instruction.
        let target = asm.forward_declare_label();
        asm.push(b_cond(Cond::Eq, target));
        asm.push(nop());
        asm.define_label(target);
        asm.push(ret(Reg::X30));
        let code = asm.finalize();
        let b0 = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // imm19 should be +2 words (skip the b.cond itself and the nop).
        let imm19 = (b0 >> 5) & 0x7ffff;
        assert_eq!(imm19, 2);
        assert_eq!(b0 & 0xf, Cond::Eq.raw());
    }

    #[test]
    fn adr_fixup_is_patched() {
        let mut asm = crate::Assembler::new();
        // adr x0, target ; nop ; nop ; target:
        let target = asm.forward_declare_label();
        asm.push(adr(Reg::X0, target));
        asm.push(nop());
        asm.push(nop());
        asm.define_label(target);
        let code = asm.finalize();
        let w = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // Offset is +12 bytes (3 instructions). immlo = 12 & 3 = 0; immhi = 12 >> 2 = 3.
        let immlo = (w >> 29) & 0b11;
        let immhi = (w >> 5) & 0x7_ffff;
        assert_eq!(immlo, 0);
        assert_eq!(immhi, 3);
        assert_eq!(w & 0x1f, 0); // Rd = x0
    }

    #[test]
    fn backward_branch_is_negative() {
        let mut asm = crate::Assembler::new();
        let head = asm.create_label();
        asm.push(nop());
        asm.push(b(head));
        let code = asm.finalize();
        let b1 = u32::from_le_bytes([code[4], code[5], code[6], code[7]]);
        let imm26 = b1 & 0x03ff_ffff;
        // -1 word in 26-bit two's complement.
        assert_eq!(imm26, 0x03ff_ffff);
    }
}
