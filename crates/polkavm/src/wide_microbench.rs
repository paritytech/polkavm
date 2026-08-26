//! Microbenchmarks for the XReviveVec wide instructions, each compared against the scalar limb
//! chain a base-ISA build would emit for the same operation. Run at both 128 and 256 bits.
//!
//! These are `#[ignore]`d: they print a table rather than asserting, and are meant to be run
//! explicitly, e.g. `cargo test -p polkavm --lib wide_microbench --release -- --ignored --nocapture`.
//!
//! Methodology. The extension keeps a wide value in a wide register, so its form of an operation is
//! one instruction over register operands. The reference keeps the value in memory -- thirteen
//! general purpose registers cannot hold several wide values plus temporaries, so a base-ISA build
//! spills, exactly as the book's "`add i256` is 30 instructions" figure reflects -- so its form
//! loads the limbs, computes with carries, and stores the result. Both are the authentic shape of
//! their respective builds. The body is unrolled (as gastool does) to amortize the call, and a
//! zero-body baseline is subtracted to remove call and setup overhead. Correctness is checked by
//! running the extension and the reference on the same inputs and comparing results, so a wrong
//! reference is caught rather than flattering the comparison.

use crate::{BackendKind, Config, Engine, Linker, Module, ProgramBlob};
use polkavm_common::program::Instruction;
use polkavm_common::program::{asm, InstructionSetKind, Reg, Reg::*, WideReg, WideWidth};
use polkavm_common::writer::ProgramBlobBuilder;

/// Unroll and call counts for cheap, constant-time operations.
const LIGHT_UNROLL: usize = 2_000;
const LIGHT_CALLS: u64 = 100;
/// Unroll and call counts for the iterative operations (division, exp, the fused modular forms),
/// each of which runs a per-bit number of internal iterations per instruction.
const HEAVY_UNROLL: usize = 40;
const HEAVY_CALLS: u64 = 20;
/// Best-of, to shed scheduling noise.
const TRIALS: usize = 5;

/// A fixed shift amount (< 64, so it stays within one limb) used by the shift benchmarks.
const SHIFT_AMOUNT: i32 = 37;

#[derive(Clone, Copy, PartialEq)]
enum Weight {
    Light,
    Heavy,
}

impl Weight {
    fn reps(self) -> (usize, u64) {
        match self {
            Weight::Light => (LIGHT_UNROLL, LIGHT_CALLS),
            Weight::Heavy => (HEAVY_UNROLL, HEAVY_CALLS),
        }
    }
}

/// Byte offsets of the two operands and the result within the read-write data region. Spaced for a
/// 256-bit value; a 128-bit value uses the low half of each slot.
const A_OFF: i32 = 0;
const B_OFF: i32 = 32;
const R_OFF: i32 = 64;
const RW_SIZE: u32 = 0x1000;

/// Two arbitrary but fixed operands, little-endian limbs. A 128-bit run uses the low two.
const OPERAND_A: [u64; 4] = [0x1234_5678_9abc_def0, 0x0f0e_0d0c_0b0a_0908, 0xdead_beef_feed_face, 0x0011_2233_4455_6677];
const OPERAND_B: [u64; 4] = [0xfedc_ba98_7654_3210, 0x1122_3344_5566_7788, 0x0000_0000_0000_002b, 0x7fff_ffff_ffff_ffff];

/// `base` holds the address of the read-write region throughout the reference bodies.
const BASE: Reg = S0;

fn limbs_of(width: WideWidth) -> usize {
    match width {
        WideWidth::W128 => 2,
        WideWidth::W256 => 4,
        WideWidth::W512 => 8,
        WideWidth::W1024 => 16,
    }
}

/// A wide operation to benchmark: its one-instruction extension form, an optional scalar limb chain
/// computing the same thing, and how to read the extension's result into `A0` for the check.
struct Op {
    name: &'static str,
    weight: Weight,
    ext_body: fn(width: WideWidth, d: WideReg, a: WideReg, b: WideReg) -> Vec<Instruction>,
    ref_body: Option<fn(limbs: usize) -> Vec<Instruction>>,
    ext_result_to_a0: fn(width: WideWidth, d: WideReg) -> Vec<Instruction>,
}

// --- extension result extractors ---

fn ext_low_to_a0(width: WideWidth, d: WideReg) -> Vec<Instruction> {
    vec![asm::wide_truncate(width, A0, d)]
}

fn ext_already_in_a0(_width: WideWidth, _d: WideReg) -> Vec<Instruction> {
    // Compares and truncate write their result straight to A0.
    vec![]
}

fn ext_read_result_memory(_width: WideWidth, _d: WideReg) -> Vec<Instruction> {
    // A store leaves its result in memory; read the low limb back.
    vec![asm::load_indirect_u64(A0, BASE, R_OFF)]
}

// --- scalar reference bodies (all over memory operands, limb by limb) ---

fn scalar_bitwise(kind: Bitwise, limbs: usize) -> Vec<Instruction> {
    let mut code = Vec::new();
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::load_indirect_u64(A2, BASE, B_OFF + i * 8));
        code.push(match kind {
            Bitwise::And => asm::and(A3, A1, A2),
            Bitwise::Or => asm::or(A3, A1, A2),
            Bitwise::Xor => asm::xor(A3, A1, A2),
        });
        code.push(asm::store_indirect_u64(A3, BASE, R_OFF + i * 8));
    }
    code
}

#[derive(Clone, Copy)]
enum Bitwise {
    And,
    Or,
    Xor,
}

fn scalar_and(limbs: usize) -> Vec<Instruction> {
    scalar_bitwise(Bitwise::And, limbs)
}
fn scalar_or(limbs: usize) -> Vec<Instruction> {
    scalar_bitwise(Bitwise::Or, limbs)
}
fn scalar_xor(limbs: usize) -> Vec<Instruction> {
    scalar_bitwise(Bitwise::Xor, limbs)
}

fn scalar_add(limbs: usize) -> Vec<Instruction> {
    // Wide add with a carry threaded through `S1`, RISC-V style (no carry flag; `sltu` recovers it).
    let mut code = vec![asm::load_imm(S1, 0)];
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::load_indirect_u64(A2, BASE, B_OFF + i * 8));
        code.push(asm::add_64(A3, A1, A2));
        code.push(asm::set_less_than_unsigned(A4, A3, A1)); // carry from a+b
        code.push(asm::add_64(A3, A3, S1));
        code.push(asm::set_less_than_unsigned(A5, A3, S1)); // carry from +carry
        code.push(asm::or(S1, A4, A5));
        code.push(asm::store_indirect_u64(A3, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_sub(limbs: usize) -> Vec<Instruction> {
    // Wide subtract with a borrow threaded through `S1`.
    let mut code = vec![asm::load_imm(S1, 0)];
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::load_indirect_u64(A2, BASE, B_OFF + i * 8));
        code.push(asm::set_less_than_unsigned(A4, A1, A2)); // borrow from a-b
        code.push(asm::sub_64(A3, A1, A2));
        code.push(asm::set_less_than_unsigned(A5, A3, S1)); // borrow from -borrow
        code.push(asm::sub_64(A3, A3, S1));
        code.push(asm::or(S1, A4, A5));
        code.push(asm::store_indirect_u64(A3, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_mul(limbs: usize) -> Vec<Instruction> {
    // Low `limbs`-worth of the product, schoolbook: result[k] accumulates a[i]*b[j] for i+j=k, the
    // high halves carried into the next limb. Only i+j < limbs contributes to the low half.
    let mut code = Vec::new();
    code.push(asm::load_imm(A5, 0));
    for k in 0..limbs as i32 {
        code.push(asm::store_indirect_u64(A5, BASE, R_OFF + k * 8));
    }
    for i in 0..limbs {
        for j in 0..(limbs - i) {
            let k = (i + j) as i32;
            code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i as i32 * 8));
            code.push(asm::load_indirect_u64(A2, BASE, B_OFF + j as i32 * 8));
            code.push(asm::mul_64(A3, A1, A2)); // low half of the partial product
            code.push(asm::load_indirect_u64(A4, BASE, R_OFF + k * 8));
            code.push(asm::add_64(A4, A4, A3));
            code.push(asm::set_less_than_unsigned(A5, A4, A3)); // carry
            code.push(asm::store_indirect_u64(A4, BASE, R_OFF + k * 8));
            if k + 1 < limbs as i32 {
                code.push(asm::mul_upper_unsigned_unsigned(A3, A1, A2)); // high half
                code.push(asm::add_64(A3, A3, A5));
                code.push(asm::load_indirect_u64(A4, BASE, R_OFF + (k + 1) * 8));
                code.push(asm::add_64(A4, A4, A3));
                code.push(asm::store_indirect_u64(A4, BASE, R_OFF + (k + 1) * 8));
            }
        }
    }
    code
}

fn scalar_set_less_than_unsigned(limbs: usize) -> Vec<Instruction> {
    // Wide unsigned compare from the top limb down; the first differing limb decides it. `S1` tracks
    // whether a decision has been made, so the result is branchless. Result (0/1) into R[0].
    let mut code = vec![asm::load_imm(A5, 0), asm::load_imm(S1, 0)];
    for i in (0..limbs as i32).rev() {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::load_indirect_u64(A2, BASE, B_OFF + i * 8));
        code.push(asm::set_less_than_unsigned(A3, A1, A2)); // a<b at this limb
        code.push(asm::set_less_than_unsigned(A4, A2, A1)); // a>b at this limb
        code.push(asm::or(A0, A3, A4)); // this limb decides
        code.push(asm::load_imm(A1, 1));
        code.push(asm::sub_64(A1, A1, S1)); // undecided = 1 - S1
        code.push(asm::and(A0, A0, A1)); // decides now = decides && undecided
        code.push(asm::cmov_if_not_zero(A5, A3, A0)); // if so, take a<b
        code.push(asm::or(S1, S1, A0)); // now decided
    }
    code.push(asm::store_indirect_u64(A5, BASE, R_OFF));
    code
}

fn scalar_set_equal(limbs: usize) -> Vec<Instruction> {
    // eq = (all limbs equal). Fold the differences into A4; the result is (A4 == 0).
    let mut code = vec![asm::load_imm(A4, 0), asm::load_imm(S1, 0)];
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::load_indirect_u64(A2, BASE, B_OFF + i * 8));
        code.push(asm::xor(A3, A1, A2));
        code.push(asm::or(A4, A4, A3));
    }
    code.push(asm::set_less_than_unsigned(A5, S1, A4)); // A5 = (0 < diff) = not-equal
    code.push(asm::load_imm(A2, 1));
    code.push(asm::xor(A5, A5, A2)); // equal = !not-equal
    code.push(asm::store_indirect_u64(A5, BASE, R_OFF));
    code
}

fn scalar_move(limbs: usize) -> Vec<Instruction> {
    let mut code = Vec::new();
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::store_indirect_u64(A1, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_byte_swap(limbs: usize) -> Vec<Instruction> {
    // Reverse the byte order of the whole value: R[i] = reverse_bytes(A[limbs-1-i]).
    let mut code = Vec::new();
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + (limbs as i32 - 1 - i) * 8));
        code.push(asm::reverse_byte(A2, A1));
        code.push(asm::store_indirect_u64(A2, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_shift_left(limbs: usize) -> Vec<Instruction> {
    // Left shift by SHIFT_AMOUNT (< 64): each limb takes its own high bits plus the low bits carried
    // up from the limb below.
    let low = SHIFT_AMOUNT;
    let high = 64 - SHIFT_AMOUNT;
    let mut code = Vec::new();
    for i in 0..limbs as i32 {
        code.push(asm::load_indirect_u64(A1, BASE, A_OFF + i * 8));
        code.push(asm::shift_logical_left_imm_64(A3, A1, low));
        if i >= 1 {
            code.push(asm::load_indirect_u64(A2, BASE, A_OFF + (i - 1) * 8));
            code.push(asm::shift_logical_right_imm_64(A4, A2, high));
            code.push(asm::or(A3, A3, A4));
        }
        code.push(asm::store_indirect_u64(A3, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_widen_unsigned(limbs: usize) -> Vec<Instruction> {
    // Zero-extend the low limb (loaded from A) into the wide result.
    let mut code = vec![asm::load_indirect_u64(A1, BASE, A_OFF)];
    code.push(asm::store_indirect_u64(A1, BASE, R_OFF));
    code.push(asm::load_imm(A2, 0));
    for i in 1..limbs as i32 {
        code.push(asm::store_indirect_u64(A2, BASE, R_OFF + i * 8));
    }
    code
}

fn scalar_truncate(_limbs: usize) -> Vec<Instruction> {
    // The low limb is the whole result.
    vec![
        asm::load_indirect_u64(A1, BASE, A_OFF),
        asm::store_indirect_u64(A1, BASE, R_OFF),
    ]
}

// --- the operation table ---

fn ops() -> Vec<Op> {
    vec![
        // Linear, one pass over the limbs.
        Op { name: "add", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_add(w, d, a, b)], ref_body: Some(scalar_add), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "sub", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_sub(w, d, a, b)], ref_body: Some(scalar_sub), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "and", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_and(w, d, a, b)], ref_body: Some(scalar_and), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "or", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_or(w, d, a, b)], ref_body: Some(scalar_or), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "xor", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_xor(w, d, a, b)], ref_body: Some(scalar_xor), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "mul", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_mul(w, d, a, b)], ref_body: Some(scalar_mul), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "move", weight: Weight::Light, ext_body: |w, d, a, _b| vec![asm::wide_move(w, d, a, a)], ref_body: Some(scalar_move), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "bswap", weight: Weight::Light, ext_body: |w, d, a, _b| vec![asm::wide_byte_swap(w, d, a, a)], ref_body: Some(scalar_byte_swap), ext_result_to_a0: ext_low_to_a0 },
        // Shifts: the amount comes from a general purpose register (A4, set in the setup).
        Op { name: "shl", weight: Weight::Light, ext_body: |w, d, a, _b| vec![asm::wide_shift_left(w, d, a, A4.into())], ref_body: Some(scalar_shift_left), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "shr_l", weight: Weight::Light, ext_body: |w, d, a, _b| vec![asm::wide_shift_right_logical(w, d, a, A4.into())], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "shr_a", weight: Weight::Light, ext_body: |w, d, a, _b| vec![asm::wide_shift_right_arithmetic(w, d, a, A4.into())], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        // Compares write their 0/1 result to a general purpose register.
        Op { name: "slt_u", weight: Weight::Light, ext_body: |w, _d, a, b| vec![asm::wide_set_less_than_unsigned(w, A0, a, b)], ref_body: Some(scalar_set_less_than_unsigned), ext_result_to_a0: ext_already_in_a0 },
        Op { name: "slt_s", weight: Weight::Light, ext_body: |w, _d, a, b| vec![asm::wide_set_less_than_signed(w, A0, a, b)], ref_body: None, ext_result_to_a0: ext_already_in_a0 },
        Op { name: "seq", weight: Weight::Light, ext_body: |w, _d, a, b| vec![asm::wide_set_equal(w, A0, a, b)], ref_body: Some(scalar_set_equal), ext_result_to_a0: ext_already_in_a0 },
        Op { name: "sne", weight: Weight::Light, ext_body: |w, _d, a, b| vec![asm::wide_set_not_equal(w, A0, a, b)], ref_body: None, ext_result_to_a0: ext_already_in_a0 },
        // Min/max: a compare and a select of the limbs (reference not reproduced).
        Op { name: "min_u", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_min_unsigned(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "min_s", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_min_signed(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "max_u", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_max_unsigned(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "max_s", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_max_signed(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        // Width conversions: widen reads a general purpose register (A1, set in the setup).
        Op { name: "zext", weight: Weight::Light, ext_body: |w, d, _a, _b| vec![asm::wide_widen_unsigned(w, d, A1.into())], ref_body: Some(scalar_widen_unsigned), ext_result_to_a0: ext_low_to_a0 },
        Op { name: "sext_w", weight: Weight::Light, ext_body: |w, d, _a, _b| vec![asm::wide_widen_signed(w, d, A1.into())], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "trunc", weight: Weight::Light, ext_body: |w, _d, a, _b| vec![asm::wide_truncate(w, A0, a)], ref_body: Some(scalar_truncate), ext_result_to_a0: ext_already_in_a0 },
        // EVM sign-extend from a byte index (index in b).
        Op { name: "signext", weight: Weight::Light, ext_body: |w, d, a, b| vec![asm::wide_sign_extend(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        // Memory: a wide load or store of the whole value, base in `BASE`.
        Op { name: "load", weight: Weight::Light, ext_body: |w, d, _a, _b| vec![asm::wide_load(w, d, BASE.into(), A_OFF)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "store", weight: Weight::Light, ext_body: |w, _d, a, _b| vec![asm::wide_store(w, a, BASE.into(), R_OFF)], ref_body: None, ext_result_to_a0: ext_read_result_memory },
        // Iterative: a per-bit number of internal steps. No short scalar reference.
        Op { name: "div_u", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_div_unsigned(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "div_s", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_div_signed(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "rem_u", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_rem_unsigned(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "rem_s", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_rem_signed(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "exp", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_exp(w, d, a, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "addmod", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_add_mod(w, d, a, b, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
        Op { name: "mulmod", weight: Weight::Heavy, ext_body: |w, d, a, b| vec![asm::wide_mul_mod(w, d, a, b, b)], ref_body: None, ext_result_to_a0: ext_low_to_a0 },
    ]
}

// --- blob construction and measurement ---

fn ext_blob(op: &Op, width: WideWidth, unroll: usize) -> ProgramBlob {
    let (d, a, b) = (WideReg::from_raw(0, width), WideReg::from_raw(8, width), WideReg::from_raw(16, width));
    let mut code = vec![
        asm::load_imm(BASE, rw_address() as i32),
        asm::wide_load(width, a, BASE.into(), A_OFF),
        asm::wide_load(width, b, BASE.into(), B_OFF),
        asm::load_indirect_u64(A1, BASE, A_OFF), // a general purpose copy of the low limb, for widen
        asm::load_imm(A4, SHIFT_AMOUNT),          // shift amount, for the shifts
    ];
    for _ in 0..unroll {
        code.extend((op.ext_body)(width, d, a, b));
    }
    code.extend((op.ext_result_to_a0)(width, d));
    code.push(asm::ret());
    blob_from(code)
}

fn ref_blob(op: &Op, width: WideWidth, unroll: usize) -> ProgramBlob {
    let body = op.ref_body.expect("only called for ops with a reference");
    let limbs = limbs_of(width);
    let mut code = vec![asm::load_imm(BASE, rw_address() as i32)];
    for _ in 0..unroll {
        code.extend(body(limbs));
    }
    code.push(asm::load_indirect_u64(A0, BASE, R_OFF));
    code.push(asm::ret());
    blob_from(code)
}

fn rw_address() -> u32 {
    polkavm_common::abi::MemoryMapBuilder::new(0x4000)
        .rw_data_size(RW_SIZE)
        .build()
        .unwrap()
        .rw_data_address()
}

fn blob_from(code: Vec<Instruction>) -> ProgramBlob {
    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::ReviveV2);
    builder.set_rw_data_size(RW_SIZE);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
}

fn engine_for(backend: BackendKind) -> Engine {
    let mut config = Config::default();
    config.set_backend(Some(backend));
    config.set_allow_experimental(true);
    Engine::new(&config).unwrap()
}

fn write_operands(instance: &mut crate::Instance<(), ()>) {
    let base = rw_address();
    for i in 0..4u32 {
        instance.write_u64(base + A_OFF as u32 + i * 8, OPERAND_A[i as usize]).unwrap();
        instance.write_u64(base + B_OFF as u32 + i * 8, OPERAND_B[i as usize]).unwrap();
    }
}

fn run_get_a0(backend: BackendKind, blob: ProgramBlob) -> u64 {
    let engine = engine_for(backend);
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
    let linker: Linker<(), ()> = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();
    write_operands(&mut instance);
    instance.call_typed_and_get_result::<u64, ()>(&mut (), "main", ()).unwrap()
}

fn time_blob(backend: BackendKind, blob: ProgramBlob, calls: u64) -> core::time::Duration {
    let engine = engine_for(backend);
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
    let linker: Linker<(), ()> = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();
    write_operands(&mut instance);

    let mut best = core::time::Duration::MAX;
    for _ in 0..TRIALS {
        let start = std::time::Instant::now();
        for _ in 0..calls {
            instance.call_typed(&mut (), "main", ()).unwrap();
        }
        best = best.min(start.elapsed());
    }
    best
}

fn ns_per_op(total: core::time::Duration, baseline_per_call: f64, unroll: usize, calls: u64) -> f64 {
    let net = (total.as_nanos() as f64 - baseline_per_call * calls as f64).max(0.0);
    net / (unroll as f64 * calls as f64)
}

fn baseline_blob() -> ProgramBlob {
    blob_from(vec![asm::load_imm(BASE, rw_address() as i32), asm::ret()])
}

fn measure(backend: BackendKind, width: WideWidth) {
    let baseline_calls = 1_000u64;
    let baseline_per_call = time_blob(backend, baseline_blob(), baseline_calls).as_nanos() as f64 / baseline_calls as f64;

    let bits = limbs_of(width) * 64;
    println!();
    println!("=== wide microbenchmark: {backend:?}, {bits}-bit (ns per operation) ===");
    println!("{:<8} {:>12} {:>12} {:>10}  {}", "op", "extension", "reference", "speedup", "correct");
    for op in ops() {
        let (unroll, calls) = op.weight.reps();
        let ext = time_blob(backend, ext_blob(&op, width, unroll), calls);
        let ext_ns = ns_per_op(ext, baseline_per_call, unroll, calls);

        if op.ref_body.is_some() {
            let reference = time_blob(backend, ref_blob(&op, width, unroll), calls);
            let ref_ns = ns_per_op(reference, baseline_per_call, unroll, calls);

            let ext_a0 = run_get_a0(backend, ext_blob(&op, width, 1));
            let ref_a0 = run_get_a0(backend, ref_blob(&op, width, 1));
            let ok = if ext_a0 == ref_a0 { "yes" } else { "MISMATCH" };

            println!("{:<8} {:>12.2} {:>12.2} {:>9.2}x  {}", op.name, ext_ns, ref_ns, ref_ns / ext_ns, ok);
        } else {
            println!("{:<8} {:>12.2} {:>12} {:>10}  {}", op.name, ext_ns, "-", "-", "(ext only)");
        }
    }
}

fn measure_all(backend: BackendKind) {
    if backend == BackendKind::Compiler && !BackendKind::Compiler.is_supported() {
        return;
    }
    measure(backend, WideWidth::W128);
    measure(backend, WideWidth::W256);
}

#[test]
#[ignore]
fn wide_microbench_interpreter() {
    measure_all(BackendKind::Interpreter);
}

#[test]
#[ignore]
fn wide_microbench_compiler() {
    measure_all(BackendKind::Compiler);
}
