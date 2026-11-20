#![deny(unused_must_use)]

use core::ops::Range;
use std::path::{Path, PathBuf};

use polkavm_common::cast::cast;
use polkavm_linker::riscv::{BranchKind, Inst, RegImmKind, RegRegKind, StoreKind};
use riscv_test_interpreter::{Exception, Interpreter, MemoryProtection, Reg, StopReason};

const RV64_CATEGORIES: &[&str] = &["rv64ua", "rv64uc", "rv64ui", "rv64um", "rv64uzbb"];
const REGISTERS_RANGE: Range<u32> = 0..32;
const END_OF_32BIT_ADDRESS_SPACE: u64 = 1 << 32;
const CODE_ADDRESS: u64 = 0x10000;
const _: () = assert!(CODE_ADDRESS < END_OF_32BIT_ADDRESS_SPACE);
const DATA_ADDRESS: u64 = 0x20000;
const _: () = assert!(DATA_ADDRESS < END_OF_32BIT_ADDRESS_SPACE);
const STOP_ADDRESS: u64 = 0xffff_0000;
const _: () = assert!(STOP_ADDRESS < END_OF_32BIT_ADDRESS_SPACE);
const REGION_ADDRESS: u64 = 0x1000;
const REGION_SIZE: u64 = 0x100;
const CORPUS_TEXT_ADDRESS: u64 = 0x10000;
const UPPER_32_BITS_SET: u64 = 0xffff_ffff_0000_0000;
const LOWEST_UPPER_BIT_SET: u64 = 1 << 32;
const SIGN_BIT_SET: u64 = 1 << 63;
const UPPER_IMMEDIATE_SIGN_BIT_SET: u32 = 1 << 19;
const ENOUGH_STEPS: u64 = 100;
const EXECUTE: MemoryProtection = MemoryProtection {
    read: false,
    write: false,
    execute: true,
};
const READ_WRITE: MemoryProtection = MemoryProtection {
    read: true,
    write: true,
    execute: false,
};
const READ_EXECUTE: MemoryProtection = MemoryProtection {
    read: true,
    write: false,
    execute: true,
};
const READ_ONLY: MemoryProtection = MemoryProtection {
    read: true,
    write: false,
    execute: false,
};
const INSTRUCTION_BYTE_LENGTH: u64 = 4;
const COMPRESSED_INSTRUCTION_BYTE_LENGTH: u64 = 2;
const NOP: u32 = i_type(Addi, Reg::Zero, Reg::Zero, 0);
const RET: u32 = i_type(Jalr, Reg::Zero, Reg::RA, 0);

/// The matching instruction word, with its opcode and funct fields
/// filled in and every operand field zero.
#[derive(Copy, Clone)]
#[repr(u32)]
enum MatchWord {
    Add = 0x0000_0033,
    Addi = 0x0000_0013,
    Slt = 0x0000_2033,
    Slti = 0x0000_2013,
    Sltu = 0x0000_3033,
    Sltiu = 0x0000_3013,
    Sll = 0x0000_1033,
    Slli = 0x0000_1013,
    Srl = 0x0000_5033,
    Srli = 0x0000_5013,
    Sra = 0x4000_5033,
    Srai = 0x4000_5013,
    Xor = 0x0000_4033,
    Xori = 0x0000_4013,
    Or = 0x0000_6033,
    Ori = 0x0000_6013,
    And = 0x0000_7033,
    Andi = 0x0000_7013,
    Mulw = 0x0200_003b,
    Mulh = 0x0200_1033,
    Mulhsu = 0x0200_2033,
    Mulhu = 0x0200_3033,
    Divw = 0x0200_403b,
    Div = 0x0200_4033,
    Divuw = 0x0200_503b,
    Divu = 0x0200_5033,
    Lui = 0x0000_0037,
    Auipc = 0x0000_0017,
    Jal = 0x0000_006f,
    Jalr = 0x0000_0067,
    Beq = 0x0000_0063,
    Bne = 0x0000_1063,
    Blt = 0x0000_4063,
    Bltu = 0x0000_6063,
    Bge = 0x0000_5063,
    Bgeu = 0x0000_7063,
    Lw = 0x0000_2003,
    Ld = 0x0000_3003,
    Sw = 0x0000_2023,
    Sd = 0x0000_3023,
}

use MatchWord::*;

fn map_region_with_data(interpreter: &mut Interpreter, address: u64, data: &[u8], protection: MemoryProtection) {
    interpreter.map_region(address, data.len() as u64, protection);
    interpreter.write_memory(address, data).unwrap();
}

fn memory_layout(interpreter: &Interpreter) -> Vec<(u64, u64, MemoryProtection)> {
    interpreter
        .regions()
        .iter()
        .map(|region| (region.address(), region.size(), region.protection()))
        .collect()
}

struct RunSetup<'a> {
    registers: &'a [(Reg, u64)],
    data: &'a [u8],
    max_steps: u64,
}

/// Calls [`run_bytes`] after converting the `code` instruction words to little-endian bytes.
fn run(code: &[u32], setup: RunSetup) -> (StopReason, Interpreter) {
    let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    run_bytes(&bytes, setup)
}

/// Runs `code` for at most `setup.max_steps` steps on a new interpreter.
/// The `code` is mapped to an execute-only memory region starting at [`CODE_ADDRESS`],
/// which the program counter is set to, and `setup.data` to a read-write memory region
/// starting at [`DATA_ADDRESS`]. The stop address and [`Reg::RA`] are set to [`STOP_ADDRESS`],
/// so that if `code` ends with a [`RET`] instruction it returns to the stop address and
/// reports [`StopReason::Finished`]. The `setup.registers` are applied and take precedence.
fn run_bytes(code: &[u8], setup: RunSetup) -> (StopReason, Interpreter) {
    let mut interpreter = Interpreter::new();
    map_region_with_data(&mut interpreter, CODE_ADDRESS, code, EXECUTE);
    map_region_with_data(&mut interpreter, DATA_ADDRESS, setup.data, READ_WRITE);
    interpreter.set_program_counter(CODE_ADDRESS);
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_reg(Reg::RA, STOP_ADDRESS);
    for &(reg, value) in setup.registers {
        interpreter.set_reg(reg, value);
        assert_eq!(interpreter.reg(reg), value);
    }

    let stop_reason = interpreter.run(setup.max_steps);

    (stop_reason, interpreter)
}

fn corpus_output_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../guest-programs/riscv-tests/output")
}

fn corpus_elfs(category: &str) -> Vec<(String, Vec<u8>)> {
    let directory = corpus_output_directory().join(category);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "elf"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());

    paths
        .into_iter()
        .map(|path| {
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&path).unwrap(),
            )
        })
        .collect()
}

fn run_corpus(category: &str) {
    const CORPUS_MAX_STEPS: u64 = 100_000;
    let mut failures = Vec::new();
    for (name, elf) in corpus_elfs(category) {
        let mut interpreter = Interpreter::new();
        interpreter.load_elf(&elf).unwrap();
        interpreter.set_stop_address(STOP_ADDRESS);
        interpreter.set_reg(Reg::RA, STOP_ADDRESS);

        let stop_reason = interpreter.run(CORPUS_MAX_STEPS);
        if stop_reason != StopReason::Finished {
            failures.push(describe_failure(&name, stop_reason, &interpreter));
        }
    }
    assert!(failures.is_empty(), "unexpected failure(s):\n{}", failures.join("\n"));
}

fn describe_failure(name: &str, stop_reason: StopReason, interpreter: &Interpreter) -> String {
    let registers: Vec<String> = REGISTERS_RANGE
        .map(Reg::decode)
        .map(|reg| format!("{}={:#x}", reg.name(), interpreter.reg(reg)))
        .collect();

    format!(
        "{name}: {stop_reason:?} at {:#x}. Registers: {}",
        interpreter.program_counter(),
        registers.join(" ")
    )
}

const fn immediate_bits(value: i32, low_bit: u32, high_bit: u32) -> u32 {
    let mask = (1 << (high_bit - low_bit + 1)) - 1;

    ((value as u32) >> low_bit) & mask
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn r_type(match_word: MatchWord, dst: Reg, src1: Reg, src2: Reg) -> u32 {
    match_word as u32 | (dst as u32) << 7 | (src1 as u32) << 15 | (src2 as u32) << 20
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn i_type(match_word: MatchWord, dst: Reg, src: Reg, imm: i32) -> u32 {
    match_word as u32 | (dst as u32) << 7 | (src as u32) << 15 | immediate_bits(imm, 0, 11) << 20
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn s_type(match_word: MatchWord, src: Reg, base: Reg, offset: i32) -> u32 {
    match_word as u32 | immediate_bits(offset, 0, 4) << 7 | (base as u32) << 15 | (src as u32) << 20 | immediate_bits(offset, 5, 11) << 25
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn b_type(match_word: MatchWord, src1: Reg, src2: Reg, offset: i32) -> u32 {
    match_word as u32
        | immediate_bits(offset, 11, 11) << 7
        | immediate_bits(offset, 1, 4) << 8
        | (src1 as u32) << 15
        | (src2 as u32) << 20
        | immediate_bits(offset, 5, 10) << 25
        | immediate_bits(offset, 12, 12) << 31
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn u_type(match_word: MatchWord, dst: Reg, upper_imm: u32) -> u32 {
    match_word as u32 | (dst as u32) << 7 | upper_imm << 12
}

/// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-3-immediate-encoding-variants
const fn j_type(match_word: MatchWord, dst: Reg, offset: i32) -> u32 {
    match_word as u32
        | (dst as u32) << 7
        | immediate_bits(offset, 12, 19) << 12
        | immediate_bits(offset, 11, 11) << 20
        | immediate_bits(offset, 1, 10) << 21
        | immediate_bits(offset, 20, 20) << 31
}

#[test]
fn format_helpers_encode_instructions() {
    // `expected_encoded` from `llvm-mc -triple=riscv64 -show-encoding` (little-endian order).
    for (encoded, expected_encoded, expected_decoded) in [
        (
            // `add a0, a1, a2`: three distinct registers, so swapped register fields change the word.
            r_type(Add, Reg::A0, Reg::A1, Reg::A2),
            0x00c58533,
            Inst::RegReg {
                kind: RegRegKind::Add64,
                dst: Reg::A0,
                src1: Reg::A1,
                src2: Reg::A2,
            },
        ),
        (
            // `addi a0, a1, -1`: every immediate bit set, so an incorrect unset bit in any of them changes the word.
            i_type(Addi, Reg::A0, Reg::A1, !0),
            0xfff58513,
            Inst::RegImm {
                kind: RegImmKind::Add64,
                dst: Reg::A0,
                src: Reg::A1,
                imm: !0,
            },
        ),
        (
            // `addi a0, a1, -2`: only immediate bit 0 unset, so `imm[11:0]` differs from incorrect adjacent `imm[12:1]`.
            i_type(Addi, Reg::A0, Reg::A1, !1),
            0xffe58513,
            Inst::RegImm {
                kind: RegImmKind::Add64,
                dst: Reg::A0,
                src: Reg::A1,
                imm: !1,
            },
        ),
        (
            // `sw a0, -1(a1)`: every immediate bit set, so an incorrect unset bit in any of them changes the word.
            s_type(Sw, Reg::A0, Reg::A1, !0),
            0xfea5afa3,
            Inst::Store {
                kind: StoreKind::U32,
                src: Reg::A0,
                base: Reg::A1,
                offset: !0,
            },
        ),
        (
            // `sw a0, -33(a1)`: only immediate bit 5 unset, so `imm[11:5]` differs from incorrect adjacent
            // `imm[12:6]` and `imm[10:4]`, and `imm[4:0]` from adjacent `imm[5:1]`.
            s_type(Sw, Reg::A0, Reg::A1, !(1 << 5)),
            0xfca5afa3,
            Inst::Store {
                kind: StoreKind::U32,
                src: Reg::A0,
                base: Reg::A1,
                offset: !(1 << 5),
            },
        ),
        (
            // `beq a0, a1, -2`: only immediate bit 0 unset, so an incorrect unset bit in any of the others changes the word.
            b_type(Beq, Reg::A0, Reg::A1, !1),
            0xfeb50fe3,
            Inst::Branch {
                kind: BranchKind::Eq64,
                src1: Reg::A0,
                src2: Reg::A1,
                target: cast(!1).bitwise_as_u32(),
            },
        ),
        (
            // `beq a0, a1, -2082`: only immediate bits 11, 5, and 0 unset, so `imm[11]` differs from incorrect `imm[12]`,
            // `imm[10:5]` from adjacent `imm[11:6]` and `imm[9:4]`, and `imm[4:1]` from adjacent `imm[5:2]` and `imm[3:0]`.
            b_type(Beq, Reg::A0, Reg::A1, !(1 << 11 | 1 << 5 | 1)),
            0xfcb50f63,
            Inst::Branch {
                kind: BranchKind::Eq64,
                src1: Reg::A0,
                src2: Reg::A1,
                target: cast(!(1 << 11 | 1 << 5 | 1)).bitwise_as_u32(),
            },
        ),
        (
            // `lui a0, 0x80000`: only the sign bit set, so an incorrect shift or mask changes the word.
            u_type(Lui, Reg::A0, UPPER_IMMEDIATE_SIGN_BIT_SET),
            0x80000537,
            Inst::LoadUpperImmediate {
                dst: Reg::A0,
                value: UPPER_IMMEDIATE_SIGN_BIT_SET << 12,
            },
        ),
        (
            // `jal ra, -2`: only immediate bit 0 unset, so an incorrect unset bit in any of the others changes the word.
            j_type(Jal, Reg::RA, !1),
            0xfffff0ef,
            Inst::JumpAndLink {
                dst: Reg::RA,
                target: cast(!1).bitwise_as_u32(),
            },
        ),
        (
            // `jal ra, -6146`: only immediate bits 12, 11, and 0 unset, so `imm[19:12]` differs from incorrect adjacent
            // `imm[20:13]` and `imm[18:11]`, `imm[11]` from `imm[20]`, and `imm[10:1]` from adjacent `imm[11:2]` and `imm[9:0]`.
            j_type(Jal, Reg::RA, !(1 << 12 | 1 << 11 | 1)),
            0xffefe0ef,
            Inst::JumpAndLink {
                dst: Reg::RA,
                target: cast(!(1 << 12 | 1 << 11 | 1)).bitwise_as_u32(),
            },
        ),
    ] {
        assert_eq!(
            encoded, expected_encoded,
            "{expected_decoded:?} should encode as {expected_encoded:#010x}, but was {encoded:#010x}"
        );
        assert_eq!(Inst::decode(encoded), Some(expected_decoded));
    }
}

#[test]
fn registers_are_zero_at_construction() {
    let interpreter = Interpreter::new();
    for index in REGISTERS_RANGE {
        assert_eq!(interpreter.reg(Reg::decode(index)), 0);
    }
}

#[test]
fn set_reg_round_trips() {
    let mut interpreter = Interpreter::new();
    interpreter.set_reg(Reg::A0, 0xdead_beef_0000_0001);
    assert_eq!(interpreter.reg(Reg::A0), 0xdead_beef_0000_0001);
    assert_eq!(interpreter.reg(Reg::A1), 0);
}

#[test]
fn writes_to_zero_register_are_noop() {
    let mut interpreter = Interpreter::new();
    interpreter.set_reg(Reg::Zero, 5);
    assert_eq!(interpreter.reg(Reg::Zero), 0);
}

#[test]
fn maps_regions_to_sorted_addresses() {
    let mut interpreter = Interpreter::new();
    let regions = [
        (REGION_ADDRESS + REGION_SIZE * 4, 10),
        (REGION_ADDRESS, 11),
        (REGION_ADDRESS + REGION_SIZE, 12),
        (REGION_ADDRESS + REGION_SIZE * 3, 13),
    ];
    for (address, repeated_data) in regions {
        map_region_with_data(&mut interpreter, address, &[repeated_data; REGION_SIZE as usize], READ_WRITE);
    }

    for (address, repeated_data) in regions {
        assert_eq!(
            interpreter.read_memory(address, REGION_SIZE),
            Some(&[repeated_data; REGION_SIZE as usize][..])
        );
    }
    assert_eq!(interpreter.read_memory(REGION_ADDRESS + REGION_SIZE * 2, 1), None);

    let mut expected_layout = regions.map(|(address, _)| (address, REGION_SIZE, READ_WRITE));
    expected_layout.sort_by_key(|(address, _, _)| *address);
    assert_eq!(memory_layout(&interpreter), expected_layout);
}

#[test]
#[should_panic]
fn region_cannot_overlap_at_start() {
    let mut interpreter = Interpreter::new();
    interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
    interpreter.map_region(REGION_ADDRESS + REGION_SIZE - 1, 2, READ_WRITE);
}

#[test]
#[should_panic]
fn region_cannot_overlap_at_end() {
    let mut interpreter = Interpreter::new();
    interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
    interpreter.map_region(REGION_ADDRESS - REGION_SIZE + 1, REGION_SIZE, READ_WRITE);
}

#[test]
#[should_panic]
fn region_cannot_overlap_fully() {
    let mut interpreter = Interpreter::new();
    interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
    interpreter.map_region(REGION_ADDRESS - REGION_SIZE, 3 * REGION_SIZE, READ_WRITE);
}

#[test]
fn accesses_memory_inside_mapped_region() {
    let mut interpreter = Interpreter::new();
    let data: Vec<u8> = (0..=u8::MAX).collect();
    map_region_with_data(&mut interpreter, REGION_ADDRESS, &data, READ_WRITE);

    assert_eq!(interpreter.read_memory(REGION_ADDRESS, data.len() as u64), Some(&data[..]));
    assert_eq!(interpreter.read_memory(REGION_ADDRESS + 0x10, 4), Some(&data[0x10..0x14]));
    assert_eq!(interpreter.read_memory(REGION_ADDRESS + 0xff, 1), Some(&data[0xff..]));

    assert!(interpreter.write_memory(REGION_ADDRESS + 0x10, &[0xaa; 4]).is_some());
    let mut expected = data;
    expected[0x10..0x14].fill(0xaa);
    assert_eq!(interpreter.read_memory(REGION_ADDRESS, expected.len() as u64), Some(&expected[..]));
}

#[test]
fn cannot_access_memory_outside_mapped_region() {
    let mut interpreter = Interpreter::new();
    interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);

    let region_end = REGION_ADDRESS + REGION_SIZE;
    for (address, length) in [
        (REGION_ADDRESS - 1, 1),
        (REGION_ADDRESS, REGION_SIZE + 1),
        (region_end, 1),
        (region_end - 1, 2),
    ] {
        assert_eq!(interpreter.read_memory(address, length), None);
        assert!(interpreter.write_memory(address, &vec![0xff; length as usize]).is_none());
    }

    assert_eq!(
        interpreter.read_memory(REGION_ADDRESS, REGION_SIZE),
        Some(&[0; REGION_SIZE as usize][..])
    );
}

#[test]
fn cannot_access_memory_across_mapped_regions() {
    let mut interpreter = Interpreter::new();
    let region_end = REGION_ADDRESS + REGION_SIZE;
    interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
    interpreter.map_region(region_end, REGION_SIZE, READ_WRITE);

    assert_eq!(
        interpreter.read_memory(REGION_ADDRESS, REGION_SIZE),
        Some(&[0; REGION_SIZE as usize][..])
    );
    assert_eq!(
        interpreter.read_memory(region_end, REGION_SIZE),
        Some(&[0; REGION_SIZE as usize][..])
    );

    assert_eq!(interpreter.read_memory(region_end - 1, 2), None);
    assert_eq!(interpreter.read_memory(REGION_ADDRESS, 2 * REGION_SIZE), None);
}

#[test]
fn host_memory_access_ignores_protection() {
    let mut interpreter = Interpreter::new();
    let data = &[1, 2, 3, 4];
    interpreter.map_region(REGION_ADDRESS, data.len() as u64, EXECUTE);

    interpreter.write_memory(REGION_ADDRESS, data).unwrap();
    assert_eq!(interpreter.read_memory(REGION_ADDRESS, data.len() as u64), Some(&data[..]));
}

#[test]
fn run_with_no_mapped_region_faults() {
    let mut interpreter = Interpreter::new();
    interpreter.set_program_counter(CODE_ADDRESS);

    assert_eq!(interpreter.run(1), StopReason::Trap(Exception::InstructionAccessFault));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_with_no_mapped_executable_region_faults() {
    let mut interpreter = Interpreter::new();
    map_region_with_data(&mut interpreter, CODE_ADDRESS, &NOP.to_le_bytes(), READ_WRITE);
    interpreter.set_program_counter(CODE_ADDRESS);

    assert_eq!(interpreter.run(1), StopReason::Trap(Exception::InstructionAccessFault));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_at_stop_address_always_finishes() {
    let mut interpreter = Interpreter::new();
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_program_counter(STOP_ADDRESS);

    assert_eq!(interpreter.run(0), StopReason::Finished);
    assert_eq!(interpreter.program_counter(), STOP_ADDRESS);
}

#[test]
fn run_with_no_budget_is_out_of_steps() {
    let mut interpreter = Interpreter::new();
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_program_counter(CODE_ADDRESS);

    assert_eq!(interpreter.run(0), StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_endless_loop_is_out_of_steps() {
    let (stop_reason, interpreter) = run(
        &[j_type(Jal, Reg::Zero, 0)],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_stops_after_budget_at_next_instruction() {
    let (stop_reason, interpreter) = run(
        &[NOP, NOP, NOP],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: 2,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS + 2 * INSTRUCTION_BYTE_LENGTH);
}

#[test]
fn run_resumes_after_out_of_steps() {
    let (stop_reason, mut interpreter) = run(
        &[NOP, RET],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: 1,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS + INSTRUCTION_BYTE_LENGTH);

    assert_eq!(interpreter.run(1), StopReason::Finished);
}

#[test]
fn run_exception_restores_program_counter() {
    let unmapped_address = CODE_ADDRESS - INSTRUCTION_BYTE_LENGTH;
    let (stop_reason, interpreter) = run(
        &[NOP, NOP, i_type(Lw, Reg::A0, Reg::A1, 0)],
        RunSetup {
            registers: &[(Reg::A1, unmapped_address)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::LoadAccessFault(unmapped_address)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS + 2 * INSTRUCTION_BYTE_LENGTH);
}

#[test]
fn run_truncated_32bit_instruction_faults() {
    let (stop_reason, interpreter) = run_bytes(
        &NOP.to_le_bytes()[..COMPRESSED_INSTRUCTION_BYTE_LENGTH as usize],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::InstructionAccessFault));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_undecodable_32bit_word_is_illegal_instruction() {
    const EBREAK: u32 = 0x0010_0073;
    let (stop_reason, interpreter) = run(
        &[EBREAK],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::IllegalInstruction(EBREAK)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_undecodable_compressed_parcel_is_illegal_instruction() {
    const COMPRESSED_EBREAK: u16 = 0x9002;
    let code = [COMPRESSED_EBREAK.to_le_bytes().as_slice(), &[0xff, 0xff]].concat();
    let (stop_reason, interpreter) = run_bytes(
        &code,
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(
        stop_reason,
        StopReason::Trap(Exception::IllegalInstruction(u32::from(COMPRESSED_EBREAK)))
    );
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn run_jump_past_end_of_code_faults() {
    let (stop_reason, interpreter) = run(
        &[j_type(Jal, Reg::Zero, INSTRUCTION_BYTE_LENGTH as i32)],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::InstructionAccessFault));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS + INSTRUCTION_BYTE_LENGTH);
}

#[test]
fn run_counts_compressed_instruction_as_one_step() {
    const COMPRESSED_NOP: u16 = 0x0001;
    let (stop_reason, interpreter) = run_bytes(
        &[COMPRESSED_NOP.to_le_bytes(), COMPRESSED_NOP.to_le_bytes()].concat(),
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: 1,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS + COMPRESSED_INSTRUCTION_BYTE_LENGTH);
}

#[test]
fn addi_to_zero_register_is_noop() {
    let (stop_reason, interpreter) = run(
        &[i_type(Addi, Reg::Zero, Reg::A0, 5), RET],
        RunSetup {
            registers: &[(Reg::A0, 1)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::Zero), 0);
}

#[test]
fn auipc_adds_offset_to_instruction_address() {
    let upper_immediate = 0x12345;
    let (stop_reason, interpreter) = run(
        &[u_type(Auipc, Reg::A0, upper_immediate), RET],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(
        interpreter.reg(Reg::A0),
        CODE_ADDRESS.wrapping_add(u64::from(upper_immediate) << 12)
    );
}

#[test]
fn auipc_sign_extends_offset() {
    let (stop_reason, interpreter) = run(
        &[u_type(Auipc, Reg::A0, UPPER_IMMEDIATE_SIGN_BIT_SET), RET],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    let expected_offset = UPPER_32_BITS_SET | (u64::from(UPPER_IMMEDIATE_SIGN_BIT_SET) << 12);
    assert_eq!(interpreter.reg(Reg::A0), CODE_ADDRESS.wrapping_add(expected_offset));
}

#[test]
fn auipc_to_zero_register_is_noop() {
    let non_zero_upper_immediate = 1;
    let (stop_reason, interpreter) = run(
        &[u_type(Auipc, Reg::Zero, non_zero_upper_immediate), RET],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::Zero), 0);
}

#[test]
fn jal_target_wraps_at_64_bits() {
    let target = UPPER_32_BITS_SET | STOP_ADDRESS;
    let offset = cast(target.wrapping_sub(CODE_ADDRESS)).truncate_to_i32();
    let (stop_reason, interpreter) = run(
        &[j_type(Jal, Reg::Zero, offset)],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: 1,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), target);
}

#[test]
fn jalr_adds_offset_to_base() {
    let base = CODE_ADDRESS + 5 * INSTRUCTION_BYTE_LENGTH;
    let offset = -(INSTRUCTION_BYTE_LENGTH as i32);
    let (stop_reason, interpreter) = run(
        &[
            i_type(Jalr, Reg::Zero, Reg::T1, offset),
            i_type(Addi, Reg::T0, Reg::T0, 1),
            i_type(Addi, Reg::T0, Reg::T0, 2),
            i_type(Addi, Reg::T0, Reg::T0, 3),
            i_type(Addi, Reg::T0, Reg::T0, 4), // <-- after Jalr
            i_type(Addi, Reg::T0, Reg::T0, 5), // <-- base
            i_type(Addi, Reg::T0, Reg::T0, 6),
            RET,
        ],
        RunSetup {
            registers: &[(Reg::T0, 1), (Reg::T1, base)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::T0), 16);
}

#[test]
fn jalr_clears_target_bit_0() {
    for (base, offset) in [(STOP_ADDRESS, 1), (STOP_ADDRESS - 1, 1)] {
        let (stop_reason, interpreter) = run(
            &[i_type(Jalr, Reg::Zero, Reg::RA, offset)],
            RunSetup {
                registers: &[(Reg::RA, base)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.program_counter(), STOP_ADDRESS);
    }
}

#[test]
fn jalr_target_truncates_to_32_bits() {
    let (stop_reason, interpreter) = run(
        &[i_type(Jalr, Reg::Zero, Reg::RA, 0)],
        RunSetup {
            registers: &[(Reg::RA, UPPER_32_BITS_SET | STOP_ADDRESS)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.program_counter(), STOP_ADDRESS);
}

#[test]
fn jalr_target_wraps_at_32_bits() {
    let (stop_reason, interpreter) = run(
        &[i_type(Jalr, Reg::Zero, Reg::Zero, -(INSTRUCTION_BYTE_LENGTH as i32))],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: 1,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), END_OF_32BIT_ADDRESS_SPACE - INSTRUCTION_BYTE_LENGTH);
}

#[test]
fn jalr_to_unmapped_address_faults() {
    let unmapped_address = CODE_ADDRESS - INSTRUCTION_BYTE_LENGTH;
    let (stop_reason, interpreter) = run(
        &[i_type(Jalr, Reg::Zero, Reg::RA, 0)],
        RunSetup {
            registers: &[(Reg::RA, unmapped_address)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::InstructionAccessFault));
    assert_eq!(interpreter.program_counter(), unmapped_address);
}

#[test]
fn branch_compares_64_bits() {
    let offset = 5 * INSTRUCTION_BYTE_LENGTH;
    for (word, lhs, rhs, expected_taken) in [
        (Blt, LOWEST_UPPER_BIT_SET, 1, false),
        (Bltu, SIGN_BIT_SET, 1, false),
        (Beq, LOWEST_UPPER_BIT_SET, 0, false),
        (Bne, LOWEST_UPPER_BIT_SET, 0, true),
        (Bge, 1, LOWEST_UPPER_BIT_SET, false),
        (Bgeu, SIGN_BIT_SET, 1, true),
        (Blt, 1, 1, false),
        (Bltu, 1, 1, false),
    ] {
        let (stop_reason, interpreter) = run(
            &[b_type(word, Reg::A0, Reg::A1, offset as i32)],
            RunSetup {
                registers: &[(Reg::A0, lhs), (Reg::A1, rhs)],
                data: &[],
                max_steps: 1,
            },
        );
        let expected_target = if expected_taken {
            CODE_ADDRESS + offset
        } else {
            CODE_ADDRESS + INSTRUCTION_BYTE_LENGTH
        };
        assert_eq!(stop_reason, StopReason::OutOfSteps);
        assert_eq!(interpreter.program_counter(), expected_target);
    }
}

#[test]
fn branch_target_wraps_at_64_bits() {
    let start_address = 0u64;
    let mut interpreter = Interpreter::new();
    map_region_with_data(
        &mut interpreter,
        start_address,
        &b_type(Beq, Reg::Zero, Reg::Zero, -(INSTRUCTION_BYTE_LENGTH as i32)).to_le_bytes(),
        EXECUTE,
    );
    interpreter.set_program_counter(start_address);

    assert_eq!(interpreter.run(1), StopReason::OutOfSteps);
    assert_eq!(interpreter.program_counter(), start_address.wrapping_sub(INSTRUCTION_BYTE_LENGTH));
}

#[test]
fn load_to_zero_register_from_mapped_address_is_noop() {
    let (stop_reason, interpreter) = run(
        &[i_type(Lw, Reg::Zero, Reg::A0, 0), RET],
        RunSetup {
            registers: &[(Reg::A0, DATA_ADDRESS)],
            data: &[1, 2, 3, 4],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::Zero), 0);
}

#[test]
fn load_to_zero_register_still_attempts_memory_access() {
    let unmapped_address = CODE_ADDRESS - INSTRUCTION_BYTE_LENGTH;
    let (stop_reason, interpreter) = run(
        &[i_type(Lw, Reg::Zero, Reg::A0, 0)],
        RunSetup {
            registers: &[(Reg::A0, unmapped_address)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::LoadAccessFault(unmapped_address)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn loads_from_read_only_address() {
    let mut interpreter = Interpreter::new();
    let code = [i_type(Lw, Reg::A0, Reg::A1, 0).to_le_bytes(), RET.to_le_bytes()].concat();
    let data: [u8; 4] = [1, 2, 3, 4];
    map_region_with_data(&mut interpreter, CODE_ADDRESS, &code, EXECUTE);
    map_region_with_data(&mut interpreter, DATA_ADDRESS, &data, READ_ONLY);
    interpreter.set_program_counter(CODE_ADDRESS);
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_reg(Reg::RA, STOP_ADDRESS);
    interpreter.set_reg(Reg::A1, DATA_ADDRESS);

    assert_eq!(interpreter.run(ENOUGH_STEPS), StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::A0), cast(i32::from_le_bytes(data)).to_u64_sign_extend());
}

#[test]
fn load_from_non_readable_address_faults() {
    let (stop_reason, interpreter) = run(
        &[i_type(Lw, Reg::A0, Reg::A1, 0)],
        RunSetup {
            registers: &[(Reg::A0, 1), (Reg::A1, CODE_ADDRESS)],
            data: &[],
            max_steps: 1,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::LoadAccessFault(CODE_ADDRESS)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    assert_eq!(interpreter.reg(Reg::A0), 1);
}

#[test]
fn load_partly_outside_mapped_region_faults() {
    let straddling_offset = 4;
    let (stop_reason, interpreter) = run(
        &[i_type(Ld, Reg::A0, Reg::A1, straddling_offset)],
        RunSetup {
            registers: &[(Reg::A0, 1), (Reg::A1, DATA_ADDRESS)],
            data: &[1, 2, 3, 4, 5, 6, 7, 8],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(
        stop_reason,
        StopReason::Trap(Exception::LoadAccessFault(DATA_ADDRESS + straddling_offset as u64))
    );
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    assert_eq!(interpreter.reg(Reg::A0), 1);
}

#[test]
fn load_truncates_address_to_32_bits() {
    let data = [1, 2, 3, 4];
    let (stop_reason, interpreter) = run(
        &[i_type(Lw, Reg::A0, Reg::A1, 0), RET],
        RunSetup {
            registers: &[(Reg::A1, UPPER_32_BITS_SET | DATA_ADDRESS)],
            data: &data,
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::A0), cast(i32::from_le_bytes(data)).to_u64_sign_extend());
}

#[test]
fn load_fault_reports_truncated_32bit_address() {
    let unmapped_address = 0;
    let (stop_reason, interpreter) = run(
        &[i_type(Lw, Reg::A0, Reg::A1, 0)],
        RunSetup {
            registers: &[(Reg::A1, UPPER_32_BITS_SET | unmapped_address)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::LoadAccessFault(unmapped_address)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn load_address_wraps_at_32_bits() {
    let mut interpreter = Interpreter::new();
    let data = [1, 2, 3, 4];
    let data_address = END_OF_32BIT_ADDRESS_SPACE - data.len() as u64;
    let code = [
        i_type(Lw, Reg::A0, Reg::Zero, -(data.len() as i32)).to_le_bytes(),
        RET.to_le_bytes(),
    ]
    .concat();
    map_region_with_data(&mut interpreter, CODE_ADDRESS, &code, EXECUTE);
    map_region_with_data(&mut interpreter, data_address, &data, READ_WRITE);
    interpreter.set_program_counter(CODE_ADDRESS);
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_reg(Reg::RA, STOP_ADDRESS);

    assert_eq!(interpreter.run(ENOUGH_STEPS), StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::A0), cast(i32::from_le_bytes(data)).to_u64_sign_extend());
}

#[test]
fn store_to_non_writable_address_faults() {
    let mut interpreter = Interpreter::new();
    let code = s_type(Sw, Reg::A0, Reg::A1, 0).to_le_bytes();
    let data = [1, 2, 3, 4];
    map_region_with_data(&mut interpreter, CODE_ADDRESS, &code, EXECUTE);
    map_region_with_data(&mut interpreter, DATA_ADDRESS, &data, READ_ONLY);
    interpreter.set_program_counter(CODE_ADDRESS);
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_reg(Reg::RA, STOP_ADDRESS);
    interpreter.set_reg(Reg::A0, u64::MAX);
    interpreter.set_reg(Reg::A1, DATA_ADDRESS);

    assert_eq!(
        interpreter.run(ENOUGH_STEPS),
        StopReason::Trap(Exception::StoreAccessFault(DATA_ADDRESS))
    );
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    assert_eq!(interpreter.read_memory(DATA_ADDRESS, data.len() as u64), Some(&data[..]));
}

#[test]
fn store_partly_outside_mapped_region_faults() {
    let straddling_offset = 4;
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let (stop_reason, interpreter) = run(
        &[s_type(Sd, Reg::A0, Reg::A1, straddling_offset)],
        RunSetup {
            registers: &[(Reg::A0, u64::MAX), (Reg::A1, DATA_ADDRESS)],
            data: &data,
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(
        stop_reason,
        StopReason::Trap(Exception::StoreAccessFault(DATA_ADDRESS + straddling_offset as u64))
    );
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    assert_eq!(interpreter.read_memory(DATA_ADDRESS, data.len() as u64), Some(&data[..]));
}

#[test]
fn store_truncates_address_to_32_bits() {
    let to_store: u32 = 0x1234_5678;
    let data = [0, 0, 0, 0];
    let (stop_reason, interpreter) = run(
        &[s_type(Sw, Reg::A0, Reg::A1, 0), RET],
        RunSetup {
            registers: &[(Reg::A0, u64::from(to_store)), (Reg::A1, UPPER_32_BITS_SET | DATA_ADDRESS)],
            data: &data,
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(
        interpreter.read_memory(DATA_ADDRESS, data.len() as u64),
        Some(&to_store.to_le_bytes()[..])
    );
}

#[test]
fn store_fault_reports_truncated_32bit_address() {
    let unmapped_address = 0;
    let (stop_reason, interpreter) = run(
        &[s_type(Sw, Reg::Zero, Reg::A1, 0)],
        RunSetup {
            registers: &[(Reg::A1, UPPER_32_BITS_SET | unmapped_address)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::StoreAccessFault(unmapped_address)));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn store_address_wraps_at_32_bits() {
    let mut interpreter = Interpreter::new();
    let to_store: u32 = 0x1234_5678;
    let data = [0, 0, 0, 0];
    let data_address = END_OF_32BIT_ADDRESS_SPACE - data.len() as u64;
    let code = [
        s_type(Sw, Reg::A0, Reg::Zero, -(data.len() as i32)).to_le_bytes(),
        RET.to_le_bytes(),
    ]
    .concat();
    map_region_with_data(&mut interpreter, CODE_ADDRESS, &code, EXECUTE);
    map_region_with_data(&mut interpreter, data_address, &data, READ_WRITE);
    interpreter.set_program_counter(CODE_ADDRESS);
    interpreter.set_stop_address(STOP_ADDRESS);
    interpreter.set_reg(Reg::RA, STOP_ADDRESS);
    interpreter.set_reg(Reg::A0, u64::from(to_store));

    assert_eq!(interpreter.run(ENOUGH_STEPS), StopReason::Finished);
    assert_eq!(
        interpreter.read_memory(data_address, data.len() as u64),
        Some(&to_store.to_le_bytes()[..])
    );
}

#[test]
fn shift_by_register_uses_low_6_bits_of_amount() {
    for (word, value, shift_amount, expected) in [
        (Sll, 1, 32, 1 << 32),
        (Sll, 1, 63, 1 << 63),
        (Sll, 1, 64, 1),
        (Sll, 1, 65, 2),
        (Srl, 1 << 63, 63, 1),
        (Sra, 1 << 63, 63, u64::MAX),
        (Sra, 1 << 63, 64, 1 << 63),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, value), (Reg::A2, shift_amount)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn shift_by_immediate_takes_6bit_amount() {
    for (word, value, shift_amount, expected) in [
        (Slli, 1, 32, 1 << 32),
        (Srli, 1 << 63, 63, 1),
        (Srai, 1 << 63, 63, u64::MAX),
        (Srai, 1 << 63, 32, 0xffff_ffff_8000_0000),
    ] {
        let (stop_reason, interpreter) = run(
            &[i_type(word, Reg::A0, Reg::A1, shift_amount), RET],
            RunSetup {
                registers: &[(Reg::A1, value)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn set_less_than_compares_64_bits() {
    for (word, lhs, rhs, expected) in [
        (Slt, LOWEST_UPPER_BIT_SET, 1, 0),
        (Slt, 1, LOWEST_UPPER_BIT_SET, 1),
        (Sltu, LOWEST_UPPER_BIT_SET, 1, 0),
        (Sltu, 1, LOWEST_UPPER_BIT_SET, 1),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }

    for (word, lhs, imm, expected) in [(Slti, LOWEST_UPPER_BIT_SET, 1, 0), (Sltiu, LOWEST_UPPER_BIT_SET, 1, 0)] {
        let (stop_reason, interpreter) = run(
            &[i_type(word, Reg::A0, Reg::A1, imm), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn bitwise_operations_use_64_bits() {
    for (word, lhs, rhs, expected) in [
        (And, LOWEST_UPPER_BIT_SET, UPPER_32_BITS_SET, LOWEST_UPPER_BIT_SET),
        (
            Or,
            LOWEST_UPPER_BIT_SET,
            UPPER_32_BITS_SET & !LOWEST_UPPER_BIT_SET,
            UPPER_32_BITS_SET,
        ),
        (
            Xor,
            LOWEST_UPPER_BIT_SET,
            UPPER_32_BITS_SET,
            UPPER_32_BITS_SET & !LOWEST_UPPER_BIT_SET,
        ),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }

    for (word, immediate) in [(Andi, -1), (Ori, 0), (Xori, 0)] {
        let (stop_reason, interpreter) = run(
            &[i_type(word, Reg::A0, Reg::A1, immediate), RET],
            RunSetup {
                registers: &[(Reg::A1, UPPER_32_BITS_SET)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), UPPER_32_BITS_SET);
    }
}

#[test]
fn mulw_sign_extends_low_32_bits_of_product() {
    let (stop_reason, interpreter) = run(
        &[r_type(Mulw, Reg::A0, Reg::A1, Reg::A2), RET],
        RunSetup {
            registers: &[(Reg::A1, 1 << 31), (Reg::A2, 1)],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Finished);
    assert_eq!(interpreter.reg(Reg::A0), cast(1_i32 << 31).to_u64_sign_extend());
}

#[test]
fn multiply_upper_returns_upper_64_bits_of_128bit_product() {
    for (word, lhs, rhs, expected) in [
        (Mulh, cast(-3 << 32).bitwise_as_u64(), 5 << 32, cast(-15).bitwise_as_u64()),
        (Mulhsu, cast(-3 << 32).bitwise_as_u64(), 5 << 32, cast(-15).bitwise_as_u64()),
        (Mulhu, 3 << 32, 5 << 32, 15),
        (Mulh, cast(i64::MIN).bitwise_as_u64(), 1, cast(-1).bitwise_as_u64()),
        (Mulhsu, cast(i64::MIN).bitwise_as_u64(), 1, cast(-1).bitwise_as_u64()),
        (Mulhu, 1 << 63, 1, 0),
        (Mulh, 1, cast(i64::MIN).bitwise_as_u64(), cast(-1).bitwise_as_u64()),
        (Mulhsu, 1, 1 << 63, 0),
        (Mulhu, 1, 1 << 63, 0),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn word_division_uses_low_32_bits_of_sources() {
    for (word, lhs, rhs, expected) in [
        (
            Divw,
            LOWEST_UPPER_BIT_SET | 7,
            cast(cast(-2).bitwise_as_u32()).to_u64(),
            cast(-3_i32).to_u64_sign_extend(),
        ),
        (Divuw, LOWEST_UPPER_BIT_SET | 7, LOWEST_UPPER_BIT_SET | 2, 3),
        (Divw, 7, LOWEST_UPPER_BIT_SET, cast(-1_i32).to_u64_sign_extend()),
        (Divuw, 7, LOWEST_UPPER_BIT_SET, u64::MAX),
        (
            Divw,
            cast(cast(i32::MIN).bitwise_as_u32()).to_u64(),
            cast(cast(-1).bitwise_as_u32()).to_u64(),
            cast(i32::MIN).to_u64_sign_extend(),
        ),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn division_uses_64_bits_of_divisor() {
    for (word, lhs, rhs, expected) in [
        (Div, LOWEST_UPPER_BIT_SET + 5, LOWEST_UPPER_BIT_SET, 1),
        (Divu, LOWEST_UPPER_BIT_SET + 5, LOWEST_UPPER_BIT_SET, 1),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn signed_division_minimum_by_negative_one_overflows() {
    for (word, lhs, rhs, expected) in [
        (
            Divw,
            cast(i32::MIN).to_u64_sign_extend(),
            cast(-1_i32).to_u64_sign_extend(),
            cast(i32::MIN).to_u64_sign_extend(),
        ),
        (
            Div,
            cast(i64::MIN).bitwise_as_u64(),
            cast(-1).bitwise_as_u64(),
            cast(i64::MIN).bitwise_as_u64(),
        ),
        (Divw, 7, cast(-1_i32).to_u64_sign_extend(), cast(-7_i32).to_u64_sign_extend()),
        (Div, 7, cast(-1).bitwise_as_u64(), cast(-7).bitwise_as_u64()),
    ] {
        let (stop_reason, interpreter) = run(
            &[r_type(word, Reg::A0, Reg::A1, Reg::A2), RET],
            RunSetup {
                registers: &[(Reg::A1, lhs), (Reg::A2, rhs)],
                data: &[],
                max_steps: ENOUGH_STEPS,
            },
        );
        assert_eq!(stop_reason, StopReason::Finished);
        assert_eq!(interpreter.reg(Reg::A0), expected);
    }
}

#[test]
fn fences_act_as_noop() {
    const FENCE: u32 = 0x0ff0_000f;
    const FENCE_TSO: u32 = 0x8330_000f;
    const FENCE_I: u32 = 0x0000_100f;
    const PAUSE: u32 = 0x0100_000f;
    let non_zero_value = 0xdead_beef_0000_0001;
    let registers = [(Reg::A0, non_zero_value)];
    let data = [1, 2, 3, 4];

    // Invoke `run` with `max_steps: 0` to get a reference interpreter with the setup
    // state unchanged, since the `0` budget does not execute any instructions.
    let (stop_reason, reference) = run(
        &[FENCE],
        RunSetup {
            registers: &registers,
            data: &data,
            max_steps: 0,
        },
    );
    assert_eq!(stop_reason, StopReason::OutOfSteps);
    assert_eq!(reference.program_counter(), CODE_ADDRESS);

    for fence in [FENCE, FENCE_TSO, FENCE_I, PAUSE] {
        let (stop_reason, interpreter) = run(
            &[fence],
            RunSetup {
                registers: &registers,
                data: &data,
                max_steps: 1,
            },
        );
        assert_eq!(stop_reason, StopReason::OutOfSteps);
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS + INSTRUCTION_BYTE_LENGTH);
        for reg in REGISTERS_RANGE.map(Reg::decode) {
            assert_eq!(interpreter.reg(reg), reference.reg(reg));
        }
        assert_eq!(interpreter.read_memory(DATA_ADDRESS, data.len() as u64), Some(&data[..]));
    }
}

#[test]
fn ecall_traps_as_environment_call() {
    const ECALL: u32 = 0x0000_0073;
    let (stop_reason, interpreter) = run(
        &[ECALL],
        RunSetup {
            registers: &[],
            data: &[],
            max_steps: ENOUGH_STEPS,
        },
    );
    assert_eq!(stop_reason, StopReason::Trap(Exception::EnvironmentCall));
    assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
}

#[test]
fn unimp_traps_as_unimplemented() {
    const UNIMP: u32 = 0xc0001073;
    const COMPRESSED_UNIMP: u16 = 0x0000;
    for code in [&UNIMP.to_le_bytes()[..], &COMPRESSED_UNIMP.to_le_bytes()[..]] {
        let (stop_reason, interpreter) = run_bytes(
            code,
            RunSetup {
                registers: &[],
                data: &[],
                max_steps: 1,
            },
        );
        assert_eq!(stop_reason, StopReason::Trap(Exception::Unimplemented));
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }
}

#[test]
fn every_rv64_category_is_represented() {
    let mut all_categories: Vec<String> = std::fs::read_dir(corpus_output_directory())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("rv64"))
        .collect();
    all_categories.sort();

    assert_eq!(RV64_CATEGORIES, all_categories);
}

#[test]
fn every_rv64_elf_loads_and_decodes_entry_instruction() {
    const ELF64_ENTRY_FIELD: Range<usize> = 0x18..0x20;
    const ELF_MAGIC: &[u8] = b"\x7fELF";
    for category in RV64_CATEGORIES {
        for (_, elf) in corpus_elfs(category) {
            let mut interpreter = Interpreter::new();
            interpreter.load_elf(&elf).unwrap();
            assert_eq!(
                interpreter.read_memory(CORPUS_TEXT_ADDRESS, ELF_MAGIC.len() as u64),
                Some(ELF_MAGIC)
            );

            let entry = u64::from_le_bytes(elf[ELF64_ENTRY_FIELD].try_into().unwrap());
            assert_eq!(interpreter.program_counter(), entry);
            let entry_offset = (entry - CORPUS_TEXT_ADDRESS) as usize;
            let entry_word = u32::from_le_bytes(elf[entry_offset..][..INSTRUCTION_BYTE_LENGTH as usize].try_into().unwrap());
            assert_eq!(
                interpreter.read_memory(entry, INSTRUCTION_BYTE_LENGTH),
                Some(&entry_word.to_le_bytes()[..])
            );
            assert!(Inst::decode(entry_word).is_some());
        }
    }
}

#[test]
fn loads_add_elf() {
    const ENTRY: u64 = 0x100f8;
    const TEXT_SIZE: usize = 0x5f4;
    let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv64ui/add.elf");
    let mut interpreter = Interpreter::new();
    interpreter.load_elf(elf).unwrap();
    assert_eq!(memory_layout(&interpreter), [(CORPUS_TEXT_ADDRESS, TEXT_SIZE as u64, READ_EXECUTE)]);
    assert_eq!(
        interpreter.read_memory(CORPUS_TEXT_ADDRESS, TEXT_SIZE as u64),
        Some(&elf[..TEXT_SIZE])
    );

    assert_eq!(interpreter.program_counter(), ENTRY);
    let word = u32::from_le_bytes(interpreter.read_memory(ENTRY, INSTRUCTION_BYTE_LENGTH).unwrap().try_into().unwrap());
    assert_eq!(
        Inst::decode(word),
        Some(Inst::RegImm {
            kind: RegImmKind::Add64,
            dst: Reg::S0,
            src: Reg::Zero,
            imm: 2
        })
    );
}

#[test]
fn loads_bss_tail_of_amoadd_d_elf() {
    const TEXT_SIZE: usize = 0x170;
    const DATA_ADDRESS: u64 = 0x11170;
    const DATA_FILE_OFFSET: usize = 0x170;
    const DATA_FILE_SIZE: usize = 0x20;
    const DATA_FILE_RANGE: Range<usize> = DATA_FILE_OFFSET..DATA_FILE_OFFSET + DATA_FILE_SIZE;
    const DATA_MEMORY_SIZE: usize = 0x28;
    const BSS_TAIL: Range<usize> = DATA_FILE_SIZE..DATA_MEMORY_SIZE;
    const BSS_TAIL_SIZE: usize = DATA_MEMORY_SIZE - DATA_FILE_SIZE;
    let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv64ua/amoadd_d.elf");
    let mut interpreter = Interpreter::new();
    interpreter.load_elf(elf).unwrap();
    assert_eq!(
        memory_layout(&interpreter),
        [
            (CORPUS_TEXT_ADDRESS, TEXT_SIZE as u64, READ_EXECUTE),
            (DATA_ADDRESS, DATA_MEMORY_SIZE as u64, READ_WRITE)
        ]
    );
    let segment = interpreter.read_memory(DATA_ADDRESS, DATA_MEMORY_SIZE as u64).unwrap();
    assert_eq!(segment[..DATA_FILE_SIZE], elf[DATA_FILE_RANGE]);
    assert_eq!(segment[BSS_TAIL], [0; BSS_TAIL_SIZE]);
}

#[test]
fn skips_loading_non_loadable_segment() {
    const ELF64_PROGRAM_HEADERS_OFFSET: usize = 0x40;
    const ELF64_PROGRAM_HEADER_SIZE: usize = 0x38;
    const P_MEMSZ_OFFSET: usize = 0x28;
    let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv64ui/add.elf");
    let mut patched_elf = elf.to_vec();
    let non_zero_size = 0x40_u64;
    let attributes_header = 0;
    let attributes_mem_size_field_start = ELF64_PROGRAM_HEADERS_OFFSET + attributes_header * ELF64_PROGRAM_HEADER_SIZE + P_MEMSZ_OFFSET;
    let attributes_mem_size_field_range = attributes_mem_size_field_start..attributes_mem_size_field_start + core::mem::size_of::<u64>();

    // The attributes header has a zero memory size. Patch it with a non-zero
    // size to test that loading is skipped due to being a non-loadable segment.
    patched_elf[attributes_mem_size_field_range].copy_from_slice(&non_zero_size.to_le_bytes());

    let mut interpreter_with_patched_elf = Interpreter::new();
    interpreter_with_patched_elf.load_elf(&patched_elf).unwrap();
    let mut interpreter = Interpreter::new();
    interpreter.load_elf(elf).unwrap();
    assert_eq!(memory_layout(&interpreter_with_patched_elf), memory_layout(&interpreter));
}

#[test]
fn rejects_loading_32bit_elf() {
    let mut interpreter = Interpreter::new();
    interpreter.set_program_counter(0x1234);
    let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv64ui/add.elf");
    let mut patched_elf = elf.to_vec();
    patched_elf[core::mem::offset_of!(object::elf::Ident, class)] = object::elf::ELFCLASS32;
    assert!(interpreter.load_elf(&patched_elf).is_err());
    assert_eq!(interpreter.program_counter(), 0x1234);
    assert_eq!(interpreter.read_memory(CORPUS_TEXT_ADDRESS, 1), None);
    assert!(interpreter.load_elf(b"not an elf").is_err());
}

#[test]
fn corpus_rv64() {
    const IMPLEMENTED_CATEGORIES: &[&str] = &["rv64uc", "rv64ui"];
    assert!(IMPLEMENTED_CATEGORIES.iter().all(|category| RV64_CATEGORIES.contains(category)));

    for category in RV64_CATEGORIES {
        if IMPLEMENTED_CATEGORIES.contains(category) {
            run_corpus(category);
        }
    }
}
