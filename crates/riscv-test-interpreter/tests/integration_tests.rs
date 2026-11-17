#![deny(unused_must_use)]

use core::ops::Range;
use std::path::{Path, PathBuf};

use polkavm_linker::riscv::{Inst, RegImmKind};
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
const INSTRUCTION_BYTE_LENGTH: u64 = 4;
const COMPRESSED_INSTRUCTION_BYTE_LENGTH: u64 = 2;
const NOP: u32 = 0x00000013;

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
/// so that if `code` ends with a `ret` instruction it returns to the stop address and
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

#[ignore = "enable once the base integer ISA is implemented"]
#[test]
fn corpus_rv64ui() {
    run_corpus("rv64ui");
}

#[ignore = "enable once the C extension is implemented"]
#[test]
fn corpus_rv64uc() {
    run_corpus("rv64uc");
}

#[ignore = "enable once the M extension is implemented"]
#[test]
fn corpus_rv64um() {
    run_corpus("rv64um");
}

#[ignore = "enable once the A extension is implemented"]
#[test]
fn corpus_rv64ua() {
    run_corpus("rv64ua");
}

#[ignore = "enable once the Zbb extension is implemented"]
#[test]
fn corpus_rv64uzbb() {
    run_corpus("rv64uzbb");
}
