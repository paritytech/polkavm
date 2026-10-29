use core::ops::Range;

use object::elf::{FileHeader64, PF_R, PF_W, PF_X, PT_LOAD};
use object::read::elf::{ElfFile, FileHeader, ProgramHeader};
use object::{LittleEndian, Object};

use crate::riscv::{DecoderConfig, Inst, Reg};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Exception {
    IllegalInstruction(u32),
    Unimplemented,
    EnvironmentCall,
    InstructionAccessFault,
    LoadAccessFault(u64),
    StoreAccessFault(u64),
    MisalignedAtomic(u64),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum StopReason {
    Finished,
    Trap(Exception),
    OutOfSteps,
}

enum MemoryAccess {
    Fetch,
    // TODO (when implementation requires it):
    // Load,
    // Store,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MemoryProtection {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

struct MemoryRegion {
    address: u64,
    data: Vec<u8>,
    protection: MemoryProtection,
}

impl MemoryRegion {
    fn end(&self) -> u64 {
        self.address + self.data.len() as u64
    }

    fn contains(&self, range: Range<u64>) -> bool {
        range.start >= self.address && range.end <= self.end()
    }

    fn overlaps(&self, range: Range<u64>) -> bool {
        range.start < self.end() && range.end > self.address
    }

    fn allows_access(&self, access: MemoryAccess) -> bool {
        match access {
            MemoryAccess::Fetch => self.protection.execute,
        }
    }
}

pub struct Interpreter {
    decoder_config: DecoderConfig,
    stop_address: Option<u64>,
    program_counter: u64,
    regs: [u64; 32],
    regions: Vec<MemoryRegion>,
}

impl Interpreter {
    pub fn new_64bit() -> Self {
        let mut decoder_config = DecoderConfig::new_32bit();
        decoder_config.set_rv64(true);

        Interpreter {
            decoder_config,
            stop_address: None,
            program_counter: 0,
            regs: [0; 32],
            regions: Vec::new(),
        }
    }

    pub fn program_counter(&self) -> u64 {
        self.program_counter
    }

    pub fn set_program_counter(&mut self, program_counter: u64) {
        self.program_counter = program_counter;
    }

    pub fn reg(&self, reg: Reg) -> u64 {
        self.regs[reg as usize]
    }

    pub fn set_reg(&mut self, reg: Reg, value: u64) {
        if reg != Reg::Zero {
            self.regs[reg as usize] = value;
        }
    }

    pub fn set_stop_address(&mut self, address: u64) {
        self.stop_address = Some(address);
    }

    pub fn load_elf(&mut self, data: &[u8]) -> Result<(), object::Error> {
        self.load_elf_class::<FileHeader64<LittleEndian>>(data)
    }

    fn load_elf_class<Elf: FileHeader>(&mut self, data: &[u8]) -> Result<(), object::Error> {
        let file: ElfFile<Elf> = ElfFile::parse(data)?;
        let endian = file.endian();
        for header in file.elf_program_headers() {
            let is_loadable = header.p_type(endian) == PT_LOAD;
            let memory_size: u64 = header.p_memsz(endian).into();
            if !is_loadable || memory_size == 0 {
                continue;
            }

            let address: u64 = header.p_vaddr(endian).into();
            let flags = header.p_flags(endian);
            let protection = MemoryProtection {
                read: flags & PF_R != 0,
                write: flags & PF_W != 0,
                execute: flags & PF_X != 0,
            };
            self.map_region(address, memory_size, protection);
            self.write_memory(address, header.data(endian, data).unwrap()).unwrap();
        }

        self.set_program_counter(file.entry());

        Ok(())
    }

    pub fn map_region(&mut self, address: u64, size: u64, protection: MemoryProtection) {
        let end = address.checked_add(size).unwrap();
        assert!(!self.regions.iter().any(|region| region.overlaps(address..end)));

        let index = self.regions.partition_point(|region| region.address < address);
        let region = MemoryRegion {
            address,
            data: vec![0; usize::try_from(size).unwrap()],
            protection,
        };
        self.regions.insert(index, region);
    }

    fn find_region(&self, address: u64, length: u64) -> Option<(usize, Range<usize>)> {
        let end = address.checked_add(length)?;
        let index = self.regions.iter().position(|region| region.contains(address..end))?;
        let offset = (address - self.regions[index].address) as usize;

        Some((index, offset..offset + length as usize))
    }

    pub fn read_memory(&self, address: u64, length: u64) -> Option<&[u8]> {
        let (index, range) = self.find_region(address, length)?;

        Some(&self.regions[index].data[range])
    }

    #[must_use]
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Option<()> {
        let (index, range) = self.find_region(address, data.len() as u64)?;
        self.regions[index].data[range].copy_from_slice(data);

        Some(())
    }

    fn read_guest_memory(&self, address: u64, length: u64, access: MemoryAccess) -> Option<&[u8]> {
        let (index, range) = self.find_region(address, length)?;
        let region = &self.regions[index];
        if !region.allows_access(access) {
            return None;
        }

        Some(&region.data[range])
    }

    pub fn run(&mut self, max_steps: u64) -> StopReason {
        let mut steps = 0;
        loop {
            if Some(self.program_counter) == self.stop_address {
                return StopReason::Finished;
            }
            if steps == max_steps {
                return StopReason::OutOfSteps;
            }
            if let Err(exception) = self.step() {
                return StopReason::Trap(exception);
            }
            steps += 1;
        }
    }

    fn step(&mut self) -> Result<(), Exception> {
        let program_counter = self.program_counter;
        let (word, length) = self.fetch_instruction(program_counter)?;
        let instruction = Inst::decode(&self.decoder_config, word).ok_or(Exception::IllegalInstruction(word))?;
        self.set_program_counter(program_counter.wrapping_add(length));

        if let Err(exception) = self.execute(instruction, program_counter) {
            self.set_program_counter(program_counter);
            return Err(exception);
        }

        Ok(())
    }

    fn fetch_instruction(&self, program_counter: u64) -> Result<(u32, u64), Exception> {
        let first_byte = self
            .read_guest_memory(program_counter, 1, MemoryAccess::Fetch)
            .ok_or(Exception::InstructionAccessFault)?[0];
        // Compressed instructions are 16 bits, otherwise 32 (https://docs.riscv.org/reference/isa/v20260120/unpriv/intro.html#base-instruction-length-encoding).
        let length = if Inst::is_compressed(first_byte) { 2 } else { 4 };
        let parcels = self
            .read_guest_memory(program_counter, length, MemoryAccess::Fetch)
            .ok_or(Exception::InstructionAccessFault)?;
        let mut word_bytes = [0; 4];
        word_bytes[..parcels.len()].copy_from_slice(parcels);

        Ok((u32::from_le_bytes(word_bytes), length))
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn execute(&mut self, _instruction: Inst, _program_counter: u64) -> Result<(), Exception> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    use crate::riscv::RegImmKind;

    const RV64_CATEGORIES: &[&str] = &["rv64ua", "rv64uc", "rv64ui", "rv64um", "rv64uzbb"];
    const REGISTERS_RANGE: Range<u32> = 0..32;
    const CODE_ADDRESS: u64 = 0x10000;
    const STOP_ADDRESS: u64 = 0xffff0000;
    const REGION_ADDRESS: u64 = 0x1000;
    const REGION_SIZE: u64 = 0x100;
    const CORPUS_TEXT_ADDRESS: u64 = 0x10000;
    const NOP: u32 = 0x00000013;
    const INSTRUCTION_BYTE_LENGTH: u64 = 4;
    const COMPRESSED_INSTRUCTION_BYTE_LENGTH: u64 = 2;
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

    fn map_region_with_data(interpreter: &mut Interpreter, address: u64, data: &[u8], protection: MemoryProtection) {
        interpreter.map_region(address, data.len() as u64, protection);
        interpreter.write_memory(address, data).unwrap();
    }

    fn memory_layout(interpreter: &Interpreter) -> Vec<(u64, usize, MemoryProtection)> {
        interpreter
            .regions
            .iter()
            .map(|region| (region.address, region.data.len(), region.protection))
            .collect()
    }

    fn run_program(code: &[u8], max_steps: u64) -> (StopReason, Interpreter) {
        let mut interpreter = Interpreter::new_64bit();
        map_region_with_data(&mut interpreter, CODE_ADDRESS, code, EXECUTE);
        interpreter.set_stop_address(STOP_ADDRESS);
        interpreter.set_program_counter(CODE_ADDRESS);
        interpreter.set_reg(Reg::RA, STOP_ADDRESS);

        let stop_reason = interpreter.run(max_steps);

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
            let mut interpreter = Interpreter::new_64bit();
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
        let interpreter = Interpreter::new_64bit();
        for index in REGISTERS_RANGE {
            assert_eq!(interpreter.reg(Reg::decode(index)), 0);
        }
    }

    #[test]
    fn set_reg_round_trips() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_reg(Reg::A0, 0xdead_beef_0000_0001);
        assert_eq!(interpreter.reg(Reg::A0), 0xdead_beef_0000_0001);
        assert_eq!(interpreter.reg(Reg::A1), 0);
    }

    #[test]
    fn writes_to_zero_register_are_noop() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_reg(Reg::Zero, 5);
        assert_eq!(interpreter.reg(Reg::Zero), 0);
    }

    #[test]
    fn decodes_in_64bit_mode() {
        assert!(Interpreter::new_64bit().decoder_config.rv64);
    }

    #[test]
    fn maps_regions_to_sorted_addresses() {
        let mut interpreter = Interpreter::new_64bit();
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

        let mut expected_layout = regions.map(|(address, _)| (address, REGION_SIZE as usize, READ_WRITE));
        expected_layout.sort_by_key(|(address, _, _)| *address);
        assert_eq!(memory_layout(&interpreter), expected_layout);
    }

    #[test]
    #[should_panic]
    fn region_cannot_overlap_at_start() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
        interpreter.map_region(REGION_ADDRESS + REGION_SIZE - 1, 2, READ_WRITE);
    }

    #[test]
    #[should_panic]
    fn region_cannot_overlap_at_end() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
        interpreter.map_region(REGION_ADDRESS - REGION_SIZE + 1, REGION_SIZE, READ_WRITE);
    }

    #[test]
    #[should_panic]
    fn region_cannot_overlap_fully() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.map_region(REGION_ADDRESS, REGION_SIZE, READ_WRITE);
        interpreter.map_region(REGION_ADDRESS - REGION_SIZE, 3 * REGION_SIZE, READ_WRITE);
    }

    #[test]
    fn accesses_memory_inside_mapped_region() {
        let mut interpreter = Interpreter::new_64bit();
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
        let mut interpreter = Interpreter::new_64bit();
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
        let mut interpreter = Interpreter::new_64bit();
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
        let mut interpreter = Interpreter::new_64bit();
        let data = &[1, 2, 3, 4];
        interpreter.map_region(REGION_ADDRESS, data.len() as u64, EXECUTE);

        interpreter.write_memory(REGION_ADDRESS, data).unwrap();
        assert_eq!(interpreter.read_memory(REGION_ADDRESS, data.len() as u64), Some(&data[..]));
    }

    #[test]
    fn run_at_stop_address_always_finishes() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_stop_address(STOP_ADDRESS);
        interpreter.set_program_counter(STOP_ADDRESS);

        assert_eq!(interpreter.run(0), StopReason::Finished);
        assert_eq!(interpreter.program_counter(), STOP_ADDRESS);
    }

    #[test]
    fn run_with_no_budget_is_out_of_steps() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_stop_address(STOP_ADDRESS);
        interpreter.set_program_counter(CODE_ADDRESS);

        assert_eq!(interpreter.run(0), StopReason::OutOfSteps);
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn run_with_no_mapped_region_faults() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_program_counter(CODE_ADDRESS);

        assert_eq!(interpreter.run(1), StopReason::Trap(Exception::InstructionAccessFault));
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn run_with_no_mapped_executable_region_faults() {
        let mut interpreter = Interpreter::new_64bit();
        map_region_with_data(&mut interpreter, CODE_ADDRESS, &NOP.to_le_bytes(), READ_WRITE);
        interpreter.set_program_counter(CODE_ADDRESS);

        assert_eq!(interpreter.run(1), StopReason::Trap(Exception::InstructionAccessFault));
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn run_truncated_32bit_instruction_faults() {
        let (stop_reason, interpreter) = run_program(&NOP.to_le_bytes()[..COMPRESSED_INSTRUCTION_BYTE_LENGTH as usize], 1);
        assert_eq!(stop_reason, StopReason::Trap(Exception::InstructionAccessFault));
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn run_undecodable_32bit_word_is_illegal_instruction() {
        const EBREAK: u32 = 0x00100073;
        let (stop_reason, interpreter) = run_program(&EBREAK.to_le_bytes(), 1);
        assert_eq!(stop_reason, StopReason::Trap(Exception::IllegalInstruction(EBREAK)));
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn run_undecodable_compressed_parcel_is_illegal_instruction() {
        const COMPRESSED_EBREAK: u16 = 0x9002;
        let code = [COMPRESSED_EBREAK.to_le_bytes().as_slice(), &[0xff, 0xff]].concat();
        let (stop_reason, interpreter) = run_program(&code, 1);
        assert_eq!(
            stop_reason,
            StopReason::Trap(Exception::IllegalInstruction(u32::from(COMPRESSED_EBREAK)))
        );
        assert_eq!(interpreter.program_counter(), CODE_ADDRESS);
    }

    #[test]
    fn fetch_returns_word_and_length() {
        const COMPRESSED_NOP: u16 = 0x0001;
        let mut interpreter = Interpreter::new_64bit();
        let code = [NOP.to_le_bytes().as_slice(), COMPRESSED_NOP.to_le_bytes().as_slice()].concat();
        map_region_with_data(&mut interpreter, CODE_ADDRESS, &code, EXECUTE);

        assert_eq!(interpreter.fetch_instruction(CODE_ADDRESS), Ok((NOP, INSTRUCTION_BYTE_LENGTH)));
        assert_eq!(
            interpreter.fetch_instruction(CODE_ADDRESS + INSTRUCTION_BYTE_LENGTH),
            Ok((u32::from(COMPRESSED_NOP), COMPRESSED_INSTRUCTION_BYTE_LENGTH))
        );
        assert_eq!(
            interpreter.fetch_instruction(CODE_ADDRESS + code.len() as u64),
            Err(Exception::InstructionAccessFault)
        );
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
                let mut interpreter = Interpreter::new_64bit();
                interpreter.load_elf(&elf).unwrap();
                assert_eq!(
                    interpreter.read_memory(CORPUS_TEXT_ADDRESS, ELF_MAGIC.len() as u64),
                    Some(ELF_MAGIC)
                );

                let entry = u64::from_le_bytes(elf[ELF64_ENTRY_FIELD].try_into().unwrap());
                assert_eq!(interpreter.program_counter(), entry);
                let entry_offset = (entry - CORPUS_TEXT_ADDRESS) as usize;
                let entry_word = u32::from_le_bytes(elf[entry_offset..][..INSTRUCTION_BYTE_LENGTH as usize].try_into().unwrap());
                assert_eq!(interpreter.fetch_instruction(entry), Ok((entry_word, INSTRUCTION_BYTE_LENGTH)));
                assert!(Inst::decode(&DecoderConfig::new_64bit(), entry_word).is_some());
            }
        }
    }

    #[test]
    fn loads_add_elf() {
        const ENTRY: u64 = 0x100f8;
        const TEXT_SIZE: usize = 0x5f4;
        let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv64ui/add.elf");
        let mut interpreter = Interpreter::new_64bit();
        interpreter.load_elf(elf).unwrap();
        assert_eq!(memory_layout(&interpreter), [(CORPUS_TEXT_ADDRESS, TEXT_SIZE, READ_EXECUTE)]);
        assert_eq!(
            interpreter.read_memory(CORPUS_TEXT_ADDRESS, TEXT_SIZE as u64),
            Some(&elf[..TEXT_SIZE])
        );

        assert_eq!(interpreter.program_counter(), ENTRY);
        let (word, length) = interpreter.fetch_instruction(ENTRY).unwrap();
        assert_eq!(length, INSTRUCTION_BYTE_LENGTH);
        assert_eq!(
            Inst::decode(&DecoderConfig::new_64bit(), word),
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
        let mut interpreter = Interpreter::new_64bit();
        interpreter.load_elf(elf).unwrap();
        assert_eq!(
            memory_layout(&interpreter),
            [
                (CORPUS_TEXT_ADDRESS, TEXT_SIZE, READ_EXECUTE),
                (DATA_ADDRESS, DATA_MEMORY_SIZE, READ_WRITE)
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
        let attributes_mem_size_field_range =
            attributes_mem_size_field_start..attributes_mem_size_field_start + core::mem::size_of::<u64>();

        // The attributes header has a zero memory size. Patch it with a non-zero
        // size to test that loading is skipped due to being a non-loadable segment.
        patched_elf[attributes_mem_size_field_range].copy_from_slice(&non_zero_size.to_le_bytes());

        let mut interpreter_with_patched_elf = Interpreter::new_64bit();
        interpreter_with_patched_elf.load_elf(&patched_elf).unwrap();
        let mut interpreter = Interpreter::new_64bit();
        interpreter.load_elf(elf).unwrap();
        assert_eq!(memory_layout(&interpreter_with_patched_elf), memory_layout(&interpreter));
    }

    #[test]
    fn rejects_loading_32bit_elf() {
        let mut interpreter = Interpreter::new_64bit();
        interpreter.set_program_counter(0x1234);
        let elf = include_bytes!("../../../guest-programs/riscv-tests/output/rv32ui/add.elf");
        assert!(interpreter.load_elf(elf).is_err());
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
}
