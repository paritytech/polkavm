use core::ops::Range;

use object::elf::{FileHeader64, PF_R, PF_W, PF_X, PT_LOAD};
use object::read::elf::{ElfFile, ProgramHeader};
use object::{LittleEndian, Object};
use polkavm_linker::riscv::{Inst, Reg};

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

pub struct MemoryRegion {
    address: u64,
    data: Vec<u8>,
    protection: MemoryProtection,
}

impl MemoryRegion {
    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn size(&self) -> u64 {
        self.data.len() as u64
    }

    pub fn protection(&self) -> MemoryProtection {
        self.protection
    }

    fn end(&self) -> u64 {
        self.address + self.size()
    }

    fn contains(&self, range: Range<u64>) -> bool {
        range.start >= self.address && range.end <= self.end()
    }

    fn overlaps(&self, range: Range<u64>) -> bool {
        range.start < self.end() && range.end > self.address
    }

    fn allows(&self, access: MemoryAccess) -> bool {
        match access {
            MemoryAccess::Fetch => self.protection.execute,
        }
    }
}

#[derive(Default)]
struct Memory {
    regions: Vec<MemoryRegion>,
}

impl Memory {
    fn map_region(&mut self, address: u64, size: u64, protection: MemoryProtection) {
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

    fn read(&self, address: u64, length: u64) -> Option<&[u8]> {
        let (index, range) = self.find_region(address, length)?;

        Some(&self.regions[index].data[range])
    }

    fn read_as_guest(&self, address: u64, length: u64, access: MemoryAccess) -> Option<&[u8]> {
        let (index, range) = self.find_region(address, length)?;
        let region = &self.regions[index];
        if !region.allows(access) {
            return None;
        }

        Some(&region.data[range])
    }

    #[must_use]
    fn write(&mut self, address: u64, data: &[u8]) -> Option<()> {
        let (index, range) = self.find_region(address, data.len() as u64)?;
        self.regions[index].data[range].copy_from_slice(data);

        Some(())
    }
}

/// A reference RV64 interpreter for a single hart with a 32-bit address space.
pub struct Interpreter {
    stop_address: Option<u64>,
    program_counter: u64,
    regs: [u64; 32],
    memory: Memory,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            stop_address: None,
            program_counter: 0,
            regs: [0; 32],
            memory: Memory::default(),
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
        let file: ElfFile<FileHeader64<LittleEndian>> = ElfFile::parse(data)?;
        let endian = file.endian();
        for header in file.elf_program_headers() {
            let is_loadable = header.p_type(endian) == PT_LOAD;
            let memory_size = header.p_memsz(endian);
            if !is_loadable || memory_size == 0 {
                continue;
            }

            let address = header.p_vaddr(endian);
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
        self.memory.map_region(address, size, protection);
    }

    pub fn regions(&self) -> &[MemoryRegion] {
        &self.memory.regions
    }

    pub fn read_memory(&self, address: u64, length: u64) -> Option<&[u8]> {
        self.memory.read(address, length)
    }

    fn read_memory_as_guest(&self, address: u64, length: u64, access: MemoryAccess) -> Result<&[u8], Exception> {
        self.memory
            .read_as_guest(address, length, access)
            .ok_or(Exception::InstructionAccessFault)
    }

    #[must_use]
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Option<()> {
        self.memory.write(address, data)
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
        let instruction = Inst::decode(word).ok_or(Exception::IllegalInstruction(word))?;
        self.set_program_counter(program_counter.wrapping_add(length));

        if let Err(exception) = self.execute(instruction, program_counter) {
            self.set_program_counter(program_counter);
            return Err(exception);
        }

        Ok(())
    }

    fn fetch_instruction(&self, program_counter: u64) -> Result<(u32, u64), Exception> {
        let first_byte = self.read_memory_as_guest(program_counter, 1, MemoryAccess::Fetch)?[0];
        // Instruction lengths: https://docs.riscv.org/reference/isa/v20260120/unpriv/intro.html#base-instruction-length-encoding
        let length = if Inst::is_compressed(first_byte) { 2 } else { 4 };
        let parcels = self.read_memory_as_guest(program_counter, length, MemoryAccess::Fetch)?;
        let mut word_bytes = [0; 4];
        word_bytes[..parcels.len()].copy_from_slice(parcels);

        Ok((u32::from_le_bytes(word_bytes), length))
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn execute(&mut self, _instruction: Inst, _instruction_address: u64) -> Result<(), Exception> {
        todo!()
    }
}
