use core::ops::Range;

use crate::riscv::{DecoderConfig, Reg};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTERS_RANGE: Range<u32> = 0..32;
    const REGION_ADDRESS: u64 = 0x1000;
    const REGION_SIZE: u64 = 0x100;
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
}
