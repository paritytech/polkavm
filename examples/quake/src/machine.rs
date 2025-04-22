use polkakernel::{Machine, MachineError, MachineError::*, Reg};
use polkavm::{ProgramCounter, RawInstance};

fn from_reg(other: Reg) -> polkavm::Reg {
    unsafe { core::mem::transmute(other) }
}

pub struct PolkaMachine(pub RawInstance);

impl Machine for PolkaMachine {
    fn program_counter(&self) -> u64 {
        self.0
            .program_counter()
            .unwrap_or_else(|| panic!("Failed to get program counter"))
            .0
            .into()
    }

    fn set_next_program_counter(&mut self, pc: u64) {
        let pc: u32 = pc.try_into().unwrap_or_else(|_| panic!("Failed to set program counter"));
        self.0.set_next_program_counter(ProgramCounter(pc));
    }

    fn reg(&self, name: Reg) -> u64 {
        self.0.reg(from_reg(name))
    }

    fn set_reg(&mut self, name: Reg, value: u64) {
        self.0.set_reg(from_reg(name), value);
    }

    fn read_u64(&self, address: u64) -> Result<u64, MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.read_u64(address).map_err(|_| BadAddress)
    }

    fn read_u32(&self, address: u64) -> Result<u32, MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.read_u32(address).map_err(|_| BadAddress)
    }

    fn read_u16(&self, address: u64) -> Result<u16, MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.read_u16(address).map_err(|_| BadAddress)
    }

    fn read_u8(&self, address: u64) -> Result<u8, MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.read_u8(address).map_err(|_| BadAddress)
    }

    fn read_memory_into(&self, address: u64, buffer: &mut [u8]) -> Result<(), MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.read_memory_into(address, buffer).map_err(|_| BadAddress)?;
        Ok(())
    }

    fn write_u64(&mut self, address: u64, value: u64) -> Result<(), MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.write_u64(address, value).map_err(|_| BadAddress)
    }

    fn write_memory(&mut self, address: u64, slice: &[u8]) -> Result<(), MachineError> {
        let address = address.try_into().map_err(|_| BadAddress)?;
        self.0.write_memory(address, slice).map_err(|_| BadAddress)
    }
}
