use core::ops::Range;

use object::elf::{FileHeader64, PF_R, PF_W, PF_X, PT_LOAD};
use object::read::elf::{ElfFile, ProgramHeader};
use object::{LittleEndian, Object};
use polkavm_linker::riscv::{BranchKind, Inst, LoadKind, Reg, RegImmKind, RegRegKind, StoreKind};

use crate::width::{
    cast_to_i32, cast_to_i64, cast_to_u32, cast_to_u64, sign_extend_16, sign_extend_32, sign_extend_8, truncate_to_i32, truncate_to_i64,
    truncate_to_u32, truncate_to_u64, zero_extend_16, zero_extend_32, zero_extend_8,
};

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

#[derive(Copy, Clone)]
enum MemoryAccess {
    Fetch,
    Load,
    Store,
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
            MemoryAccess::Load => self.protection.read,
            MemoryAccess::Store => self.protection.write,
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

    #[must_use]
    fn write_as_guest(&mut self, address: u64, data: &[u8]) -> Option<()> {
        let (index, range) = self.find_region(address, data.len() as u64)?;
        let region = &mut self.regions[index];
        if !region.allows(MemoryAccess::Store) {
            return None;
        }
        region.data[range].copy_from_slice(data);

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
        self.memory.read_as_guest(address, length, access).ok_or(match access {
            MemoryAccess::Fetch => Exception::InstructionAccessFault,
            MemoryAccess::Load => Exception::LoadAccessFault(address),
            // Handle `Store` when AMO is supported. AMO should raise a store access-fault exception.
            // (https://docs.riscv.org/reference/isa/v20260120/priv/machine.html#norm:pmp_store_fault)
            MemoryAccess::Store => todo!(),
        })
    }

    #[must_use]
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Option<()> {
        self.memory.write(address, data)
    }

    fn write_memory_as_guest(&mut self, address: u64, data: &[u8]) -> Result<(), Exception> {
        self.memory
            .write_as_guest(address, data)
            .ok_or(Exception::StoreAccessFault(address))
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

    fn execute(&mut self, instruction: Inst, instruction_address: u64) -> Result<(), Exception> {
        match instruction {
            Inst::RegImm { kind, dst, src, imm } => self.register_immediate(kind, dst, src, imm),
            Inst::RegReg { kind, dst, src1, src2 } => self.register_register(kind, dst, src1, src2),
            Inst::LoadUpperImmediate { dst, value } => self.load_upper_immediate(dst, value, None),
            Inst::AddUpperImmediateToPc { dst, value } => self.load_upper_immediate(dst, value, Some(instruction_address)),
            Inst::JumpAndLink { dst, target } => self.jump_and_link(dst, target, instruction_address),
            Inst::JumpAndLinkRegister { dst, base, value } => self.jump_and_link_register(dst, base, value),
            Inst::Branch { kind, src1, src2, target } => self.branch(kind, src1, src2, target, instruction_address),
            Inst::Load { kind, dst, base, offset } => self.load(kind, dst, base, offset)?,
            Inst::Store { kind, src, base, offset } => self.store(kind, src, base, offset)?,

            // https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#fence
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/zifencei.html
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/zihintpause.html
            Inst::Fence { .. } | Inst::FenceI => {}

            // https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#ecall-ebreak
            Inst::Ecall => return Err(Exception::EnvironmentCall),

            // https://github.com/riscv-non-isa/riscv-asm-manual/blob/7303c035b1be5b7de673a66db046221da237da8e/src/asm-manual.adoc#instruction-aliases
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/c-st-ext.html#27-1-5-4-defined-illegal-instruction
            Inst::Unimplemented => return Err(Exception::Unimplemented),

            // "B" extension (Zbb):
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-2-count-leadingtrailing-zero-bits
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-3-count-population
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-5-sign-extension-and-zero-extension
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-7-or-combine
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-8-byte-reverse
            Inst::Reg { .. } => todo!(),

            // "A" extension:
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/a-st-ext.html#sec:lrsc
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/a-st-ext.html#sec:amo
            Inst::LoadReserved32 { .. }
            | Inst::StoreConditional32 { .. }
            | Inst::LoadReserved64 { .. }
            | Inst::StoreConditional64 { .. }
            | Inst::Atomic { .. } => todo!(),

            // Vendor conditional moves:
            // https://github.com/XUANTIE-RV/thead-extension-spec/blob/2.4.0/xtheadcondmov/mveqz.adoc
            // https://github.com/XUANTIE-RV/thead-extension-spec/blob/2.4.0/xtheadcondmov/mvnez.adoc
            // https://github.com/llvm/llvm-project/blob/e30e644266fbc9ba638ee2c6aa23b5691397163f/llvm/lib/Target/RISCV/RISCVInstrInfoXMips.td#L148-L166
            Inst::Cmov { .. } | Inst::MipsCmov { .. } => todo!(),
        }

        Ok(())
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-4-1-integer-register-immediate-instructions
    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv64.html#3-1-2-1-integer-register-immediate-instructions
    fn register_immediate(&mut self, kind: RegImmKind, dst: Reg, src: Reg, imm: i32) {
        let lhs = self.reg(src);
        let rhs = sign_extend_32(imm);
        let result = match kind {
            RegImmKind::Add32AndSignExtend => sign_extend_32(truncate_to_i32(lhs.wrapping_add(rhs))),
            RegImmKind::Add64 => lhs.wrapping_add(rhs),
            RegImmKind::SetLessThanSigned64 => u64::from(cast_to_i64(lhs) < cast_to_i64(rhs)),
            RegImmKind::SetLessThanUnsigned64 => u64::from(lhs < rhs),
            RegImmKind::ShiftLogicalLeft32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_shl(cast_to_u32(imm))),
            RegImmKind::ShiftLogicalLeft64 => lhs.wrapping_shl(cast_to_u32(imm)),
            // `lhs` for SRLI and SRLIW are unsigned since `wrapping_shr` is logical for unsigned values, otherwise arithmetic.
            RegImmKind::ShiftLogicalRight32AndSignExtend => {
                sign_extend_32(cast_to_i32(truncate_to_u32(lhs).wrapping_shr(cast_to_u32(imm))))
            }
            RegImmKind::ShiftLogicalRight64 => lhs.wrapping_shr(cast_to_u32(imm)),
            // `lhs` for SRAI and SRAIW are signed since `wrapping_shr` is arithmetic for signed values, otherwise logical.
            RegImmKind::ShiftArithmeticRight32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_shr(cast_to_u32(imm))),
            RegImmKind::ShiftArithmeticRight64 => cast_to_u64(cast_to_i64(lhs).wrapping_shr(cast_to_u32(imm))),
            RegImmKind::Xor64 => lhs ^ rhs,
            RegImmKind::Or64 => lhs | rhs,
            RegImmKind::And64 => lhs & rhs,

            // "B" extension (Zbb):
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-6-bitwise-rotation
            RegImmKind::RotateRight32AndSignExtend | RegImmKind::RotateRight64 => todo!(),
        };
        self.set_reg(dst, result);
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-4-2-integer-register-register-instructions
    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv64.html#3-1-2-2-integer-register-register-operations
    fn register_register(&mut self, kind: RegRegKind, dst: Reg, src1: Reg, src2: Reg) {
        let lhs = self.reg(src1);
        let rhs = self.reg(src2);
        let result = match kind {
            RegRegKind::Add32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_add(truncate_to_i32(rhs))),
            RegRegKind::Add64 => lhs.wrapping_add(rhs),
            RegRegKind::Sub32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_sub(truncate_to_i32(rhs))),
            RegRegKind::Sub64 => lhs.wrapping_sub(rhs),
            RegRegKind::SetLessThanSigned64 => u64::from(cast_to_i64(lhs) < cast_to_i64(rhs)),
            RegRegKind::SetLessThanUnsigned64 => u64::from(lhs < rhs),
            RegRegKind::ShiftLogicalLeft32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_shl(truncate_to_u32(rhs))),
            RegRegKind::ShiftLogicalLeft64 => lhs.wrapping_shl(truncate_to_u32(rhs)),
            // `lhs` for SRL and SRLW are unsigned since `wrapping_shr` is logical for unsigned values, otherwise arithmetic.
            RegRegKind::ShiftLogicalRight32AndSignExtend => {
                sign_extend_32(cast_to_i32(truncate_to_u32(lhs).wrapping_shr(truncate_to_u32(rhs))))
            }
            RegRegKind::ShiftLogicalRight64 => lhs.wrapping_shr(truncate_to_u32(rhs)),
            // `lhs` for SRA and SRAW are signed since `wrapping_shr` is arithmetic for signed values, otherwise logical.
            RegRegKind::ShiftArithmeticRight32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_shr(truncate_to_u32(rhs))),
            RegRegKind::ShiftArithmeticRight64 => cast_to_u64(cast_to_i64(lhs).wrapping_shr(truncate_to_u32(rhs))),
            RegRegKind::Xor64 => lhs ^ rhs,
            RegRegKind::Or64 => lhs | rhs,
            RegRegKind::And64 => lhs & rhs,

            // "M" extension:
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/m-st-ext.html#mult-ops
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/m-st-ext.html#11-1-2-division-operations
            RegRegKind::Mul32AndSignExtend => sign_extend_32(truncate_to_i32(lhs).wrapping_mul(truncate_to_i32(rhs))),
            RegRegKind::Mul64 => lhs.wrapping_mul(rhs),
            RegRegKind::MulUpperSignedSigned64 => {
                let product = i128::from(cast_to_i64(lhs)) * i128::from(cast_to_i64(rhs));
                cast_to_u64(truncate_to_i64(product >> u64::BITS))
            }
            RegRegKind::MulUpperSignedUnsigned64 => {
                let product = i128::from(cast_to_i64(lhs)) * i128::from(rhs);
                cast_to_u64(truncate_to_i64(product >> u64::BITS))
            }
            RegRegKind::MulUpperUnsignedUnsigned64 => {
                let product = u128::from(lhs) * u128::from(rhs);
                truncate_to_u64(product >> u64::BITS)
            }
            RegRegKind::Div32AndSignExtend => {
                let (dividend, divisor) = (truncate_to_i32(lhs), truncate_to_i32(rhs));
                let quotient = if divisor == 0 { -1 } else { dividend.wrapping_div(divisor) };
                sign_extend_32(quotient)
            }
            RegRegKind::Div64 => {
                let (dividend, divisor) = (cast_to_i64(lhs), cast_to_i64(rhs));
                let quotient = if divisor == 0 { -1 } else { dividend.wrapping_div(divisor) };
                cast_to_u64(quotient)
            }
            RegRegKind::DivUnsigned32AndSignExtend => {
                let (dividend, divisor) = (truncate_to_u32(lhs), truncate_to_u32(rhs));
                let quotient = if divisor == 0 { u32::MAX } else { dividend / divisor };
                sign_extend_32(cast_to_i32(quotient))
            }
            RegRegKind::DivUnsigned64 => {
                if rhs == 0 {
                    u64::MAX
                } else {
                    lhs / rhs
                }
            }
            RegRegKind::Rem32AndSignExtend => {
                let (dividend, divisor) = (truncate_to_i32(lhs), truncate_to_i32(rhs));
                let remainder = if divisor == 0 { dividend } else { dividend.wrapping_rem(divisor) };
                sign_extend_32(remainder)
            }
            RegRegKind::Rem64 => {
                let (dividend, divisor) = (cast_to_i64(lhs), cast_to_i64(rhs));
                let remainder = if divisor == 0 { dividend } else { dividend.wrapping_rem(divisor) };
                cast_to_u64(remainder)
            }
            RegRegKind::RemUnsigned32AndSignExtend => {
                let (dividend, divisor) = (truncate_to_u32(lhs), truncate_to_u32(rhs));
                let remainder = if divisor == 0 { dividend } else { dividend % divisor };
                sign_extend_32(cast_to_i32(remainder))
            }
            RegRegKind::RemUnsigned64 => {
                if rhs == 0 {
                    lhs
                } else {
                    lhs % rhs
                }
            }

            // "B" extension (Zbb):
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-1-logical-with-negate
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-4-integer-minimummaximum
            // https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#29-1-3-6-bitwise-rotation
            RegRegKind::AndInverted
            | RegRegKind::OrInverted
            | RegRegKind::Xnor
            | RegRegKind::Maximum
            | RegRegKind::MaximumUnsigned
            | RegRegKind::Minimum
            | RegRegKind::MinimumUnsigned
            | RegRegKind::RotateLeft32AndSignExtend
            | RegRegKind::RotateLeft64
            | RegRegKind::RotateRight32AndSignExtend
            | RegRegKind::RotateRight64 => todo!(),
        };
        self.set_reg(dst, result);
    }

    /// LUI and AUIPC of https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-4-1-integer-register-immediate-instructions
    fn load_upper_immediate(&mut self, dst: Reg, imm: u32, base: Option<u64>) {
        let value = match base {
            Some(base) => base.wrapping_add(sign_extend_32(cast_to_i32(imm))),
            None => sign_extend_32(cast_to_i32(imm)),
        };
        self.set_reg(dst, value);
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-5-1-unconditional-jumps
    fn jump_and_link(&mut self, dst: Reg, offset: u32, instruction_address: u64) {
        let target = instruction_address.wrapping_add(sign_extend_32(cast_to_i32(offset)));
        self.set_reg(dst, self.program_counter);
        self.set_program_counter(target);
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#ct-unconditional
    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/c-st-ext.html#27-1-4-control-transfer-instructions
    fn jump_and_link_register(&mut self, dst: Reg, base: Reg, value: i32) {
        let target = self.reg(base).wrapping_add(sign_extend_32(value)) & !1;
        self.set_reg(dst, self.program_counter);
        self.set_program_counter(Self::to_32bit_address(target));
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#1-1-5-2-conditional-branches
    fn branch(&mut self, kind: BranchKind, src1: Reg, src2: Reg, offset: u32, instruction_address: u64) {
        let lhs = self.reg(src1);
        let rhs = self.reg(src2);
        let taken = match kind {
            BranchKind::Eq64 => lhs == rhs,
            BranchKind::NotEq64 => lhs != rhs,
            BranchKind::LessSigned64 => cast_to_i64(lhs) < cast_to_i64(rhs),
            BranchKind::LessUnsigned64 => lhs < rhs,
            BranchKind::GreaterOrEqualSigned64 => cast_to_i64(lhs) >= cast_to_i64(rhs),
            BranchKind::GreaterOrEqualUnsigned64 => lhs >= rhs,
        };
        if taken {
            let target = instruction_address.wrapping_add(sign_extend_32(cast_to_i32(offset)));
            self.set_program_counter(target);
        }
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#ldst
    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv64.html#3-1-3-load-and-store-instructions
    fn load(&mut self, kind: LoadKind, dst: Reg, base: Reg, offset: i32) -> Result<(), Exception> {
        let address = Self::to_32bit_address(self.reg(base).wrapping_add(sign_extend_32(offset)));
        let bytes = self.read_memory_as_guest(address, kind.size(), MemoryAccess::Load)?;
        let value = match kind {
            LoadKind::I8 => sign_extend_8(i8::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::U8 => zero_extend_8(u8::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::I16 => sign_extend_16(i16::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::U16 => zero_extend_16(u16::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::I32 => sign_extend_32(i32::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::U32 => zero_extend_32(u32::from_le_bytes(bytes.try_into().unwrap())),
            LoadKind::U64 => u64::from_le_bytes(bytes.try_into().unwrap()),
        };
        self.set_reg(dst, value);

        Ok(())
    }

    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv32.html#ldst
    /// https://docs.riscv.org/reference/isa/v20260120/unpriv/rv64.html#3-1-3-load-and-store-instructions
    fn store(&mut self, kind: StoreKind, src: Reg, base: Reg, offset: i32) -> Result<(), Exception> {
        let address = Self::to_32bit_address(self.reg(base).wrapping_add(sign_extend_32(offset)));
        let width = match kind {
            StoreKind::U8 => 1,
            StoreKind::U16 => 2,
            StoreKind::U32 => 4,
            StoreKind::U64 => 8,
        };
        let bytes = self.reg(src).to_le_bytes();

        self.write_memory_as_guest(address, &bytes[..width])
    }

    /// RISC-V address space is 2^XLEN bytes and address arithmetic wraps at 2^XLEN (https://docs.riscv.org/reference/isa/v20260120/unpriv/intro.html#memory),
    /// but PolkaVM's address space is 2^32 bytes. Thus, this RISC-V execution environment always uses
    /// only the low 32 bits of load/store addresses and indirect jump targets before resolving them,
    /// so that sign-extended 32-bit pointers resolve as they do on PolkaVM, ensuring the differential
    /// comparison during fuzzing does not falsely diverge on them.
    fn to_32bit_address(address: u64) -> u64 {
        zero_extend_32(truncate_to_u32(address))
    }
}
