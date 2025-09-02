#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[repr(u8)]
#[derive(Arbitrary, Debug)]
pub enum Reg {
    Zero = 0,
    RA = 1,
    SP = 2,
    // These registers are not supported
    // GP = 3,
    // TP = 4,
    T0 = 5,
    T1 = 6,
    T2 = 7,
    S0 = 8,
    S1 = 9,
    A0 = 10,
    A1 = 11,
    A2 = 12,
    A3 = 13,
    A4 = 14,
    A5 = 15,
}

#[derive(Arbitrary, Debug)]
enum RegKind {
    CountLeadingZeroBits,
    CountLeadingZeroBitsInWord,
    CountSetBits,
    CountSetBitsInWord,
    CountTrailingZeroBits,
    CountTrailingZeroBitsInWord,
    OrCombineByte,
    ReverseByte,
    SignExtend8,
    SignExtend16,
    ZeroExtend16,
}

#[derive(Arbitrary, Debug)]
enum RegRegKind {
    Add32AndSignExtend,
    Add64,
    Sub32AndSignExtend,
    Sub64,
    Mul32AndSignExtend,
    Mul64,
    MulUpperSignedSigned64,
    MulUpperSignedUnsigned64,
    MulUpperUnsignedUnsigned64,
    Div32AndSignExtend,
    Div64,
    DivUnsigned32AndSignExtend,
    DivUnsigned64,
    Rem32AndSignExtend,
    Rem64,
    RemUnsigned32AndSignExtend,
    RemUnsigned64,
    Xor64,
    Or64,
    And64,
    ShiftLogicalLeft32AndSignExtend,
    ShiftLogicalLeft64,
    ShiftLogicalRight32AndSignExtend,
    ShiftLogicalRight64,
    ShiftArithmeticRight32AndSignExtend,
    ShiftArithmeticRight64,
    SetLessThanSigned64,
    SetLessThanUnsigned64,
    AndInverted,
    OrInverted,
    Xnor,
    Maximum,
    MaximumUnsigned,
    Minimum,
    MinimumUnsigned,
    RotateLeft32AndSignExtend,
    RotateLeft64,
    RotateRight32AndSignExtend,
    RotateRight64,
}

#[derive(Arbitrary, Debug)]
enum RegImmKind {
    Add32AndSignExtend,
    Add64,
    SetLessThanSigned64,
    SetLessThanUnsigned64,
    Xor64,
    Or64,
    And64,
    ShiftLogicalLeft32AndSignExtend,
    ShiftLogicalLeft64,
    ShiftLogicalRight32AndSignExtend,
    ShiftLogicalRight64,
    ShiftArithmeticRight32AndSignExtend,
    ShiftArithmeticRight64,
    RotateRight32AndSignExtend,
    RotateRight64,
}

#[derive(Arbitrary, Debug)]
enum Instruction {
    Reg { kind: RegKind, dst: Reg, src: Reg },
    RegReg { kind: RegRegKind, dst: Reg, src1: Reg, src2: Reg },
    RegImm { kind: RegImmKind, dst: Reg, src: Reg, imm: u32 },
}

fn serialize_instructions(data: Vec<Instruction>) -> Vec<u8> {
    let mut buffer = Vec::new();
    for instruction in data {
        match instruction {
            Instruction::Reg { kind, dst, src } => {
                let mut encoding: u32 = match kind {
                    RegKind::CountLeadingZeroBits => 0b0110000_00000_00000_001_00000_0010011,
                    RegKind::CountLeadingZeroBitsInWord => 0b0110000_00000_00000_001_00000_0011011,
                    RegKind::CountSetBits => 0b0110000_00010_00000_001_00000_0010011,
                    RegKind::CountSetBitsInWord => 0b0110000_00010_00000_001_00000_0011011,
                    RegKind::CountTrailingZeroBits => 0b0110000_00001_00000_001_00000_0010011,
                    RegKind::CountTrailingZeroBitsInWord => 0b0110000_00001_00000_001_00000_0011011,
                    RegKind::OrCombineByte => 0b0010100_00111_00000_101_00000_0010011,
                    RegKind::ReverseByte => 0b0110101_11000_00000_101_00000_0010011,
                    RegKind::SignExtend8 => 0b0110000_00100_00000_001_00000_0010011,
                    RegKind::SignExtend16 => 0b0110000_00101_00000_001_00000_0010011,
                    RegKind::ZeroExtend16 => 0b0000100_00000_00000_100_00000_0111011,
                };

                encoding |= (dst as u32) << 7;
                encoding |= (src as u32) << 15;
                buffer.extend_from_slice(&encoding.to_le_bytes());
            }
            Instruction::RegReg { kind, dst, src1, src2 } => {
                let mut encoding: u32 = match kind {
                    RegRegKind::Add32AndSignExtend => 0b0000000_00000_00000_000_00000_0111011,
                    RegRegKind::Sub32AndSignExtend => 0b0100000_00000_00000_000_00000_0111011,
                    RegRegKind::Mul32AndSignExtend => 0b0000001_00000_00000_000_00000_0111011,
                    RegRegKind::Div32AndSignExtend => 0b0000001_00000_00000_100_00000_0111011,
                    RegRegKind::DivUnsigned32AndSignExtend => 0b0000001_00000_00000_101_00000_0111011,
                    RegRegKind::Rem32AndSignExtend => 0b0000001_00000_00000_110_00000_0111011,
                    RegRegKind::RemUnsigned32AndSignExtend => 0b0000001_00000_00000_111_00000_0111011,
                    RegRegKind::ShiftLogicalLeft32AndSignExtend => 0b0000000_00000_00000_001_00000_0111011,
                    RegRegKind::ShiftLogicalRight32AndSignExtend => 0b0000001_00000_00000_000_00000_0111011,
                    RegRegKind::ShiftArithmeticRight32AndSignExtend => 0b0100000_00000_00000_101_00000_0111011,
                    RegRegKind::RotateLeft32AndSignExtend => 0b0110000_00000_00000_001_00000_0111011,
                    RegRegKind::RotateRight32AndSignExtend => 0b0110000_00000_00000_101_00000_0111011,

                    RegRegKind::Add64 => 0b0000000_00000_00000_000_00000_0110011,
                    RegRegKind::Sub64 => 0b0100000_00000_00000_000_00000_0110011,
                    RegRegKind::Mul64 => 0b0000001_00000_00000_000_00000_0110011,
                    RegRegKind::Div64 => 0b0000001_00000_00000_100_00000_0110011,
                    RegRegKind::DivUnsigned64 => 0b0000001_00000_00000_101_00000_0110011,
                    RegRegKind::Rem64 => 0b0000001_00000_00000_110_00000_0110011,
                    RegRegKind::RemUnsigned64 => 0b0000001_00000_00000_111_00000_0110011,
                    RegRegKind::ShiftLogicalLeft64 => 0b0000000_00000_00000_001_00000_0110011,
                    RegRegKind::ShiftLogicalRight64 => 0b0000000_00000_00000_101_00000_0110011,
                    RegRegKind::ShiftArithmeticRight64 => 0b0100000_00000_00000_101_00000_0110011,
                    RegRegKind::RotateLeft64 => 0b0110000_00000_00000_001_00000_0110011,
                    RegRegKind::RotateRight64 => 0b0110000_00000_00000_101_00000_0110011,

                    RegRegKind::MulUpperSignedSigned64 => 0b0000001_00000_00000_001_00000_0110011,
                    RegRegKind::MulUpperSignedUnsigned64 => 0b0000001_00000_00000_010_00000_0110011,
                    RegRegKind::MulUpperUnsignedUnsigned64 => 0b0000001_00000_00000_011_00000_0110011,

                    RegRegKind::SetLessThanSigned64 => 0b0000000_00000_00000_010_00000_0110011,
                    RegRegKind::SetLessThanUnsigned64 => 0b0000000_00000_00000_011_00000_0110011,
                    RegRegKind::Xor64 => 0b0000000_00000_00000_100_00000_0110011,
                    RegRegKind::Or64 => 0b0000000_00000_00000_110_00000_0110011,
                    RegRegKind::And64 => 0b0000000_00000_00000_111_00000_0110011,

                    RegRegKind::Minimum => 0b0000101_00000_00000_100_00000_0110011,
                    RegRegKind::MinimumUnsigned => 0b0000101_00000_00000_101_00000_0110011,
                    RegRegKind::Maximum => 0b0000101_00000_00000_110_00000_0110011,
                    RegRegKind::MaximumUnsigned => 0b0000101_00000_00000_111_00000_0110011,
                    RegRegKind::Xnor => 0b0100000_00000_00000_100_00000_0110011,
                    RegRegKind::OrInverted => 0b0100000_00000_00000_110_00000_0110011,
                    RegRegKind::AndInverted => 0b0100000_00000_00000_111_00000_0110011,
                };

                encoding |= (dst as u32) << 7;
                encoding |= (src1 as u32) << 15;
                encoding |= (src2 as u32) << 20;
                buffer.extend_from_slice(&encoding.to_le_bytes());
            }
            Instruction::RegImm { kind, dst, src, imm } => {
                let mut encoding: u32 = match kind {
                    RegImmKind::Add32AndSignExtend => 0b0000000_00000_00000_000_00000_0011011,
                    RegImmKind::ShiftLogicalLeft32AndSignExtend => 0b0000000_00000_00000_001_00000_0011011,
                    RegImmKind::ShiftLogicalRight32AndSignExtend => 0b0000000_00000_00000_101_00000_0011011,
                    RegImmKind::ShiftArithmeticRight32AndSignExtend => 0b0100000_00000_00000_101_00000_0011011,
                    RegImmKind::RotateRight32AndSignExtend => 0b0110000_00000_00000_101_00000_0011011,

                    RegImmKind::ShiftLogicalLeft64 => 0b0000000_00000_00000_001_00000_0010011,
                    RegImmKind::ShiftLogicalRight64 => 0b0000000_00000_00000_101_00000_0010011,
                    RegImmKind::ShiftArithmeticRight64 => 0b0100000_00000_00000_101_00000_0010011,

                    RegImmKind::RotateRight64 => 0b0110000_00000_00000_101_00000_0010011,

                    RegImmKind::Add64 => 0b0000000_00000_00000_000_00000_0010011,
                    RegImmKind::SetLessThanSigned64 => 0b0000000_00000_00000_010_00000_0010011,
                    RegImmKind::SetLessThanUnsigned64 => 0b0000000_00000_00000_011_00000_0010011,
                    RegImmKind::Xor64 => 0b0000000_00000_00000_100_00000_0010011,
                    RegImmKind::Or64 => 0b0000000_00000_00000_110_00000_0010011,
                    RegImmKind::And64 => 0b0000000_00000_00000_111_00000_0010011,
                };

                let imm_mask: u32 = match kind {
                    RegImmKind::Add32AndSignExtend => 0b111111111111,
                    _ => 0b11111,
                };

                encoding |= (dst as u32) << 7;
                encoding |= (src as u32) << 15;
                encoding |= ((imm & imm_mask) as u32) << 20;
                buffer.extend_from_slice(&encoding.to_le_bytes());
            }
        }
    }
    buffer
}

fuzz_target!(|instructions: Vec<Instruction>| {
    let bytecode = serialize_instructions(instructions);
    let program = create_minimal_elf(&bytecode);

    let mut config = polkavm_linker::Config::default();
    config.set_strip(true);
    config.set_optimize(true);

    polkavm_linker::program_from_elf(config, &program).unwrap();
});

#[repr(C)]
struct Elf64Ehdr {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[repr(C)]
struct Elf64Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

#[repr(C)]
struct Elf64Shdr {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

fn create_minimal_elf(bytecode: &[u8]) -> Vec<u8> {
    assert!(std::mem::size_of::<Elf64Ehdr>() == 64);
    assert!(std::mem::size_of::<Elf64Phdr>() == 56);
    assert!(std::mem::size_of::<Elf64Shdr>() == 64);

    let elf_hdr = Elf64Ehdr {
        e_ident: [
            0x7f, b'E', b'L', b'F', // Magic number
            2,    // 64-bit
            1,    // Little-endian
            1,    // ELF version
            0,    // ABI
            0,    // ABI version
            0, 0, 0, 0, 0, 0, 0,
        ],
        e_type: 2,       // Executable file
        e_machine: 0xF3, // RISC-V architecture
        e_version: 1,    // ELF version
        e_entry: 0x1000, // Entry point address
        e_phoff: 0,      // Program header table file offset
        e_shoff: 64,     // Section header table file offset
        e_flags: 0,      // Processor-specific flags
        e_ehsize: 64,    // ELF header size
        e_phentsize: 0,  // Program header table entry size
        e_phnum: 0,      // Number of program header entries
        e_shentsize: 64, // Section header table entry size
        e_shnum: 3,      // Number of section header entries
        e_shstrndx: 2,   // Section header string table index
    };

    let null_shdr = Elf64Shdr {
        sh_name: 0,      // Section name (index into the section header string table)
        sh_type: 0,      // Section type (NULL)
        sh_flags: 0,     // Section flags
        sh_addr: 0,      // Section virtual address
        sh_offset: 0,    // Section file offset
        sh_size: 0,      // Section size in file
        sh_link: 0,      // Link to another section
        sh_info: 0,      // Additional section information
        sh_addralign: 0, // Section alignment
        sh_entsize: 0,   // Entry size if section holds table
    };

    let text_shdr = Elf64Shdr {
        sh_name: 1,                     // Section name (index into the section header string table)
        sh_type: 1,                     // Section type (PROGBITS)
        sh_flags: 6,                    // Section flags (ALLOC + EXECINSTR)
        sh_addr: 0x1000,                // Section virtual address
        sh_offset: 64 * 4 + 32,         // Section file offset
        sh_size: bytecode.len() as u64, // Section size in file
        sh_link: 0,                     // Link to another section
        sh_info: 0,                     // Additional section information
        sh_addralign: 4,                // Section alignment
        sh_entsize: 0,                  // Entry size if section holds table
    };

    let shstrtab_shdr = Elf64Shdr {
        sh_name: 7,        // Section name (index into the section header string table)
        sh_type: 3,        // Section type (STRTAB)
        sh_flags: 0,       // Section flags
        sh_addr: 0,        // Section virtual address
        sh_offset: 64 * 4, // Section file offset
        sh_size: 32,       // Section size in file
        sh_link: 0,        // Link to another section
        sh_info: 0,        // Additional section information
        sh_addralign: 1,   // Section alignment
        sh_entsize: 0,     // Entry size if section holds table
    };

    let mut program = Vec::new();
    program.extend_from_slice(unsafe {
        std::slice::from_raw_parts((&elf_hdr as *const Elf64Ehdr) as *const u8, std::mem::size_of::<Elf64Ehdr>())
    });
    program.extend_from_slice(unsafe {
        std::slice::from_raw_parts((&null_shdr as *const Elf64Shdr) as *const u8, std::mem::size_of::<Elf64Shdr>())
    });
    program.extend_from_slice(unsafe {
        std::slice::from_raw_parts((&text_shdr as *const Elf64Shdr) as *const u8, std::mem::size_of::<Elf64Shdr>())
    });
    program.extend_from_slice(unsafe {
        std::slice::from_raw_parts((&shstrtab_shdr as *const Elf64Shdr) as *const u8, std::mem::size_of::<Elf64Shdr>())
    });
    program.extend_from_slice(b"\0.text\0.shstrtab\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0");
    program.extend_from_slice(&bytecode);

    program
}
