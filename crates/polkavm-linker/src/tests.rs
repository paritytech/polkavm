use crate::{program_from_elf, Config, TargetInstructionSet};
use polkavm_common::cast::cast;

#[cfg(test)]
fn create_elf(code_u32: &[u32]) -> Vec<u8> {
    use object::write::{Object, Relocation, StandardSegment, Symbol, SymbolSection};
    use object::{Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags, SymbolKind, SymbolScope};

    let mut obj = Object::new(BinaryFormat::Elf, Architecture::Riscv64, Endianness::Little);

    let text_section = obj.add_section(
        obj.segment_name(StandardSegment::Text).to_vec(),
        b".text".to_vec(),
        SectionKind::Text,
    );

    let mut code = Vec::new();
    for &inst in code_u32 {
        code.extend_from_slice(&inst.to_le_bytes());
    }

    obj.append_section_data(text_section, &code, 1);

    let symbol_start = obj.add_symbol(Symbol {
        name: b"_start".to_vec(),
        value: 0,
        size: cast(code.len()).to_u64(),
        kind: SymbolKind::Text,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    let metadata_section = obj.add_section(
        obj.segment_name(StandardSegment::Text).to_vec(),
        b".polkavm_metadata".to_vec(),
        SectionKind::ReadOnlyData,
    );

    let symbol_name = b"_start";
    let symbol_name_offset = 0u64;
    let metadata_offset = symbol_name_offset + symbol_name.len() as u64;

    obj.append_section_data(metadata_section, symbol_name, 1);

    let mut metadata_bytes = Vec::new();
    // version: u8 = 1
    metadata_bytes.push(1u8);
    // flags: u32 = 0
    metadata_bytes.extend_from_slice(&0u32.to_le_bytes());
    // symbol_length: u32
    metadata_bytes.extend_from_slice(&(symbol_name.len() as u32).to_le_bytes());
    // symbol: u64
    metadata_bytes.extend_from_slice(&0u64.to_le_bytes());
    // input_regs: u8 = 0
    metadata_bytes.push(0u8);
    // output_regs: u8 = 0
    metadata_bytes.push(0u8);

    obj.append_section_data(metadata_section, &metadata_bytes, 1);

    let metadata_symbol = obj.add_symbol(Symbol {
        name: b"_polkavm_export_metadata__start".to_vec(),
        value: metadata_offset,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(metadata_section),
        flags: SymbolFlags::None,
    });

    let metadata_section_symbol = obj.section_symbol(metadata_section);
    obj.add_relocation(
        metadata_section,
        Relocation {
            offset: metadata_offset + 9,
            symbol: metadata_section_symbol,
            addend: 0,
            flags: RelocationFlags::Elf {
                r_type: object::elf::R_RISCV_64,
            },
        },
    )
    .unwrap();

    let exports_section = obj.add_section(
        obj.segment_name(StandardSegment::Text).to_vec(),
        b".polkavm_exports".to_vec(),
        SectionKind::ReadOnlyData,
    );

    let mut exports_data = Vec::new();
    exports_data.push(1u8);
    exports_data.extend_from_slice(&[0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    exports_data.extend_from_slice(&[0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

    obj.append_section_data(exports_section, &exports_data, 1);

    obj.add_relocation(
        exports_section,
        Relocation {
            offset: 1,
            symbol: metadata_symbol,
            addend: 0,
            flags: RelocationFlags::Elf {
                r_type: object::elf::R_RISCV_PCREL_HI20,
            },
        },
    )
    .unwrap();

    obj.add_relocation(
        exports_section,
        Relocation {
            offset: 9,
            symbol: symbol_start,
            addend: 0,
            flags: RelocationFlags::Elf {
                r_type: object::elf::R_RISCV_PCREL_HI20,
            },
        },
    )
    .unwrap();

    obj.write().unwrap()
}

fn disassemble(program: &[u8]) -> String {
    let blob = polkavm_common::program::ProgramBlob::parse(program.into()).unwrap();
    let mut disassembler = polkavm_disassembler::Disassembler::new(&blob, polkavm_disassembler::DisassemblyFormat::Guest).unwrap();
    disassembler.emit_header(false);
    disassembler.show_offsets(false);
    let mut buf = Vec::new();
    disassembler.disassemble_into(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

#[test]
fn trap_is_injected_at_the_end() {
    let _ = env_logger::try_init();

    let bytes = create_elf(&[0x00b50533, 0xfe050ee3]);
    let mut config = Config::default();
    config.set_optimize(false);
    let program = program_from_elf(config, TargetInstructionSet::Latest, &bytes).unwrap();
    let disassembly = disassemble(&program);

    assert_eq!(
        disassembly.trim(),
        "<_start>:
        @0 [@dyn 1] [export #0: '_start']
        a0 = a0 + a1
        jump @0 if a0 == 0

        @1 [@dyn 2]
        trap"
            .trim()
            .replace("        ", "")
    );
}

#[test]
fn trap_is_not_injected_at_the_end() {
    let _ = env_logger::try_init();

    let bytes = create_elf(&[0x00b50533, 0x00008067]);
    let mut config = Config::default();
    config.set_optimize(false);
    let program = program_from_elf(config, TargetInstructionSet::Latest, &bytes).unwrap();
    let disassembly = disassemble(&program);

    assert_eq!(
        disassembly.trim(),
        "<_start>:
        @0 [export #0: '_start']
        a0 = a0 + a1
        ret"
        .trim()
        .replace("        ", "")
    );
}
