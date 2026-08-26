use crate::{program_from_elf, Config, TargetInstructionSet};
use polkavm_common::cast::cast;

#[cfg(test)]
fn create_elf(code_u32: &[u32]) -> Vec<u8> {
    create_elf_with_code_relocations(code_u32, &[])
}

/// What a code relocation names.
#[derive(Copy, Clone)]
enum RelocationTarget {
    /// A read-only datum the builder places for that purpose.
    Datum,
    /// The start of the code, which is what a `%pcrel_lo` names.
    Code,
}

/// An ELF whose code carries the given relocations, each as an offset into the code, an ELF
/// relocation type and what it names. The datum exists only when a relocation asks for it.
#[cfg(test)]
fn create_elf_with_code_relocations(code_u32: &[u32], code_relocations: &[(u64, u32, RelocationTarget)]) -> Vec<u8> {
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

    let datum_symbol = (!code_relocations.is_empty()).then(|| {
        let rodata_section = obj.add_section(
            obj.segment_name(StandardSegment::Data).to_vec(),
            b".rodata".to_vec(),
            SectionKind::ReadOnlyData,
        );

        obj.append_section_data(rodata_section, &[0xaa; 32], 16);

        obj.add_symbol(Symbol {
            name: b"DATUM".to_vec(),
            value: 0,
            size: 32,
            kind: SymbolKind::Data,
            scope: SymbolScope::Linkage,
            weak: false,
            section: SymbolSection::Section(rodata_section),
            flags: SymbolFlags::None,
        })
    });

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

    for &(offset, r_type, target) in code_relocations {
        obj.add_relocation(
            text_section,
            Relocation {
                offset,
                symbol: match target {
                    RelocationTarget::Datum => datum_symbol.unwrap(),
                    RelocationTarget::Code => symbol_start,
                },
                addend: 0,
                flags: RelocationFlags::Elf { r_type },
            },
        )
        .unwrap();
    }

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

#[test]
fn metadata_hash_is_embedded_in_blob() {
    let _ = env_logger::try_init();

    let bytes = create_elf(&[0x00b50533, 0x00008067]);
    let metadata_hash = [0x42u8; 32];

    let mut config = Config::default();
    config.set_optimize(false);
    config.set_metadata_hash(Some(metadata_hash.to_vec()));
    let program = program_from_elf(config, TargetInstructionSet::Latest, &bytes).unwrap();

    let blob = polkavm_common::program::ProgramBlob::parse(program.as_slice().into()).unwrap();
    assert_eq!(blob.metadata_hash(), metadata_hash);
}

// The words below are custom-2 instructions as the compiler assembles them, `.i128` being the
// 128-bit spelling of a mnemonic, so a change on either side shows up here. Every fold is
// exercised at both widths, and the two words of a pair differ in a single bit -- the top of
// `funct7` for a register shape, the middle of `funct3` for a memory one -- so each case also
// shows that neither width is translated as the other.
const WIDE_ADD_128: u32 = 0x80a4045b; // revive.wadd.i128 v8, v8, v10
const WIDE_LOAD_128: u32 = 0x0105645b; // revive.wld.i128 v8, 16(a0)
const WIDE_LESS_THAN_128: u32 = 0x84a4155b; // revive.wsltu.i128 a0, v8, v10
const WIDE_MOVE_128: u32 = 0x980404db; // revive.wmv.i128 v9, v8
const WIDE_TO_REG_128: u32 = 0x9a04055b; // revive.wtrunc.i128 a0, v8
const WIDE_COUNT_128: u32 = 0xa004055b; // revive.wclz.i128 a0, v8
const WIDE_STORE_128: u32 = 0x0085785b; // revive.wst.i128 v8, 16(a0)
const WIDE_SHIFT_128: u32 = 0x80a4245b; // revive.wsll.i128 v8, v8, a0
const WIDE_SHIFT: u32 = 0x00a4245b; // revive.wsll w4, w4, a0
const WIDE_FROM_REG_128: u32 = 0xa405045b; // revive.wzext.i128 v8, a0
const WIDE_FROM_REG: u32 = 0x2405045b; // revive.wzext v8, a0
const WIDE_LOAD_128_A1: u32 = 0x0105e45b; // revive.wld.i128 v8, 16(a1)
const WIDE_LOAD_A1: u32 = 0x0105c45b; // revive.wld w4, 16(a1)
const LOAD_UPPER_A1: u32 = 0x000005b7; // lui a1, 0
const ADD_UPPER_TO_PC_A1: u32 = 0x00000597; // auipc a1, 0
const LOAD_IMMEDIATE_A0: u32 = 0x00500513; // li a0, 5
const RETURN: u32 = 0x00008067; // ret

/// The code linked for the revive instruction set, disassembled.
fn link_revive(code_u32: &[u32], code_relocations: &[(u64, u32, RelocationTarget)], optimize: bool) -> String {
    let _ = env_logger::try_init();

    let bytes = create_elf_with_code_relocations(code_u32, code_relocations);
    let mut config = Config::default();
    config.set_optimize(optimize);
    let program = program_from_elf(config, TargetInstructionSet::ReviveV1, &bytes).unwrap();
    disassemble(&program)
}

fn assert_disassembly(disassembly: &str, expected: &str) {
    assert_eq!(disassembly.trim(), expected.trim().replace("        ", ""));
}

#[test]
fn the_128_bit_instructions_translate_to_their_own_opcodes() {
    // One case per operand shape a 128-bit instruction has, so an operand order transposed on
    // the way through the encoding shows up here rather than silently.
    let disassembly = link_revive(
        &[
            WIDE_ADD_128,
            WIDE_LESS_THAN_128,
            WIDE_MOVE_128,
            WIDE_TO_REG_128,
            WIDE_COUNT_128,
            WIDE_LOAD_128,
            WIDE_STORE_128,
            RETURN,
        ],
        &[],
        false,
    );

    assert_disassembly(
        &disassembly,
        "<_start>:
        @0 [export #0: '_start']
        v8 = v8 +w128 v10
        a0 = v8 <uw128 v10
        v9 = v8
        a0 = truncate v8
        a0 = clz v8
        v8 = u128 [a0 + 0x10]
        u128 [a0 + 0x10] = v8
        ret",
    );
}

#[test]
fn a_shift_by_a_known_amount_folds_at_either_width() {
    let disassembly = link_revive(&[LOAD_IMMEDIATE_A0, WIDE_SHIFT_128, WIDE_SHIFT, RETURN], &[], true);

    assert_disassembly(
        &disassembly,
        "<_start>:
        @0 [export #0: '_start']
        a0 = 0x5
        v8 = v8 <<w128 0x5
        w4 = w4 <<w 0x5
        ret",
    );
}

#[test]
fn widening_a_known_value_folds_at_either_width() {
    let disassembly = link_revive(&[LOAD_IMMEDIATE_A0, WIDE_FROM_REG_128, WIDE_FROM_REG, RETURN], &[], true);

    assert_disassembly(
        &disassembly,
        "<_start>:
        @0 [export #0: '_start']
        a0 = 0x5
        v8 = u64 0x5
        w4 = u64 0x5
        ret",
    );
}

#[test]
fn a_load_from_a_known_address_folds_with_the_reconstructed_offset() {
    // The datum is at 0x10000 and both loads ask for sixteen bytes past it, so both fold to the
    // same address. The two widths reach it through different instructions, and an address the
    // fold got wrong would still be a valid instruction.
    let disassembly = link_revive(
        &[LOAD_UPPER_A1, WIDE_LOAD_128_A1, WIDE_LOAD_A1, RETURN],
        &[(0, object::elf::R_RISCV_HI20, RelocationTarget::Datum)],
        true,
    );

    assert_disassembly(
        &disassembly,
        "<_start>:
        @0 [export #0: '_start']
        a1 = 0x10000
        v8 = u128 [0x10010]
        w4 = u256 [0x10010]
        ret",
    );
}

#[test]
fn a_relocation_against_a_wide_instruction_is_a_link_error() {
    // The compiler never emits one, and the linker never quietly accepts one: patching a low
    // twelve bits means replacing the whole instruction with one built for a known address, and
    // only the instructions on an allowlist have such a form. No wide access is on it, so both
    // widths are rejected the same way.
    for (word, instruction) in [(WIDE_LOAD_128_A1, "Wide128Load"), (WIDE_LOAD_A1, "WideLoad")] {
        let bytes = create_elf_with_code_relocations(&[word, RETURN], &[(0, object::elf::R_RISCV_LO12_I, RelocationTarget::Datum)]);

        let error = program_from_elf(Config::default(), TargetInstructionSet::ReviveV1, &bytes)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("R_RISCV_LO12_I for an unsupported instruction") && error.contains(instruction),
            "{error}"
        );
    }

    for (word, instruction) in [(WIDE_LOAD_128_A1, "Wide128Load"), (WIDE_LOAD_A1, "WideLoad")] {
        let bytes = create_elf_with_code_relocations(
            &[ADD_UPPER_TO_PC_A1, word, RETURN],
            &[
                (0, object::elf::R_RISCV_PCREL_HI20, RelocationTarget::Datum),
                (4, object::elf::R_RISCV_PCREL_LO12_I, RelocationTarget::Code),
            ],
        );

        let error = program_from_elf(Config::default(), TargetInstructionSet::ReviveV1, &bytes)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("relocation (with R_RISCV_PCREL_HI20 as the upper relocation) for an unsupported instruction")
                && error.contains(instruction),
            "{error}"
        );
    }
}

#[test]
fn metadata_hash_is_absent_by_default() {
    let _ = env_logger::try_init();

    let bytes = create_elf(&[0x00b50533, 0x00008067]);
    let mut config = Config::default();
    config.set_optimize(false);
    let program = program_from_elf(config, TargetInstructionSet::Latest, &bytes).unwrap();

    let blob = polkavm_common::program::ProgramBlob::parse(program.as_slice().into()).unwrap();
    assert!(blob.metadata_hash().is_empty());
}
