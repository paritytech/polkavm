use object::{read::elf::Sym, LittleEndian, Object, ObjectSection, ObjectSymbol};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use crate::program_from_elf::{ProgramFromElfError, SectionTarget};

type ElfFile<'a, H> = object::read::elf::ElfFile<'a, H, &'a [u8]>;
type ElfSectionIndex = object::read::SectionIndex;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SectionIndex(usize);

impl SectionIndex {
    #[cfg(test)]
    pub fn new(value: usize) -> Self {
        SectionIndex(value)
    }

    pub fn raw(self) -> usize {
        self.0
    }
}

impl core::fmt::Display for SectionIndex {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(fmt, "section #{}", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
enum SymbolSection {
    None,
    Undefined,
    Absolute,
    Section(SectionTarget),
}

pub struct Section<'a> {
    index: SectionIndex,
    name: String,
    original_address: u64,
    align: u64,
    size: u64,
    data: Cow<'a, [u8]>,
    flags: u64,
    raw_section_index: Option<ElfSectionIndex>,
    relocations: Vec<(u64, Relocation)>,
    elf_section_type: u32,
}

impl<'a> Section<'a> {
    pub fn index(&self) -> SectionIndex {
        self.index
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn original_address(&self) -> u64 {
        self.original_address
    }

    pub fn align(&self) -> u64 {
        self.align
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn is_writable(&self) -> bool {
        self.flags & u64::from(object::elf::SHF_WRITE) != 0
    }

    pub fn is_allocated(&self) -> bool {
        self.flags & u64::from(object::elf::SHF_ALLOC) != 0
    }

    pub fn is_executable(&self) -> bool {
        self.flags & u64::from(object::elf::SHF_EXECINSTR) != 0
    }

    pub fn is_progbits(&self) -> bool {
        self.elf_section_type == object::elf::SHT_PROGBITS
    }

    pub fn is_nobits(&self) -> bool {
        self.elf_section_type == object::elf::SHT_NOBITS
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn relocations(&'_ self) -> impl Iterator<Item = (u64, Relocation)> + '_ {
        self.relocations.iter().copied()
    }

    pub fn elf_section_type(&self) -> u32 {
        self.elf_section_type
    }
}

#[derive(Clone)]
pub struct Symbol {
    name: Option<String>,
    address: u64,
    size: u64,
    kind: u8,
    section: SymbolSection,
}

impl Symbol {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self.section, SymbolSection::Undefined)
    }

    pub fn section_target(&self) -> Result<SectionTarget, ProgramFromElfError> {
        match self.section {
            SymbolSection::Section(section_target) => Ok(section_target),
            SymbolSection::Undefined => Err(ProgramFromElfError::other(format!(
                "found undefined symbol: '{}'",
                self.name().unwrap_or("")
            ))),
            section => Err(ProgramFromElfError::other(format!(
                "found symbol in an unhandled section: {:?}",
                section
            ))),
        }
    }

    pub fn original_address(&self) -> u64 {
        self.address
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn kind(&self) -> u8 {
        self.kind
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Relocation {
    kind: object::RelocationKind,
    encoding: object::RelocationEncoding,
    flags: object::RelocationFlags,
    target: object::RelocationTarget,
    size: u8,
    addend: i64,
}

impl Relocation {
    pub fn kind(&self) -> object::RelocationKind {
        self.kind
    }

    pub fn encoding(&self) -> object::RelocationEncoding {
        self.encoding
    }

    pub fn size(&self) -> u8 {
        self.size
    }

    pub fn target(&self) -> object::RelocationTarget {
        self.target
    }

    pub fn addend(&self) -> i64 {
        self.addend
    }

    pub fn flags(&self) -> object::RelocationFlags {
        self.flags
    }

    #[allow(clippy::unused_self)]
    pub fn has_implicit_addend(&self) -> bool {
        false
    }
}

#[cfg_attr(test, derive(Default))]
pub struct Elf<'data> {
    sections: Vec<Section<'data>>,
    section_index_by_name: HashMap<String, Vec<SectionIndex>>,
    symbols: Vec<Symbol>,
    is_64_bit: bool,
}

impl<'data> Elf<'data> {
    pub fn parse<H>(data: &'data [u8]) -> Result<Self, ProgramFromElfError>
    where
        H: object::read::elf::FileHeader<Endian = object::LittleEndian>,
    {
        let elf: ElfFile<H> = ElfFile::parse(data)?;
        if elf.elf_header().e_ident().data != object::elf::ELFDATA2LSB {
            return Err(ProgramFromElfError::other("file is not a little endian ELF file"));
        }

        let os_abi = elf.elf_header().e_ident().os_abi;
        if os_abi != object::elf::ELFOSABI_SYSV && os_abi != object::elf::ELFOSABI_GNU {
            return Err(ProgramFromElfError::other("file doesn't use the System V nor GNU ABI"));
        }

        if !matches!(
            elf.elf_header().e_type(LittleEndian),
            object::elf::ET_EXEC | object::elf::ET_REL | object::elf::ET_DYN
        ) {
            return Err(ProgramFromElfError::other(
                "file is not a supported ELF file (ET_EXEC or ET_REL or ET_DYN)",
            ));
        }

        if elf.elf_header().e_machine(LittleEndian) != object::elf::EM_RISCV {
            return Err(ProgramFromElfError::other("file is not a RISC-V file (EM_RISCV)"));
        }

        let is_64_bit = elf.is_64();

        let mut relocation_sections_for_section = HashMap::new();
        let mut relocations_in_section = HashMap::new();
        let mut sections = Vec::new();
        let mut section_index_by_name = HashMap::new();
        let mut section_index_map = HashMap::new();
        for section in elf.sections() {
            let name = section.name()?;
            let file_range = section.file_range().unwrap_or((0, 0));
            let file_start = usize::try_from(file_range.0)
                .map_err(|_| ProgramFromElfError::other(format!("out of range offset for '{name}' section")))?;

            let file_size =
                usize::try_from(file_range.1).map_err(|_| ProgramFromElfError::other(format!("out of range size for '{name}' section")))?;

            let file_end = file_start
                .checked_add(file_size)
                .ok_or_else(|| ProgramFromElfError::other(format!("out of range '{name}' section (overflow)")))?;

            let file_range = file_start..file_end;
            let file_data = data
                .get(file_range)
                .ok_or_else(|| ProgramFromElfError::other(format!("out of range '{name}' section (out of bounds of ELF file)")))?;

            if section.size() < file_data.len() as u64 {
                return Err(ProgramFromElfError::other(format!(
                    "section '{name}' has bigger physical size than virtual size"
                )));
            }

            let object::SectionFlags::Elf { sh_flags: flags } = section.flags() else {
                unreachable!()
            };

            let index = SectionIndex(sections.len());

            // The way the 'object' crate handles relocations is broken if there are multiple
            // '.rela' sections per section, so let's just manually parse the relocations ourselves.
            use object::read::elf::SectionHeader;
            if section.elf_section_header().sh_type(LittleEndian) == object::elf::SHT_RELA {
                use object::read::elf::Rela;

                // '.rela' sections have the index of the section they should be applied to in their 'sh_type'.
                let info = section.elf_section_header().sh_info(LittleEndian);
                relocation_sections_for_section
                    .entry(info as usize)
                    .or_insert_with(Vec::new)
                    .push(index);

                let mut relocations: Vec<(u64, Relocation)> = Vec::new();
                let section_data = section.data()?;
                let mut struct_offset = 0;
                while struct_offset < section_data.len() {
                    let struct_size = if is_64_bit {
                        core::mem::size_of::<object::elf::Rela64<LittleEndian>>()
                    } else {
                        core::mem::size_of::<object::elf::Rela32<LittleEndian>>()
                    };
                    let rela = section_data
                        .get(struct_offset..struct_offset + struct_size)
                        .ok_or_else(|| ProgramFromElfError::other("failed to parse relocations"))?;
                    struct_offset += struct_size;

                    let r_type;
                    let symbol;
                    let addend;
                    let offset;
                    if is_64_bit {
                        // SAFETY: The pointer is always within bounds since we used '.get'.
                        let rela = unsafe { rela.as_ptr().cast::<object::elf::Rela64<LittleEndian>>().read_unaligned() };
                        r_type = rela.r_type(LittleEndian, false);
                        symbol = rela.symbol(LittleEndian, false);
                        addend = rela.r_addend(LittleEndian);
                        offset = rela.r_offset(LittleEndian);
                    } else {
                        // SAFETY: The pointer is always within bounds since we used '.get'.
                        let rela = unsafe { rela.as_ptr().cast::<object::elf::Rela32<LittleEndian>>().read_unaligned() };
                        r_type = rela.r_type(LittleEndian);
                        symbol = rela.symbol(LittleEndian, false);
                        addend = i64::from(rela.r_addend(LittleEndian));
                        offset = u64::from(rela.r_offset(LittleEndian));
                    }
                    let (kind, encoding, size) = match r_type {
                        object::elf::R_RISCV_32 => (object::RelocationKind::Absolute, object::RelocationEncoding::Generic, 32),
                        object::elf::R_RISCV_64 => (object::RelocationKind::Absolute, object::RelocationEncoding::Generic, 64),
                        _ => (object::RelocationKind::Unknown, object::RelocationEncoding::Generic, 0),
                    };

                    let flags = object::RelocationFlags::Elf { r_type };
                    let target = match symbol {
                        None => object::RelocationTarget::Absolute,
                        Some(symbol) => object::RelocationTarget::Symbol(symbol),
                    };

                    relocations.push((
                        offset,
                        Relocation {
                            kind,
                            encoding,
                            flags,
                            target,
                            size,
                            addend,
                        },
                    ));
                }

                relocations_in_section.insert(index, relocations);
            }

            sections.push(Section {
                index,
                name: name.to_owned(),
                original_address: section.address(),
                align: section.align(),
                size: section.size(),
                data: Cow::Borrowed(file_data),
                flags,
                raw_section_index: Some(section.index()),
                relocations: Vec::new(),
                elf_section_type: section.elf_section_header().sh_type(LittleEndian),
            });

            section_index_by_name.entry(name.to_owned()).or_insert_with(Vec::new).push(index);
            if section_index_map.insert(section.index(), index).is_some() {
                return Err(ProgramFromElfError::other("multiple sections with the same section index"));
            }
        }

        for section_index in (0..sections.len()).map(SectionIndex) {
            let elf_section_index = sections[section_index.raw()].raw_section_index.unwrap();
            if let Some(relocation_sections) = relocation_sections_for_section.remove(&elf_section_index.0) {
                for relocation_section in relocation_sections {
                    sections[section_index.raw()]
                        .relocations
                        .extend(relocations_in_section.remove(&relocation_section).unwrap());
                }
            }
        }

        assert_eq!(sections.len(), elf.sections().count());

        let mut fake_empty_sections = HashMap::new();
        let mut symbols = Vec::new();
        for symbol in elf.symbols() {
            let mut name = symbol.name().ok();
            if name == Some("") {
                name = None;
            }
            let size = symbol.size();
            let address = symbol.address();
            let address_end = address.wrapping_add(size);
            let kind = symbol.elf_symbol().st_type();

            let symbol_section = match symbol.section() {
                object::read::SymbolSection::Undefined => SymbolSection::Undefined,
                object::read::SymbolSection::None => SymbolSection::None,
                object::read::SymbolSection::Absolute => SymbolSection::Absolute,
                object::read::SymbolSection::Section(elf_section_index) => {
                    let mut section_index = section_index_map
                        .get(&elf_section_index)
                        .copied()
                        .expect("unable to map section index");

                    let section = &sections[section_index.0];
                    let section_size = section.size();
                    let mut section_address = section.original_address();
                    let section_address_end = section_address + section_size;

                    if address < section_address || address_end < section_address || address_end > section_address_end {
                        let name_s = name.map(|name| format!("'{name}'")).unwrap_or_else(|| "''".to_owned());
                        let section_name = section.name();
                        let error = format!("symbol {name_s} with address 0x{address:x}..0x{address_end:x} in {section_index} ('{section_name}') which is at 0x{section_address:x}..0x{section_address_end:x}");

                        if size > 0 {
                            // We could support these, but I haven't seen any in the wild yet so let's just return an error.
                            return Err(ProgramFromElfError::other(error));
                        }

                        // The compiler sometimes emits these zero-sized symbols which get assigned to the wrong section.
                        // As far as I can see this happens because the original section to which these symbols belong ends
                        // up being empty, so the section gets deleted, and then the compiler/linker instead of correctly marking
                        // those symbols as not belonging to any section it just picks the next section that still exists.
                        log::warn!("Broken symbol: {error}");

                        match fake_empty_sections.entry(address) {
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                section_index = SectionIndex(sections.len());
                                sections.push(Section {
                                    index: section_index,
                                    name: ".data.fake".to_owned(),
                                    original_address: address,
                                    align: 1,
                                    size,
                                    data: Cow::Borrowed(b""),
                                    flags: u64::from(object::elf::SHF_ALLOC | object::elf::SHF_WRITE),
                                    raw_section_index: None,
                                    relocations: Vec::new(),
                                    elf_section_type: object::elf::SHT_PROGBITS,
                                });
                                entry.insert(section_index);
                            }
                            std::collections::hash_map::Entry::Occupied(entry) => {
                                section_index = *entry.get();
                            }
                        }

                        section_address = sections[section_index.0].original_address;
                    }

                    let target = SectionTarget {
                        section_index,
                        offset: address - section_address,
                    };

                    SymbolSection::Section(target)
                }
                section => {
                    return Err(ProgramFromElfError::other(format!(
                        "failed to parse ELF file: found symbol in an unhandled section: {:?}",
                        section
                    )));
                }
            };

            let symbol = Symbol {
                name: name.map(|name| name.to_owned()),
                address,
                size,
                kind,
                section: symbol_section,
            };
            symbols.push(symbol);
        }

        assert_eq!(symbols.len(), elf.symbols().count());

        Ok(Elf {
            sections,
            section_index_by_name,
            symbols,
            is_64_bit,
        })
    }

    pub fn symbol_by_index(&self, symbol_index: object::SymbolIndex) -> Result<&Symbol, ProgramFromElfError> {
        self.symbols
            .get(symbol_index.0 - 1)
            .ok_or_else(|| ProgramFromElfError::other("symbol not found"))
    }

    pub fn section_by_name(&self, name: &str) -> impl ExactSizeIterator<Item = &Section> {
        #[allow(clippy::map_unwrap_or)]
        let indexes = self.section_index_by_name.get(name).map(|vec| vec.as_slice()).unwrap_or(&[]);
        indexes.iter().map(|&index| &self.sections[index.0])
    }

    pub fn section_by_index(&self, index: SectionIndex) -> &Section<'data> {
        &self.sections[index.0]
    }

    pub fn section_by_raw_index(&self, index: usize) -> Option<&Section<'data>> {
        self.sections.get(index)
    }

    pub fn sections<'r>(&'r self) -> impl ExactSizeIterator<Item = &'r Section<'data>> + 'r {
        self.sections.iter()
    }

    pub fn symbols(&self) -> impl ExactSizeIterator<Item = &'_ Symbol> {
        self.symbols.iter()
    }

    pub fn add_empty_data_section(&mut self, name: &str) -> SectionIndex {
        let index = SectionIndex(self.sections.len());

        self.section_index_by_name
            .entry(name.to_owned())
            .or_insert_with(Vec::new)
            .push(index);
        self.sections.push(Section {
            index,
            name: name.to_owned(),
            original_address: 0,
            align: 4,
            size: 0,
            data: Cow::Owned(Vec::new()),
            flags: u64::from(object::elf::SHF_ALLOC),
            raw_section_index: None,
            relocations: Vec::new(),
            elf_section_type: 0,
        });

        index
    }

    pub fn extend_section_to_at_least(&mut self, index: SectionIndex, size: usize) {
        let section = &mut self.sections[index.0];
        section.data.to_mut().resize(size, 0);
        section.size = core::cmp::max(section.size, size as u64);
    }

    pub fn section_data_mut(&mut self, index: SectionIndex) -> &mut [u8] {
        let section = &mut self.sections[index.0];
        section.data.to_mut()
    }

    pub fn is_64(&self) -> bool {
        self.is_64_bit
    }

    pub fn section_to_function_name(&self) -> BTreeMap<SectionTarget, String> {
        self.symbols()
            .filter_map(|symbol| {
                if symbol.kind() != object::elf::STT_FUNC {
                    return None;
                }

                let name = symbol.name()?.to_owned();
                let target = symbol.section_target().ok()?;
                Some((target, name))
            })
            .collect()
    }
}
