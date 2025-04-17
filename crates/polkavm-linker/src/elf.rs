use object::{read::elf::Sym, LittleEndian, Object, ObjectSection, ObjectSymbol};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use crate::program_from_elf::{ProgramFromElfError, SectionTarget};

type ElfFile<'a, H> = object::read::elf::ElfFile<'a, H, &'a [u8]>;
type ElfSymbol<'data, 'file, H> = object::read::elf::ElfSymbol<'data, 'file, H, &'data [u8]>;
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

pub struct Symbol<'data, 'file, H>
where
    H: object::read::elf::FileHeader,
{
    elf: &'file Elf<'data, H>,
    elf_symbol: ElfSymbol<'data, 'file, H>,
}

impl<'data, 'file, H> Symbol<'data, 'file, H>
where
    H: object::read::elf::FileHeader<Endian = object::LittleEndian>,
{
    fn new(elf: &'file Elf<'data, H>, elf_symbol: ElfSymbol<'data, 'file, H>) -> Self {
        Symbol { elf, elf_symbol }
    }

    pub fn name(&self) -> Option<&'data str> {
        let name = self.elf_symbol.name().ok()?;
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self.elf_symbol.section(), object::read::SymbolSection::Undefined)
    }

    pub fn section_and_offset(&self) -> Result<(&Section, u64), ProgramFromElfError> {
        let elf_section_index = match self.elf_symbol.section() {
            object::read::SymbolSection::Section(section_index) => section_index,
            object::read::SymbolSection::Undefined => {
                return Err(ProgramFromElfError::other(format!(
                    "found undefined symbol: '{}'",
                    self.name().unwrap_or("")
                )));
            }
            section => {
                return Err(ProgramFromElfError::other(format!(
                    "found symbol in an unhandled section: {:?}",
                    section
                )));
            }
        };

        let section_index = self
            .elf
            .section_index_map
            .get(&elf_section_index)
            .copied()
            .expect("unable to map section index");
        let section = self.elf.section_by_index(section_index);
        let Some(offset) = self.elf_symbol.address().checked_sub(section.original_address()) else {
            return Err(ProgramFromElfError::other("relative symbol address underflow"));
        };

        Ok((section, offset))
    }

    pub fn size(&self) -> u64 {
        self.elf_symbol.size()
    }

    pub fn kind(&self) -> u8 {
        self.elf_symbol.elf_symbol().st_type()
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

pub struct Elf<'data, H>
where
    H: object::read::elf::FileHeader,
{
    sections: Vec<Section<'data>>,
    section_index_by_name: HashMap<String, Vec<SectionIndex>>,
    section_index_map: HashMap<ElfSectionIndex, SectionIndex>,
    // TODO: Always have a dummy ELF file for testing?
    raw_elf: Option<ElfFile<'data, H>>,
    is_64_bit: bool,
}

#[cfg(test)]
impl<'data, H> Default for Elf<'data, H>
where
    H: object::read::elf::FileHeader,
{
    fn default() -> Self {
        Elf {
            sections: Default::default(),
            section_index_by_name: Default::default(),
            section_index_map: Default::default(),
            raw_elf: Default::default(),
            is_64_bit: false,
        }
    }
}

impl<'data, H> Elf<'data, H>
where
    H: object::read::elf::FileHeader<Endian = object::LittleEndian>,
{
    pub fn parse(data: &'data [u8]) -> Result<Self, ProgramFromElfError> {
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

        Ok(Elf {
            sections,
            section_index_by_name,
            section_index_map,
            raw_elf: Some(elf),
            is_64_bit,
        })
    }

    pub fn symbol_by_index(&self, symbol_index: object::SymbolIndex) -> Result<Symbol<H>, object::Error> {
        self.raw_elf
            .as_ref()
            .unwrap()
            .symbol_by_index(symbol_index)
            .map(|elf_symbol| Symbol::new(self, elf_symbol))
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

    pub fn symbols<'r>(&'r self) -> impl Iterator<Item = Symbol<'data, 'r, H>> + 'r {
        self.raw_elf
            .as_ref()
            .unwrap()
            .symbols()
            .map(|elf_symbol| Symbol::new(self, elf_symbol))
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

    pub fn section_to_function_name(&self) -> BTreeMap<SectionTarget, &'data str> {
        self.symbols()
            .filter_map(|symbol| {
                if symbol.kind() != object::elf::STT_FUNC {
                    return None;
                }

                let name = symbol.name()?;
                let (section, offset) = symbol.section_and_offset().ok()?;
                let target = SectionTarget {
                    section_index: section.index(),
                    offset,
                };
                Some((target, name))
            })
            .collect()
    }
}
