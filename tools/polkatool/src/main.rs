#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::exit)]

use clap::Parser;
use polkavm_common::program::{InstructionSetKind, Opcode, ProgramBlob};
use polkavm_disassembler::DisassemblyFormat;
use polkavm_linker::TargetInstructionSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Bitness {
    #[clap(name = "32")]
    B32,
    #[clap(name = "64")]
    B64,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum OptLevel {
    #[clap(name = "0")]
    O0,
    #[clap(name = "1")]
    O1,
    #[clap(name = "2")]
    O2,
    #[clap(name = "experimental")]
    Oexperimental,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Isa {
    #[clap(name = "revive_v1")]
    ReviveV1,
    #[clap(name = "latest32")]
    Latest32,
    #[clap(name = "latest64")]
    Latest64,
}

impl Isa {
    fn convert(self) -> InstructionSetKind {
        match self {
            Isa::ReviveV1 => InstructionSetKind::ReviveV1,
            Isa::Latest32 => InstructionSetKind::Latest32,
            Isa::Latest64 => InstructionSetKind::Latest64,
        }
    }
}

impl core::fmt::Display for Isa {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str(self.convert().name())
    }
}

#[derive(Parser, Debug)]
#[clap(version)]
enum Args {
    /// Links a given ELF file into a `.polkavm` program blob.
    Link {
        /// The output file.
        #[clap(short = 'o', long)]
        output: PathBuf,

        #[clap(short = 's', long)]
        strip: bool,

        /// Will disable optimizations.
        #[clap(long, hide = true)] // For backwards-compatibility.
        disable_optimizations: bool,

        /// The optimization level.
        #[clap(long, default_value = "2")]
        opt_level: OptLevel,

        /// Will only run if the output file doesn't exist, or the input is newer.
        #[clap(long)]
        run_only_if_newer: bool,

        /// Sets the minimum stack size.
        #[clap(long)]
        min_stack_size: Option<u32>,

        /// Exports to use to build a dispatch table, separated by a comma.
        #[clap(long)]
        dispatch_table: Option<String>,

        /// The input file.
        input: PathBuf,
    },

    /// Disassembles a .polkavm blob into its human-readable assembly.
    Disassemble {
        /// The output file.
        #[clap(short = 'o', long)]
        output: Option<PathBuf>,

        #[clap(short = 'f', long, value_enum, default_value_t = DisassemblyFormat::Guest)]
        format: DisassemblyFormat,

        #[clap(long)]
        display_gas: bool,

        #[clap(long)]
        show_raw_bytes: bool,

        #[clap(long)]
        no_show_offsets: bool,

        #[clap(long)]
        no_show_native_raw_bytes: bool,

        #[clap(long)]
        no_show_native_offsets: bool,

        /// The input file.
        input: PathBuf,
    },

    /// Assembles a .polkavm blob from human-readable assembly.
    Assemble {
        /// The output file.
        #[clap(short = 'o', long)]
        output: PathBuf,

        /// The target instruction set.
        #[clap(short = 'i', long, default_value_t = Isa::Latest64)]
        instruction_set: Isa,

        /// The input file.
        input: PathBuf,
    },

    /// Calculates various statistics for given program blobs.
    Stats {
        /// The input files.
        inputs: Vec<PathBuf>,
    },

    /// Writes a path to a JSON target file for rustc to stdout.
    GetTargetJsonPath {
        #[clap(short = 'b', long, value_enum, default_value_t = Bitness::B64)]
        bitness: Bitness,
    },
}

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(format!($($arg)*))
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let result = match args {
        Args::Link {
            output,
            input,
            strip,
            disable_optimizations,
            opt_level,
            run_only_if_newer,
            min_stack_size,
            dispatch_table,
        } => main_link(
            input,
            output,
            strip,
            disable_optimizations,
            opt_level,
            run_only_if_newer,
            min_stack_size,
            dispatch_table,
        ),
        Args::Disassemble {
            output,
            format,
            display_gas,
            show_raw_bytes,
            no_show_offsets,
            no_show_native_raw_bytes,
            no_show_native_offsets,
            input,
        } => main_disassemble(
            input,
            format,
            display_gas,
            show_raw_bytes,
            !no_show_offsets,
            !no_show_native_raw_bytes,
            !no_show_native_offsets,
            output,
        ),
        Args::Assemble {
            input,
            output,
            instruction_set,
        } => main_assemble(input, output, instruction_set),
        Args::Stats { inputs } => main_stats(inputs),
        Args::GetTargetJsonPath { bitness } => {
            let mut args = polkavm_linker::TargetJsonArgs::default();
            args.is_64_bit = match bitness {
                Bitness::B32 => false,
                Bitness::B64 => true,
            };

            let result = polkavm_linker::target_json_path(args);
            result.map(|path| print!("{}", path.to_str().unwrap()))
        }
    };

    if let Err(error) = result {
        eprintln!("ERROR: {}", error);
        std::process::exit(1);
    }
}

fn main_link(
    input: PathBuf,
    output: PathBuf,
    strip: bool,
    disable_optimizations: bool,
    opt_level: OptLevel,
    run_only_if_newer: bool,
    min_stack_size: Option<u32>,
    dispatch_table: Option<String>,
) -> Result<(), String> {
    if run_only_if_newer {
        if let Ok(output_mtime) = std::fs::metadata(&output).and_then(|m| m.modified()) {
            if let Ok(input_mtime) = std::fs::metadata(&input).and_then(|m| m.modified()) {
                if output_mtime >= input_mtime {
                    return Ok(());
                }
            }
        }
    }

    let opt_level = match opt_level {
        _ if disable_optimizations => polkavm_linker::OptLevel::O0,
        OptLevel::O0 => polkavm_linker::OptLevel::O0,
        OptLevel::O1 => polkavm_linker::OptLevel::O1,
        OptLevel::O2 => polkavm_linker::OptLevel::O2,
        OptLevel::Oexperimental => polkavm_linker::OptLevel::Oexperimental,
    };

    let mut config = polkavm_linker::Config::default();
    config.set_strip(strip);
    config.set_opt_level(opt_level);
    if let Some(min_stack_size) = min_stack_size {
        config.set_min_stack_size(min_stack_size);
    }
    if let Some(dispatch_table) = dispatch_table {
        let mut table = Vec::new();
        for name in dispatch_table.split(',') {
            table.push(name.trim().as_bytes().to_owned());
        }
        config.set_dispatch_table(table);
    }

    let data = match std::fs::read(&input) {
        Ok(data) => data,
        Err(error) => {
            bail!("failed to read {input:?}: {error}");
        }
    };

    let blob = match polkavm_linker::program_from_elf(config, TargetInstructionSet::Latest, &data) {
        Ok(blob) => blob,
        Err(error) => {
            bail!("failed to link {input:?}: {error}");
        }
    };

    if let Err(error) = std::fs::write(&output, blob) {
        bail!("failed to write the program blob to {output:?}: {error}");
    }

    Ok(())
}

fn load_blob(input: &Path) -> Result<ProgramBlob, String> {
    let data = match std::fs::read(input) {
        Ok(data) => data,
        Err(error) => {
            bail!("failed to read {input:?}: {error}");
        }
    };

    let blob = match polkavm_linker::ProgramBlob::parse(data[..].into()) {
        Ok(blob) => blob,
        Err(error) => {
            bail!("failed to parse {input:?}: {error}");
        }
    };

    Ok(blob)
}

fn main_stats(inputs: Vec<PathBuf>) -> Result<(), String> {
    let mut map = HashMap::new();
    for opcode in Opcode::ALL {
        map.insert(opcode, 0);
    }

    for input in inputs {
        let blob = load_blob(&input)?;
        let instructions: Vec<_> = blob.instructions().collect();
        for instruction in instructions {
            *map.get_mut(&instruction.opcode()).unwrap() += 1;
        }
    }

    let mut list: Vec<_> = map.into_iter().collect();
    list.sort_by_key(|(_, count)| core::cmp::Reverse(*count));

    println!("Instruction distribution:");
    for (opcode, count) in list {
        println!("{opcode:>40}: {count}", opcode = format!("{:?}", opcode));
    }

    Ok(())
}

#[allow(clippy::fn_params_excessive_bools)]
fn main_disassemble(
    input: PathBuf,
    format: DisassemblyFormat,
    display_gas: bool,
    show_raw_bytes: bool,
    show_offsets: bool,
    show_native_raw_bytes: bool,
    show_native_offsets: bool,
    output: Option<PathBuf>,
) -> Result<(), String> {
    let blob = load_blob(&input)?;

    let mut disassembler = polkavm_disassembler::Disassembler::new(&blob, format).map_err(|error| error.to_string())?;
    disassembler.show_raw_bytes(show_raw_bytes);
    disassembler.show_native_raw_bytes(show_native_raw_bytes);
    disassembler.show_offsets(show_offsets);
    disassembler.show_native_offsets(show_native_offsets);

    // TODO: Should be set with command line arguments.
    if let Ok(config) = polkavm::Config::from_env() {
        disassembler.cost_model(config.default_cost_model());
    }

    if display_gas {
        disassembler.display_gas().map_err(|error| error.to_string())?;
    }

    match output {
        Some(output) => {
            let fp = match std::fs::File::create(&output) {
                Ok(fp) => fp,
                Err(error) => {
                    bail!("failed to create output file {output:?}: {error}");
                }
            };

            disassembler.disassemble_into(std::io::BufWriter::new(fp))
        }
        None => {
            let stdout = std::io::stdout();
            disassembler.disassemble_into(std::io::BufWriter::new(stdout))
        }
    }
    .map_err(|error| error.to_string())
}

fn main_assemble(input_path: PathBuf, output_path: PathBuf, instruction_set: Isa) -> Result<(), String> {
    let input = match std::fs::read_to_string(&input_path) {
        Ok(input) => input,
        Err(error) => {
            bail!("failed to read {input_path:?}: {error}");
        }
    };

    let instruction_set = instruction_set.convert();
    let blob = match polkavm_common::assembler::assemble(Some(instruction_set), &input) {
        Ok(blob) => blob,
        Err(error) => {
            bail!("failed to assemble {input_path:?}: {error}");
        }
    };

    if let Err(error) = std::fs::write(&output_path, blob) {
        bail!("failed to write to {output_path:?}: {error}");
    }

    Ok(())
}
