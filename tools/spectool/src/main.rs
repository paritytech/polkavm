#![allow(clippy::exit)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::use_debug)]

use clap::Parser;
use core::fmt::Write;
use polkavm::{CacheModel, CostModelKind, Engine, InterruptKind, MemoryProtection, Module, ModuleConfig, ProgramBlob, Reg};
use polkavm_common::assembler::assemble;
use polkavm_common::program::{asm, ProgramCounter, ProgramParts, ISA64_V1};
use polkavm_common::utils::parse_slice;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[clap(version)]
enum Args {
    Generate,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    match args {
        Args::Generate => main_generate(),
    }
}

struct Testcase {
    disassembly: String,
    timelines: Vec<(String, u32, u32)>,
    initial_page_map: Vec<Page>,
    final_page_map: Vec<Page>,
    initial_memory: Vec<MemoryChunk>,
    final_memory: Vec<MemoryChunk>,
    initial_state: State,
    final_state: State,
    json: TestcaseJson,
}

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Page {
    address: u32,
    length: u32,
    is_writable: bool,
}

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct MemoryChunk {
    address: u32,
    contents: Vec<u8>,
}

#[derive(Clone)]
struct State {
    status: &'static str,
    page_fault_address: Option<u32>,
    hostcall: Option<u32>,
    gas: i64,
    pc: ProgramCounter,
    regs: Vec<u64>,
    memory: Vec<MemoryChunk>,
}

fn are_regs_empty(regs: &[Option<u64>; 13]) -> bool {
    regs.iter().all(|value| value.is_none())
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[serde(tag = "kind")]
enum TestcaseStep {
    Run,
    Map {
        address: u32,
        length: u32,
        is_writable: bool,
    },
    Write {
        address: u32,
        contents: Vec<u8>,
    },
    SetReg {
        reg: u32,
        value: u64,
    },
    Assert {
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        page_fault_address: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hostcall: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        gas: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pc: Option<u32>,
        #[serde(skip_serializing_if = "are_regs_empty")]
        regs: [Option<u64>; 13],
        #[serde(skip_serializing_if = "Option::is_none")]
        memory: Option<Vec<MemoryChunk>>,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TestcaseJson {
    name: String,
    initial_pc: u32,
    initial_gas: i64,
    program: Vec<u8>,
    steps: Vec<TestcaseStep>,
    block_gas_costs: BTreeMap<u32, u32>,
}

fn extract_chunks(base_address: u32, slice: &[u8]) -> Vec<MemoryChunk> {
    let mut output = Vec::new();
    let mut position = 0;
    while let Some(next_position) = slice[position..].iter().position(|&byte| byte != 0).map(|offset| position + offset) {
        position = next_position;
        let length = slice[position..].iter().take_while(|&&byte| byte != 0).count();
        output.push(MemoryChunk {
            address: base_address + position as u32,
            contents: slice[position..position + length].into(),
        });
        position += length;
    }

    output
}

enum ProgramCounterRef {
    ByLabel { label: String, instruction_offset: u32 },
    Preset(ProgramCounter),
}

#[derive(Default)]
struct PrePost {
    gas: Option<i64>,
    regs: [Option<u64>; 13],
    pc: Option<ProgramCounterRef>,
}

fn parse_pre_post(line: &str, output: &mut PrePost) {
    let line = line.trim();
    let index = line.find('=').expect("invalid 'pre' / 'post' directive: no '=' found");
    let lhs = line[..index].trim();
    let rhs = line[index + 1..].trim();
    if lhs == "gas" {
        output.gas = Some(rhs.parse::<i64>().expect("invalid 'pre' / 'post' directive: failed to parse rhs"));
    } else if lhs == "pc" {
        let rhs = rhs
            .strip_prefix('@')
            .expect("invalid 'pre' / 'post' directive: failed to parse 'pc': no '@' found")
            .trim();
        let index = rhs
            .find('[')
            .expect("invalid 'pre' / 'post' directive: failed to parse 'pc': no '[' found");
        let label = &rhs[..index];
        let rhs = &rhs[index + 1..];
        let index = rhs
            .find(']')
            .expect("invalid 'pre' / 'post' directive: failed to parse 'pc': no ']' found");
        let offset = rhs[..index]
            .parse::<u32>()
            .expect("invalid 'pre' / 'post' directive: failed to parse 'pc': invalid offset");
        if !rhs[index + 1..].trim().is_empty() {
            panic!("invalid 'pre' / 'post' directive: failed to parse 'pc': junk after ']'");
        }

        output.pc = Some(ProgramCounterRef::ByLabel {
            label: label.to_owned(),
            instruction_offset: offset,
        });
    } else {
        let lhs = polkavm_common::utils::parse_reg(lhs).expect("invalid 'pre' / 'post' directive: failed to parse lhs");
        let rhs = polkavm_common::utils::parse_immediate(rhs)
            .map(Into::into)
            .expect("invalid 'pre' / 'post' directive: failed to parse rhs");
        output.regs[lhs as usize] = Some(rhs);
    }
}

fn parse_u32(text: &str) -> Option<u32> {
    let text = text.trim();
    if let Some(text) = text.strip_prefix("0x") {
        u32::from_str_radix(text, 16).ok()
    } else if let Some(text) = text.strip_prefix("0b") {
        u32::from_str_radix(text, 2).ok()
    } else {
        text.parse::<u32>().ok()
    }
}

fn parse_u64(text: &str) -> Option<u64> {
    let text = text.trim();
    if let Some(text) = text.strip_prefix("0x") {
        u64::from_str_radix(text, 16).ok()
    } else if let Some(text) = text.strip_prefix("0b") {
        u64::from_str_radix(text, 2).ok()
    } else if let Ok(value) = text.parse::<i64>() {
        Some(value as u64)
    } else {
        text.parse::<u64>().ok()
    }
}

fn parse_step(line: &str, steps: &mut Vec<TestcaseStep>) {
    let line = line.trim();
    if line == "run" {
        steps.push(TestcaseStep::Run);
        return;
    } else if let Some(line) = line.strip_prefix("map ") {
        let error = "invalid 'step': failed to parse 'map'";
        let line = line.trim();
        let is_writable = if line.strip_prefix("RO ").is_some() {
            false
        } else if line.strip_prefix("RW ").is_some() {
            true
        } else {
            panic!("{}", error);
        };

        let line = line[3..].trim();
        let mut xs = line.split("..");
        let start = xs.next().expect(error);
        let end = xs.next().expect(error);
        assert!(xs.next().is_none(), "{}", error);
        let start = parse_u32(start).expect(error);
        let end = parse_u32(end).expect(error);
        if (start % 4096) != 0 || (end % 4096) != 0 {
            panic!("invalid 'step': 'map' has address or length that is not page-aligned");
        }

        steps.push(TestcaseStep::Map {
            address: start,
            length: end - start,
            is_writable,
        });
        return;
    } else if let Some(line) = line.strip_prefix("write ") {
        let error = "invalid 'step': failed to parse 'write'";
        let line = line.trim();
        if let Some(index) = line.find(' ') {
            let address = line[..index].trim();
            let contents = line[index + 1..].trim();
            let address = parse_u32(address).expect(error);
            let contents = parse_slice(contents).expect(error);
            steps.push(TestcaseStep::Write { address, contents });
            return;
        }
    } else if let Some(index) = line.find("=") {
        let lhs = line[..index].trim();
        let rhs = line[index + 1..].trim();
        if let Some(reg) = Reg::ALL.iter().position(|reg| reg.name() == lhs) {
            if let Some(value) = parse_u64(rhs) {
                steps.push(TestcaseStep::SetReg { reg: reg as u32, value });
                return;
            }
        }
    }

    panic!("invalid 'step': failed to parse line: '{line}'");
}

fn main_generate() {
    let mut tests = Vec::new();

    let mut config = polkavm::Config::new();
    config.set_backend(Some(polkavm::BackendKind::Interpreter));
    config.set_allow_dynamic_paging(true);

    let engine = Engine::new(&config).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("spec");
    let mut found_errors = false;

    let mut paths: Vec<PathBuf> = std::fs::read_dir(root.join("src"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();

    paths.sort_by_key(|entry| entry.file_stem().unwrap().to_string_lossy().to_string());

    struct RawTestcase {
        name: String,
        internal_name: String,
        blob: Vec<u8>,
        pre: PrePost,
        post: PrePost,
        steps: Vec<TestcaseStep>,
        expected_status: Option<&'static str>,
    }

    let mut testcases = Vec::new();
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy();

        let mut pre = PrePost::default();
        let mut post = PrePost::default();
        let mut steps = Vec::new();

        let input = std::fs::read_to_string(&path).unwrap();
        let mut input_lines = Vec::new();
        for line in input.lines() {
            if let Some(line) = line.strip_prefix("pre:") {
                parse_pre_post(line, &mut pre);
                input_lines.push(""); // Insert dummy line to not mess up the line count.
                continue;
            }

            if let Some(line) = line.strip_prefix("post:") {
                parse_pre_post(line, &mut post);
                input_lines.push(""); // Insert dummy line to not mess up the line count.
                continue;
            }

            if let Some(line) = line.strip_prefix("step:") {
                parse_step(line, &mut steps);
                input_lines.push(""); // Insert dummy line to not mess up the line count.
                continue;
            }

            input_lines.push(line);
        }

        if steps.is_empty() {
            steps.push(TestcaseStep::Run);
        }

        let input = input_lines.join("\n");
        let blob = match assemble(&input) {
            Ok(blob) => blob,
            Err(error) => {
                eprintln!("Failed to assemble {path:?}: {error}");
                found_errors = true;
                continue;
            }
        };

        testcases.push(RawTestcase {
            name: name.into_owned(),
            internal_name: format!("{path:?}"),
            blob,
            pre,
            post,
            steps,
            expected_status: None,
        });
    }

    // This is kind of a hack, but whatever.
    for line in include_str!("../../../crates/polkavm/src/tests_riscv.rs").lines() {
        let prefix = "riscv_test!(riscv_unoptimized_rv64";
        if !line.starts_with(prefix) {
            continue;
        }

        let line = &line[prefix.len()..];
        let mut xs = line.split(',');
        let name = xs.next().unwrap();
        let path = xs.next().unwrap().trim();
        let path = &path[1..path.len() - 1];

        let path = root.join("../../../crates/polkavm/src").join(path);
        let path = path.canonicalize().unwrap();
        let elf = std::fs::read(&path).unwrap();

        let mut linker_config = polkavm_linker::Config::default();
        linker_config.set_opt_level(polkavm_linker::OptLevel::O1);
        linker_config.set_strip(true);
        linker_config.set_min_stack_size(0);
        let blob = polkavm_linker::program_from_elf(linker_config, &elf).unwrap();

        let mut post = PrePost::default();

        let program_blob = ProgramBlob::parse(blob.clone().into()).unwrap();
        post.pc = Some(ProgramCounterRef::Preset(
            program_blob
                .instructions(ISA64_V1)
                .find(|inst| inst.kind == asm::ret())
                .unwrap()
                .offset,
        ));

        testcases.push(RawTestcase {
            name: format!("riscv_rv64{name}"),
            internal_name: format!("{path:?}"),
            blob,
            pre: PrePost::default(),
            post,
            steps: vec![TestcaseStep::Run],
            expected_status: Some("halt"),
        });
    }

    'next_testcase: for testcase in testcases {
        let RawTestcase {
            name,
            internal_name,
            blob,
            pre,
            post,
            mut steps,
            expected_status,
        } = testcase;

        assert!(!steps.is_empty());

        let initial_gas = pre.gas.unwrap_or(10000);
        let initial_regs = pre.regs.map(|value| value.unwrap_or(0));
        assert!(pre.pc.is_none(), "'pre: pc = ...' is currently unsupported");

        let parts = ProgramParts::from_bytes(blob.into()).unwrap();
        let blob = ProgramBlob::from_parts(parts.clone()).unwrap();

        let cache_model = CacheModel::L2Hit;
        let mut module_config = ModuleConfig::default();
        module_config.set_strict(true);
        module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
        module_config.set_step_tracing(true);
        module_config.set_dynamic_paging(true);
        module_config.set_cost_model(Some(CostModelKind::Full(cache_model)));

        let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();
        let mut instance = module.instantiate().unwrap();

        let mut initial_steps = Vec::new();
        if module.memory_map().ro_data_size() > 0 {
            initial_steps.push(TestcaseStep::Map {
                address: module.memory_map().ro_data_address(),
                length: module.memory_map().ro_data_size(),
                is_writable: false,
            });

            for chunk in extract_chunks(module.memory_map().ro_data_address(), blob.ro_data()) {
                initial_steps.push(TestcaseStep::Write {
                    address: chunk.address,
                    contents: chunk.contents,
                });
            }
        }

        if module.memory_map().rw_data_size() > 0 {
            initial_steps.push(TestcaseStep::Map {
                address: module.memory_map().rw_data_address(),
                length: module.memory_map().rw_data_size(),
                is_writable: true,
            });

            for chunk in extract_chunks(module.memory_map().rw_data_address(), blob.rw_data()) {
                initial_steps.push(TestcaseStep::Write {
                    address: chunk.address,
                    contents: chunk.contents,
                });
            }
        }

        if module.memory_map().stack_size() > 0 {
            initial_steps.push(TestcaseStep::Map {
                address: module.memory_map().stack_address_low(),
                length: module.memory_map().stack_size(),
                is_writable: true,
            });
        }

        for (reg, value) in pre.regs.into_iter().enumerate() {
            if let Some(value) = value {
                if value != 0 {
                    initial_steps.push(TestcaseStep::SetReg { reg: reg as u32, value });
                }
            }
        }

        initial_steps.extend(steps);
        steps = initial_steps;

        let initial_pc = blob.exports().find(|export| export.symbol() == "main").unwrap().program_counter();

        let expected_final_pc = if let Some(export) = blob.exports().find(|export| export.symbol() == "expected_exit") {
            assert!(
                post.pc.is_none(),
                "'@expected_exit' label and 'post: pc = ...' should not be used together"
            );
            export.program_counter().0
        } else if let Some(ProgramCounterRef::ByLabel { label, instruction_offset }) = post.pc {
            let Some(export) = blob.exports().find(|export| export.symbol().as_bytes() == label.as_bytes()) else {
                panic!("label specified in 'post: pc = ...' is missing: @{label}");
            };

            let instructions: Vec<_> = blob.instructions(ISA64_V1).collect();
            let index = instructions
                .iter()
                .position(|inst| inst.offset == export.program_counter())
                .expect("failed to find label specified in 'post: pc = ...'");
            let instruction = instructions
                .get(index + instruction_offset as usize)
                .expect("invalid 'post: pc = ...': offset goes out of bounds of the basic block");
            instruction.offset.0
        } else if let Some(ProgramCounterRef::Preset(pc)) = post.pc {
            pc.0
        } else {
            blob.code().len() as u32
        };

        instance.set_gas(initial_gas);
        instance.set_next_program_counter(initial_pc);

        for (reg, value) in Reg::ALL.into_iter().zip(initial_regs) {
            instance.set_reg(reg, value);
        }

        let mut final_state = State {
            status: "",
            page_fault_address: None,
            hostcall: None,
            gas: instance.gas(),
            pc: initial_pc,
            regs: initial_regs.to_vec(),
            memory: vec![],
        };

        let mut initial_state = final_state.clone();
        let mut initial_page_map = Vec::new();
        let mut final_page_map = Vec::new();
        let mut initial_memory = Vec::new();
        let mut nth_step = 0;
        while nth_step < steps.len() {
            let step = steps[nth_step].clone();
            match step {
                TestcaseStep::Map {
                    address,
                    length,
                    is_writable,
                } => {
                    instance
                        .zero_memory_with_memory_protection(
                            address,
                            length,
                            if is_writable {
                                MemoryProtection::ReadWrite
                            } else {
                                MemoryProtection::Read
                            },
                        )
                        .unwrap();

                    if final_state.status.is_empty() {
                        initial_page_map.push(Page {
                            address,
                            length,
                            is_writable,
                        });
                    }

                    final_page_map.push(Page {
                        address,
                        length,
                        is_writable,
                    });
                    nth_step += 1;
                    continue;
                }
                TestcaseStep::Write { address, contents } => {
                    let mut pages_made_writable = Vec::new();
                    for address in ((address / 4096 * 4096)..(address + contents.len() as u32).next_multiple_of(4096)).step_by(4096) {
                        assert!(instance.is_memory_accessible(address, 4096, MemoryProtection::Read));
                        if !instance.is_memory_accessible(address, 4096, MemoryProtection::ReadWrite) {
                            pages_made_writable.push(address);
                            instance.unprotect_memory(address, 4096).unwrap();
                        }
                    }

                    instance.write_memory(address, &contents).unwrap();
                    for address in pages_made_writable {
                        instance.protect_memory(address, 4096).unwrap();
                    }

                    nth_step += 1;
                    continue;
                }
                TestcaseStep::SetReg { reg, value } => {
                    instance.set_reg(Reg::ALL[reg as usize], value);
                    nth_step += 1;
                    continue;
                }
                TestcaseStep::Assert {
                    status,
                    page_fault_address,
                    hostcall,
                    gas,
                    pc,
                    regs,
                    memory,
                } => {
                    let mut found_local_errors = false;
                    if let Some(status) = status {
                        if status != final_state.status {
                            eprintln!(
                                "Unexpected status for {internal_name}: expected {status}, is {}",
                                final_state.status
                            );
                            found_local_errors = true;
                        }
                    }

                    if page_fault_address != final_state.page_fault_address {
                        eprintln!(
                            "Unexpected page fault address for {internal_name}: expected {page_fault_address:?}, is {:?}",
                            final_state.page_fault_address
                        );
                        found_local_errors = true;
                    }

                    if hostcall != final_state.hostcall {
                        eprintln!(
                            "Unexpected hostcall for {internal_name}: expected {hostcall:?}, is {:?}",
                            final_state.hostcall
                        );
                        found_local_errors = true;
                    }

                    if let Some(gas) = gas {
                        if gas != final_state.gas {
                            eprintln!("Unexpected gas for {internal_name}: expected {gas}, is {}", final_state.gas);
                            found_local_errors = true;
                        }
                    }

                    if let Some(pc) = pc {
                        if pc != final_state.pc.0 {
                            eprintln!("Unexpected PC for {internal_name}: expected {pc}, is {}", final_state.pc);
                            found_local_errors = true;
                        }
                    }

                    assert_eq!(regs.len(), Reg::ALL.len());
                    assert_eq!(final_state.regs.len(), Reg::ALL.len());
                    for ((reg, expected_value), actual_value) in Reg::ALL
                        .iter()
                        .copied()
                        .zip(regs.iter().copied())
                        .zip(final_state.regs.iter().copied())
                    {
                        let Some(expected_value) = expected_value else { continue };
                        if actual_value != expected_value {
                            eprintln!("Unexpected value of {reg} for {internal_name}: expected {expected_value}, is {actual_value}");
                            found_local_errors = true;
                        }
                    }

                    let mut current_memory = Vec::new();
                    for page in &final_page_map {
                        let contents = instance.read_memory(page.address, page.length).unwrap();
                        current_memory.extend(extract_chunks(page.address, &contents));
                    }

                    if let Some(memory) = memory {
                        if current_memory != memory {
                            eprintln!("Memory contents mismatch for {internal_name}");
                            found_local_errors = true;
                        }
                    }

                    if found_local_errors {
                        found_errors = true;
                        continue 'next_testcase;
                    }

                    nth_step += 1;
                    continue;
                }
                TestcaseStep::Run => {
                    nth_step += 1;
                }
            }

            if final_state.status.is_empty() {
                for page in &initial_page_map {
                    let memory = instance.read_memory(page.address, page.length).unwrap();
                    initial_memory.extend(extract_chunks(page.address, &memory));
                }

                initial_state = final_state.clone();
            }

            let (new_status, new_page_fault_address, new_hostcall) = loop {
                match instance.run().unwrap() {
                    InterruptKind::Finished => break ("halt", None, None),
                    InterruptKind::Trap => break ("panic", None, None),
                    InterruptKind::Ecalli(hostcall) => break ("ecalli", None, Some(hostcall)),
                    InterruptKind::NotEnoughGas => break ("out-of-gas", None, None),
                    InterruptKind::Segfault(segfault) => break ("page-fault", Some(segfault.page_address), None),
                    InterruptKind::Step => {
                        final_state.pc = instance.program_counter().unwrap();
                        continue;
                    }
                }
            };

            final_state.status = new_status;
            final_state.page_fault_address = new_page_fault_address;
            final_state.hostcall = new_hostcall;
            final_state.gas = instance.gas();
            if final_state.status != "halt" {
                final_state.pc = instance.program_counter().unwrap();
            }

            for reg in Reg::ALL {
                final_state.regs[reg.to_usize()] = instance.reg(reg);
            }

            final_state.memory.clear();
            for page in &final_page_map {
                let contents = instance.read_memory(page.address, page.length).unwrap();
                final_state.memory.extend(extract_chunks(page.address, &contents));
            }

            if nth_step >= steps.len() || !matches!(steps[nth_step], TestcaseStep::Assert { .. }) {
                steps.insert(
                    nth_step,
                    TestcaseStep::Assert {
                        status: None,
                        page_fault_address: None,
                        hostcall: None,
                        gas: None,
                        pc: None,
                        regs: [None; 13],
                        memory: None,
                    },
                );
            }

            let TestcaseStep::Assert {
                ref mut status,
                ref mut gas,
                ref mut pc,
                ref mut regs,
                ref mut page_fault_address,
                ref mut hostcall,
                ref mut memory,
            } = steps[nth_step]
            else {
                unreachable!()
            };
            if status.is_none() {
                *status = Some(final_state.status.to_owned());
            }

            if gas.is_none() {
                *gas = Some(final_state.gas);
            }

            if pc.is_none() {
                *pc = Some(final_state.pc.0);
            }

            for reg in Reg::ALL {
                if regs[reg.to_usize()].is_none() {
                    regs[reg.to_usize()] = Some(final_state.regs[reg.to_usize()]);
                }
            }

            if final_state.status == "page-fault" {
                if page_fault_address.is_none() {
                    *page_fault_address = final_state.page_fault_address;
                }
                assert!(page_fault_address.is_some());
            } else {
                assert!(page_fault_address.is_none());
            }

            if final_state.status == "ecalli" {
                if hostcall.is_none() {
                    *hostcall = final_state.hostcall;
                }
                assert!(hostcall.is_some());
            } else {
                assert!(hostcall.is_none());
            }

            if memory.is_none() {
                *memory = Some(final_state.memory.clone());
            }
        }

        let mut final_memory = Vec::new();
        for page in &final_page_map {
            let memory = instance.read_memory(page.address, page.length).unwrap();
            final_memory.extend(extract_chunks(page.address, &memory));
        }

        if let Some(expected_status) = expected_status {
            if final_state.status != expected_status {
                eprintln!(
                    "Unexpected final status for {internal_name}: expected {expected_status}, is {}",
                    final_state.status
                );
                found_errors = true;
                continue;
            }
        }

        if final_state.pc.0 != expected_final_pc {
            eprintln!(
                "Unexpected final program counter for {internal_name}: expected {expected_final_pc}, is {}",
                final_state.pc.0
            );
            found_errors = true;
            continue;
        }

        let mut found_post_check_errors = false;

        for ((final_value, reg), required_value) in final_state.regs.iter().zip(Reg::ALL).zip(post.regs.iter()) {
            if let Some(required_value) = required_value {
                if final_value != required_value {
                    eprintln!("{internal_name}: unexpected {reg}: 0x{final_value:x} (expected: 0x{required_value:x})");
                    found_post_check_errors = true;
                }
            }
        }

        if let Some(post_gas) = post.gas {
            if final_state.gas != post_gas {
                eprintln!("{internal_name}: unexpected gas: {} (expected: {post_gas})", final_state.gas);
                found_post_check_errors = true;
            }
        }

        if found_post_check_errors {
            found_errors = true;
            continue;
        }

        let mut blocks = Vec::new();
        let mut buffer = Vec::new();
        for instruction in blob.instructions_bounded_at(polkavm::program::ISA64_V1, ProgramCounter(0)) {
            buffer.push(instruction);
            if instruction.starts_new_basic_block() {
                blocks.push(core::mem::take(&mut buffer));
            }
        }

        if !buffer.is_empty() {
            blocks.push(buffer);
        }

        let mut disassembler = polkavm_disassembler::Disassembler::new(&blob, polkavm_disassembler::DisassemblyFormat::Guest).unwrap();
        disassembler.show_raw_bytes(true);
        disassembler.prefer_non_abi_reg_names(true);
        disassembler.prefer_unaliased(true);
        disassembler.prefer_offset_jump_targets(true);
        disassembler.emit_header(false);
        disassembler.emit_exports(false);

        let mut disassembly = Vec::new();
        disassembler.disassemble_into(&mut disassembly).unwrap();
        let mut disassembly = String::from_utf8(disassembly).unwrap().replace(" // INVALID", "");
        if initial_pc.0 != 0 {
            let mut disassembly_new = String::new();
            for line in disassembly.lines() {
                if line.trim().starts_with(&format!("{}:", initial_pc.0)) {
                    disassembly_new.push_str("     // Start execution HERE:\n");
                }
                disassembly_new.push_str(line);
                disassembly_new.push('\n');
            }
            disassembly = disassembly_new;
        }

        let mut block_gas_costs = BTreeMap::new();
        let mut timelines = Vec::new();
        let mut timeline_config = polkavm_common::simulator::TimelineConfig::default();
        timeline_config.instruction_format.prefer_non_abi_reg_names = true;
        timeline_config.instruction_format.prefer_unaliased = true;
        let jump_target_formatter = |target: u32, fmt: &mut core::fmt::Formatter| write!(fmt, "{}", target);
        timeline_config.instruction_format.jump_target_formatter = Some(&jump_target_formatter);
        for block in blocks {
            let (timeline, block_cycles) =
                polkavm_common::simulator::timeline_for_instructions(blob.code(), cache_model, &block, timeline_config.clone());
            timelines.push((timeline, block[0].offset.0, block_cycles));

            block_gas_costs.insert(block[0].offset.0, block_cycles);

            // Just a sanity check.
            assert_eq!(i64::from(block_cycles), module.calculate_gas_cost_for(block[0].offset).unwrap());

            // Another sanity check.
            let mut timeline_config_clone = timeline_config.clone();
            timeline_config_clone.should_enable_fast_forward = true;
            let (_, block_cycles_fast_forward) =
                polkavm_common::simulator::timeline_for_instructions(blob.code(), cache_model, &block, timeline_config.clone());
            assert_eq!(block_cycles, block_cycles_fast_forward);
        }

        tests.push(Testcase {
            disassembly,
            timelines,
            initial_page_map,
            final_page_map,
            initial_memory,
            final_memory,
            initial_state,
            final_state,
            json: TestcaseJson {
                name,
                initial_pc: initial_pc.0,
                initial_gas,
                program: parts.code_and_jump_table.to_vec(),
                steps,
                block_gas_costs,
            },
        });
    }

    tests.sort_by_key(|test| test.json.name.clone());

    let output_programs_root = root.join("output").join("programs");
    std::fs::create_dir_all(&output_programs_root).unwrap();

    let mut index_md = String::new();
    writeln!(&mut index_md, "# Testcases\n").unwrap();
    writeln!(&mut index_md, "This file contains a human-readable index of all of the testcases,").unwrap();
    writeln!(&mut index_md, "along with their disassemblies and other relevant information.\n\n").unwrap();

    for test in tests {
        let payload = serde_json::to_string_pretty(&test.json).unwrap();
        let output_path = output_programs_root.join(format!("{}.json", test.json.name));
        if !std::fs::read(&output_path)
            .map(|old_payload| old_payload == payload.as_bytes())
            .unwrap_or(false)
        {
            println!("Generating {output_path:?}...");
            std::fs::write(output_path, payload).unwrap();
        }

        writeln!(&mut index_md, "## {}\n", test.json.name).unwrap();

        if test.json.steps.iter().filter(|step| matches!(step, TestcaseStep::Run)).count() > 1 {
            writeln!(&mut index_md, "Execution steps:").unwrap();
            let mut started = false;
            for step in test.json.steps.iter().skip_while(|step| {
                matches!(
                    step,
                    TestcaseStep::Write { .. } | TestcaseStep::SetReg { .. } | TestcaseStep::Map { .. }
                )
            }) {
                match step {
                    TestcaseStep::Map {
                        address,
                        length,
                        is_writable,
                    } => {
                        let access = if *is_writable { "RW" } else { "RO" };
                        writeln!(
                            &mut index_md,
                            "   * Map page: 0x{:x}-0x{:x} (0x{:x} bytes, {access})",
                            address,
                            address + length,
                            length
                        )
                        .unwrap();
                    }
                    TestcaseStep::Write { address, contents } => {
                        let contents_len = contents.len();
                        let contents: Vec<_> = contents.iter().map(|byte| format!("0x{:02x}", byte)).collect();
                        let contents = contents.join(", ");
                        writeln!(
                            &mut index_md,
                            "   * Write: 0x{:x}-0x{:x} (0x{:x} bytes) = [{}]",
                            address,
                            address + contents_len as u32,
                            contents_len,
                            contents
                        )
                        .unwrap();
                    }
                    TestcaseStep::SetReg { reg, value } => {
                        writeln!(
                            &mut index_md,
                            "   * Set: {} = 0x{:x}",
                            Reg::ALL[*reg as usize].name_non_abi(),
                            value
                        )
                        .unwrap();
                    }
                    TestcaseStep::Assert {
                        status,
                        page_fault_address,
                        hostcall,
                        gas,
                        pc,
                        regs: _,
                        memory: _,
                    } => {
                        let status = status.as_ref().unwrap();
                        let gas = gas.unwrap();
                        let pc = pc.unwrap();
                        let status_extra = if let Some(address) = page_fault_address {
                            format!(" (address = 0x{address:x})")
                        } else if let Some(hostcall) = hostcall {
                            format!(" (hostcall = {hostcall})")
                        } else {
                            String::new()
                        };

                        writeln!(
                            &mut index_md,
                            "   * Execution interrupted: status = '{status}'{status_extra}, gas = {gas}, pc = {pc}",
                        )
                        .unwrap();
                    }
                    TestcaseStep::Run => {
                        if started {
                            writeln!(&mut index_md, "   * Resume execution",).unwrap();
                        } else {
                            writeln!(&mut index_md, "   * Start execution",).unwrap();
                        }
                        started = true;
                    }
                }
            }

            writeln!(&mut index_md).unwrap();
        }

        if !test.initial_page_map.is_empty() {
            writeln!(&mut index_md, "Initial page map:").unwrap();
            for page in &test.initial_page_map {
                let access = if page.is_writable { "RW" } else { "RO" };

                writeln!(
                    &mut index_md,
                    "   * {access}: 0x{:x}-0x{:x} (0x{:x} bytes)",
                    page.address,
                    page.address + page.length,
                    page.length
                )
                .unwrap();
            }

            writeln!(&mut index_md).unwrap();
        }

        if test.initial_page_map != test.final_page_map {
            writeln!(&mut index_md, "Final page map:").unwrap();
            for page in &test.final_page_map {
                let access = if page.is_writable { "RW" } else { "RO" };

                writeln!(
                    &mut index_md,
                    "   * {access}: 0x{:x}-0x{:x} (0x{:x} bytes)",
                    page.address,
                    page.address + page.length,
                    page.length
                )
                .unwrap();
            }

            writeln!(&mut index_md).unwrap();
        }

        if !test.initial_memory.is_empty() {
            writeln!(&mut index_md, "Initial non-zero memory chunks:").unwrap();
            for chunk in &test.initial_memory {
                let contents: Vec<_> = chunk.contents.iter().map(|byte| format!("0x{:02x}", byte)).collect();
                let contents = contents.join(", ");
                writeln!(
                    &mut index_md,
                    "   * 0x{:x}-0x{:x} (0x{:x} bytes) = [{}]",
                    chunk.address,
                    chunk.address + chunk.contents.len() as u32,
                    chunk.contents.len(),
                    contents
                )
                .unwrap();
            }

            writeln!(&mut index_md).unwrap();
        }

        if test.initial_state.regs.iter().any(|value| *value != 0) {
            writeln!(&mut index_md, "Initial non-zero registers:").unwrap();
            for reg in Reg::ALL {
                let value = test.initial_state.regs[reg as usize];
                if value != 0 {
                    writeln!(&mut index_md, "   * {} = 0x{:x}", reg.name_non_abi(), value).unwrap();
                }
            }

            writeln!(&mut index_md).unwrap();
        }

        if test.json.initial_pc != 0 {
            writeln!(&mut index_md, "Initial program counter: {}\n", test.json.initial_pc).unwrap();
        }

        writeln!(&mut index_md, "```\n{}```\n", test.disassembly).unwrap();

        if test
            .initial_state
            .regs
            .iter()
            .zip(test.final_state.regs.iter())
            .any(|(old_value, new_value)| *old_value != *new_value)
        {
            writeln!(&mut index_md, "Registers after execution (only changed registers):").unwrap();
            for reg in Reg::ALL {
                let value_before = test.initial_state.regs[reg as usize];
                let value_after = test.final_state.regs[reg as usize];
                if value_before != value_after {
                    writeln!(
                        &mut index_md,
                        "   * {} = 0x{:x} (initially was 0x{:x})",
                        reg.name_non_abi(),
                        value_after,
                        value_before
                    )
                    .unwrap();
                }
            }

            writeln!(&mut index_md).unwrap();
        }

        if !test.final_memory.is_empty() {
            if test.final_memory == test.initial_memory {
                writeln!(&mut index_md, "The memory contents after execution should be unchanged.").unwrap();
            } else {
                writeln!(&mut index_md, "Final non-zero memory chunks:").unwrap();
                for chunk in &test.final_memory {
                    let contents: Vec<_> = chunk.contents.iter().map(|byte| format!("0x{:02x}", byte)).collect();
                    let contents = contents.join(", ");
                    writeln!(
                        &mut index_md,
                        "   * 0x{:x}-0x{:x} (0x{:x} bytes) = [{}]",
                        chunk.address,
                        chunk.address + chunk.contents.len() as u32,
                        chunk.contents.len(),
                        contents
                    )
                    .unwrap();
                }
            }

            writeln!(&mut index_md).unwrap();
        }

        assert_eq!(
            test.final_state.status == "page-fault",
            test.final_state.page_fault_address.is_some()
        );
        write!(&mut index_md, "Program should end with: {}", test.final_state.status).unwrap();

        if let Some(address) = test.final_state.page_fault_address {
            write!(&mut index_md, " (page address = 0x{:x})", address).unwrap();
        }

        writeln!(&mut index_md, "\n").unwrap();
        writeln!(&mut index_md, "Final value of the program counter: {}\n", test.final_state.pc).unwrap();
        writeln!(
            &mut index_md,
            "Gas consumed: {} -> {}\n",
            test.json.initial_gas, test.final_state.gas
        )
        .unwrap();

        for (timeline, pc, gas_cost) in &test.timelines {
            writeln!(&mut index_md, "Gas simulation at offset {pc} with total cost of {gas_cost}:\n").unwrap();
            writeln!(&mut index_md, "```").unwrap();
            for line in timeline.lines() {
                writeln!(&mut index_md, "    {line}").unwrap();
            }
            writeln!(&mut index_md, "```\n").unwrap();
        }
    }

    std::fs::write(root.join("output").join("TESTCASES.md"), index_md).unwrap();

    if found_errors {
        std::process::exit(1);
    }
}
