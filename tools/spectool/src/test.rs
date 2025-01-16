use std::path::PathBuf;

use log::debug;
use polkavm::{Engine, InterruptKind, Module, ModuleConfig, ProgramBlob, ProgramCounter, ProgramParts, Reg};

use crate::{extract_chunks, MemoryChunk, Page, TestcaseJson};

pub fn main(files: Vec<PathBuf>) -> Result<(), String> {
    let mut fail_count = 0;
    for path in &files {
        let file = std::fs::File::open(path).unwrap();
        let testcase = serde_json::from_reader(file).unwrap();
        if let Err(errors) = run(testcase) {
            fail_count += 1;
            if let Some(path) = path.to_str() {
                eprintln!("Errors in {path}:");
            }
            for e in errors {
                eprintln!("  {e}");
            }
        }
    }

    let count = files.len();
    if fail_count > 0 {
        let okay = count - fail_count;
        eprintln!("{okay}/{count}: OK");
        Err("Some of the files produced errors.".into())
    } else {
        println!("{count}/{count}: OK");
        Ok(())
    }
}

fn run(test: TestcaseJson) -> Result<(), Vec<String>> {
    let mut config = polkavm::Config::new();
    config.set_backend(Some(polkavm::BackendKind::Interpreter));
    let engine = Engine::new(&config).unwrap();

    let mut parts = ProgramParts::default();
    parts.is_64_bit = true;
    parts.code_and_jump_table = test.program.into();
    setup_memory(&mut parts, &test.initial_page_map, &test.initial_memory).map_err(|x| vec![x.to_owned()])?;

    let blob = ProgramBlob::from_parts(parts.clone()).unwrap();

    let mut module_config = ModuleConfig::default();
    module_config.set_strict(true);
    module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
    module_config.set_step_tracing(true);

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    instance.set_gas(test.initial_gas);
    instance.set_next_program_counter(ProgramCounter(test.initial_pc));

    for (reg, value) in Reg::ALL.into_iter().zip(test.initial_regs) {
        instance.set_reg(reg, value);
    }

    let name = test.name;
    debug!("Running {name}");
    let mut final_pc = test.initial_pc;
    let status = loop {
        match instance.run().unwrap() {
            InterruptKind::Finished => break "halt",
            InterruptKind::Trap => break "trap",
            InterruptKind::Ecalli(..) => break "host",
            InterruptKind::NotEnoughGas => break "out-of-gas",
            InterruptKind::Segfault(..) => break "fault",
            InterruptKind::Step => {
                final_pc = instance.program_counter().unwrap().0;
                continue;
            }
        }
    };

    let mut errors = vec![];
    ensure(&mut errors, "Status", status, &test.expected_status);
    ensure(&mut errors, "PC", final_pc, test.expected_pc);
    ensure(&mut errors, "Gas", instance.gas(), test.expected_gas);

    // Registers
    for (reg, expected_value) in Reg::ALL.into_iter().zip(test.expected_regs) {
        let actual_value = instance.reg(reg);
        ensure(&mut errors, &format!("Register[{reg}]"), actual_value, expected_value);
    }

    // Memory
    let mut actual_memory = Vec::new();
    for page in &test.initial_page_map {
        let memory = instance.read_memory(page.address, page.length).unwrap();
        actual_memory.extend(extract_chunks(page.address, &memory));
    }

    for (actual, expected) in actual_memory.iter().zip(&test.expected_memory) {
        let id = format!("Memory @ {:?}", actual.address);
        ensure(&mut errors, &id, format!("{:?}", actual), format!("{:?}", expected));
    }

    ensure(
        &mut errors,
        "Number of memory chunks",
        actual_memory.len(),
        test.expected_memory.len(),
    );

    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(())
    }
}

fn ensure<T: core::fmt::Display + Eq>(errors: &mut Vec<String>, id: &str, actual: T, expected: T) {
    if actual != expected {
        errors.push(format!("{id:>12} | expected: {expected:10}, got: {actual:10}"));
    }
}

fn setup_memory(parts: &mut ProgramParts, pages: &[Page], chunks: &[MemoryChunk]) -> Result<(), &'static str> {
    let mut ro_start = None;
    let mut rw_start = None;
    let mut stack_start = None;

    for page in pages {
        if page.is_writable {
            if rw_start.is_some() {
                if stack_start.is_some() {
                    return Err("Can't set STACK/RW memory twice");
                }
                parts.stack_size = page.length;
                stack_start = Some(page.address);
            } else {
                parts.rw_data_size = page.length;
                rw_start = Some(page.address);
            }
        } else {
            if ro_start.is_some() {
                return Err("Can't set RO memory twice");
            }
            parts.ro_data_size = page.length;
            ro_start = Some(page.address);
        }
    }

    let mut ro_data = vec![0; parts.ro_data_size as usize];
    let mut rw_data = vec![0; parts.rw_data_size as usize];

    let copy_chunk = |chunk: &MemoryChunk, start, size, into: &mut Vec<u8>| {
        if let Some(start) = start {
            if chunk.address >= start {
                let rel_address = chunk.address - start;
                if rel_address < size {
                    let rel_address = rel_address as usize;
                    let rel_end = rel_address + chunk.contents.len();
                    into[rel_address..rel_end].copy_from_slice(&chunk.contents);
                    return true;
                }
            }
        }
        false
    };

    if let Some(ro_start) = ro_start {
        if ro_start != 0x10000 {
            return Err("Unsupported address of RO data.");
        }
    }

    for chunk in chunks {
        let is_in_ro = copy_chunk(chunk, ro_start, parts.ro_data_size, &mut ro_data);
        let is_in_rw = copy_chunk(chunk, rw_start, parts.rw_data_size, &mut rw_data);
        if !is_in_ro && !is_in_rw {
            return Err("Invalid chunk!");
        }
    }

    parts.ro_data = ro_data.into();
    parts.rw_data = rw_data.into();

    Ok(())
}
