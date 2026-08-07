//! Counts executed instructions per opcode for a benchmark blob.
//!
//! Runs the blob's `initialize` export once, then the `run` export (both are
//! the standard benchtool guest protocol), interpreting with step tracing.
//! Reports the dynamic per-opcode instruction counts and the gas (as priced
//! by the module's cost model) consumed by each export.
//!
//! This is the profiler used to verify the wide-arithmetic PoC's predicted
//! per-verify counts (see designs/parachain-service-on-jam in polkadot-sdk).
//!
//! Usage: opcount <blob.polkavm> [<export> ...]
//!        (exports default to: initialize run)

use polkavm::{Config, Engine, GasMeteringKind, InterruptKind, Module, ModuleConfig, ProgramBlob, ProgramCounter};
use std::collections::HashMap;

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: opcount <blob.polkavm> [<export> ...]");
        std::process::exit(1);
    };

    let exports: Vec<String> = {
        let list: Vec<String> = args.collect();
        if list.is_empty() {
            vec!["initialize".to_owned(), "run".to_owned()]
        } else {
            list
        }
    };

    let raw = std::fs::read(&path).expect("failed to read the blob");
    let blob = ProgramBlob::parse(raw.into()).expect("failed to parse the blob");

    // Build a program-counter -> opcode name map by decoding the whole blob once.
    let mut opcode_for_pc: HashMap<u32, &'static str> = HashMap::new();
    for instruction in blob.instructions() {
        opcode_for_pc.insert(instruction.offset.0, instruction.kind.opcode().name());
    }

    let mut config = Config::from_env().expect("invalid config");
    config.set_backend(Some(polkavm::BackendKind::Interpreter));
    let engine = Engine::new(&config).expect("failed to create engine");

    let mut module_config = ModuleConfig::new();
    module_config.set_step_tracing(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).expect("failed to create module");

    let mut instance = module.instantiate().expect("failed to instantiate");

    for export_name in &exports {
        let export_pc = module
            .exports()
            .find(|export| export.symbol().as_bytes() == export_name.as_bytes())
            .map(|export| export.program_counter())
            .unwrap_or_else(|| panic!("export not found: {export_name}"));

        let (counts, gas_used) = run_and_count(&module, &mut instance, export_pc, &opcode_for_pc);
        let total: u64 = counts.values().sum();

        println!("\n=== {export_name}: {total} instructions, {gas_used} gas ===");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by_key(|&(_, &count)| core::cmp::Reverse(count));
        for (name, &count) in sorted {
            println!("{count:>12} {name}");
        }
    }
}

fn run_and_count(
    module: &Module,
    instance: &mut polkavm::RawInstance,
    export_pc: ProgramCounter,
    opcode_for_pc: &HashMap<u32, &'static str>,
) -> (HashMap<&'static str, u64>, i64) {
    let mut counts: HashMap<&'static str, u64> = HashMap::new();

    const INITIAL_GAS: i64 = i64::MAX / 2;
    instance.set_gas(INITIAL_GAS);
    instance.set_reg(polkavm::Reg::RA, polkavm::RETURN_TO_HOST);
    instance.set_reg(polkavm::Reg::SP, module.default_sp());
    instance.set_next_program_counter(export_pc);

    let gas_used;
    loop {
        match instance.run().expect("run failed") {
            InterruptKind::Step => {
                let pc = instance.program_counter().expect("no program counter during step");
                let name = opcode_for_pc.get(&pc.0).copied().unwrap_or("<unknown>");
                *counts.entry(name).or_insert(0) += 1;
            }
            InterruptKind::Finished => {
                gas_used = INITIAL_GAS - instance.gas();
                break;
            }
            InterruptKind::Ecalli(num) => panic!("unexpected hostcall: {num}"),
            interrupt => panic!("unexpected interrupt: {interrupt:?}"),
        }
    }

    (counts, gas_used)
}
