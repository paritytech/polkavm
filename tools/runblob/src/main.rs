// Loads a .polkavm blob and runs its exports in the interpreter, reporting how far each got.
// Host imports get a fallback that returns zero, so a contract runs until it needs something
// real rather than refusing to start.
fn main() {
    let _ = env_logger::try_init();
    let path = std::env::args().nth(1).expect("usage: run_blob <blob>");
    let bytes = std::fs::read(&path).expect("read blob");
    let blob = polkavm::ProgramBlob::parse(bytes.into()).expect("parse blob");

    let mut config = polkavm::Config::default();
    // `RUNBLOB_BACKEND=compiler` runs the blob through the recompiler instead.
    let backend = match std::env::var("RUNBLOB_BACKEND").as_deref() {
        Ok("compiler") => polkavm::BackendKind::Compiler,
        _ => polkavm::BackendKind::Interpreter,
    };
    config.set_backend(Some(backend));
    config.set_allow_experimental(true);
    let engine = polkavm::Engine::new(&config).expect("engine");
    // Gas metering is what makes the run measurable: the count is deterministic and identical on
    // both backends, so it compares the work two builds do without wall-clock noise.
    let mut module_config = polkavm::ModuleConfig::default();
    module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
    let module = polkavm::Module::from_blob(&engine, &module_config, blob).expect("module");

    let mut linker: polkavm::Linker<(), ()> = polkavm::Linker::new();
    linker.define_fallback(|_caller: polkavm::Caller<()>, _num: u32| -> Result<(), ()> { Ok(()) });
    let pre = linker.instantiate_pre(&module).expect("pre");
    let mut instance = pre.instantiate().expect("instance");

    // `RUNBLOB_ITERS=N` re-runs each export N times off the one instance and reports the fastest
    // wall-clock, so the recompiler's per-module JIT and the sandbox worker spawn -- both paid once
    // at instantiate -- are amortized away and what is left is execution time. Memory and storage
    // drift across the reruns, so this is a proxy for the run cost, not an exact replay.
    let iters: u64 = std::env::var("RUNBLOB_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

    // Host calls that end the call in production (return / revert / self-destruct). Every host call
    // is otherwise stubbed as a no-op, which lets a contract's return or revert path fall through
    // into the dead code after the (noreturn) host call and spin until it runs out of gas -- so its
    // reported time and gas become "how long to burn the budget" rather than real work. Ending the
    // run at these makes it terminate where production does. Matched by symbol (the import index
    // varies per contract).
    let terminating: std::collections::HashSet<u32> = module
        .imports()
        .iter()
        .enumerate()
        .filter_map(|(index, symbol)| {
            let symbol = symbol?;
            matches!(symbol.as_bytes(), b"seal_return" | b"consume_all_gas" | b"terminate").then_some(index as u32)
        })
        .collect();

    let exports: Vec<_> = module.exports().map(|e| (e.symbol().to_string(), e.program_counter())).collect();
    println!("{} exports", exports.len());
    for (name, pc) in exports {
        const BUDGET: i64 = 2_000_000;
        let run_once = |instance: &mut polkavm::RawInstance| -> (String, u64) {
            instance.set_gas(BUDGET);
            instance.set_reg(polkavm::Reg::SP, module.default_sp());
            instance.set_next_program_counter(pc);
            let mut steps = 0u64;
            let outcome = loop {
                match instance.run() {
                    Ok(polkavm::InterruptKind::Finished) => break "finished".to_string(),
                    Ok(polkavm::InterruptKind::Ecalli(n)) if terminating.contains(&n) => break "ended".to_string(),
                    Ok(polkavm::InterruptKind::Ecalli(_)) => { steps += 1; if steps > 200 { break format!("ran ({steps} hostcalls)") } }
                    Ok(other) => break format!("{other:?}"),
                    Err(error) => break format!("error: {error}"),
                }
            };
            (outcome, steps)
        };

        let (outcome, steps) = run_once(&mut instance);
        // Gas is the deterministic measure of work done, and it is identical across backends, so
        // it compares execution cost between builds without depending on wall-clock noise.
        let used = BUDGET - instance.gas();

        let timing = if iters > 1 {
            let mut best = std::time::Duration::MAX;
            for _ in 0..iters {
                let start = std::time::Instant::now();
                let _ = run_once(&mut instance);
                best = best.min(start.elapsed());
            }
            format!(" time_ns={}", best.as_nanos())
        } else {
            String::new()
        };
        println!("  {name}: {outcome} gas={used} steps={steps}{timing}");
    }
}
