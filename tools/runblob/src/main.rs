// Loads a .polkavm blob and runs its exports in the interpreter, reporting how far each got.
// Host imports get a fallback that returns zero, so a contract runs until it needs something
// real rather than refusing to start.
//
// RUNBLOB_TRACE=1 switches to a value-correctness trace: instead of timing, it records each
// export's observable outputs -- the sequence of host calls, every storage write as key -> value,
// and the final return data -- as a canonical, layout-independent text block. Two builds that
// compute the same thing produce byte-identical traces; a value miscompile (e.g. a truncated wide
// copy) shows up as a differing store value or return payload. See tools diffcheck.py.
/// Zero every wide-instruction cost, leaving scalar ops at their 1-gas cost. Running under this model
/// yields the executed scalar-instruction gas of a build, so `full - scalar_only` is the total gas the
/// wide ops contribute.
fn zero_all_wide(m: &mut polkavm::CostModel) {
    zero_convert(m);
    zero_memory(m);
    m.wide_move = 0;
    zero_linear(m);
    m.wide_mul = 0;
    zero_divrem(m);
    zero_modexp(m);
}
fn zero_convert(m: &mut polkavm::CostModel) {
    m.wide_widen_unsigned = 0;
    m.wide_widen_signed = 0;
    m.wide_truncate = 0;
}
fn zero_memory(m: &mut polkavm::CostModel) {
    m.wide_load = 0;
    m.wide_store = 0;
}
fn zero_linear(m: &mut polkavm::CostModel) {
    m.wide_add = 0;
    m.wide_sub = 0;
    m.wide_and = 0;
    m.wide_or = 0;
    m.wide_xor = 0;
    m.wide_min_unsigned = 0;
    m.wide_min_signed = 0;
    m.wide_max_unsigned = 0;
    m.wide_max_signed = 0;
    m.wide_byte_swap = 0;
    m.wide_sign_extend = 0;
    m.wide_shift_left = 0;
    m.wide_shift_right_logical = 0;
    m.wide_shift_right_arithmetic = 0;
    m.wide_set_equal = 0;
    m.wide_set_not_equal = 0;
    m.wide_set_less_than_unsigned = 0;
    m.wide_set_less_than_signed = 0;
}
fn zero_divrem(m: &mut polkavm::CostModel) {
    m.wide_div_unsigned = 0;
    m.wide_div_signed = 0;
    m.wide_rem_unsigned = 0;
    m.wide_rem_signed = 0;
}
fn zero_modexp(m: &mut polkavm::CostModel) {
    m.wide_add_mod = 0;
    m.wide_mul_mod = 0;
    m.wide_exp = 0;
}

/// RUNBLOB_GASDECOMP: total executed gas over all exports, under a series of cost models that each
/// zero out one wide-op class. Since gas metering never changes control flow (as long as the budget
/// is not exhausted), every model executes the identical instruction stream, so `full - z_<class>`
/// is exactly the gas that class of wide op contributed to this build. Deterministic (interpreter).
fn gas_decomp(engine: &polkavm::Engine, bytes: &[u8]) {
    let variants: Vec<(&str, fn(&mut polkavm::CostModel))> = vec![
        ("full", |_m| {}),
        ("scalar_only", zero_all_wide),
        ("z_convert", zero_convert),
        ("z_memory", zero_memory),
        ("z_move", |m| m.wide_move = 0),
        ("z_linear", zero_linear),
        ("z_mul", |m| m.wide_mul = 0),
        ("z_divrem", zero_divrem),
        ("z_modexp", zero_modexp),
        // Per-opcode isolation (full - z_<op> = that op's executed gas). Scalar buckets first.
        ("z_scalar_load", |m| { m.load_u8=0; m.load_u16=0; m.load_u32=0; m.load_u64=0; m.load_i8=0; m.load_i16=0; m.load_i32=0; }),
        ("z_scalar_store", |m| { m.store_u8=0; m.store_u16=0; m.store_u32=0; m.store_u64=0; }),
        ("z_scalar_reverse_byte", |m| m.reverse_byte = 0),
        // Individual wide opcodes.
        ("z_wide_load", |m| m.wide_load = 0),
        ("z_wide_store", |m| m.wide_store = 0),
        ("z_wide_zext", |m| m.wide_widen_unsigned = 0),
        ("z_wide_trunc", |m| m.wide_truncate = 0),
        ("z_wide_seq", |m| m.wide_set_equal = 0),
        ("z_wide_or", |m| m.wide_or = 0),
        ("z_wide_and", |m| m.wide_and = 0),
        ("z_wide_add", |m| m.wide_add = 0),
        ("z_wide_bswap", |m| m.wide_byte_swap = 0),
        ("z_wide_shift_left", |m| m.wide_shift_left = 0),
        ("z_wide_shift_right_logical", |m| m.wide_shift_right_logical = 0),
    ];
    println!("gas_decomp (interpreter): model<TAB>total_gas");
    for (name, apply) in &variants {
        let mut model = polkavm::CostModel::naive();
        apply(&mut model);
        let kind = polkavm::CostModelKind::from(std::sync::Arc::new(model));
        println!("  {name}\t{}", total_gas_under(engine, bytes, kind));
    }
}

/// Sum executed gas across every export of the blob, under the given cost model. Ends each export
/// where production would (return/revert/self-destruct) and caps stray host-call loops, matching the
/// normal run's termination so the gas reflects the real deploy/call path.
fn total_gas_under(engine: &polkavm::Engine, bytes: &[u8], cost_model: polkavm::CostModelKind) -> i64 {
    let blob = polkavm::ProgramBlob::parse(bytes.to_vec().into()).expect("parse blob");
    let mut module_config = polkavm::ModuleConfig::default();
    module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
    module_config.set_cost_model(Some(cost_model));
    let module = polkavm::Module::from_blob(engine, &module_config, blob).expect("module");
    let symbols: Vec<Option<String>> = module
        .imports()
        .iter()
        .map(|s| s.map(|s| String::from_utf8_lossy(s.as_bytes()).into_owned()))
        .collect();
    let sym = |n: u32| symbols.get(n as usize).and_then(|s| s.as_deref()).unwrap_or("?");
    let terminating: std::collections::HashSet<u32> = (0..symbols.len() as u32)
        .filter(|&n| matches!(sym(n), "seal_return" | "consume_all_gas" | "terminate"))
        .collect();
    let mut linker: polkavm::Linker<(), ()> = polkavm::Linker::new();
    linker.define_fallback(|_caller: polkavm::Caller<()>, _num: u32| -> Result<(), ()> { Ok(()) });
    let mut instance = linker.instantiate_pre(&module).expect("pre").instantiate().expect("instance");
    let exports: Vec<_> = module.exports().map(|e| e.program_counter()).collect();
    const BUDGET: i64 = 2_000_000;
    let mut total = 0i64;
    for pc in exports {
        instance.set_gas(BUDGET);
        instance.set_reg(polkavm::Reg::SP, module.default_sp());
        instance.set_next_program_counter(pc);
        let mut steps = 0u64;
        loop {
            match instance.run() {
                Ok(polkavm::InterruptKind::Finished) => break,
                Ok(polkavm::InterruptKind::Ecalli(n)) => {
                    if terminating.contains(&n) {
                        break;
                    }
                    steps += 1;
                    if steps > 2000 {
                        break;
                    }
                }
                Ok(_) => break,
                Err(_) => break,
            }
        }
        total += BUDGET - instance.gas();
    }
    total
}

fn main() {
    let _ = env_logger::try_init();
    let path = std::env::args().nth(1).expect("usage: run_blob <blob>");
    let bytes = std::fs::read(&path).expect("read blob");
    let blob = polkavm::ProgramBlob::parse(bytes.clone().into()).expect("parse blob");

    let mut config = polkavm::Config::default();
    // `RUNBLOB_BACKEND=compiler` runs the blob through the recompiler instead.
    let backend = match std::env::var("RUNBLOB_BACKEND").as_deref() {
        Ok("compiler") => polkavm::BackendKind::Compiler,
        _ => polkavm::BackendKind::Interpreter,
    };
    config.set_backend(Some(backend));
    config.set_allow_experimental(true);
    let engine = polkavm::Engine::new(&config).expect("engine");

    // RUNBLOB_GASDECOMP: attribute total gas to each wide-op class (see gas_decomp), then exit.
    if std::env::var("RUNBLOB_GASDECOMP").is_ok() {
        gas_decomp(&engine, &bytes);
        return;
    }
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
    let trace = std::env::var("RUNBLOB_TRACE").is_ok();

    // Import index -> symbol. Host-call indices are per-contract, so everything keys off the symbol.
    let symbols: Vec<Option<String>> = module
        .imports()
        .iter()
        .map(|s| s.map(|s| String::from_utf8_lossy(s.as_bytes()).into_owned()))
        .collect();
    let sym = |n: u32| symbols.get(n as usize).and_then(|s| s.as_deref()).unwrap_or("?");

    // Host calls that end the call in production (return / revert / self-destruct). Every host call
    // is otherwise stubbed as a no-op, which lets a contract's return or revert path fall through
    // into the dead code after the (noreturn) host call and spin until it runs out of gas -- so its
    // reported time and gas become "how long to burn the budget" rather than real work. Ending the
    // run at these makes it terminate where production does. Matched by symbol (the import index
    // varies per contract).
    let terminating: std::collections::HashSet<u32> = (0..symbols.len() as u32)
        .filter(|&n| matches!(sym(n), "seal_return" | "consume_all_gas" | "terminate"))
        .collect();

    let exports: Vec<_> = module.exports().map(|e| (e.symbol().to_string(), e.program_counter())).collect();
    println!("{} exports", exports.len());

    for (name, pc) in exports {
        const BUDGET: i64 = 2_000_000;

        // A single run that records the value trace (host-call symbols, storage writes, return data).
        // The trace is deliberately layout-independent: it captures the *data* behind the pointer
        // arguments (storage key/value, return payload), not the pointer values, which differ with
        // stack/heap layout between builds.
        let run_traced = |instance: &mut polkavm::RawInstance| -> (String, Vec<String>) {
            instance.set_gas(BUDGET);
            instance.set_reg(polkavm::Reg::SP, module.default_sp());
            instance.set_next_program_counter(pc);
            let read_hex = |instance: &mut polkavm::RawInstance, ptr: u64, len: u64| -> String {
                let len = len.min(4096) as u32;
                match instance.read_memory(ptr as u32, len) {
                    Ok(bytes) => bytes.iter().map(|b| format!("{b:02x}")).collect(),
                    Err(_) => "<unreadable>".to_string(),
                }
            };
            let mut events = Vec::new();
            let mut steps = 0u64;
            let outcome = loop {
                match instance.run() {
                    Ok(polkavm::InterruptKind::Finished) => break "finished".to_string(),
                    Ok(polkavm::InterruptKind::Ecalli(n)) => {
                        let s = sym(n);
                        match s {
                            // key and value are both 32-byte EVM words at a1/a2.
                            "set_storage_or_clear" => {
                                let (k, v) = (instance.reg(polkavm::Reg::A1), instance.reg(polkavm::Reg::A2));
                                let key = read_hex(instance, k, 32);
                                let val = read_hex(instance, v, 32);
                                events.push(format!("store key={key} val={val}"));
                            }
                            // seal_return(flags, data_ptr, data_len): the return payload.
                            "seal_return" => {
                                let flags = instance.reg(polkavm::Reg::A0);
                                let (ptr, len) = (instance.reg(polkavm::Reg::A1), instance.reg(polkavm::Reg::A2));
                                let data = read_hex(instance, ptr, len);
                                events.push(format!("return flags={flags} data={data}"));
                                break "ended".to_string();
                            }
                            _ if terminating.contains(&n) => {
                                events.push(format!("terminate {s}"));
                                break "ended".to_string();
                            }
                            _ => {
                                events.push(format!("call {s}"));
                                steps += 1;
                                if steps > 2000 {
                                    break format!("ran ({steps} hostcalls)");
                                }
                            }
                        }
                    }
                    Ok(other) => break format!("{other:?}"),
                    Err(error) => break format!("error: {error}"),
                }
            };
            (outcome, events)
        };

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

        if trace {
            let (outcome, events) = run_traced(&mut instance);
            let used = BUDGET - instance.gas();
            // Canonical block: outcome + gas, then one line per observable event, indented so the
            // diff driver can slice per export.
            println!("  {name}: {outcome} gas={used}");
            for e in events {
                println!("    {e}");
            }
            continue;
        }

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
