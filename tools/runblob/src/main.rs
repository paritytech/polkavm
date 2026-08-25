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
    let module = polkavm::Module::from_blob(&engine, &Default::default(), blob).expect("module");

    let mut linker: polkavm::Linker<(), ()> = polkavm::Linker::new();
    linker.define_fallback(|_caller: polkavm::Caller<()>, _num: u32| -> Result<(), ()> { Ok(()) });
    let pre = linker.instantiate_pre(&module).expect("pre");
    let mut instance = pre.instantiate().expect("instance");

    let exports: Vec<_> = module.exports().map(|e| (e.symbol().to_string(), e.program_counter())).collect();
    println!("{} exports", exports.len());
    for (name, pc) in exports {
        instance.set_gas(2_000_000);
        instance.set_reg(polkavm::Reg::SP, module.default_sp());
        instance.set_next_program_counter(pc);
        let mut steps = 0u64;
        let outcome = loop {
            match instance.run() {
                Ok(polkavm::InterruptKind::Finished) => break "finished".to_string(),
                Ok(polkavm::InterruptKind::Ecalli(_)) => { steps += 1; if steps > 200 { break format!("ran ({steps} hostcalls)") } }
                Ok(other) => break format!("{other:?}"),
                Err(error) => break format!("error: {error}"),
            }
        };
        println!("  {name}: {outcome}");
    }
}
