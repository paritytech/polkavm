use polkavm::{
    BackendKind, Config, Engine, GasMeteringKind, InterpreterCompilationCost, InterruptKind, Module, ModuleConfig, ProgramBlob,
    ProgramCounter, RawInstance, Reg, SetCacheSizeLimitArgs,
};
use polkavm_common::program::{asm, interpreter_calculate_cache_num_entries, interpreter_calculate_cache_size, InstructionSetKind};
use polkavm_common::writer::ProgramBlobBuilder;

fn chain_of_blocks_test_blob(isa: InstructionSetKind) -> ProgramBlob {
    let mut code = Vec::new();
    for block in 0..4 {
        code.extend([
            asm::add_imm_32(Reg::A0, Reg::A0, 1),
            asm::add_imm_32(Reg::A0, Reg::A0, 1),
            asm::add_imm_32(Reg::A0, Reg::A0, 1),
        ]);
        code.push(if block < 3 { asm::jump(block + 1) } else { asm::ret() });
    }

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
}

fn run_and_get_gas_used(instance: &mut RawInstance, program_counter: ProgramCounter) -> i64 {
    instance.set_gas(1000);
    instance.set_reg(Reg::RA, polkavm::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(program_counter);
    loop {
        match instance.run().unwrap() {
            InterruptKind::Step => {}
            interrupt => {
                assert_eq!(interrupt, InterruptKind::Finished);
                return 1000 - instance.gas();
            }
        }
    }
}

// Unit tests always emit the debug logging handlers; without a logger here, they aren't emitted.
#[test]
fn cache_size_limit_does_not_depend_on_step_tracing_without_debug_logging() {
    let mut config = Config::default();
    config.set_backend(Some(BackendKind::Interpreter));
    let engine = Engine::new(&config).unwrap();
    let cost = InterpreterCompilationCost {
        per_block: 10,
        per_instruction_millis: 500,
        per_code_byte_millis: 0,
    };

    for isa in [InstructionSetKind::Latest64, InstructionSetKind::ReviveV1] {
        let run = |step_tracing: bool, max_cache_entries: usize| {
            let mut module_config = ModuleConfig::new();
            module_config.set_gas_metering(Some(GasMeteringKind::Sync));
            module_config.set_step_tracing(step_tracing);
            module_config.set_interpreter_compilation_cost(Some(cost));
            let blob = chain_of_blocks_test_blob(isa);
            let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();
            let module = Module::from_blob(&engine, &module_config, blob).unwrap();

            let mut instance = module.instantiate().unwrap();
            instance
                .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                    max_block_size: 5,
                    max_cache_size_bytes: interpreter_calculate_cache_size(max_cache_entries),
                }))
                .unwrap();

            [offsets[0], offsets[0], offsets[1]].map(|program_counter| {
                let gas = run_and_get_gas_used(&mut instance, program_counter);
                (
                    gas,
                    interpreter_calculate_cache_num_entries(instance.interpreter_cache_size().unwrap()),
                )
            })
        };

        // Same as in the `cache_size_limit_does_not_depend_on_step_tracing` unit test.
        let fits_one_block = [(16 + 7 * 12, 5), (16 + 8 * 12, 5), (16 + 8 * 12, 5)];
        let fits_all_blocks = [(16 + 4 * 12, 20), (16, 20), (16, 20)];
        for (max_cache_entries, expected) in [(14, fits_one_block), (27, fits_all_blocks)] {
            for step_tracing in [false, true] {
                assert_eq!(
                    run(step_tracing, max_cache_entries),
                    expected,
                    "expected the same cache contents with max_cache_entries={max_cache_entries}, step_tracing={step_tracing}"
                );
            }
        }
    }
}
