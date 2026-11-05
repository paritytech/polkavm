use crate::mutex::Mutex;
use crate::{
    BackendKind, CallError, Caller, CompileError, Config, Engine, GasMeteringKind, InterpreterCompilationCost, InterruptKind, Linker,
    MemoryAccessError, MemoryProtection, Module, ModuleConfig, ProgramBlob, ProgramCounter, RawInstance, Reg, Segfault,
    SetCacheSizeLimitArgs,
};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use polkavm_common::abi::MemoryMapBuilder;
use polkavm_common::cast::cast;
use polkavm_common::program::{asm, InstructionSet, InstructionSetKind, Opcode};
use polkavm_common::program::{
    interpreter_calculate_cache_num_entries, interpreter_calculate_cache_size, BlobLen, Reg::*, INTERPRETER_CACHE_ENTRY_SIZE,
};
use polkavm_common::utils::align_to_next_page_u32;
use polkavm_common::writer::ProgramBlobBuilder;
use polkavm_linker::TargetInstructionSet;

use paste::paste;

const INVALID_RAW_OPCODE: u8 = 254;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum TestProgram {
    Pinky,
    TestBlob,
    TestBlobSo,
}

#[cfg(feature = "std")]
fn get_test_program(kind: TestProgram, is_64_bit: bool) -> &'static [u8] {
    static ELF_MAP: Mutex<BTreeMap<(TestProgram, bool), &'static [u8]>> = Mutex::new(BTreeMap::new());
    let mut elf_map = ELF_MAP.lock();
    if let Some(blob) = elf_map.get(&(kind, is_64_bit)) {
        return blob;
    }

    let path = std::path::PathBuf::new()
        .join(std::env!("CARGO_MANIFEST_DIR"))
        .join("../../guest-programs");

    let mut envs: alloc::collections::BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| !["CARGO", "RUSTC", "RUSTUP"].iter().any(|e| k.contains(e)))
        .collect();

    let target_dir = if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        envs.insert("CARGO_TARGET_DIR".to_owned(), target_dir.clone());
        std::path::PathBuf::from(target_dir)
    } else {
        path.join("target")
    };

    let mut args = polkavm_linker::TargetJsonArgs::default();
    args.is_64_bit = is_64_bit;
    args.rustc_version = polkavm_linker::RustcVersion::Legacy;

    let (target, target_path) = if is_64_bit {
        ("riscv64emac-unknown-none-polkavm", polkavm_linker::target_json_path(args).unwrap())
    } else {
        ("riscv32emac-unknown-none-polkavm", polkavm_linker::target_json_path(args).unwrap())
    };

    let (project, filename, profile, is_bin) = match kind {
        TestProgram::Pinky => ("bench-pinky", "bench-pinky", "release", true),
        TestProgram::TestBlob => ("test-blob", "test-blob", "no-lto", true),
        TestProgram::TestBlobSo => ("test-blob", "test_blob.elf", "no-lto", false),
    };

    let mut cmd = std::process::Command::new("cargo");
    cmd.env_clear()
        .arg("build")
        .arg("-q")
        .arg("--profile")
        .arg(profile)
        .arg("-p")
        .arg(project)
        .arg("--target")
        .arg(target_path)
        .arg("-Zbuild-std=core,alloc")
        .current_dir(path.to_str().unwrap())
        .envs(envs);

    if is_bin {
        cmd.arg("--bin").arg(project);
    } else {
        cmd.arg("--lib");
    }

    let res = cmd.output().unwrap();
    if !res.status.success() {
        core::mem::drop(elf_map);
        panic!("{}", String::from_utf8_lossy(&res.stderr));
    }

    let blob = std::fs::read(target_dir.join(target).join(profile).join(filename)).unwrap().leak();

    elf_map.insert((kind, is_64_bit), blob);
    blob
}

#[cfg(not(feature = "std"))]
fn get_test_program(kind: TestProgram, is_64_bit: bool) -> &'static [u8] {
    match (kind, is_64_bit) {
        (TestProgram::Pinky, true) => include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/release/bench-pinky"),
        (TestProgram::Pinky, false) => unreachable!(),
        (TestProgram::TestBlob, true) => include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/no-lto/test-blob"),
        (TestProgram::TestBlob, false) => {
            include_bytes!("../../../guest-programs/target/riscv32emac-unknown-none-polkavm/no-lto/test-blob")
        }
        (TestProgram::TestBlobSo, true) => {
            include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/no-lto/test_blob.elf")
        }
        (TestProgram::TestBlobSo, false) => {
            include_bytes!("../../../guest-programs/target/riscv32emac-unknown-none-polkavm/no-lto/test_blob.elf")
        }
    }
}

fn get_native_page_size() -> usize {
    if_compiler_is_supported! {
        { crate::sandbox::get_native_page_size() } else { 4096 }
    }
}

#[track_caller]
fn assert_out_of_range_access<T>(result: Result<T, MemoryAccessError>, expected_address: u32, expected_length: u32) {
    match result {
        Ok(_) => panic!("expected Err(MemoryAccessError::OutOfRangeAccess), got Ok"),
        Err(MemoryAccessError::OutOfRangeAccess { address, length })
            if address == expected_address && length == u64::from(expected_length) => {}
        Err(error) => panic!(
            "expected Err(MemoryAccessError::OutOfRangeAccess {{ address: {expected_address}, length: {expected_length} }}), got {error:?}"
        ),
    }
}

macro_rules! run_tests_on_isa {
    ($isa_suffix:ident, $isa:expr, $($test_name:ident)+) => {
        if_compiler_is_supported! {
            $(
                paste! {
                    #[cfg(target_os = "linux")]
                    #[test]
                    fn [<compiler_linux_ $isa_suffix _ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_worker_count(1);
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Linux));
                        $test_name(config, $isa);
                    }

                    #[cfg(target_os = "linux")]
                    #[test]
                    fn [<tracing_linux_ $isa_suffix _ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Linux));
                        config.set_allow_experimental(true);
                        config.set_crosscheck(true);
                        $test_name(config, $isa);
                    }

                    #[cfg(feature = "generic-sandbox")]
                    #[test]
                    fn [<compiler_generic_ $isa_suffix _ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Generic));
                        config.set_allow_experimental(true);
                        $test_name(config, $isa);
                    }

                    #[cfg(feature = "generic-sandbox")]
                    #[test]
                    fn [<tracing_generic_ $isa_suffix _ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Generic));
                        config.set_allow_experimental(true);
                        config.set_crosscheck(true);
                        $test_name(config, $isa);
                    }
                }
            )+
        }

        $(
            paste! {
                #[test]
                fn [<interpreter_ $isa_suffix _ $test_name>]() {
                    let mut config = crate::Config::default();
                    config.set_backend(Some(crate::BackendKind::Interpreter));
                    $test_name(config, $isa);
                }
            }
        )+
    }
}

macro_rules! run_tests {
    ($($test_name:ident)+) => {
        run_tests_on_isa! { latest64, InstructionSetKind::Latest64, $($test_name)+ }
        run_tests_on_isa! { revive_v1, InstructionSetKind::ReviveV1, $($test_name)+ }
    }
}

macro_rules! run_test_blob_tests {
    ($($test_name:ident)+) => {
        paste! {
            run_tests_on_isa! { latest32, InstructionSetKind::Latest32,
                $([<unoptimized_bin_32_ $test_name>])+
                $([<unoptimized_cdylib_32_ $test_name>])+
                $([<optimized_bin_32_ $test_name>])+
                $([<optimized_cdylib_32_ $test_name>])+
            }
            run_tests! {
                $([<unoptimized_bin_64_ $test_name>])+
                $([<unoptimized_cdylib_64_ $test_name>])+
                $([<optimized_bin_64_ $test_name>])+
                $([<optimized_cdylib_64_ $test_name>])+
            }
        }

        $(
            paste! {
                fn [<unoptimized_bin_32_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: false,
                        is_64_bit: false,
                        is_cdylib: false
                    })
                }

                fn [<unoptimized_bin_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: false,
                        is_64_bit: true,
                        is_cdylib: false
                    })
                }

                fn [<unoptimized_cdylib_32_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: false,
                        is_64_bit: false,
                        is_cdylib: true
                    })
                }

                fn [<unoptimized_cdylib_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: false,
                        is_64_bit: true,
                        is_cdylib: true
                    })
                }

                fn [<optimized_bin_32_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: true,
                        is_64_bit: false,
                        is_cdylib: false
                    })
                }

                fn [<optimized_bin_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: true,
                        is_64_bit: true,
                        is_cdylib: false
                    })
                }

                fn [<optimized_cdylib_32_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: true,
                        is_64_bit: false,
                        is_cdylib: true
                    })
                }

                fn [<optimized_cdylib_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(TestBlobArgs {
                        config,
                        isa,
                        optimize: true,
                        is_64_bit: true,
                        is_cdylib: true
                    })
                }
            }
        )+
    }
}

macro_rules! run_asm_tests {
    ($($test_name:ident)+) => {
        paste! {
            run_tests! {
                $([<unoptimized_64_ $test_name>])+
                $([<optimized_64_ $test_name>])+
            }
        }

        $(
            paste! {
                fn [<unoptimized_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(config, isa, false)
                }

                fn [<optimized_64_ $test_name>](config: Config, isa: InstructionSetKind) {
                    $test_name(config, isa, true)
                }
            }
        )+
    }
}

fn basic_test_blob(isa: InstructionSetKind) -> ProgramBlob {
    let memory_map = MemoryMapBuilder::new(0x4000).rw_data_size(0x4000).build().unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_rw_data_size(0x4000);
    builder.add_export_by_basic_block(0, b"main");
    builder.add_import(b"hostcall");
    builder.set_code(
        &[
            asm::store_imm_u32(memory_map.rw_data_address().try_into().unwrap(), 0x12345678),
            asm::add_32(S0, A0, A1),
            asm::ecalli(0),
            asm::add_32(A0, A0, S0),
            asm::ret(),
        ],
        &[],
    );
    ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
}

fn basic_test(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let blob = basic_test_blob(isa);
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
    let mut linker: Linker<State, MemoryAccessError> = Linker::new();

    #[derive(Default)]
    struct State {}

    let address = module.memory_map().rw_data_address();
    linker
        .define_typed("hostcall", move |caller: Caller<State>| -> Result<u32, MemoryAccessError> {
            let value = caller.instance.read_u32(address)?;
            assert_eq!(value, 0x12345678);

            Ok(100)
        })
        .unwrap();

    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();
    let mut state = State::default();
    let result = instance
        .call_typed_and_get_result::<u32, (u32, u32)>(&mut state, "main", (1, 10))
        .unwrap();

    assert_eq!(result, 111);
}

fn fallback_hostcall_handler_works(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let blob = basic_test_blob(isa);
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
    let mut linker = Linker::new();

    linker.define_fallback(move |caller: Caller<()>, num: u32| -> Result<(), ()> {
        assert_eq!(num, 0);
        caller.instance.set_reg(Reg::A0, 100);
        Ok(())
    });

    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();
    let result = instance
        .call_typed_and_get_result::<u32, (u32, u32)>(&mut (), "main", (1, 10))
        .unwrap();

    assert_eq!(result, 111);
}

macro_rules! match_interrupt {
    ($interrupt:expr, $pattern:pat) => {
        let i = $interrupt;
        assert!(
            matches!(i, $pattern),
            "unexpected interrupt: {i:?}, expected: {:?}",
            stringify!($pattern)
        );
    };
}

fn step_tracing_basic(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let blob = basic_test_blob(isa);
    let engine = Engine::new(&engine_config).unwrap();
    let mut config = ModuleConfig::new();
    config.set_step_tracing(true);
    let code_length = blob.code().len() as u32;

    let module = Module::from_blob(&engine, &config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();
    assert_eq!(instance.program_counter(), None);
    assert_eq!(instance.next_program_counter(), None);
    assert!(instance.next_native_program_counter().is_none());

    for pc in 0..=code_length + 1 {
        let pc = ProgramCounter(pc);
        instance.set_next_program_counter(pc);
        assert_eq!(instance.program_counter(), None);
        assert_eq!(instance.next_program_counter(), Some(pc));
    }

    let entry_point = module.exports().find(|export| export == "main").unwrap().program_counter();
    assert_eq!(entry_point.0, 0);

    let list: Vec<_> = module.blob().instructions().collect();
    let address = module.memory_map().rw_data_address();

    instance.prepare_call_typed(entry_point, (1, 10));
    assert_eq!(instance.program_counter(), None);
    assert_eq!(instance.next_program_counter(), Some(entry_point));

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.program_counter(), Some(list[0].offset));
    assert_eq!(instance.next_program_counter(), Some(list[0].offset));
    assert_eq!(instance.read_u32(address).unwrap(), 0);

    // u32 [0x20000] = 305419896

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.program_counter(), Some(list[1].offset));
    assert_eq!(instance.next_program_counter(), Some(list[1].offset));
    assert_eq!(instance.read_u32(address).unwrap(), 0x12345678);
    assert_eq!(instance.reg(Reg::S0), 0);

    // s0 = a0 + a1

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.program_counter(), Some(list[2].offset));
    assert_eq!(instance.next_program_counter(), Some(list[2].offset));
    assert_eq!(instance.reg(Reg::S0), 11);

    // ecalli 0

    match_interrupt!(instance.run().unwrap(), InterruptKind::Ecalli(0));
    assert_eq!(instance.program_counter(), Some(list[2].offset));
    assert_eq!(instance.next_program_counter(), Some(list[2].next_offset));
    instance.set_reg(Reg::A0, 100);

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.program_counter(), Some(list[3].offset));
    assert_eq!(instance.next_program_counter(), Some(list[3].offset));
    assert_eq!(instance.reg(Reg::A0), 100);

    // a0 = a0 + s0

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.program_counter(), Some(list[4].offset));
    assert_eq!(instance.next_program_counter(), Some(list[4].offset));
    assert_eq!(instance.reg(Reg::A0), 111);

    // ret

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.program_counter(), None);
    assert_eq!(instance.next_program_counter(), None);
    assert_eq!(instance.reg(Reg::A0), 111);

    assert_eq!(
        instance.run().unwrap_err().to_string(),
        "failed to run: next program counter is not set"
    );

    // trap, implicit and misaligned

    for offset in [code_length, code_length + 1, code_length + 1000, 0xffffffff, 1] {
        log::trace!("Testing trap at: {}", offset);
        instance.set_next_program_counter(ProgramCounter(offset));
        assert!(instance.program_counter().is_none()); // Calling `set_next_program_counter` clears the program counter.
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.program_counter(), Some(ProgramCounter(offset)));
        assert_eq!(instance.next_program_counter(), Some(ProgramCounter(offset)));

        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.program_counter(), Some(ProgramCounter(offset)));
        assert_eq!(instance.next_program_counter(), Some(ProgramCounter(offset)));
    }
}

fn reclaim_cache_memory(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_imm(A0, 0x1234),
            asm::add_imm_32(A1, A1, 100),
            asm::ecalli(0),
            asm::load_imm(A2, 0x4567),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let list: Vec<_> = module.blob().instructions().collect();

    let mut instance = module.instantiate().unwrap();

    instance.reset_interpreter_cache().unwrap();

    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A1, 0);
    instance.set_next_program_counter(list[0].offset);
    instance.reset_interpreter_cache().unwrap();

    // ecalli 0

    match_interrupt!(instance.run().unwrap(), InterruptKind::Ecalli(0));
    assert_eq!(instance.program_counter(), Some(list[2].offset));
    assert_eq!(instance.next_program_counter(), Some(list[2].next_offset));
    assert_eq!(instance.reg(Reg::A0), 0x1234);
    assert_eq!(instance.reg(Reg::A1), 100);

    // ret

    instance.reset_interpreter_cache().unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A2), 0x4567);
}

fn bounded_interpreter_cache(config: Config, isa: InstructionSetKind) {
    // this test is only relevant for the interpreter backend
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"entry_0");
    builder.add_export_by_basic_block(1, b"entry_1");
    builder.add_export_by_basic_block(2, b"entry_2");
    builder.add_export_by_basic_block(3, b"entry_3");
    builder.add_export_by_basic_block(4, b"entry_4");
    builder.add_export_by_basic_block(5, b"entry_5");

    // 0 -> 4 -> 1 -> 3 -> 2 -> 5
    builder.set_code(
        &[
            asm::add_imm_32(A0, A0, 0x1),
            asm::jump(4),
            asm::add_imm_32(A1, A1, 0x1),
            asm::add_imm_32(A1, A1, 0x10),
            asm::add_imm_32(A1, A1, 0x100),
            asm::add_imm_32(A1, A1, 0x1),
            asm::add_imm_32(A1, A1, 0x10),
            asm::add_imm_32(A1, A1, 0x100),
            asm::jump(3),
            asm::add_imm_32(A2, A2, 0x1),
            asm::add_imm_32(A2, A2, 0x10),
            asm::add_imm_32(A2, A2, 0x100),
            asm::add_imm_32(A2, A2, 0x1000),
            asm::add_imm_32(A2, A2, 0x1),
            asm::add_imm_32(A2, A2, 0x10),
            asm::add_imm_32(A2, A2, 0x100),
            asm::add_imm_32(A2, A2, 0x1000),
            asm::add_imm_32(A2, A2, 0x1),
            asm::add_imm_32(A2, A2, 0x10),
            asm::add_imm_32(A2, A2, 0x100),
            asm::add_imm_32(A2, A2, 0x1000),
            asm::jump(5),
            asm::add_imm_32(A3, A3, 0x1),
            asm::add_imm_32(A3, A3, 0x10),
            asm::add_imm_32(A3, A3, 0x100),
            asm::add_imm_32(A3, A3, 0x1),
            asm::add_imm_32(A3, A3, 0x10),
            asm::add_imm_32(A3, A3, 0x100),
            asm::jump(2),
            asm::add_imm_32(A4, A4, 0x1),
            asm::add_imm_32(A4, A4, 0x10),
            asm::jump(1),
            asm::load_imm(A5, 0x1),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let exports: Vec<_> = module.exports().map(|export| export.program_counter()).collect();

    let mut instance = module.instantiate().unwrap();

    assert_eq!(INTERPRETER_CACHE_ENTRY_SIZE, 24);

    let minimum_cache_size = 24 * 2;

    for max_cache_size_bytes in 0..minimum_cache_size {
        log::debug!("Testing with max_cache_size_bytes: {}", max_cache_size_bytes);
        assert!(instance
            .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                max_block_size: 0,
                max_cache_size_bytes,
            }))
            .is_err());
    }

    assert!(instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 0,
            max_cache_size_bytes: minimum_cache_size,
        }))
        .is_ok());

    for (max_block_size, max_cache_size_bytes) in [(0, 24 * 12), (5, 24 * 22), (12, 1000000)] {
        for (start_block, export) in exports.iter().enumerate() {
            log::debug!(
                "Testing with max_block_size: {}, max_cache_size_bytes: {}, start_block: {}",
                max_block_size,
                max_cache_size_bytes,
                start_block
            );

            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_reg(Reg::A0, 0);
            instance.set_reg(Reg::A1, 0);
            instance.set_reg(Reg::A2, 0);
            instance.set_reg(Reg::A3, 0);
            instance.set_reg(Reg::A4, 0);
            instance.set_reg(Reg::A5, 0);
            instance.set_next_program_counter(*export);

            assert!(instance
                .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                    max_block_size,
                    max_cache_size_bytes,
                }))
                .is_ok());

            match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

            if start_block == 0 {
                assert_eq!(instance.reg(Reg::A0), 0x1);
            }

            if matches!(start_block, 0 | 1 | 4) {
                assert_eq!(instance.reg(Reg::A1), 0x222);
            }

            if matches!(start_block, 0 | 1 | 2 | 3 | 4) {
                assert_eq!(instance.reg(Reg::A2), 0x3333);
            }

            if matches!(start_block, 0 | 1 | 3 | 4) {
                assert_eq!(instance.reg(Reg::A3), 0x222);
            }

            if matches!(start_block, 0 | 4) {
                assert_eq!(instance.reg(Reg::A4), 0x11);
            }

            if matches!(start_block, 0 | 1 | 2 | 3 | 4 | 5) {
                assert_eq!(instance.reg(Reg::A5), 0x1);
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum InterpreterBranchTestSide {
    Fallthrough,
    Target,
}

impl InterpreterBranchTestSide {
    pub(crate) fn a0(self) -> u64 {
        match self {
            InterpreterBranchTestSide::Fallthrough => 1,
            InterpreterBranchTestSide::Target => 0,
        }
    }
}

pub(crate) struct InterpreterBranchTestBlob {
    pub(crate) blob: ProgramBlob,
    pub(crate) branch: ProgramCounter,
    pub(crate) fallthrough_block: Vec<ProgramCounter>,
    pub(crate) target_block: Vec<ProgramCounter>,
}

impl InterpreterBranchTestBlob {
    const TARGET_BLOCK_ADD_COUNT: u64 = 32;

    pub(crate) const BRANCH_SIDE_ORDERS: [(InterpreterBranchTestSide, InterpreterBranchTestSide); 2] = [
        (InterpreterBranchTestSide::Fallthrough, InterpreterBranchTestSide::Target),
        (InterpreterBranchTestSide::Target, InterpreterBranchTestSide::Fallthrough),
    ];

    fn path(&self, side: InterpreterBranchTestSide) -> Vec<ProgramCounter> {
        let block = match side {
            InterpreterBranchTestSide::Fallthrough => &self.fallthrough_block,
            InterpreterBranchTestSide::Target => &self.target_block,
        };

        core::iter::once(self.branch).chain(block.iter().copied()).collect()
    }
}

pub(crate) fn interpreter_branch_test_blob(isa: InstructionSetKind) -> InterpreterBranchTestBlob {
    let mut code = vec![asm::branch_eq_imm(A0, 0, 2), asm::add_imm_32(A1, A1, 1), asm::ret()];
    code.extend((0..InterpreterBranchTestBlob::TARGET_BLOCK_ADD_COUNT).map(|_| asm::add_imm_32(A2, A2, 1)));
    code.push(asm::ret());

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let [branch_block, fallthrough_block, target_block] = <[Vec<ProgramCounter>; 3]>::try_from(basic_blocks(&blob)).unwrap();
    let [branch] = <[ProgramCounter; 1]>::try_from(branch_block).unwrap();
    InterpreterBranchTestBlob {
        blob,
        branch,
        fallthrough_block,
        target_block,
    }
}

fn basic_blocks(blob: &ProgramBlob) -> Vec<Vec<ProgramCounter>> {
    let instructions: Vec<_> = blob.instructions().collect();
    instructions
        .split_inclusive(|instruction| instruction.opcode().starts_new_basic_block())
        .map(|block| block.iter().map(|instruction| instruction.offset).collect())
        .collect()
}

fn run_interpreter_branch_test(instance: &mut RawInstance, test_blob: &InterpreterBranchTestBlob, side: InterpreterBranchTestSide) {
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, side.a0());
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::A2, 0);
    instance.set_next_program_counter(test_blob.branch);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);

    let expected = match side {
        InterpreterBranchTestSide::Fallthrough => (1, 0),
        InterpreterBranchTestSide::Target => (0, InterpreterBranchTestBlob::TARGET_BLOCK_ADD_COUNT),
    };
    assert_eq!((instance.reg(Reg::A1), instance.reg(Reg::A2)), expected);
}

fn lazy_branch_target_compilation_on_first_take(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let module = Module::from_blob(&engine, &ModuleConfig::new(), test_blob.blob.clone()).unwrap();
    let target_block_minimum_size = interpreter_calculate_cache_size(test_blob.target_block.len());
    for (first, second) in InterpreterBranchTestBlob::BRANCH_SIDE_ORDERS {
        let mut instance = module.instantiate().unwrap();
        let mut run = |side: InterpreterBranchTestSide| {
            run_interpreter_branch_test(&mut instance, &test_blob, side);
            instance.interpreter_cache_size()
        };

        let (Some(size_1), Some(size_2), Some(size_3), Some(size_4)) = (run(first), run(second), run(second), run(first)) else {
            continue;
        };

        if second == InterpreterBranchTestSide::Target {
            assert!(
                size_1 < target_block_minimum_size,
                "expected the untaken big block to not be compiled"
            );
            assert!(
                size_2 - size_1 >= target_block_minimum_size,
                "expected the big block to be compiled when taken"
            );
        }

        assert!(size_2 > size_1, "expected the deferred target to be compiled when taken");
        assert_eq!(size_3, size_2, "expected nothing to be compiled on the second take");
        assert_eq!(size_4, size_3, "expected nothing to be compiled on the second take");
    }
}

fn lazy_branch_target_cache_reset(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let module = Module::from_blob(&engine, &ModuleConfig::new(), test_blob.blob.clone()).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 0,
            max_cache_size_bytes: 1000000,
        }))
        .unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(test_blob.target_block[0]);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    let target_block_size = instance.interpreter_cache_size();
    let fallthrough_path_size = target_block_size.map(|target_block_size| {
        // Every instruction here compiles to the same number of handlers.
        assert_eq!(
            target_block_size % interpreter_calculate_cache_size(test_blob.target_block.len()),
            0
        );
        target_block_size / test_blob.target_block.len() * test_blob.path(InterpreterBranchTestSide::Fallthrough).len()
    });

    // The soft limit is the hard limit minus `max_block_size + 1`, so the cache fits just the big block
    // and compiling it after anything else resets the cache.
    let mut instance = module.instantiate().unwrap();
    instance
        .set_interpreter_cache_size_limit(target_block_size.map(|target_block_size| SetCacheSizeLimitArgs {
            max_block_size: 0,
            max_cache_size_bytes: target_block_size + interpreter_calculate_cache_size(1),
        }))
        .unwrap();

    for side in [
        InterpreterBranchTestSide::Fallthrough,
        InterpreterBranchTestSide::Target,
        InterpreterBranchTestSide::Fallthrough,
        InterpreterBranchTestSide::Target,
    ] {
        run_interpreter_branch_test(&mut instance, &test_blob, side);
        let Some(cache_size) = instance.interpreter_cache_size() else {
            continue;
        };

        match side {
            InterpreterBranchTestSide::Fallthrough => assert_eq!(
                Some(cache_size),
                fallthrough_path_size,
                "expected only the first two blocks to be compiled"
            ),
            InterpreterBranchTestSide::Target => assert_eq!(
                Some(cache_size),
                target_block_size,
                "expected the cache to be reset when the big block was compiled"
            ),
        }
    }
}

fn cache_size_limits(module: &Module, largest_block_start: ProgramCounter) -> Vec<Option<SetCacheSizeLimitArgs>> {
    const GAS_METERING_ENTRIES: usize = 1;

    let mut instance = module.instantiate().unwrap();
    instance.set_gas(0);
    instance.set_next_program_counter(largest_block_start);
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    let Some(largest_block_size) = instance.interpreter_cache_size() else {
        return vec![None];
    };

    let largest_block_entries = interpreter_calculate_cache_num_entries(largest_block_size);
    let max_block_size = u32::try_from(largest_block_entries - GAS_METERING_ENTRIES).unwrap();
    (2 * largest_block_entries..100)
        .map(|max_cache_entries| {
            Some(SetCacheSizeLimitArgs {
                max_block_size,
                max_cache_size_bytes: interpreter_calculate_cache_size(max_cache_entries),
            })
        })
        .collect()
}

fn lazy_branch_target_cache_limit(config: Config, isa: InstructionSetKind) {
    const BRANCH_COUNT: u32 = 8;

    let _ = env_logger::try_init();

    // Every branch jumps to the same final block; the fallthrough side is never taken.
    let mut code = Vec::new();
    for _ in 0..BRANCH_COUNT {
        code.push(asm::add_imm_32(A1, A1, 1));
        code.push(asm::branch_eq_imm(A0, 0, 2 * BRANCH_COUNT));
        code.push(asm::trap());
    }
    code.push(asm::ret());

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_code(&code, &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    if !config.crosscheck() {
        module_config.set_interpreter_compilation_cost(Some(ZERO_COMPILATION_COST));
    }
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();
    let (blocks, ret) = offsets.split_at(offsets.len() - 1);

    for limit in cache_size_limits(&module, blocks[0]) {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_cache_size_limit(limit).unwrap();

        let mut run = |program_counter: ProgramCounter, gas: i64| {
            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_reg(Reg::A0, 0);
            instance.set_gas(gas);
            instance.set_next_program_counter(program_counter);
            let interrupt = instance.run().unwrap();
            if let (Some(cache_size), Some(limit)) = (instance.interpreter_cache_size(), limit) {
                let max_cache_entries = interpreter_calculate_cache_num_entries(limit.max_cache_size_bytes);
                assert!(
                    cache_size <= limit.max_cache_size_bytes,
                    "expected the cache to stay within {max_cache_entries} entries"
                );
            }
            interrupt
        };

        assert_eq!(run(ret[0], 10), InterruptKind::Finished);

        // Entering a block in its middle without enough gas compiles it without entering it, so a block
        // compiled past the soft limit stays in the cache until it's entered at its start.
        for block in blocks.chunks(3) {
            assert_eq!(run(block[1], 0), InterruptKind::NotEnoughGas);
            assert_eq!(run(block[0], 0), InterruptKind::NotEnoughGas);
        }

        // Resolving a branch compiles at most its taken side.
        for block in blocks.chunks(3) {
            assert_eq!(run(block[0], 10), InterruptKind::Finished);
        }
    }
}

fn interpreter_cache_limit_test_module(config: &Config, isa: InstructionSetKind) -> Module {
    let mut code = Vec::new();
    for add_count in [2, 6, 3, 9] {
        code.extend((0..add_count).map(|_| asm::add_imm_32(A0, A0, 1)));
        code.push(asm::ret());
    }

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_code(&code, &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(config).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    if !config.crosscheck() {
        module_config.set_interpreter_compilation_cost(Some(ZERO_COMPILATION_COST));
    }
    Module::from_blob(&engine, &module_config, blob).unwrap()
}

fn jump_into_middle_of_each_basic_block(instance: &mut RawInstance, blocks: &[Vec<ProgramCounter>]) -> Vec<Option<usize>> {
    let mut cache_sizes = Vec::new();
    for block in blocks.iter().chain(blocks) {
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, 0);
        instance.set_gas(1000);
        instance.set_next_program_counter(block[1]);
        assert_eq!(instance.run().unwrap(), InterruptKind::Finished);

        let add_count_after_entry = cast(block.len() - 2).to_u64();
        let block_cost = i64::try_from(block.len()).unwrap();
        assert_eq!((instance.reg(Reg::A0), instance.gas()), (add_count_after_entry, 1000 - block_cost));
        cache_sizes.push(instance.interpreter_cache_size());
    }
    cache_sizes
}

fn cache_limit_with_middle_of_basic_block_entries(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let module = interpreter_cache_limit_test_module(&config, isa);
    let blocks = basic_blocks(module.blob());
    let largest_block = blocks.iter().max_by_key(|block| block.len()).unwrap();
    for limit in cache_size_limits(&module, largest_block[0]) {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_cache_size_limit(limit).unwrap();
        for cache_size in jump_into_middle_of_each_basic_block(&mut instance, &blocks) {
            if let (Some(cache_size), Some(limit)) = (cache_size, limit) {
                assert!(cache_size <= limit.max_cache_size_bytes);
            }
        }
    }
}

fn undersized_max_block_size_with_middle_of_basic_block_entries(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let module = interpreter_cache_limit_test_module(&config, isa);
    let mut instance = module.instantiate().unwrap();
    instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 0,
            max_cache_size_bytes: interpreter_calculate_cache_size(2),
        }))
        .unwrap();

    jump_into_middle_of_each_basic_block(&mut instance, &basic_blocks(module.blob()));
}

fn lazy_branch_target_steps_and_charges_gas(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_step_tracing(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let test_blob = interpreter_branch_test_blob(isa);
    let module = Module::from_blob(&engine, &module_config, test_blob.blob.clone()).unwrap();

    let mut instance = module.instantiate().unwrap();
    let mut run = |side: InterpreterBranchTestSide, gas: i64| {
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, side.a0());
        instance.set_reg(Reg::A2, 0);
        instance.set_gas(gas);
        instance.set_next_program_counter(test_blob.branch);

        let mut steps = Vec::new();
        loop {
            match instance.run().unwrap() {
                InterruptKind::Step => steps.push(instance.program_counter().unwrap()),
                interrupt => return (steps, interrupt, instance.program_counter(), instance.gas(), instance.reg(Reg::A2)),
            }
        }
    };

    // Every instruction costs one unit of gas.
    let fallthrough_path = test_blob.path(InterpreterBranchTestSide::Fallthrough);
    let fallthrough_path_cost = i64::try_from(fallthrough_path.len()).unwrap();
    let branch_cost = 1;
    let target_block_start = test_blob.target_block[0];
    let target_block_cost = i64::try_from(test_blob.target_block.len()).unwrap();

    assert_eq!(
        run(InterpreterBranchTestSide::Fallthrough, 100),
        (fallthrough_path, InterruptKind::Finished, None, 100 - fallthrough_path_cost, 0)
    );
    assert_eq!(
        run(InterpreterBranchTestSide::Target, target_block_cost),
        (
            vec![test_blob.branch, target_block_start],
            InterruptKind::NotEnoughGas,
            Some(target_block_start),
            target_block_cost - branch_cost,
            0
        )
    );

    for _ in 0..2 {
        assert_eq!(
            run(InterpreterBranchTestSide::Target, 100),
            (
                test_blob.path(InterpreterBranchTestSide::Target),
                InterruptKind::Finished,
                None,
                100 - branch_cost - target_block_cost,
                InterpreterBranchTestBlob::TARGET_BLOCK_ADD_COUNT
            )
        );
    }
}

/// Charges 10 per block plus 1 for every two instructions, rounded up.
const TEST_COMPILATION_COST: InterpreterCompilationCost = InterpreterCompilationCost {
    per_block: 10,
    per_instruction_millis: 500,
    per_code_byte_millis: 0,
};

/// Lets a test limit the cache size with gas metering without charging for compilation.
const ZERO_COMPILATION_COST: InterpreterCompilationCost = InterpreterCompilationCost {
    per_block: 0,
    per_instruction_millis: 0,
    per_code_byte_millis: 0,
};

fn two_blocks_test_blob(isa: InstructionSetKind) -> ProgramBlob {
    padded_two_blocks_test_blob(isa, 0)
}

fn chain_of_blocks_test_blob(isa: InstructionSetKind) -> ProgramBlob {
    let mut code = Vec::new();
    for block in 0..4 {
        code.extend([asm::add_imm_32(A0, A0, 1), asm::add_imm_32(A0, A0, 1), asm::add_imm_32(A0, A0, 1)]);
        code.push(if block < 3 { asm::jump(block + 1) } else { asm::ret() });
    }

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
}

/// Runs to completion with 1000 gas, past steps and host calls, and returns the gas used.
fn run_and_get_gas_used(instance: &mut RawInstance, program_counter: ProgramCounter, a0: u64) -> i64 {
    instance.set_gas(1000);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, a0);
    instance.set_next_program_counter(program_counter);
    loop {
        match instance.run().unwrap() {
            InterruptKind::Step | InterruptKind::Ecalli(..) => {}
            interrupt => {
                match_interrupt!(interrupt, InterruptKind::Finished);
                return 1000 - instance.gas();
            }
        }
    }
}

fn instantiate_with_compilation_cost(
    engine: &Engine,
    gas_metering: GasMeteringKind,
    blob: ProgramBlob,
    cost: Option<InterpreterCompilationCost>,
) -> RawInstance {
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(gas_metering));
    module_config.set_interpreter_compilation_cost(cost);
    let module = Module::from_blob(engine, &module_config, blob).unwrap();
    module.instantiate().unwrap()
}

fn compilation_is_charged_when_compiling(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let run = |instance: &mut RawInstance, side: InterpreterBranchTestSide| run_and_get_gas_used(instance, test_blob.branch, side.a0());
    for gas_metering in [GasMeteringKind::Sync, GasMeteringKind::Async] {
        let mut instance = instantiate_with_compilation_cost(&engine, gas_metering, test_blob.blob.clone(), Some(TEST_COMPILATION_COST));

        // The branch costs 1 and the block after it 2; compiling either costs 11.
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Fallthrough), 1 + 11 + 2 + 11);
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Fallthrough), 1 + 2);

        // The branch target is compiled on its first take; it costs 33 to run and 27 to compile.
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Target), 1 + 33 + 27);
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Target), 1 + 33);
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Fallthrough), 1 + 2);

        instance.reset_interpreter_cache().unwrap();
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Fallthrough), 1 + 11 + 2 + 11);
        assert_eq!(run(&mut instance, InterpreterBranchTestSide::Fallthrough), 1 + 2);

        let mut instance = instantiate_with_compilation_cost(&engine, gas_metering, test_blob.blob.clone(), None);
        assert!(instance.reset_interpreter_cache().is_err());
    }

    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_per_instruction_metering(true);
    assert!(Module::from_blob(&engine, &module_config, test_blob.blob.clone()).is_ok());
    module_config.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));
    assert!(Module::from_blob(&engine, &module_config, test_blob.blob.clone()).is_err());
}

fn compilation_cost_does_nothing_on_other_backends(config: Config, isa: InstructionSetKind) {
    if config.backend() == Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    if config.crosscheck() {
        let mut module_config = ModuleConfig::new();
        module_config.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));
        assert!(Module::from_blob(&engine, &module_config, test_blob.blob).is_err());
        return;
    }

    let gas_used = |cost| {
        let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, test_blob.blob.clone(), cost);
        assert!(instance.reset_interpreter_cache().is_ok());
        assert!(instance
            .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                max_block_size: 0,
                max_cache_size_bytes: interpreter_calculate_cache_size(4),
            }))
            .is_ok());

        let gas = [
            InterpreterBranchTestSide::Fallthrough,
            InterpreterBranchTestSide::Fallthrough,
            InterpreterBranchTestSide::Target,
            InterpreterBranchTestSide::Target,
        ]
        .map(|side| run_and_get_gas_used(&mut instance, test_blob.branch, side.a0()));
        assert!(instance.reset_interpreter_cache().is_ok());
        gas
    };

    assert_eq!(gas_used(Some(TEST_COMPILATION_COST)), gas_used(None));
}

fn compilation_cost_errors(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let blob = interpreter_branch_test_blob(isa).blob;
    let limit = Some(SetCacheSizeLimitArgs {
        max_block_size: 0,
        max_cache_size_bytes: 1000,
    });

    // The branch and the block after it cost 1 + 11 and 2 + 11.
    let mut crosscheck_config = config;
    crosscheck_config.set_allow_experimental(true);
    crosscheck_config.set_crosscheck(true);
    let crosscheck_engine = Engine::new(&crosscheck_config).unwrap();
    let mut instance =
        instantiate_with_compilation_cost(&crosscheck_engine, GasMeteringKind::Sync, blob.clone(), Some(TEST_COMPILATION_COST));
    assert_eq!(
        run_and_get_gas_used(&mut instance, ProgramCounter(0), InterpreterBranchTestSide::Fallthrough.a0()),
        1 + 11 + 2 + 11
    );

    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_per_instruction_metering(true);
    let mut instance = Module::from_blob(&engine, &module_config, blob.clone())
        .unwrap()
        .instantiate()
        .unwrap();
    assert!(instance.set_interpreter_cache_size_limit(limit).is_ok());
    assert!(instance.reset_interpreter_cache().is_ok());

    let mut instance = Module::from_blob(&engine, &ModuleConfig::new(), blob.clone())
        .unwrap()
        .instantiate()
        .unwrap();
    assert!(instance.set_interpreter_cache_size_limit(limit).is_ok());
    assert!(instance.reset_interpreter_cache().is_ok());

    let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, blob.clone(), None);
    assert!(instance.set_interpreter_cache_size_limit(limit).is_err());
    assert!(instance.set_interpreter_cache_size_limit(None).is_ok());

    let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, blob, Some(TEST_COMPILATION_COST));
    assert!(instance.set_interpreter_cache_size_limit(limit).is_ok());
    assert!(instance.reset_interpreter_cache().is_ok());
}

fn large_compilation_cost_is_charged_in_full(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let blob = two_blocks_test_blob(isa);
    let code_length = i64::try_from(blob.code().len()).unwrap();
    let max = i64::from(u32::MAX);
    let costs = [
        (
            InterpreterCompilationCost {
                per_block: u32::MAX,
                ..ZERO_COMPILATION_COST
            },
            max,
        ),
        (
            InterpreterCompilationCost {
                per_block: u32::MAX,
                per_instruction_millis: u32::MAX,
                per_code_byte_millis: u32::MAX,
            },
            max + (4 * max + 999) / 1000 + (code_length * max + 999) / 1000,
        ),
    ];

    for (cost, expected_compilation_cost) in costs {
        let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, blob.clone(), Some(cost));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_gas(100);
        instance.set_next_program_counter(ProgramCounter(0));
        assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
        assert_eq!(instance.gas(), 100 - expected_compilation_cost);
    }
}

fn code_length_dependent_state_is_charged(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let cost = InterpreterCompilationCost {
        per_code_byte_millis: 1000,
        ..ZERO_COMPILATION_COST
    };

    let blob = two_blocks_test_blob(isa);
    let code_length = i64::try_from(blob.code().len()).unwrap();

    // The blocks cost 4 and 2, plus one per byte of code for the first compilation.
    let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, blob, Some(cost));
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(4 + code_length - 1);
    instance.set_next_program_counter(ProgramCounter(0));
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.gas(), 4 - 1);

    assert_eq!(run_and_get_gas_used(&mut instance, ProgramCounter(0), 0), 6);

    instance.reset_interpreter_cache().unwrap();
    assert_eq!(run_and_get_gas_used(&mut instance, ProgramCounter(0), 0), 6 + code_length);

    // The blocks take 5 and 3 entries and the limit fits one, so running resets the cache.
    instance.reset_interpreter_cache().unwrap();
    instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 4,
            max_cache_size_bytes: interpreter_calculate_cache_size(10),
        }))
        .unwrap();

    assert_eq!(run_and_get_gas_used(&mut instance, ProgramCounter(0), 0), 6 + code_length);
    assert_eq!(instance.interpreter_cache_size(), Some(interpreter_calculate_cache_size(3)));
    assert_eq!(run_and_get_gas_used(&mut instance, ProgramCounter(0), 0), 6);
}

fn out_of_gas_after_a_compilation(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let fallthrough = InterpreterBranchTestSide::Fallthrough.a0();
    for gas_metering in [GasMeteringKind::Sync, GasMeteringKind::Async] {
        let mut instance = instantiate_with_compilation_cost(&engine, gas_metering, test_blob.blob.clone(), Some(TEST_COMPILATION_COST));

        // Enough to pay 1 + 11 for the branch, but not 11 to compile the block after it, which costs 2 to run.
        let branch_cost = 1 + 11;
        let gas_left = 5;
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, fallthrough);
        instance.set_reg(Reg::A1, 0);
        instance.set_gas(branch_cost + gas_left);
        instance.set_next_program_counter(test_blob.branch);
        assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
        assert_eq!(instance.reg(Reg::A1), 0);

        assert_eq!(
            instance.interpreter_cache_size(),
            Some(interpreter_calculate_cache_size(3 + 5)),
            "expected the block after the branch to be compiled"
        );

        if gas_metering == GasMeteringKind::Async {
            assert_eq!(instance.gas(), gas_left - 11 - 2);
            assert_eq!(run_and_get_gas_used(&mut instance, test_blob.branch, fallthrough), 1 + 2);
            continue;
        }

        assert_eq!(instance.gas(), gas_left - 11);
        assert_eq!(instance.program_counter(), Some(test_blob.fallthrough_block[0]));

        instance.set_gas(100);
        assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.reg(Reg::A1), 1);
        assert_eq!(instance.gas(), 100 - 2, "expected the compilation to be charged once");

        assert_eq!(run_and_get_gas_used(&mut instance, test_blob.branch, fallthrough), 1 + 2);
    }
}

fn compilation_thrown_away_by_a_cache_reset_is_charged(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let target_block_add_count = InterpreterBranchTestBlob::TARGET_BLOCK_ADD_COUNT;

    // The limit fits only the 34 entry big block, so compiling it after the branch throws it away
    // and compiles it again. The big block costs 33 to run and 27 to compile.
    let run_out_of_gas_entering_big_block = |gas_metering, gas_left_after_branch| {
        let mut instance = instantiate_with_compilation_cost(&engine, gas_metering, test_blob.blob.clone(), Some(TEST_COMPILATION_COST));
        instance
            .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                max_block_size: 0,
                max_cache_size_bytes: interpreter_calculate_cache_size(35),
            }))
            .unwrap();

        assert_eq!(
            run_and_get_gas_used(&mut instance, test_blob.branch, InterpreterBranchTestSide::Fallthrough.a0()),
            1 + 11 + 2 + 11
        );

        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, InterpreterBranchTestSide::Target.a0());
        instance.set_reg(Reg::A2, 0);
        instance.set_gas(1 + gas_left_after_branch);
        instance.set_next_program_counter(test_blob.branch);
        assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
        assert_eq!(instance.reg(Reg::A2), 0);
        instance
    };

    let gas_left = 27 + 27 + 33 - 1;
    let mut instance = run_out_of_gas_entering_big_block(GasMeteringKind::Sync, gas_left);
    assert_eq!(instance.gas(), gas_left - 27 - 27);
    assert_eq!(instance.program_counter(), Some(test_blob.target_block[0]));
    assert_eq!(
        instance.interpreter_cache_size(),
        Some(interpreter_calculate_cache_size(34)),
        "expected only the big block to be in the cache"
    );

    instance.set_gas(100);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A2), target_block_add_count);
    assert_eq!(instance.gas(), 100 - 33, "expected both compilations to be charged once");

    let instance = run_out_of_gas_entering_big_block(GasMeteringKind::Async, gas_left);
    assert_eq!(instance.gas(), gas_left - (27 + 27 + 33));

    let mut instance = run_out_of_gas_entering_big_block(GasMeteringKind::Sync, gas_left);
    assert_eq!(run_and_get_gas_used(&mut instance, test_blob.target_block[1], 0), 33);
    assert_eq!(instance.reg(Reg::A2), target_block_add_count - 1);
    assert_eq!(
        instance.interpreter_cache_size(),
        Some(interpreter_calculate_cache_size(34)),
        "expected only the big block to be in the cache"
    );

    // The branch doesn't fit next to the big block either, so it's also compiled twice.
    instance.set_reg(Reg::A0, InterpreterBranchTestSide::Fallthrough.a0());
    instance.set_gas(1 + 11);
    instance.set_next_program_counter(test_blob.branch);
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.gas(), 1 + 11 - 2 * 11);

    instance.set_gas(100);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.gas(), 100 - 1 - (2 + 11));
}

fn compilation_thrown_away_on_entry_into_the_middle_of_a_block_is_charged(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let entry = test_blob.target_block[1];
    let mut instance =
        instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, test_blob.blob.clone(), Some(TEST_COMPILATION_COST));
    instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 0,
            max_cache_size_bytes: interpreter_calculate_cache_size(35),
        }))
        .unwrap();

    assert_eq!(
        run_and_get_gas_used(&mut instance, test_blob.branch, InterpreterBranchTestSide::Fallthrough.a0()),
        1 + 11 + 2 + 11
    );

    // The big block doesn't fit next to the other two, so it's compiled twice before it's
    // entered. It costs 33 to run and 27 to compile.
    let gas = 33 + 27 + 27 - 1;
    instance.set_gas(gas);
    instance.set_next_program_counter(entry);
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.gas(), gas - 27 - 27);
    assert_eq!(
        instance.interpreter_cache_size(),
        Some(interpreter_calculate_cache_size(34)),
        "expected only the big block to be in the cache"
    );

    assert_eq!(
        run_and_get_gas_used(&mut instance, entry, 0),
        33,
        "expected both compilations to be charged once"
    );
}

fn compilation_charge_on_entry_into_the_middle_of_a_block(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[asm::add_imm_32(A0, A0, 1), asm::ecalli(0), asm::add_imm_32(A1, A1, 1), asm::ret()],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();
    let run_until_interrupt = |instance: &mut RawInstance| loop {
        match instance.run().unwrap() {
            InterruptKind::Step => {}
            interrupt => return interrupt,
        }
    };

    for step_tracing in [false, true] {
        let mut module_config = ModuleConfig::new();
        module_config.set_gas_metering(Some(GasMeteringKind::Sync));
        module_config.set_step_tracing(step_tracing);
        module_config.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));
        let mut instance = Module::from_blob(&engine, &module_config, blob.clone())
            .unwrap()
            .instantiate()
            .unwrap();

        // The block costs 4 and compiling it 12.
        assert_eq!(run_and_get_gas_used(&mut instance, offsets[2], 0), 4 + 12);
        assert_eq!(run_and_get_gas_used(&mut instance, offsets[2], 0), 4);

        instance.set_reg(Reg::A1, 0);
        instance.set_gas(1000);
        instance.set_next_program_counter(offsets[0]);
        assert_eq!(run_until_interrupt(&mut instance), InterruptKind::Ecalli(0));
        assert_eq!(instance.gas(), 1000 - 4);

        instance.reset_interpreter_cache().unwrap();
        assert_eq!(run_until_interrupt(&mut instance), InterruptKind::Finished);
        assert_eq!(
            instance.gas(),
            1000 - 4 - 12,
            "expected only the compilation to be charged with step_tracing={step_tracing}"
        );
        assert_eq!(instance.reg(Reg::A1), 1);

        instance.set_reg(Reg::A1, 0);
        instance.set_gas(1000);
        instance.set_next_program_counter(offsets[0]);
        assert_eq!(run_until_interrupt(&mut instance), InterruptKind::Ecalli(0));

        instance.reset_interpreter_cache().unwrap();
        instance.set_gas(11);
        assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
        assert_eq!(instance.gas(), 11 - 12);
        assert_eq!(instance.reg(Reg::A1), 0);

        instance.set_gas(100);
        assert_eq!(run_until_interrupt(&mut instance), InterruptKind::Finished);
        assert_eq!(instance.gas(), 100, "expected the compilation to be charged once");
        assert_eq!(instance.reg(Reg::A1), 1);

        assert_eq!(run_and_get_gas_used(&mut instance, offsets[0], 0), 4);
    }
}

fn cache_reset_while_suspended_resumes_at_the_next_program_counter(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let test_blob = interpreter_branch_test_blob(isa);
    let fallthrough = InterpreterBranchTestSide::Fallthrough.a0();
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));

    // The branch costs 1 and the block after it 2; compiling either costs 11.
    let mut instance = Module::from_blob(&engine, &module_config, test_blob.blob.clone())
        .unwrap()
        .instantiate()
        .unwrap();
    assert_eq!(run_and_get_gas_used(&mut instance, test_blob.branch, fallthrough), 1 + 11 + 2 + 11);

    instance.set_reg(Reg::A1, 0);
    instance.set_gas(1 + 1);
    instance.set_next_program_counter(test_blob.branch);
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.gas(), 1);

    instance.reset_interpreter_cache().unwrap();
    instance.set_gas(100);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 1);
    assert_eq!(instance.gas(), 100 - 2 - 11);

    module_config.set_step_tracing(true);
    let mut instance = Module::from_blob(&engine, &module_config, test_blob.blob.clone())
        .unwrap()
        .instantiate()
        .unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, fallthrough);
    instance.set_gas(1000);
    instance.set_next_program_counter(test_blob.branch);

    let reset_at = test_blob.fallthrough_block[1];
    let mut steps = Vec::new();
    loop {
        match instance.run().unwrap() {
            InterruptKind::Step => {
                let program_counter = instance.program_counter().unwrap();
                if program_counter == reset_at && !steps.contains(&program_counter) {
                    instance.reset_interpreter_cache().unwrap();
                }

                steps.push(program_counter);
            }
            interrupt => {
                assert_eq!(interrupt, InterruptKind::Finished);
                break;
            }
        }
    }

    let mut expected_steps = test_blob.path(InterpreterBranchTestSide::Fallthrough);
    expected_steps.push(reset_at);
    assert_eq!(steps, expected_steps);
    assert_eq!(instance.reg(Reg::A1), 1);
    assert_eq!(instance.gas(), 1000 - (1 + 11 + 2 + 11) - 11);
}

fn compilation_thrown_away_by_a_host_cache_reset_is_charged(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut instance = instantiate_with_compilation_cost(
        &engine,
        GasMeteringKind::Sync,
        two_blocks_test_blob(isa),
        Some(TEST_COMPILATION_COST),
    );

    // The first block costs 4 to run and 12 to compile, the second 2 and 11.
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(12 + 3);
    instance.set_next_program_counter(ProgramCounter(0));
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.gas(), 3);

    instance.reset_interpreter_cache().unwrap();
    instance.set_gas(100);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.gas(), 100 - (12 + 4) - (11 + 2));
}

fn step_interrupt_right_after_a_compilation(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_step_tracing(true);
    module_config.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));
    let module = Module::from_blob(&engine, &module_config, two_blocks_test_blob(isa)).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();
    let mut instance = module.instantiate().unwrap();

    // The first block costs 4 to run and 12 to compile, the second 2 and 11.
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(12 + 4 + 5);
    instance.set_next_program_counter(offsets[0]);
    let mut interrupts = Vec::new();
    loop {
        let interrupt = instance.run().unwrap();
        interrupts.push((interrupt.clone(), instance.program_counter(), instance.gas()));
        match interrupt {
            InterruptKind::Step => {}
            InterruptKind::NotEnoughGas => instance.set_gas(10),
            _ => break,
        }
    }

    assert_eq!(
        interrupts,
        [
            (InterruptKind::Step, Some(offsets[0]), 5 + 4),
            (InterruptKind::Step, Some(offsets[1]), 5),
            (InterruptKind::Step, Some(offsets[2]), 5),
            (InterruptKind::Step, Some(offsets[3]), 5),
            (InterruptKind::NotEnoughGas, Some(offsets[4]), 5 - 11),
            (InterruptKind::Step, Some(offsets[5]), 10 - 2),
            (InterruptKind::Finished, None, 10 - 2),
        ]
    );
}

fn arbitrary_jump_inside_of_an_instruction_does_not_compile_again(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let blob = two_blocks_test_blob(isa);
    let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();
    let mut instance = instantiate_with_compilation_cost(&engine, GasMeteringKind::Sync, blob, Some(TEST_COMPILATION_COST));
    assert_eq!(run_and_get_gas_used(&mut instance, offsets[0], 0), (12 + 4) + (11 + 2));
    let cache_size = instance.interpreter_cache_size();
    let gas = instance.gas();

    instance.set_next_program_counter(ProgramCounter(offsets[1].0 + 1));
    assert_eq!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(
        instance.interpreter_cache_size(),
        cache_size,
        "expected the already compiled block to not be compiled again"
    );
    assert_eq!(instance.gas(), gas);
}

fn cache_size_limit_disables_debug_handlers(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), chain_of_blocks_test_blob(isa)).unwrap();
    let limit = SetCacheSizeLimitArgs {
        max_block_size: 5,
        max_cache_size_bytes: interpreter_calculate_cache_size(27),
    };

    let mut instance = module.instantiate().unwrap();
    let run = |instance: &mut RawInstance| {
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, 0);
        instance.set_next_program_counter(ProgramCounter(0));
        assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.reg(Reg::A0), 12);
        interpreter_calculate_cache_num_entries(instance.interpreter_cache_size().unwrap())
    };

    // Unit tests always run the interpreter in debug mode, which adds a `trace_pc` handler to each of the 16 instructions.
    assert_eq!(run(&mut instance), 32);

    instance.set_interpreter_cache_size_limit(Some(limit)).unwrap();
    assert_eq!(instance.interpreter_cache_size(), Some(0));
    assert_eq!(run(&mut instance), 16);

    instance.set_interpreter_cache_size_limit(None).unwrap();
    assert_eq!(instance.interpreter_cache_size(), Some(0));
    assert_eq!(run(&mut instance), 32);

    let mut module_config = ModuleConfig::new();
    module_config.set_step_tracing(true);
    let module = Module::from_blob(&engine, &module_config, chain_of_blocks_test_blob(isa)).unwrap();
    let mut instance = module.instantiate().unwrap();
    assert!(instance.set_interpreter_cache_size_limit(Some(limit)).is_err());
}

fn lowering_the_cache_size_limit_clears_the_cache(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), chain_of_blocks_test_blob(isa)).unwrap();
    let mut instance = module.instantiate().unwrap();
    let mut run_with_limit = |max_cache_entries: usize| {
        instance
            .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
                max_block_size: 5,
                max_cache_size_bytes: interpreter_calculate_cache_size(max_cache_entries),
            }))
            .unwrap();

        let cache_size_before_running = instance.interpreter_cache_size();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, 0);
        instance.set_next_program_counter(ProgramCounter(0));
        assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.reg(Reg::A0), 12);
        (cache_size_before_running, instance.interpreter_cache_size())
    };

    // Without gas metering every block takes four entries; this limit fits all four.
    let (_, Some(cache_size)) = run_with_limit(27) else {
        return;
    };
    assert_eq!(cache_size, interpreter_calculate_cache_size(16));

    // This limit fits two blocks, so running resets the cache.
    let (cache_size_before_running, cache_size) = run_with_limit(15);
    assert_eq!(cache_size_before_running, Some(0));
    assert_eq!(cache_size, Some(interpreter_calculate_cache_size(8)));
}

const PER_CODE_BYTE_COMPILATION_COST: InterpreterCompilationCost = InterpreterCompilationCost {
    per_code_byte_millis: 1,
    ..TEST_COMPILATION_COST
};

// With 256 KiB of padding the code map is rewritten after every 1023 cache clears.
const PADDING: usize = 1 << 18;
const CLEARS_PER_REWRITE: usize = 1023;

fn padded_two_blocks_test_blob(isa: InstructionSetKind, padding: usize) -> ProgramBlob {
    let mut code = vec![
        asm::add_imm_32(A0, A0, 1),
        asm::add_imm_32(A0, A0, 1),
        asm::add_imm_32(A0, A0, 1),
        asm::jump(1),
        asm::add_imm_32(A1, A1, 1),
        asm::ret(),
    ];
    code.resize(code.len() + padding, asm::trap());

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
}

fn instantiate_padded_two_blocks(engine: &Engine, isa: InstructionSetKind, padding: usize) -> (RawInstance, i64) {
    let blob = padded_two_blocks_test_blob(isa, padding);
    let code_map_cost = i64::try_from(blob.code().len().div_ceil(1000)).unwrap();
    let instance = instantiate_with_compilation_cost(engine, GasMeteringKind::Sync, blob, Some(PER_CODE_BYTE_COMPILATION_COST));
    (instance, code_map_cost)
}

// The first block costs 4 to run and 12 to compile, the second 2 and 11.
const RUN_COST: i64 = 4 + 2;
const COMPILATION_COST: i64 = 12 + 11;

fn run_padded_two_blocks(instance: &mut RawInstance) -> i64 {
    let gas_used = run_and_get_gas_used(instance, ProgramCounter(0), 0);
    assert_eq!(instance.reg(Reg::A0), 3);
    gas_used
}

fn run_padded_two_blocks_from_scratch(instance: &mut RawInstance, code_map_cost: i64) {
    assert_eq!(run_padded_two_blocks(instance), code_map_cost + COMPILATION_COST + RUN_COST);
}

fn cache_resets_past_the_epoch_range(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let (mut instance, code_map_cost) = instantiate_padded_two_blocks(&engine, isa, PADDING);
    instance
        .set_interpreter_cache_size_limit(Some(SetCacheSizeLimitArgs {
            max_block_size: 4,
            max_cache_size_bytes: interpreter_calculate_cache_size(10),
        }))
        .unwrap();

    // The limit fits one block: the first run resets the cache once and compiles the second block
    // twice, every later run resets it twice and compiles both blocks twice.
    assert_eq!(
        run_padded_two_blocks(&mut instance),
        code_map_cost + COMPILATION_COST + 11 + RUN_COST
    );

    let runs = 2..=CLEARS_PER_REWRITE + 2;
    let code_map_costs: Vec<i64> = runs
        .clone()
        .map(|_| run_padded_two_blocks(&mut instance) - (2 * COMPILATION_COST + RUN_COST))
        .collect();

    let expected: Vec<i64> = runs
        .map(|run| {
            if (2 * run - 2) % CLEARS_PER_REWRITE == 0 || (2 * run - 1) % CLEARS_PER_REWRITE == 0 {
                code_map_cost
            } else {
                0
            }
        })
        .collect();
    assert_eq!(expected.iter().filter(|&&cost| cost != 0).count(), 2);
    assert_eq!(code_map_costs, expected);
}

fn clear_cache_by_switching_the_limit(instance: &mut RawInstance, limited: &mut bool, clears: usize) -> Vec<i64> {
    let limit = SetCacheSizeLimitArgs {
        max_block_size: 4,
        max_cache_size_bytes: interpreter_calculate_cache_size(100),
    };

    (0..clears)
        .map(|_| {
            *limited = !*limited;
            let gas = instance.gas();
            instance.set_interpreter_cache_size_limit(limited.then_some(limit)).unwrap();
            gas - instance.gas()
        })
        .collect()
}

fn charged_on_the_last_clear(clears: usize, code_map_cost: i64) -> Vec<i64> {
    let mut charges = vec![0; clears];
    charges[clears - 1] = code_map_cost;
    charges
}

fn cache_clears_past_the_epoch_range_overwrite_the_code_map(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    for (padding, clears_per_rewrite) in [(PADDING, CLEARS_PER_REWRITE), (2 * PADDING, 511)] {
        let (mut instance, code_map_cost) = instantiate_padded_two_blocks(&engine, isa, padding);
        let mut limited = false;

        run_padded_two_blocks_from_scratch(&mut instance, code_map_cost);
        assert_eq!(run_padded_two_blocks(&mut instance), RUN_COST);

        assert_eq!(
            clear_cache_by_switching_the_limit(&mut instance, &mut limited, clears_per_rewrite),
            charged_on_the_last_clear(clears_per_rewrite, code_map_cost)
        );
        assert_eq!(run_padded_two_blocks(&mut instance), COMPILATION_COST + RUN_COST);
        assert_eq!(run_padded_two_blocks(&mut instance), RUN_COST);
    }
}

fn reset_interpreter_cache_restarts_the_epoch_count(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let (mut instance, code_map_cost) = instantiate_padded_two_blocks(&engine, isa, PADDING);
    let mut limited = false;

    run_padded_two_blocks_from_scratch(&mut instance, code_map_cost);
    assert_eq!(clear_cache_by_switching_the_limit(&mut instance, &mut limited, 4), [0; 4]);

    instance.reset_interpreter_cache().unwrap();
    run_padded_two_blocks_from_scratch(&mut instance, code_map_cost);
    assert_eq!(
        clear_cache_by_switching_the_limit(&mut instance, &mut limited, CLEARS_PER_REWRITE),
        charged_on_the_last_clear(CLEARS_PER_REWRITE, code_map_cost)
    );
}

fn epoch_wrap_without_block_gas_metering_charges_nothing(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let blob = padded_two_blocks_test_blob(isa, PADDING);
    for (gas_metering, is_per_instruction_metering, cost) in [
        (None, false, Some(PER_CODE_BYTE_COMPILATION_COST)),
        (Some(GasMeteringKind::Sync), true, None),
    ] {
        let mut module_config = ModuleConfig::new();
        module_config.set_gas_metering(gas_metering);
        module_config.set_per_instruction_metering(is_per_instruction_metering);
        module_config.set_interpreter_compilation_cost(cost);
        let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();
        let mut instance = module.instantiate().unwrap();
        let mut limited = false;

        run_padded_two_blocks(&mut instance);
        assert_eq!(
            clear_cache_by_switching_the_limit(&mut instance, &mut limited, CLEARS_PER_REWRITE),
            [0; CLEARS_PER_REWRITE]
        );
        run_padded_two_blocks(&mut instance);
    }
}

fn host_cache_clear_leaving_negative_gas_stops_the_next_run(config: Config, isa: InstructionSetKind) {
    if config.backend() != Some(crate::BackendKind::Interpreter) {
        return;
    }

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let (mut instance, code_map_cost) = instantiate_padded_two_blocks(&engine, isa, PADDING);
    let mut limited = false;

    run_padded_two_blocks_from_scratch(&mut instance, code_map_cost);
    assert_eq!(
        clear_cache_by_switching_the_limit(&mut instance, &mut limited, CLEARS_PER_REWRITE - 1),
        [0; CLEARS_PER_REWRITE - 1]
    );

    instance.set_gas(code_map_cost - 1);
    assert_eq!(clear_cache_by_switching_the_limit(&mut instance, &mut limited, 1), [code_map_cost]);
    assert_eq!(instance.gas(), -1);

    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(ProgramCounter(0));
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.reg(Reg::A0), 0);
    assert_eq!(instance.gas(), -1);

    instance.set_gas(100);
    assert_eq!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 3);
    assert_eq!(instance.gas(), 100 - COMPILATION_COST - RUN_COST);
}

fn step_tracing_invalid_store(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut config = ModuleConfig::new();
    config.set_step_tracing(true);

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::fallthrough(), asm::store_imm_u32(0, 0x12345678), asm::ret()], &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    instance.set_next_program_counter(ProgramCounter(1));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter(), Some(ProgramCounter(1)));
    assert_eq!(instance.next_program_counter(), None);
}

fn step_tracing_invalid_load(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut config = ModuleConfig::new();
    config.set_step_tracing(true);

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::fallthrough(), asm::load_i32(Reg::A0, 0), asm::ret()], &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    instance.set_next_program_counter(ProgramCounter(1));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter(), Some(ProgramCounter(1)));
    assert_eq!(instance.next_program_counter(), None);
}

fn step_tracing_out_of_gas(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut config = ModuleConfig::new();
    config.set_step_tracing(true);
    config.set_gas_metering(Some(GasMeteringKind::Sync));

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::fallthrough(),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::fallthrough(),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::fallthrough(),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::move_reg(Reg::A0, Reg::A0),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();
    let mut instance = module.instantiate().unwrap();

    instance.set_gas(4);
    instance.set_next_program_counter(offsets[1]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.gas(), 4);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));
    if engine_config.backend() == Some(BackendKind::Compiler) {
        assert!(instance.next_native_program_counter().is_some());
    }

    // Setting the program counter again resets stepping.
    instance.set_next_program_counter(offsets[1]); // move_reg, fallthrough
    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.gas(), 4);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));
    if engine_config.backend() == Some(BackendKind::Compiler) {
        assert!(instance.next_native_program_counter().is_some());
    }

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.gas(), 2);
    assert_eq!(instance.program_counter(), Some(offsets[2])); // fallthrough
    assert_eq!(instance.next_program_counter(), Some(offsets[2]));
    if engine_config.backend() == Some(BackendKind::Compiler) {
        assert!(instance.next_native_program_counter().is_some());
    }

    match_interrupt!(instance.run().unwrap(), InterruptKind::Step);
    assert_eq!(instance.gas(), 2);
    assert_eq!(instance.program_counter(), Some(offsets[3])); // move_reg, move_reg, fallthrough
    assert_eq!(instance.next_program_counter(), Some(offsets[3]));
    if engine_config.backend() == Some(BackendKind::Compiler) {
        assert!(instance.next_native_program_counter().is_some());
    }

    for _ in 0..2 {
        match_interrupt!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
        assert_eq!(instance.gas(), 2);
        assert_eq!(instance.program_counter(), Some(offsets[3]));
        assert_eq!(instance.next_program_counter(), Some(offsets[3]));
        if engine_config.backend() == Some(BackendKind::Compiler) {
            assert!(instance.next_native_program_counter().is_some());
        }
    }

    instance.set_next_program_counter(ProgramCounter(cast(module.blob().code().len()).to_u32_or_panic()));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.gas(), 2);

    instance.set_next_program_counter(ProgramCounter(10000));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.gas(), 2);
}

fn zero_memory(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();

    let memory_map = MemoryMapBuilder::new(0x4000).rw_data_size(0x4000).build().unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_rw_data_size(0x4000);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::store_imm_u32(memory_map.rw_data_address().try_into().unwrap(), 0x12345678),
            asm::ecalli(0),
            asm::load_i32(A0, memory_map.rw_data_address().try_into().unwrap()),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    assert_out_of_range_access(
        instance.zero_memory(memory_map.ro_data_address(), 1),
        memory_map.ro_data_address(),
        1,
    );
    instance.set_next_program_counter(offsets[0]);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Ecalli(..));
    assert_eq!(instance.read_u32(memory_map.rw_data_address()).unwrap(), 0x12345678);
    instance.zero_memory(memory_map.rw_data_address(), 2).unwrap();
    let value = instance.read_u32(memory_map.rw_data_address()).unwrap();
    assert_eq!(value, 0x12340000, "unexpected value: 0x{value:x}");
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(A0), 0x12340000);
}

#[track_caller]
fn expect_segfault(interrupt: InterruptKind) -> Segfault {
    match interrupt {
        InterruptKind::Segfault(segfault) => segfault,
        interrupt => unreachable!("expected segfault, got: {interrupt:?}"),
    }
}

fn dynamic_jump_to_null(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let programs = [
        vec![asm::move_reg(Reg::A0, Reg::A0), asm::ret()],
        vec![asm::move_reg(Reg::A0, Reg::A0), asm::ret(), asm::move_reg(Reg::A0, Reg::A0)],
    ];

    for code in programs {
        log::info!("Testing program...");
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&code, &[]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
        let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

        let mut instance = module.instantiate().unwrap();
        instance.set_next_program_counter(offsets[0]);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.program_counter(), Some(offsets[1]));
        assert_eq!(instance.next_program_counter(), None);
    }
}

fn simple_test(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(A0, 0x1234), asm::add_imm_32(A1, A1, 100), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A1, 0);
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x1234);
    assert_eq!(instance.reg(Reg::A1), 100);
}

fn out_of_range_execution(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(A0, 1), asm::load_imm(A0, 2), asm::branch_eq_imm(RA, 0, 0)], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter(), Some(offsets[2]));
}

fn jump_into_middle_of_basic_block_from_outside(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::add_imm_32(A0, A0, 2),
            asm::add_imm_32(A0, A0, 4),
            asm::add_imm_32(A0, A0, 8),
            asm::add_imm_32(A0, A0, 16),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);

    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[4]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);
    assert_eq!(instance.gas(), 995);

    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[3]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 16);
    assert_eq!(instance.gas(), 990);

    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[1]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 4 + 8 + 16);
    assert_eq!(instance.gas(), 985);

    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[2]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 8 + 16);
    assert_eq!(instance.gas(), 980);

    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 2 + 4 + 8 + 16);
    assert_eq!(instance.gas(), 975);
}

fn jump_into_middle_of_basic_block_from_within(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::jump(1), asm::add_imm_32(A0, A0, 100), asm::ret()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    // First, sanity check: does this program execute correctly as-is?
    let instructions = {
        let mut module_config: ModuleConfig = ModuleConfig::new();
        module_config.set_page_size(get_native_page_size().try_into().unwrap());
        module_config.set_gas_metering(Some(GasMeteringKind::Sync));
        let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();
        let instructions: Vec<_> = module.blob().instructions().collect();

        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_gas(1000);

        instance.set_next_program_counter(instructions[0].offset);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.reg(Reg::A0), 100);
        assert_eq!(instance.gas(), 997);
        instructions
    };

    use polkavm_common::program::{Instruction, ParsedInstruction};

    // Then, let's patch the code to jump somewhere invalid.
    assert_eq!(
        instructions[0],
        ParsedInstruction {
            kind: Instruction::jump(instructions[1].offset.0),
            offset: ProgramCounter(0),
            next_offset: ProgramCounter(2)
        }
    );
    assert_eq!(instructions[2].kind, asm::ret());

    // Patch the jump so that it jumps after the `add_imm_32`/before the `ret`.
    let mut raw_code = blob.code().to_vec();
    raw_code[1] = (instructions[2].offset.0 - instructions[0].offset.0) as u8;

    blob.set_code(raw_code.into());
    let new_instructions: Vec<_> = blob.instructions().collect();
    assert_eq!(&instructions[1..], &new_instructions[1..]);
    assert_eq!(new_instructions[0].kind, asm::jump(new_instructions[2].offset.0));

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();
    let instructions: Vec<_> = module.blob().instructions().collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);

    instance.set_next_program_counter(instructions[0].offset);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.gas(), 999);
}

fn entry_into_the_middle_of_an_instruction_is_rejected(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(A0, 0x11223344), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let instructions: Vec<_> = module.blob().instructions().collect();
    assert!(instructions[0].next_offset.0 > instructions[0].offset.0 + 1);

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    for offset in instructions[0].offset.0 + 1..instructions[0].next_offset.0 {
        instance.set_gas(1000);
        instance.set_next_program_counter(ProgramCounter(offset));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.program_counter(), Some(ProgramCounter(offset)));
        assert_eq!(instance.gas(), 1000);
    }
}

fn invalid_instruction_is_a_trap(engine_config: Config, isa: InstructionSetKind, make_invalid: impl FnOnce(&mut ProgramBlob)) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap(), asm::load_imm(A0, 42), asm::ret(), asm::jump(1)], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    make_invalid(&mut blob);

    let instructions: Vec<_> = blob.instructions().collect();
    assert_eq!(
        instructions[0],
        polkavm_common::program::ParsedInstruction {
            kind: crate::program::Instruction::invalid,
            offset: ProgramCounter(0),
            next_offset: ProgramCounter(1),
        }
    );

    assert_eq!(instructions[3].kind, asm::jump(instructions[0].next_offset.0));

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let run_from = |program_counter| {
        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_gas(1000);
        instance.set_next_program_counter(program_counter);
        let interrupt = instance.run().unwrap();
        (interrupt, instance.reg(A0), instance.gas())
    };

    let (interrupt, a0, gas) = run_from(instructions[3].offset);
    match_interrupt!(interrupt, InterruptKind::Finished);
    assert_eq!((a0, gas), (42, 997));

    let (interrupt, a0, gas) = run_from(instructions[1].offset);
    match_interrupt!(interrupt, InterruptKind::Finished);
    assert_eq!((a0, gas), (42, 998));

    let (interrupt, a0, gas) = run_from(instructions[0].offset);
    match_interrupt!(interrupt, InterruptKind::Trap);
    assert_eq!((a0, gas), (0, 999));
}

fn jump_after_invalid_instruction_from_within(engine_config: Config, isa: InstructionSetKind) {
    invalid_instruction_is_a_trap(engine_config, isa, |blob| {
        let mut raw_code = blob.code().to_vec();
        raw_code[0] = INVALID_RAW_OPCODE;
        blob.set_code(raw_code.into());
    });
}

fn jump_after_an_injected_invalid_instruction(engine_config: Config, isa: InstructionSetKind) {
    if !isa.is_legacy() {
        return;
    }

    invalid_instruction_is_a_trap(engine_config, isa, |blob| {
        let mut raw_bitmask = blob.bitmask().to_vec();
        raw_bitmask[0] &= !1;
        blob.set_bitmask(raw_bitmask.into());
    });
}

fn serialize_padded(isa: InstructionSetKind, length: usize, groups: &[(u32, &[polkavm_common::program::Instruction])]) -> Vec<u8> {
    use polkavm_common::program::MAX_INSTRUCTION_LENGTH;

    let opcode_unlikely = isa.opcode_to_raw(Opcode::unlikely, 0).unwrap().0 as u8;
    let mut code = vec![opcode_unlikely; length];
    for &(start, instructions) in groups {
        let mut position = start;
        for &instruction in instructions {
            let mut buffer = [0; MAX_INSTRUCTION_LENGTH];
            let instruction_length = instruction.serialize_into(isa, position, &mut buffer) as u32;
            code[position as usize..(position + instruction_length) as usize].copy_from_slice(&buffer[..instruction_length as usize]);
            position += instruction_length;
        }
    }

    code
}

fn jump_to_macro_block_boundary_is_always_valid(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    if isa.is_legacy() {
        return;
    }

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let code = serialize_padded(
        isa,
        1040,
        &[
            (0, &[asm::jump(1024)]),
            (1023, &[asm::fallthrough()]),
            (1024, &[asm::load_imm(A1, 42), asm::ret()]),
        ],
    );
    blob.set_code(code.into());

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    // A direct jump to the boundary.
    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 42);
    assert_eq!(instance.gas(), 997);

    // An arbitrary entry landing exactly on the boundary.
    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(1024));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 42);
    assert_eq!(instance.gas(), 998);

    // An arbitrary entry into the `unlikely` run before the boundary.
    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(3000);
    instance.set_next_program_counter(ProgramCounter(5));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 42);
    assert_eq!(instance.gas(), 3000 - ((1024 - 3) + 2));
}

fn unterminated_macro_block_traps_at_its_last_instruction(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    if isa.is_legacy() {
        return;
    }

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let code = serialize_padded(isa, 1040, &[(1024, &[asm::load_imm(A1, 42), asm::ret()])]);
    blob.set_code(code.into());

    let instructions: Vec<_> = blob.instructions_bounded_at(ProgramCounter(1023)).collect();
    assert_eq!(instructions[0].kind, crate::program::Instruction::invalid);
    assert_eq!(instructions[0].next_offset, ProgramCounter(1024));

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(3000);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.program_counter(), Some(ProgramCounter(1023)));
    assert_eq!(instance.gas(), 3000 - 1024);
}

fn instructions_across_macro_block_boundary_are_traps_1(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    if isa.is_legacy() {
        return;
    }

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let code = serialize_padded(
        isa,
        1040,
        &[
            (0, &[asm::jump(1019)]),
            (1018, &[asm::fallthrough()]),
            (1019, &[asm::load_imm64(A1, 0x112233)]),
        ],
    );
    blob.set_code(code.into());

    let instructions: Vec<_> = blob.instructions_bounded_at(ProgramCounter(1019)).collect();
    assert_eq!(instructions[0].kind, crate::program::Instruction::invalid);
    assert_eq!(instructions[0].next_offset, ProgramCounter(1024));

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.program_counter(), Some(ProgramCounter(1019)));
    assert_eq!(instance.gas(), 998);
}

fn instructions_across_macro_block_boundary_are_traps_2(engine_config: Config, isa: InstructionSetKind) {
    use polkavm_common::program::MAX_INSTRUCTION_LENGTH;

    let _ = env_logger::try_init();
    if isa.is_legacy() {
        return;
    }

    const BOUNDARY: u32 = 1024;
    const TARGET: u32 = 2048;

    let branch = asm::branch_eq(A0, A1, TARGET);
    let branch_offset = BOUNDARY - 2;
    let branch_length = {
        let mut buffer = [0; MAX_INSTRUCTION_LENGTH];
        branch.serialize_into(isa, branch_offset, &mut buffer) as u32
    };

    assert!(branch_offset + branch_length > BOUNDARY);

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let code = serialize_padded(
        isa,
        3072,
        &[
            (branch_offset - 1, &[asm::fallthrough()]),
            (branch_offset, &[branch]),
            (branch_offset + branch_length, &[asm::add_imm_64(A0, A0, 1)]),
            (TARGET, &[asm::load_imm(A1, 42), asm::ret()]),
        ],
    );
    blob.set_code(code.into());

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_cost_model(Some(crate::CostModelKind::Full(crate::CacheModel::L1Hit)));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(branch_offset));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.gas(), 998);
}

fn instructions_across_code_blob_boundary_are_traps(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    if isa.is_legacy() {
        return;
    }

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut code = serialize_padded(isa, 17, &[]);

    // A `load_imm` whose argument byte is past the end of the code; the decoder pads with zeros.
    *code.last_mut().unwrap() = isa.opcode_to_raw(Opcode::load_imm, 1).unwrap().0 as u8;
    blob.set_code(code.into());

    let instructions: Vec<_> = blob.instructions().collect();
    let [.., overshooting] = instructions.as_slice() else {
        unreachable!()
    };
    assert_eq!(
        (overshooting.kind, overshooting.offset, overshooting.next_offset),
        (crate::program::Instruction::invalid, ProgramCounter(16), ProgramCounter(17))
    );

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter(), Some(ProgramCounter(16)));
}

#[test]
fn linear_decode_matches_the_instruction_iterator() {
    use polkavm_common::program::OpcodeVisitor;

    #[derive(Copy, Clone)]
    struct CollectOffsets(InstructionSetKind);

    impl OpcodeVisitor for CollectOffsets {
        type State = Vec<ProgramCounter>;
        type ReturnTy = ();
        type InstructionSet = InstructionSetKind;

        fn instruction_set(self) -> Self::InstructionSet {
            self.0
        }

        fn dispatch(self, state: &mut Self::State, _opcode: usize, _chunk: u128, offset: u32, _length: u32) {
            state.push(ProgramCounter(offset));
        }
    }

    let _ = env_logger::try_init();
    let isa = InstructionSetKind::Latest64;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_code(&[asm::trap()], &[]);

    let code_length = 32;
    for tail in [&[asm::trap()][..], &[][..]] {
        let mut blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
        let code = serialize_padded(
            isa,
            code_length,
            &[(0, &[asm::load_imm(A1, 42), asm::ret()]), (code_length as u32 - 1, tail)],
        );
        blob.set_code(code.into());

        let mut linear = Vec::new();
        blob.visit(CollectOffsets(isa), &mut linear);

        let iterated: Vec<_> = blob.instructions().map(|instruction| instruction.offset).collect();
        assert_eq!(linear, iterated);
    }
}

fn jump_to_an_injected_invalid_instruction(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    if !isa.is_legacy() {
        return;
    }

    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(A1, 1), asm::load_imm(A1, 42), asm::ret(), asm::jump(0)], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut raw_bitmask = blob.bitmask().to_vec();
    raw_bitmask[0] &= !1;
    blob.set_bitmask(raw_bitmask.into());

    let instructions: Vec<_> = blob.instructions().collect();
    assert_eq!(instructions[0].kind, crate::program::Instruction::invalid);
    assert_eq!(instructions[0].offset, ProgramCounter(0));
    assert_eq!(instructions[1].kind, asm::load_imm(A1, 42));
    assert_eq!(instructions[3].kind, asm::jump(0));

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.gas(), 999);

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(instructions[3].offset);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.gas(), 998);
}

fn jump_indirect_simple(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::jump_indirect(A0, 0),
            asm::load_imm(A1, 100),
            asm::ret(),
            asm::load_imm(A1, 200),
            asm::ret(),
        ],
        &[1, 2],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, 2);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 100);

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, 4);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 200);

    for pointer in [0, 1, 3, 5, 6, 7, 1024 * 1024 - 1, 1024 * 1024, 0xffffffffffffffff] {
        log::info!("Trying pointer: {pointer}");
        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, pointer);
        instance.set_next_program_counter(ProgramCounter(0));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    }
}

fn jump_indirect_into_return_to_host_page(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::jump_indirect(A0, 0)], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    for pointer in (crate::RETURN_TO_HOST + 1..=crate::RETURN_TO_HOST + 0x1000).chain([u64::from(u32::MAX)]) {
        instance.set_reg(Reg::A0, pointer);
        instance.set_next_program_counter(ProgramCounter(0));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    }
}

fn jump_indirect_big_table(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[asm::jump_indirect(A0, 1024 * 1024), asm::trap(), asm::ret()],
        &vec![2; 1024 * 1024],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
}

fn jump_indirect_truncates_the_target_to_32_bits(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();

    let jump_to = |offset: i32, target: u64| {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&[asm::jump_indirect(A0, offset), asm::load_imm(A1, 100), asm::ret()], &[1]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();

        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, target);
        instance.set_next_program_counter(ProgramCounter(0));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        instance.reg(Reg::A1)
    };

    assert_eq!(jump_to(0, 0x00000001_00000002), 100);
    assert_eq!(jump_to(2, 0x00000001_00000000), 100);
    assert_eq!(jump_to(4, 0x00000000_fffffffe), 100);
}

fn dynamic_paging_basic(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_imm(Reg::A3, 0x1234),
            asm::store_imm_u32(0x10004, 1),
            asm::load_i32(Reg::A0, 0x10004),
            asm::load_i32(Reg::A1, 0x10008),
            asm::load_i32(Reg::A2, 0x10000 + cast(page_size).to_i32_or_panic()),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A0, 0x10); // Just clobber the registers.
    instance.set_reg(Reg::A1, 0x11);
    instance.set_reg(Reg::A2, 0x12);
    instance.set_reg(Reg::A3, 0x13);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert_eq!(segfault.page_size, page_size);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));
    if engine_config.backend() == Some(BackendKind::Compiler) {
        assert!(instance.next_native_program_counter().is_some());
    }
    assert_eq!(instance.reg(Reg::A3), 0x1234); // Registers are properly fetched.
    instance.set_reg(Reg::T0, 0x5678);

    let segfault = expect_segfault(instance.run().unwrap());
    // Segfault was not handled.
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));
    assert_eq!(segfault.page_address, 0x10000);
    assert_eq!(segfault.page_size, page_size);

    // Both normal 'zero_memory' and 'write_memory' cannot resolve pagefaults.
    assert_out_of_range_access(
        instance.zero_memory(segfault.page_address, page_size),
        segfault.page_address,
        page_size,
    );
    assert_out_of_range_access(instance.write_memory(segfault.page_address, &[0, 0]), segfault.page_address, 2);
    assert_out_of_range_access(instance.read_u8(segfault.page_address), segfault.page_address, 1);

    // Now handle it.
    instance
        .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    assert!(instance.is_memory_accessible(0x10000, 0x4, MemoryProtection::Read));
    assert!(!instance.is_memory_accessible(0x10000 + page_size, 0x4, MemoryProtection::Read));

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000 + page_size);
    assert_eq!(segfault.page_size, page_size);
    assert_eq!(instance.program_counter(), Some(offsets[4]));
    assert_eq!(instance.next_program_counter(), Some(offsets[4]));
    assert_eq!(instance.reg(Reg::A0), 1);
    assert_eq!(instance.reg(Reg::A1), 0);
    assert_eq!(instance.reg(Reg::A2), 0x12);
    assert_eq!(instance.reg(Reg::T0), 0x5678);
    instance
        .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A2), 0);
    assert_eq!(instance.reg(Reg::T0), 0x5678);

    // Running the program again produces no more segfaults, since everything is faulted already.
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    // Clear the first page and make it read-only.
    instance
        .zero_memory_with_memory_protection(0x10000, page_size, MemoryProtection::Read)
        .unwrap();

    // Cannot write to the page anymore, but can read it.
    assert_out_of_range_access(instance.zero_memory(0x10000, page_size), 0x10000, page_size);
    assert_out_of_range_access(instance.zero_memory(0x10000, 1), 0x10000, 1);
    assert_out_of_range_access(instance.write_memory(0x10000, &[0]), 0x10000, 1);
    assert_eq!(instance.read_u8(0x10000).unwrap(), 0);

    // The program cannot store anything there either.
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert_eq!(segfault.page_size, page_size);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));

    // But it can read.
    instance.set_next_program_counter(offsets[2]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
}

fn dynamic_paging_freeing_pages(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, 0x10000), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    instance
        .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    instance.free_pages(0x10000, page_size).unwrap();

    instance.set_next_program_counter(offsets[0]);
    expect_segfault(instance.run().unwrap());
}

fn dynamic_paging_protect_memory(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    if engine_config.crosscheck() {
        // TODO: This is currently broken due to a different stepping behavior when page faults are involved.
        return;
    }

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[asm::load_i32(Reg::A0, 0x10000), asm::store_imm_u32(0x10000, 0x12345678), asm::ret()],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    #[allow(clippy::match_wildcard_for_single_variants)]
    match instance.protect_memory(0x10000, page_size).unwrap_err() {
        MemoryAccessError::OutOfRangeAccess { address, length } => {
            assert_eq!(address, 0x10000);
            assert_eq!(length, u64::from(page_size));
        }
        error => panic!("unexpected error: {error}"),
    }

    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert!(!segfault.is_write_protected);
    assert_eq!(instance.program_counter(), Some(offsets[0]));
    instance
        .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    instance.protect_memory(segfault.page_address, page_size).unwrap();

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert!(segfault.is_write_protected);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert!(segfault.is_write_protected);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));

    instance.unprotect_memory(segfault.page_address, page_size).unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
}

#[cfg(feature = "std")]
mod stress_test {
    use core::sync::atomic::{AtomicBool, Ordering};

    static STRESS_TEST_LOCK: AtomicBool = AtomicBool::new(false);
    pub struct StressTestLock;

    impl StressTestLock {
        pub fn new() -> StressTestLock {
            while STRESS_TEST_LOCK
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::thread::sleep(core::time::Duration::from_millis(50));
            }

            Self
        }
    }

    impl Drop for StressTestLock {
        fn drop(&mut self) {
            STRESS_TEST_LOCK.store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "std")]
use self::stress_test::StressTestLock;

#[cfg(not(feature = "std"))]
fn dynamic_paging_stress_test(_engine_config: Config, _: InstructionSetKind) {}

#[cfg(feature = "std")]
fn dynamic_paging_stress_test(mut engine_config: Config, isa: InstructionSetKind) {
    let _lock = StressTestLock::new();
    let _ = env_logger::try_init();
    engine_config.set_allow_dynamic_paging(true);
    engine_config.set_worker_count(0);

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, 0x10000), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    for _ in 0..4 {
        let mut threads = Vec::new();
        for _ in 0..16 {
            let engine_config = engine_config.clone();
            let blob = blob.clone();
            let thread = std::thread::spawn(move || {
                let engine = Engine::new(&engine_config).unwrap();
                let page_size = get_native_page_size() as u32;
                let mut module_config = ModuleConfig::new();
                module_config.set_page_size(page_size);
                module_config.set_dynamic_paging(true);
                let module = Module::from_blob(&engine, &module_config, blob).unwrap();
                let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

                let mut instance = module.instantiate().unwrap();
                instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
                instance.set_next_program_counter(offsets[0]);
                let segfault = expect_segfault(instance.run().unwrap());
                instance
                    .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
                    .unwrap();
                match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
            });
            threads.push(thread);
        }

        for thread in threads {
            thread.join().unwrap();
        }
    }
}

fn dynamic_paging_initialize_multiple_pages(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_i32(Reg::A0, 0x10004),
            asm::load_i32(Reg::A1, 0x10004 + cast(page_size).to_i32_or_panic()),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size * 2, MemoryProtection::ReadWrite)
        .unwrap();
    // We've zeroed two pages, so we don't get a segfault anymore.
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
}

fn dynamic_paging_preinitialize_pages(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_i32(Reg::A0, 0x10004),
            asm::load_i32(Reg::A1, 0x10004 + cast(page_size).to_i32_or_panic()),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size * 2, MemoryProtection::ReadWrite)
        .unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
}

fn dynamic_paging_reading_does_not_resolve_segfaults(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, 0x10000), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert!(instance.read_u32(0x10000).is_err());

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
}

fn dynamic_paging_read_at_page_boundary(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, 0x10ffe), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size * 2, MemoryProtection::ReadWrite)
        .unwrap();
    instance.write_memory(0x10fff, &[0xaa, 0xbb]).unwrap();

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00bbaa00);

    instance.set_reg(Reg::A0, 0);
    instance.protect_memory(0x10000, page_size * 2).unwrap();
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00bbaa00);
}

fn dynamic_paging_read_at_top_of_address_space(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, cast(0xffffffff_u32).bitwise_as_i32()), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0xfffff000);
}

fn dynamic_paging_read_with_upper_bits_set(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_imm64(Reg::A0, 0xffffffff10000001),
            asm::load_indirect_i32(Reg::A1, Reg::A0, 0),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000000);
}

fn dynamic_paging_read_at_bottom_of_address_space(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_i32(Reg::A0, 1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    assert_eq!(instance.run().unwrap(), InterruptKind::Trap);
}

fn dynamic_paging_read_below_the_guard_threshold(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;

    // Runs a program which does a single 4-byte load from `address` and returns how it stopped.
    let run_load = |address: i32| {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&[asm::load_i32(Reg::A0, address), asm::ret()], &[]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let mut module_config = ModuleConfig::new();
        module_config.set_page_size(page_size);
        module_config.set_dynamic_paging(true);
        let module = Module::from_blob(&engine, &module_config, blob).unwrap();
        let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(offsets[0]);
        instance.run().unwrap()
    };

    // Reads fully inside the lowest 64KB of the address space must trap instead of
    // producing a recoverable segfault, on every backend. (See #390.)
    for address in [0x1000, 0x4000, 0xf000] {
        assert_eq!(run_load(address), InterruptKind::Trap);
    }

    // A read straddling the boundary (partially below 0x10000 and partially at/above it) must
    // also trap, since its lowest byte still lands in the inaccessible zone.
    assert_eq!(run_load(0xfffe), InterruptKind::Trap);

    // ...while a read fully at 0x10000 is the first one which segfaults recoverably.
    let segfault = expect_segfault(run_load(0x10000));
    assert_eq!(segfault.page_address, 0x10000);
}

fn dynamic_paging_read_memory_which_is_not_paged_in(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    #[allow(clippy::match_wildcard_for_single_variants)]
    match instance.read_memory(0x10000, page_size).unwrap_err() {
        MemoryAccessError::OutOfRangeAccess { address, length } => {
            assert_eq!(address, 0x10000);
            assert_eq!(length, u64::from(page_size));
        }
        error => panic!("unexpected error: {error}"),
    }
}

fn dynamic_paging_write_at_page_boundary_with_no_pages(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_u32(0x10ffe, 0x12345678), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x11000);
    assert_eq!(instance.read_memory(0x10ffe, 2).unwrap(), vec![0, 0]);
    instance
        .zero_memory_with_memory_protection(0x11000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_memory(0x10ffe, 2).unwrap(), vec![0x78, 0x56]);
}

fn dynamic_paging_write_at_page_boundary_with_first_page(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_u32(0x10ffe, 0x12345678), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x11000);
    assert_eq!(instance.read_memory(0x10ffe, 2).unwrap(), vec![0, 0]);
    instance
        .zero_memory_with_memory_protection(0x11000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_memory(0x10ffe, 2).unwrap(), vec![0x78, 0x56]);
}

fn dynamic_paging_write_at_page_boundary_with_second_page(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_u32(0x10ffe, 0x12345678), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    instance
        .zero_memory_with_memory_protection(0x11000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    assert_eq!(instance.read_memory(0x11000, 2).unwrap(), vec![0, 0]);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size, MemoryProtection::ReadWrite)
        .unwrap();

    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_memory(0x11000, 2).unwrap(), vec![0x34, 0x12]);
}

fn dynamic_paging_change_written_value_and_address_during_segfault(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_indirect_u32(Reg::A0, Reg::A1, 0), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    instance.set_reg(Reg::A0, 0x11223344);
    instance.set_reg(Reg::A1, 0x10001);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
    instance
        .zero_memory_with_memory_protection(0x10000, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    instance.set_reg(Reg::A0, 0x55667788);
    instance.set_reg(Reg::A1, 0x10002);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_memory(0x10000, 6).unwrap(), vec![0, 0, 0x88, 0x77, 0x66, 0x55]);
}

fn dynamic_paging_cancel_segfault_by_changing_address(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    // The offset here matters: the recompiler can compute the address of an access into a
    // scratch register, which then has to be recalculated when the address is changed.
    for store_offset in [0, 4] {
        let engine = Engine::new(&engine_config).unwrap();
        let page_size = get_native_page_size() as u32;
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&[asm::store_imm_indirect_u32(Reg::A0, store_offset, 0x12345678), asm::ret()], &[]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let mut module_config = ModuleConfig::new();
        module_config.set_page_size(page_size);
        module_config.set_dynamic_paging(true);
        let module = Module::from_blob(&engine, &module_config, blob).unwrap();
        let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

        let mut instance = module.instantiate().unwrap();
        instance
            .zero_memory_with_memory_protection(0x11000, page_size, MemoryProtection::ReadWrite)
            .unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(offsets[0]);
        instance.set_reg(Reg::A0, 0x10000);
        let segfault = expect_segfault(instance.run().unwrap());
        assert_eq!(segfault.page_address, 0x10000);
        instance.set_reg(Reg::A0, 0x11000);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(
            instance.read_memory(0x11000 + store_offset as u32, 4).unwrap(),
            vec![0x78, 0x56, 0x34, 0x12]
        );
    }
}

fn dynamic_paging_worker_recycle_turn_dynamic_paging_on_and_off(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);
    engine_config.set_worker_count(1);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_rw_data_size(1);
    builder.set_code(&[asm::store_imm_u32(0x20000, 0x12345678), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module_dynamic = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();
    module_config.set_dynamic_paging(false);
    let module_static = Module::from_blob(&engine, &module_config, blob).unwrap();

    for is_dynamic in [false, true, false, true] {
        let mut instance = if is_dynamic {
            module_dynamic.instantiate().unwrap()
        } else {
            module_static.instantiate().unwrap()
        };

        if !is_dynamic {
            assert_eq!(instance.read_u32(0x20000).unwrap(), 0);
        } else {
            assert_out_of_range_access(instance.read_u32(0x20000), 0x20000, 4);
        }

        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        if is_dynamic {
            let segfault = expect_segfault(instance.run().unwrap());
            assert_eq!(segfault.page_address, 0x20000);
            assert_eq!(segfault.page_size, page_size);
            let segfault = expect_segfault(instance.run().unwrap());
            assert_out_of_range_access(instance.read_u32(0x21000), 0x21000, 4);
            instance
                .zero_memory_with_memory_protection(segfault.page_address, page_size + 4, MemoryProtection::ReadWrite)
                .unwrap();
            match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
            assert_eq!(instance.read_u32(0x20000).unwrap(), 0x12345678);
            assert_eq!(instance.read_u32(0x21000).unwrap(), 0);
            instance.set_next_program_counter(ProgramCounter(0));
            match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        } else {
            assert_out_of_range_access(instance.read_u32(0x21000), 0x21000, 4);
            assert_eq!(instance.read_u32(0x20000).unwrap(), 0);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
            assert_eq!(instance.read_u32(0x20000).unwrap(), 0x12345678);
        }
    }
}

fn dynamic_paging_worker_recycle_during_segfault(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);
    engine_config.set_worker_count(1);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let blob_1 = {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_rw_data_size(1);
        builder.set_code(&[asm::store_imm_u32(0x20000, 0x12345678), asm::ret()], &[]);

        ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
    };

    let blob_2 = {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_rw_data_size(1);
        builder.set_code(&[asm::store_imm_u32(0x20000, 0x11223344), asm::ret()], &[]);

        ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap()
    };

    let module_1 = {
        let mut module_config = ModuleConfig::new();
        module_config.set_page_size(page_size);
        module_config.set_dynamic_paging(true);
        Module::from_blob(&engine, &module_config, blob_1).unwrap()
    };

    let module_2 = {
        let mut module_config = ModuleConfig::new();
        module_config.set_page_size(page_size);
        module_config.set_dynamic_paging(false);
        Module::from_blob(&engine, &module_config, blob_2).unwrap()
    };

    {
        let mut instance = module_1.instantiate().unwrap();
        instance.set_next_program_counter(ProgramCounter(0));
        expect_segfault(instance.run().unwrap());
    }

    let mut instance = module_2.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_u32(0x20000).unwrap(), 0x11223344);
}

fn dynamic_paging_change_program_counter_during_segfault(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::store_imm_u32(0x10000, 1),
            asm::ret(),
            asm::store_imm_u32(0x11000, 2),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);

    instance.set_next_program_counter(offsets[2]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x11000);
    instance
        .zero_memory_with_memory_protection(segfault.page_address, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_u32(0x11000).unwrap(), 2);
}

fn dynamic_paging_run_out_of_gas(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[asm::load_imm(Reg::A0, 1), asm::fallthrough(), asm::load_imm(Reg::A0, 2), asm::ret()],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    instance.set_gas(2);
    match_interrupt!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.program_counter(), Some(offsets[2]));
    assert_eq!(instance.gas(), 0);
}

#[cfg(not(feature = "std"))]
fn dynamic_paging_receive_from_another_thread_and_run(_: Config, _: InstructionSetKind) {}

#[cfg(feature = "std")]
fn dynamic_paging_receive_from_another_thread_and_run(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let mut instance = std::thread::spawn(move || {
        let engine = Engine::new(&engine_config).unwrap();
        let page_size = get_native_page_size() as u32;
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(
            &[
                asm::load_imm(Reg::A0, 0x10000),
                asm::fallthrough(),
                asm::store_indirect_u64(Reg::A0, Reg::A0, 0),
                asm::add_imm_64(Reg::A0, Reg::A0, 0x1000),
                asm::jump(1),
            ],
            &[],
        );

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let mut module_config = ModuleConfig::new();
        module_config.set_page_size(page_size);
        module_config.set_dynamic_paging(true);
        module_config.set_gas_metering(Some(GasMeteringKind::Sync));
        let module = Module::from_blob(&engine, &module_config, blob).unwrap();
        let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(offsets[0]);
        instance.set_gas(1000000);
        instance
    })
    .join()
    .unwrap();

    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, 0x10000);
}

#[cfg(not(feature = "std"))]
fn dynamic_paging_instantiate_on_another_thread(_: Config, _: InstructionSetKind) {}

#[cfg(feature = "std")]
fn dynamic_paging_instantiate_on_another_thread(mut engine_config: Config, isa: InstructionSetKind) {
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_imm(Reg::A0, 0x10000),
            asm::fallthrough(),
            asm::store_indirect_u64(Reg::A0, Reg::A0, 0),
            asm::add_imm_64(Reg::A0, Reg::A0, 0x1000),
            asm::jump(1),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    {
        let module = module.clone();
        let mut instance = std::thread::spawn(move || module.instantiate()).join().unwrap().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(offsets[0]);
        instance.set_gas(1000000);

        let segfault = expect_segfault(instance.run().unwrap());
        assert_eq!(segfault.page_address, 0x10000);
    }

    const THREAD_COUNT: usize = 32;

    let barrier = alloc::sync::Arc::new(std::sync::Barrier::new(THREAD_COUNT));
    for _ in 0..32 {
        let mut threads = Vec::new();
        for _ in 0..THREAD_COUNT {
            let module = module.clone();
            let barrier = alloc::sync::Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                module.instantiate()
            }));
        }
        for thread in threads {
            let mut instance = thread.join().unwrap().unwrap();
            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_next_program_counter(offsets[0]);
            instance.set_gas(1000000);

            let segfault = expect_segfault(instance.run().unwrap());
            assert_eq!(segfault.page_address, 0x10000);
        }
    }
}

#[cfg(not(feature = "std"))]
fn dynamic_paging_parallel_page_fault_stress_test(_: Config, _: InstructionSetKind) {}

#[cfg(feature = "std")]
fn dynamic_paging_parallel_page_fault_stress_test(mut engine_config: Config, isa: InstructionSetKind) {
    let _lock = StressTestLock::new();
    engine_config.set_allow_dynamic_paging(true);

    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_imm(Reg::A0, 0x10000),
            asm::fallthrough(),
            asm::store_indirect_u64(Reg::A0, Reg::A0, 0),
            asm::add_imm_64(Reg::A0, Reg::A0, 0x1000),
            asm::jump(1),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();
    let initial_offset = offsets[0];

    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    const THREAD_COUNT: usize = 32;
    let barrier = alloc::sync::Arc::new(std::sync::Barrier::new(THREAD_COUNT));
    let flag = Arc::new(AtomicBool::new(false));

    let mut threads = Vec::new();
    for nth_thread in 0..THREAD_COUNT {
        let module = module.clone();
        let barrier = alloc::sync::Arc::clone(&barrier);
        struct InterruptOnDrop(Option<Arc<AtomicBool>>);
        impl Drop for InterruptOnDrop {
            fn drop(&mut self) {
                if let Some(should_interrupt) = self.0.take() {
                    should_interrupt.store(true, Ordering::Relaxed);
                }
            }
        }
        impl InterruptOnDrop {
            fn disarm(&mut self) {
                self.0.take();
            }

            fn should_interrupt(&self) -> bool {
                self.0.as_ref().map_or(false, |flag| flag.load(Ordering::Relaxed))
            }
        }
        let mut flag = InterruptOnDrop(Some(Arc::clone(&flag)));
        let thread = std::thread::spawn(move || {
            let mut instance = module.instantiate().unwrap();
            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_next_program_counter(initial_offset);
            instance.set_gas(1000000);
            let mut address = 0x10000;

            barrier.wait();
            log::info!("Starting thread #{nth_thread}... (child PID = {:?})", instance.pid());
            for _ in 0..1000 {
                if flag.should_interrupt() {
                    break;
                }
                let segfault = expect_segfault(instance.run().unwrap());
                if flag.should_interrupt() {
                    break;
                }
                assert_eq!(segfault.page_address, address);
                instance
                    .zero_memory_with_memory_protection(segfault.page_address, segfault.page_size, MemoryProtection::ReadWrite)
                    .unwrap();
                address += segfault.page_size;
            }
            flag.disarm();
            log::info!("Finished thread #{nth_thread} (child PID = {:?})", instance.pid());
        });
        threads.push(thread);
    }

    let mut results = Vec::new();
    for thread in threads {
        results.push(thread.join());
    }

    for result in results {
        result.unwrap();
    }
}

fn memory_access_with_upper_address_bits_set(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let engine = Engine::new(&engine_config).unwrap();
    const ADDRESS: u64 = 0xffffffff_fffffffc;

    let run = |code: &[polkavm_common::program::Instruction]| {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(code, &[]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
        let offset = module.blob().instructions().next().unwrap().offset;

        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_reg(Reg::A0, ADDRESS);
        instance.set_next_program_counter(offset);
        instance.run().unwrap()
    };

    match_interrupt!(run(&[asm::load_indirect_u32(Reg::A1, Reg::A0, 0), asm::ret()]), InterruptKind::Trap);
    match_interrupt!(
        run(&[asm::store_indirect_u32(Reg::A1, Reg::A0, 0), asm::ret()]),
        InterruptKind::Trap
    );
}

fn decompress_zstd(mut bytes: &[u8]) -> Vec<u8> {
    use ruzstd::io::Read;
    let mut output = Vec::new();
    let mut fp = ruzstd::streaming_decoder::StreamingDecoder::new(&mut bytes).unwrap();

    let mut buffer = vec![0_u8; 32 * 1024];
    loop {
        let count = fp.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }

        output.extend_from_slice(&buffer);
    }

    output
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BlobMapKey {
    optimize: bool,
    strip: bool,
    isa: InstructionSetKind,
    elf: &'static [u8],
    is_permissive: bool,
}

static BLOB_MAP: Mutex<Option<BTreeMap<BlobMapKey, ProgramBlob>>> = Mutex::new(None);

fn get_blob(elf: &'static [u8], isa: InstructionSetKind) -> ProgramBlob {
    get_blob_impl(BlobMapKey {
        optimize: true,
        strip: false,
        isa,
        elf,
        is_permissive: false,
    })
}

fn get_blob_impl(key: BlobMapKey) -> ProgramBlob {
    let mut blob_map = BLOB_MAP.lock();
    let blob_map = blob_map.get_or_insert_with(BTreeMap::new);
    blob_map
        .entry(key)
        .or_insert_with(|| {
            // This is slow, so cache it.
            let decompress = !key.elf.starts_with(&[0x7f, b'E', b'L', b'F']);
            let elf = if decompress { decompress_zstd(key.elf) } else { key.elf.to_vec() };
            let mut config = polkavm_linker::Config::default();
            config.set_optimize(key.optimize);
            config.set_strip(key.strip);
            config.set_allow_unsupported_instructions(key.is_permissive && !key.isa.supports_opcode(Opcode::sbrk));

            let bytes = polkavm_linker::program_from_elf(config, key.isa.into(), &elf).unwrap();
            let blob = ProgramBlob::parse(bytes.into()).unwrap();
            assert_eq!(blob.isa(), key.isa);

            blob
        })
        .clone()
}

fn doom_impl(config: Config, isa: InstructionSetKind, elf: &'static [u8]) {
    if config.backend() == Some(crate::BackendKind::Interpreter) || config.crosscheck() {
        // The interpreter is currently too slow to run doom.
        return;
    }

    if cfg!(debug_assertions) {
        // The linker is currently very slow in debug mode.
        return;
    }

    const DOOM_WAD: &[u8] = include_bytes!("../../../examples/doom/roms/doom1.wad");

    let _ = env_logger::try_init();
    let blob = get_blob(elf, isa);
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_page_size(16 * 1024); // TODO: Also test with other page sizes.
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let mut linker: Linker<State, String> = Linker::new();

    struct State {
        frame: Vec<u8>,
        frame_width: u32,
        frame_height: u32,
    }

    linker
        .define_typed(
            "ext_output_video",
            |caller: Caller<State>, address: u32, width: u32, height: u32| -> Result<(), String> {
                let length = width * height * 4;
                caller.user_data.frame.clear();
                caller.user_data.frame.reserve(length as usize);
                caller
                    .instance
                    .read_memory_into(address, &mut caller.user_data.frame.spare_capacity_mut()[..length as usize])
                    .map_err(|err| err.to_string())?;
                // SAFETY: We've successfully read this many bytes into this Vec.
                unsafe {
                    caller.user_data.frame.set_len(length as usize);
                }
                caller.user_data.frame_width = width;
                caller.user_data.frame_height = height;
                Ok(())
            },
        )
        .unwrap();

    linker
        .define_typed("ext_output_audio", |_caller: Caller<State>, _address: u32, _samples: u32| {})
        .unwrap();

    linker
        .define_typed("ext_rom_size", |_caller: Caller<State>| -> u32 { DOOM_WAD.len() as u32 })
        .unwrap();

    linker
        .define_typed(
            "ext_rom_read",
            |caller: Caller<State>, pointer: u32, offset: u32, length: u32| -> Result<(), String> {
                let chunk = DOOM_WAD
                    .get(offset as usize..offset as usize + length as usize)
                    .ok_or_else(|| format!("invalid ROM read: offset = 0x{offset:x}, length = {length}"))?;

                caller.instance.write_memory(pointer, chunk).map_err(|err| err.to_string())
            },
        )
        .unwrap();

    linker
        .define_typed("ext_stdout", |_caller: Caller<State>, _buffer: u32, length: u32| -> i32 {
            length as i32
        })
        .unwrap();

    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    let mut state = State {
        frame: Vec::new(),
        frame_width: 0,
        frame_height: 0,
    };

    instance.call_typed(&mut state, "ext_initialize", ()).unwrap();
    for nth_frame in 0..=10440 {
        instance.call_typed(&mut state, "ext_tick", ()).unwrap();

        let expected_frame_raw = match nth_frame {
            120 => decompress_zstd(include_bytes!("../../../test-data/doom_00120.tga.zst")),
            1320 => decompress_zstd(include_bytes!("../../../test-data/doom_01320.tga.zst")),
            9000 => decompress_zstd(include_bytes!("../../../test-data/doom_09000.tga.zst")),
            10440 => decompress_zstd(include_bytes!("../../../test-data/doom_10440.tga.zst")),
            _ => continue,
        };

        for pixel in state.frame.chunks_exact_mut(4) {
            pixel.swap(0, 2);
            pixel[3] = 0xff;
        }

        let expected_frame = image::load_from_memory_with_format(&expected_frame_raw, image::ImageFormat::Tga)
            .unwrap()
            .to_rgba8();

        if state.frame != *expected_frame.as_raw() {
            panic!("frame {nth_frame:05} doesn't match!");
        }
    }

    // Generate frames to pick:
    // for nth_frame in 0..20000 {
    //     ext_tick.call(&mut state, ()).unwrap();
    //     if nth_frame % 120 == 0 {
    //         for pixel in state.frame.chunks_exact_mut(4) {
    //             pixel.swap(0, 2);
    //             pixel[3] = 0xff;
    //         }
    //         let filename = format!("/tmp/doom-frames/doom_{:05}.tga", nth_frame);
    //         image::save_buffer(filename, &state.frame, state.frame_width, state.frame_height, image::ColorType::Rgba8).unwrap();
    //     }
    // }
}

fn doom_o3_dwarf5(config: Config, isa: InstructionSetKind) {
    if isa.is_64_bit() {
        return;
    }

    doom_impl(config, isa, include_bytes!("../../../test-data/doom_O3_dwarf5.elf.zst"));
}

fn doom_o1_dwarf5(config: Config, isa: InstructionSetKind) {
    if isa.is_64_bit() {
        return;
    }

    doom_impl(config, isa, include_bytes!("../../../test-data/doom_O1_dwarf5.elf.zst"));
}

fn doom_o3_dwarf2(config: Config, isa: InstructionSetKind) {
    if isa.is_64_bit() {
        return;
    }

    doom_impl(config, isa, include_bytes!("../../../test-data/doom_O3_dwarf2.elf.zst"));
}

fn doom(config: Config, isa: InstructionSetKind) {
    if !isa.is_64_bit() {
        return;
    }

    doom_impl(config, isa, include_bytes!("../../../test-data/doom_64.elf.zst"));
}

fn pinky_dynamic_paging(mut config: Config, isa: InstructionSetKind) {
    config.set_allow_dynamic_paging(true);
    pinky_impl(config, isa);
}

fn pinky_standard(config: Config, isa: InstructionSetKind) {
    pinky_impl(config, isa)
}

fn pinky_impl(config: Config, isa: InstructionSetKind) {
    if (config.backend() == Some(crate::BackendKind::Interpreter) && cfg!(debug_assertions)) || config.crosscheck() {
        return; // Too slow.
    }

    let _ = env_logger::try_init();
    let blob = get_blob(get_test_program(TestProgram::Pinky, isa.is_64_bit()), isa);

    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    if config.allow_dynamic_paging() {
        module_config.set_dynamic_paging(true);
    }
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    instance.call_typed(&mut (), "initialize", ()).unwrap();
    for _ in 0..256 {
        instance.call_typed(&mut (), "run", ()).unwrap();
    }

    let address: u32 = instance.call_typed_and_get_result(&mut (), "get_framebuffer", ()).unwrap();
    let framebuffer = instance.read_memory(address, 256 * 240 * 4).unwrap();

    let expected_frame_raw = decompress_zstd(include_bytes!("../../../test-data/pinky_00256.tga.zst"));
    let expected_frame = image::load_from_memory_with_format(&expected_frame_raw, image::ImageFormat::Tga)
        .unwrap()
        .to_rgba8();

    if framebuffer != *expected_frame.as_raw() {
        panic!("frames doesn't match!");
    }
}

fn dispatch_table(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"block_0");
    builder.add_export_by_basic_block(1, b"block_1");
    builder.add_export_by_basic_block(2, b"block_2");
    builder.add_dispatch_table_entry("block_2");
    builder.add_dispatch_table_entry("block_0");
    builder.add_dispatch_table_entry("block_1");
    let code = vec![
        asm::load_imm(Reg::A0, 10),
        asm::ret(),
        asm::load_imm(Reg::A0, 11),
        asm::ret(),
        asm::load_imm(Reg::A0, 12),
        asm::ret(),
    ];

    builder.set_code(&code, &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();
    assert_eq!(offsets[0], ProgramCounter(0));
    assert_eq!(offsets[1], ProgramCounter(5));
    assert_eq!(offsets[2], ProgramCounter(10));

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);

    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 12);

    instance.set_next_program_counter(ProgramCounter(5));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 10);

    instance.set_next_program_counter(ProgramCounter(10));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 11);
}

fn fallthrough_into_already_compiled_block(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::jump(2),
            asm::add_imm_32(A0, A0, 2),
            asm::fallthrough(),
            asm::add_imm_32(A0, A0, 4),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 4);

    instance.set_reg(Reg::A0, 0);
    instance.set_gas(1000);
    instance.set_next_program_counter(offsets[1]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 6);
    let gas = instance.gas();

    instance.set_reg(Reg::A0, 0);
    instance.set_gas(1000);
    instance.set_next_program_counter(offsets[1]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 6);
    assert_eq!(gas, instance.gas());
}

fn implicit_trap_after_fallthrough(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::fallthrough()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter().unwrap().0, 0);
    assert_eq!(instance.next_program_counter(), None);
}

fn invalid_instruction_after_fallthrough(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::fallthrough(), asm::fallthrough(), asm::ret()], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let instructions: Vec<_> = blob.instructions().collect();

    let mut raw_code = blob.code().to_vec();
    raw_code[instructions[1].offset.0 as usize] = INVALID_RAW_OPCODE;
    blob.set_code(raw_code.into());

    let instructions: Vec<_> = blob.instructions().collect();
    assert_eq!(
        instructions[1],
        polkavm_common::program::ParsedInstruction {
            kind: crate::program::Instruction::invalid,
            offset: ProgramCounter(1),
            next_offset: ProgramCounter(2),
        }
    );

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);

    instance.set_next_program_counter(instructions[0].offset);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.gas(), 998);
    assert_eq!(instance.program_counter().unwrap(), instructions[1].offset);
    assert_eq!(instance.next_program_counter(), None);
}

fn invalid_branch_target(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::branch_eq_imm(Reg::A0, 33, 2),
            asm::load_imm(Reg::A1, 1),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 3),
        ],
        &[],
    );

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let instructions: Vec<_>;

    // Valid branch.
    {
        let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
        instructions = blob.instructions().collect();

        let module = Module::from_blob(&engine, &module_config, blob).unwrap();

        // False branch.
        let mut instance = module.instantiate().unwrap();
        instance.set_gas(100);
        instance.set_next_program_counter(instructions[0].offset);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.gas(), 97);
        assert_eq!(instance.program_counter().unwrap(), instructions[2].offset);
        assert_eq!(instance.next_program_counter(), None);
        assert_eq!(instance.reg(Reg::A1), 1);

        // True branch.
        let mut instance = module.instantiate().unwrap();
        instance.set_gas(100);
        instance.set_next_program_counter(instructions[0].offset);
        instance.set_reg(Reg::A0, 33);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.gas(), 97);
        assert_eq!(instance.program_counter().unwrap(), instructions[4].offset);
        assert_eq!(instance.next_program_counter(), None);
        assert_eq!(instance.reg(Reg::A1), 2);
    }

    // Invalid branch (true case).
    {
        let mut blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
        let mut raw_code = blob.code().to_vec();
        raw_code[instructions[0].next_offset.0 as usize - 1] -= 1;
        blob.set_code(raw_code.into());

        let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();

        for _ in 0..2 {
            // False branch.
            let mut instance = module.instantiate().unwrap();
            instance.set_gas(100);
            instance.set_next_program_counter(instructions[0].offset);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
            assert_eq!(instance.gas(), 99);
            assert_eq!(instance.program_counter().unwrap(), instructions[0].offset);
            assert_eq!(instance.next_program_counter(), None);
            assert_eq!(instance.reg(Reg::A1), 0);

            // True branch.
            let mut instance = module.instantiate().unwrap();
            instance.set_reg(Reg::A0, 33);
            instance.set_gas(100);
            instance.set_next_program_counter(instructions[0].offset);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
            assert_eq!(instance.gas(), 99);
            assert_eq!(instance.program_counter().unwrap(), instructions[0].offset);
            assert_eq!(instance.next_program_counter(), None);
            assert_eq!(instance.reg(Reg::A1), 0);
        }
    }

    // Invalid branch (false case).
    if isa.is_legacy() {
        // TODO: Also test on non-legacy.
        let mut blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
        let mut raw_bitmask = blob.bitmask().to_vec();
        raw_bitmask.fill(0);
        raw_bitmask[0] = 1;
        blob.set_bitmask(raw_bitmask.into());

        let instructions: Vec<_> = blob.instructions().collect();
        let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();

        for _ in 0..2 {
            // False branch.
            let mut instance = module.instantiate().unwrap();
            instance.set_gas(100);
            instance.set_next_program_counter(instructions[0].offset);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
            assert_eq!(instance.gas(), 99);
            assert_eq!(instance.program_counter().unwrap(), instructions[0].offset);
            assert_eq!(instance.next_program_counter(), None);
            assert_eq!(instance.reg(Reg::A1), 0);

            // True branch.
            let mut instance = module.instantiate().unwrap();
            instance.set_reg(Reg::A0, 33);
            instance.set_gas(100);
            instance.set_next_program_counter(instructions[0].offset);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
            assert_eq!(instance.gas(), 99);
            assert_eq!(instance.program_counter().unwrap(), instructions[0].offset);
            assert_eq!(instance.next_program_counter(), None);
            assert_eq!(instance.reg(Reg::A1), 0);
        }
    }
}

fn branch_gas_cost_consistent_across_backends(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let blob = crate::program::assemble(
        Some(isa),
        "
            a0 = a1 + a2
            jump @skip if a0 == a1
            trap
            @skip:
            a0 = a0 + a1
            trap
        ",
    )
    .unwrap();
    let blob = ProgramBlob::parse(blob.into()).unwrap();

    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    module_config.set_cost_model(Some(crate::CostModelKind::Full(crate::CacheModel::L1Hit)));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_gas(100);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert!(instance.gas() >= 0);
}

fn aux_data_works(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_indirect_i32(Reg::A1, Reg::A0, 0),
            asm::store_imm_indirect_u32(Reg::A0, 0, 0x11223344),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(1);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.write_u32(module.memory_map().aux_data_address(), 0x12345678).unwrap();
    instance.set_reg(Reg::A0, u64::from(module.memory_map().aux_data_address()));
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    assert_eq!(instance.program_counter().unwrap(), offsets[1]);
    assert_eq!(instance.reg(Reg::A1), 0x12345678);

    instance.zero_memory(module.memory_map().aux_data_address(), 1).unwrap();
    assert_eq!(instance.read_u32(module.memory_map().aux_data_address()).unwrap(), 0x12345600);
    instance
        .zero_memory(module.memory_map().aux_data_address(), module.memory_map().aux_data_size())
        .unwrap();
    assert_eq!(instance.read_u32(module.memory_map().aux_data_address()).unwrap(), 0);
}

fn two_page_aux_data_module(engine: &Engine, isa: InstructionSetKind) -> Module {
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_indirect_u32(Reg::A1, Reg::A0, 0), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(page_size * 2);
    Module::from_blob(engine, &module_config, blob).unwrap()
}

#[derive(PartialEq, Debug)]
struct AuxDataPage {
    host_read: Option<Vec<u8>>,
    guest_load_of_last_word: Option<u64>,
}

fn aux_data_pages(instance: &mut crate::RawInstance) -> Vec<AuxDataPage> {
    let page_size = get_native_page_size() as u32;
    let aux_data_range = instance.module().memory_map().aux_data_range();
    aux_data_range
        .step_by(page_size as usize)
        .map(|page_address| {
            let host_read = instance.read_memory(page_address, page_size).ok();
            instance.set_reg(Reg::A0, u64::from(page_address + page_size - 4));
            instance.set_reg(Reg::A1, 0xdeadbeef);
            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_next_program_counter(ProgramCounter(0));
            let guest_load_of_last_word = match instance.run().unwrap() {
                InterruptKind::Finished => Some(instance.reg(Reg::A1)),
                InterruptKind::Trap => None,
                interrupt => panic!("unexpected interrupt: {interrupt:?}"),
            };

            AuxDataPage {
                host_read,
                guest_load_of_last_word,
            }
        })
        .collect()
}

fn aux_data_page_with_content(byte: u8) -> AuxDataPage {
    AuxDataPage {
        host_read: Some(vec![byte; get_native_page_size()]),
        guest_load_of_last_word: Some(u64::from(u32::from_le_bytes([byte; 4]))),
    }
}

fn aux_data_content_on_new_instance(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let module = two_page_aux_data_module(&engine, isa);
    let aux_data_range = module.memory_map().aux_data_range();

    let mut instance = module.instantiate().unwrap();
    instance
        .write_memory(aux_data_range.start, &vec![0xff; aux_data_range.len()])
        .unwrap();
    core::mem::drop(instance);

    let mut instance = module.instantiate().unwrap();
    assert_eq!(
        aux_data_pages(&mut instance),
        [aux_data_page_with_content(0), aux_data_page_with_content(0)]
    );
}

fn aux_data_after_memory_reset(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let module = two_page_aux_data_module(&engine, isa);
    let aux_data_range = module.memory_map().aux_data_range();
    let page_size = get_native_page_size() as u32;

    let mut instance = module.instantiate().unwrap();
    instance
        .write_memory(aux_data_range.start, &vec![0xff; aux_data_range.len()])
        .unwrap();
    instance.set_accessible_aux_size(page_size).unwrap();
    instance.reset_memory().unwrap();

    let mut new_instance = module.instantiate().unwrap();
    assert_eq!(aux_data_pages(&mut instance), aux_data_pages(&mut new_instance));
}

fn aux_data_content_after_shrink(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let module = two_page_aux_data_module(&engine, isa);
    let aux_data_range = module.memory_map().aux_data_range();
    let page_size = get_native_page_size() as u32;

    let mut instance = module.instantiate().unwrap();
    instance
        .write_memory(aux_data_range.start, &vec![0xff; aux_data_range.len()])
        .unwrap();
    instance.set_accessible_aux_size(page_size).unwrap();
    assert_eq!(
        aux_data_pages(&mut instance),
        [
            aux_data_page_with_content(0xff),
            AuxDataPage {
                host_read: None,
                guest_load_of_last_word: None
            }
        ]
    );

    instance.set_accessible_aux_size(page_size * 2).unwrap();
    assert_eq!(
        aux_data_pages(&mut instance),
        [aux_data_page_with_content(0xff), aux_data_page_with_content(0)]
    );
}

fn aux_data_accessible_area(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_indirect_i32(Reg::A1, Reg::A0, 0), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(2_u32.pow(24));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let mut instance = module.instantiate().unwrap();
    instance.set_accessible_aux_size(1).unwrap();
    instance.write_u32(module.memory_map().aux_data_address(), 0x12345678).unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);

    instance.set_reg(Reg::A0, u64::from(module.memory_map().aux_data_address()));
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A1), 0x12345678);

    instance.set_reg(Reg::A0, u64::from(module.memory_map().aux_data_address() + page_size - 4));
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    assert!(instance.read_u32(module.memory_map().aux_data_address() + page_size - 4).is_ok());

    instance.set_reg(Reg::A0, u64::from(module.memory_map().aux_data_address() + page_size - 3));
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    assert!(instance.read_u32(module.memory_map().aux_data_address() + page_size - 3).is_err());

    instance.set_accessible_aux_size(page_size + 1).unwrap();

    instance.set_reg(Reg::A0, u64::from(module.memory_map().aux_data_address() + page_size - 3));
    instance.set_next_program_counter(offsets[0]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    assert!(instance.is_memory_accessible(module.memory_map().aux_data_address() + page_size - 3, 4, MemoryProtection::Read));
    assert!(instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 4,
        4,
        MemoryProtection::Read
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 3,
        4,
        MemoryProtection::Read
    ));
    assert!(instance.read_u32(module.memory_map().aux_data_address() + page_size - 3).is_ok());
    assert!(instance
        .read_u32(module.memory_map().aux_data_address() + page_size * 2 - 4)
        .is_ok());
    assert!(instance
        .read_u32(module.memory_map().aux_data_address() + page_size * 2 - 3)
        .is_err());

    assert!(instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size - 3,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 4,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 3,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size - 3, 0)
        .is_ok());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2 - 4, 0)
        .is_ok());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2 - 3, 0)
        .is_err());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2, 0)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size - 3, 4)
        .is_ok());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2 - 4, 4)
        .is_ok());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2 - 3, 4)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2, 4)
        .is_err());

    instance.set_host_side_aux_write_protect(true).unwrap();

    // Still readable as before.
    assert!(instance.is_memory_accessible(module.memory_map().aux_data_address() + page_size - 3, 4, MemoryProtection::Read));
    assert!(instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 4,
        4,
        MemoryProtection::Read
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 3,
        4,
        MemoryProtection::Read
    ));
    assert!(instance.read_u32(module.memory_map().aux_data_address() + page_size - 3).is_ok());
    assert!(instance
        .read_u32(module.memory_map().aux_data_address() + page_size * 2 - 4)
        .is_ok());
    assert!(instance
        .read_u32(module.memory_map().aux_data_address() + page_size * 2 - 3)
        .is_err());

    // Not writable anymore.
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size - 3,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 4,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2 - 3,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(!instance.is_memory_accessible(
        module.memory_map().aux_data_address() + page_size * 2,
        4,
        MemoryProtection::ReadWrite
    ));
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size - 3, 0)
        .is_err());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2 - 4, 0)
        .is_err());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2 - 3, 0)
        .is_err());
    assert!(instance
        .write_u32(module.memory_map().aux_data_address() + page_size * 2, 0)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size - 3, 4)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2 - 4, 4)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2 - 3, 4)
        .is_err());
    assert!(instance
        .zero_memory(module.memory_map().aux_data_address() + page_size * 2, 4)
        .is_err());
}

fn access_memory_from_host(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);
    builder.set_ro_data_size(1);
    builder.set_rw_data_size(1);
    builder.set_stack_size(1);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(1);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map();

    let mut instance = module.instantiate().unwrap();

    let mut page_size_blob = Vec::new();
    let mut page_size_blob_plus_1 = Vec::new();
    page_size_blob.resize(page_size as usize, 1);
    page_size_blob_plus_1.resize(page_size as usize + 1, 1);

    let list = [
        (memory_map.ro_data_range(), true),
        (memory_map.rw_data_range(), false),
        (memory_map.stack_range(), false),
        (memory_map.aux_data_range(), false),
    ];

    for (range, is_read_only) in list {
        log::debug!("Testing host access for range: 0x{:x}-0x{:x}", range.start, range.end);

        // Partial writes should not clobber the memory region, so do the failing writes first.
        assert!(instance.write_memory(range.start - 1, &[1]).is_err());
        assert!(instance.write_memory(range.start + page_size, &[1]).is_err());
        assert!(instance.write_memory(range.start, &page_size_blob_plus_1).is_err());
        assert!(instance.read_memory(range.start, page_size).unwrap().iter().all(|&byte| byte == 0));

        assert_eq!(instance.read_memory(range.start, 1).unwrap(), vec![0]);
        assert_eq!(instance.read_memory(range.start + page_size - 1, 1).unwrap(), vec![0]);
        assert_eq!(instance.read_memory(range.start, page_size).unwrap().len(), page_size as usize);
        assert!(instance.read_memory(range.start - 1, 1).is_err());
        assert!(instance.read_memory(range.start + page_size, 1).is_err());
        assert!(instance.read_memory(range.start, page_size + 1).is_err());

        if is_read_only {
            assert!(instance.write_memory(range.start, &[1]).is_err());
            assert!(instance.write_memory(range.start + page_size - 1, &[1]).is_err());
            assert!(instance.write_memory(range.start, &page_size_blob).is_err());

            assert!(instance.zero_memory(range.start, 1).is_err());
            assert!(instance.zero_memory(range.start + page_size - 1, 1).is_err());
            assert!(instance.zero_memory(range.start, page_size).is_err());
        } else {
            assert!(instance.write_memory(range.start, &[1]).is_ok());
            assert_eq!(instance.read_memory(range.start, 2).unwrap(), vec![1, 0]);
            assert!(instance.write_memory(range.start + page_size - 1, &[1]).is_ok());
            assert!(instance.write_memory(range.start, &page_size_blob).is_ok());
            assert!(instance.read_memory(range.start, page_size).unwrap().iter().all(|&byte| byte == 1));

            assert!(instance.zero_memory(range.start, 1).is_ok());
            assert!(instance.zero_memory(range.start + page_size - 1, 1).is_ok());
            assert!(instance.zero_memory(range.start, page_size).is_ok());
        }

        assert_eq!(instance.read_memory(range.start, 0).unwrap(), vec![]);
    }

    // If length is zero then these should always succeed.
    assert_eq!(instance.read_memory(0, 0).unwrap(), vec![]);
    assert_eq!(instance.read_memory(0xffffffff, 0).unwrap(), vec![]);
    assert!(instance.write_memory(0, &[]).is_ok());
    assert!(instance.write_memory(0xffffffff, &[]).is_ok());
    assert!(instance.zero_memory(0, 0).is_ok());
    assert!(instance.zero_memory(0xffffffff, 0).is_ok());

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);
    builder.set_ro_data_size(4);
    builder.set_ro_data([1, 2, 3, 4].into());
    builder.set_rw_data_size(4);
    builder.set_rw_data([5, 6, 7, 8].into());
    builder.set_stack_size(4096 + 1);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(4096 + 1);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map();

    let mut instance = module.instantiate().unwrap();
    assert_eq!(
        instance.read_memory(memory_map.ro_data_range().start + 2, 4).unwrap(),
        vec![3, 4, 0, 0]
    );
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().start + 2, 4).unwrap(),
        vec![7, 8, 0, 0]
    );
    assert!(instance.write_memory(memory_map.aux_data_range().start, &[9, 10, 11, 12]).is_ok());
    assert_eq!(
        instance.read_memory(memory_map.aux_data_range().start + 2, 4).unwrap(),
        vec![11, 12, 0, 0]
    );

    assert!(instance
        .write_memory(memory_map.aux_data_range().start + 4096 - 2, &[13, 14])
        .is_ok());
    assert_eq!(
        instance.read_memory(memory_map.aux_data_range().start + 4096 - 2, 4).unwrap(),
        vec![13, 14, 0, 0]
    );

    assert!(instance.write_memory(memory_map.stack_range().start + 4096 - 2, &[15, 16]).is_ok());
    assert_eq!(
        instance.read_memory(memory_map.stack_range().start + 4096 - 2, 4).unwrap(),
        vec![15, 16, 0, 0]
    );
}

fn access_memory_from_within(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::ret()], &[]);
    builder.set_ro_data_size(page_size * 3);
    builder.set_rw_data_size(page_size * 3);
    builder.set_stack_size(page_size * 2);
    builder.set_ro_data((0..page_size - 2).map(|_| 0x42_u8).collect());
    builder.set_rw_data((0..page_size - 2).map(|_| 0x73_u8).collect());

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(page_size * 2);

    let memory_map = Module::from_blob(
        &engine,
        &module_config,
        ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap(),
    )
    .unwrap()
    .memory_map()
    .clone();

    let mut spawn = move |code: &[polkavm_common::program::Instruction]| -> crate::RawInstance {
        builder.set_code(code, &[]);
        let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
        let module = Module::from_blob(&engine, &module_config, blob).unwrap();
        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance
    };

    // RO data
    let mut instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address() + page_size * 3 - 3).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address() + page_size * 3 - 4).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address()).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x42424242);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address() + page_size - 6).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x42424242);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address() + page_size - 2).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.ro_data_address() + page_size - 4).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00004242);

    // RW data
    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size * 3 - 3).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size * 3 - 4).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address()).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x73737373);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size - 6).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x73737373);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size - 2).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size - 4).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00007373);

    instance = spawn(&[
        asm::load_imm(Reg::A1, 0x12345678),
        asm::store_u32(Reg::A1, cast(memory_map.rw_data_address() + page_size - 4).bitwise_as_i32()),
        asm::load_u32(Reg::A0, cast(memory_map.rw_data_address() + page_size - 3).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00123456);

    instance = spawn(&[
        asm::store_imm_u32(cast(memory_map.rw_data_address() + page_size * 3 - 3).bitwise_as_i32(), 0),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    // Stack.
    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.stack_address_high() - 4).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.stack_address_high() - 3).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.stack_address_low()).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0);

    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.stack_address_low() - 1).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    instance = spawn(&[
        asm::load_imm(Reg::A1, 0x12345678),
        asm::store_u32(Reg::A1, cast(memory_map.stack_address_low() + page_size - 4).bitwise_as_i32()),
        asm::load_u32(Reg::A0, cast(memory_map.stack_address_low() + page_size - 2).bitwise_as_i32()),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00001234);

    instance = spawn(&[
        asm::store_imm_u32(cast(memory_map.stack_address_low() - 1).bitwise_as_i32(), 0),
        asm::ret(),
    ]);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

    // Aux data.
    instance = spawn(&[
        asm::load_u32(Reg::A0, cast(memory_map.aux_data_address() + page_size - 2).bitwise_as_i32()),
        asm::ret(),
    ]);
    instance
        .write_memory(memory_map.aux_data_address(), &vec![0x23_u8; cast(page_size - 1).to_usize()])
        .unwrap();
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.reg(Reg::A0), 0x00000023);
}

#[test]
fn interpreter_max_allocation_size() {
    let _ = env_logger::try_init();
    let mut config = crate::Config::default();
    config.set_backend(Some(crate::BackendKind::Interpreter));
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_indirect_u32(Reg::A0, 0, 0x12345678), asm::ret()], &[]);
    builder.set_rw_data_size(1024 * 1024 * 512);
    builder.set_stack_size(128 * 1024);
    let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();

    let limit: usize = 1024 * 32;
    let limit_u32 = cast(limit).to_u32_or_panic();
    let mut module_config = ModuleConfig::new();
    module_config.set_aux_data_size(128 * 1024);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map();
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_max_allocation_size(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address()));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    }
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_max_allocation_size(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 - 4));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    }

    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_max_allocation_size(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 - 3));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

        assert!(instance.write_u32(memory_map.rw_data_address() + limit_u32 - 4, 0x77777777).is_ok());
        assert!(matches!(
            instance.write_u32(memory_map.rw_data_address() + limit_u32 - 3, 0x66666666),
            Err(MemoryAccessError::MemoryLimitReached)
        ));
        assert_eq!(instance.read_u32(memory_map.rw_data_address() + limit_u32 - 3).unwrap(), 0x00777777);
    }
}

#[test]
fn interpreter_guest_memory_limit() {
    let _ = env_logger::try_init();
    let mut config = crate::Config::default();
    config.set_backend(Some(crate::BackendKind::Interpreter));
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_indirect_u32(Reg::A0, 0, 0x12345678), asm::ret()], &[]);
    builder.set_rw_data_size(1024 * 1024 * 512);
    builder.set_stack_size(1024 * 1024 * 512);
    let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();

    let limit: usize = 1024 * 32;
    let limit_u32 = cast(limit).to_u32_or_panic();
    let mut module_config = ModuleConfig::new();
    module_config.set_aux_data_size(128 * 1024);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map();
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_max_allocation_size(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 - 4));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    }
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_max_allocation_size(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 - 3));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
    }
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_guest_memory_limit(Some(limit));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);

        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 / 2 - 4));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.stack_address_high() - limit_u32 / 2));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_address() + limit_u32 / 2 - 3));
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);

        assert!(matches!(
            instance.write_u32(memory_map.rw_data_range().end - 4, 0x66666666),
            Err(MemoryAccessError::MemoryLimitReached)
        ));
        assert!(matches!(
            instance.write_u32(memory_map.stack_range().start, 0x66666666),
            Err(MemoryAccessError::MemoryLimitReached)
        ));
        assert!(matches!(
            instance.write_u32(memory_map.aux_data_range().end - 4, 0x66666666),
            Err(MemoryAccessError::MemoryLimitReached)
        ));

        assert_eq!(instance.read_u32(memory_map.rw_data_range().end - 4).unwrap(), 0);
        assert_eq!(instance.read_u32(memory_map.stack_range().end - 4).unwrap(), 0);
        assert_eq!(instance.read_u32(memory_map.aux_data_range().end - 4).unwrap(), 0);
    }
}

#[test]
fn interpreter_stack_size_not_power_of_two() {
    let _ = env_logger::try_init();
    let mut config = crate::Config::default();
    config.set_backend(Some(crate::BackendKind::Interpreter));
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::load_indirect_u64(Reg::A1, Reg::A0, 0),
            asm::store_indirect_u64(Reg::A2, Reg::A0, 0),
            asm::ret(),
        ],
        &[],
    );
    builder.set_ro_data_size(0xffd60000);
    builder.set_stack_size(0x41000);
    let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();

    let mut module_config = ModuleConfig::new();
    module_config.set_aux_data_size(0x100000);
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map();
    let stack_size = memory_map.stack_size();
    let low = memory_map.stack_address_low();
    let heap_address = memory_map.heap_base() + memory_map.max_heap_size() - 8;
    {
        let mut instance = module.instantiate().unwrap();
        assert!(instance.sbrk(memory_map.max_heap_size()).unwrap().is_some());
        instance.write_u64(heap_address, 0x1111).unwrap();
        instance.write_memory(low, &[1]).unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::A0, u64::from(heap_address));
        instance.set_reg(Reg::A2, 0x2222);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.reg(Reg::A1), 0x1111);
        assert_eq!(instance.read_u64(heap_address).unwrap(), 0x2222);
    }
    {
        let mut instance = module.instantiate().unwrap();
        instance.set_interpreter_guest_memory_limit(Some(cast(stack_size).to_usize()));
        assert!(matches!(
            instance.write_memory(low, &[1]),
            Err(MemoryAccessError::MemoryLimitReached)
        ));
        instance.write_memory(memory_map.stack_address_high() - 0x40000, &[1]).unwrap();
    }
}

fn write_read_memory_from_host(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);
    builder.set_rw_data_size(page_size * 32);
    builder.set_stack_size(page_size * 32);

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_aux_data_size(page_size * 32);

    let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let memory_map = module.memory_map().clone();
    let mut instance = module.instantiate().unwrap();

    instance.write_memory(memory_map.rw_data_address() + page_size * 4, &[1]).unwrap();
    instance.write_memory(memory_map.rw_data_address() + page_size, &[2]).unwrap();
    instance.write_memory(memory_map.rw_data_address() + page_size * 16, &[3]).unwrap();
    assert_eq!(
        instance.read_memory(memory_map.rw_data_address() + page_size * 4, 1).unwrap(),
        vec![1]
    );
    assert_eq!(instance.read_memory(memory_map.rw_data_address() + page_size, 1).unwrap(), vec![2]);
    assert_eq!(
        instance.read_memory(memory_map.rw_data_address() + page_size * 16, 1).unwrap(),
        vec![3]
    );

    instance.write_memory(memory_map.aux_data_address() + page_size * 4, &[4]).unwrap();
    instance.write_memory(memory_map.aux_data_address() + page_size, &[5]).unwrap();
    instance.write_memory(memory_map.aux_data_address() + page_size * 16, &[6]).unwrap();
    assert_eq!(
        instance.read_memory(memory_map.aux_data_address() + page_size * 4, 1).unwrap(),
        vec![4]
    );
    assert_eq!(instance.read_memory(memory_map.aux_data_address() + page_size, 1).unwrap(), vec![5]);
    assert_eq!(
        instance.read_memory(memory_map.aux_data_address() + page_size * 16, 1).unwrap(),
        vec![6]
    );

    instance.write_memory(memory_map.stack_address_low() + page_size * 4, &[7]).unwrap();
    instance.write_memory(memory_map.stack_address_low() + page_size, &[8]).unwrap();
    instance
        .write_memory(memory_map.stack_address_low() + page_size * 16, &[9])
        .unwrap();
    assert_eq!(
        instance.read_memory(memory_map.stack_address_low() + page_size * 4, 1).unwrap(),
        vec![7]
    );
    assert_eq!(
        instance.read_memory(memory_map.stack_address_low() + page_size, 1).unwrap(),
        vec![8]
    );
    assert_eq!(
        instance.read_memory(memory_map.stack_address_low() + page_size * 16, 1).unwrap(),
        vec![9]
    );

    let mut instance = module.instantiate().unwrap();
    instance
        .write_memory(memory_map.stack_address_high() - page_size * 4, &[7])
        .unwrap();
    instance.write_memory(memory_map.stack_address_high() - page_size, &[8]).unwrap();
    instance
        .write_memory(memory_map.stack_address_high() - page_size * 16, &[9])
        .unwrap();
    assert_eq!(
        instance.read_memory(memory_map.stack_address_high() - page_size * 4, 1).unwrap(),
        vec![7]
    );
    assert_eq!(
        instance.read_memory(memory_map.stack_address_high() - page_size, 1).unwrap(),
        vec![8]
    );
    assert_eq!(
        instance.read_memory(memory_map.stack_address_high() - page_size * 16, 1).unwrap(),
        vec![9]
    );
}

fn stack_size_not_power_of_two(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"load");
    builder.add_export_by_basic_block(1, b"store");
    builder.set_code(
        &[
            asm::load_indirect_u64(Reg::A1, Reg::A0, 0),
            asm::ret(),
            asm::store_indirect_u64(Reg::A1, Reg::A0, 0),
            asm::ret(),
        ],
        &[],
    );
    builder.set_stack_size(page_size * 24);

    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);

    let blob = ProgramBlob::parse(builder.to_vec().unwrap().into()).unwrap();
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let exports: Vec<_> = module.exports().map(|export| export.program_counter()).collect();
    let low = module.memory_map().stack_address_low();
    let high = module.memory_map().stack_address_high();

    let mut instance = module.instantiate().unwrap();
    let writes = [(high - 1, 1), (high - page_size - 1, 2), (low + page_size, 3), (low, 4)];
    for (address, value) in writes {
        instance.write_memory(address, &[value]).unwrap();
    }
    for (address, value) in writes {
        assert_eq!(instance.read_memory(address, 1).unwrap(), vec![value]);
    }

    assert!(instance.read_memory(low - 1, 1).is_err());
    assert!(instance.write_memory(low - 1, &[1]).is_err());
    for address in [low - 4, low - 8, low - page_size] {
        for &pc in &exports {
            instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
            instance.set_reg(Reg::A0, u64::from(address));
            instance.set_next_program_counter(pc);
            match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        }
    }
}

fn sbrk_knob_works(config: Config, isa: InstructionSetKind) {
    let sbrk_allowed = isa.supports_opcode(Opcode::sbrk);

    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::sbrk(Reg::A0, Reg::A0), asm::ret()], &[]);
    builder.set_ignore_instruction_set_incompatibility(true);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 0);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);

    instance.set_gas(5);
    instance.set_next_program_counter(ProgramCounter(0));

    #[allow(clippy::branches_sharing_code)]
    if sbrk_allowed {
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
        assert_eq!(instance.gas(), 3);
    } else {
        match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
        assert_eq!(instance.program_counter(), Some(ProgramCounter(0)));
        assert_eq!(instance.gas(), 4);
    }
}

struct TestInstance {
    module: crate::Module,
    instance: crate::Instance,
}

impl TestInstance {
    fn new(args: &TestBlobArgs, elf: &'static [u8]) -> Self {
        let _ = env_logger::try_init();
        let blob = get_blob_impl(BlobMapKey {
            optimize: args.optimize,
            strip: false,
            isa: args.isa,
            elf,
            is_permissive: true,
        });

        let engine = Engine::new(&args.config).unwrap();
        let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
        let mut linker = Linker::new();
        linker
            .define_typed("multiply_by_2", |_caller: Caller<()>, value: u32| -> u32 { value * 2 })
            .unwrap();

        linker
            .define_typed("identity", |_caller: Caller<()>, value: u32| -> u32 { value })
            .unwrap();

        linker
            .define_untyped("multiply_all_input_registers", |caller: Caller<()>| {
                let mut value = 1;

                use Reg as R;
                for reg in [R::A0, R::A1, R::A2, R::A3, R::A4, R::A5, R::T0, R::T1, R::T2] {
                    value *= caller.instance.reg(reg);
                }

                caller.instance.set_reg(Reg::A0, value);
                Ok(())
            })
            .unwrap();

        linker
            .define_typed("call_sbrk_indirectly_impl", |caller: Caller<()>, size: u32| -> u32 {
                caller.instance.sbrk(size).unwrap().unwrap_or(0)
            })
            .unwrap();

        linker
            .define_untyped("return_tuple_u32", |caller: Caller<()>| {
                caller.instance.set_reg(Reg::A0, 0x12345678);
                caller.instance.set_reg(Reg::A1, 0x9abcdefe);
                Ok(())
            })
            .unwrap();

        linker
            .define_untyped("return_tuple_u64", |caller: Caller<()>| {
                caller.instance.set_reg(Reg::A0, 0x123456789abcdefe);
                caller.instance.set_reg(Reg::A1, 0x1122334455667788);
                Ok(())
            })
            .unwrap();

        linker
            .define_untyped("return_tuple_usize", move |caller: Caller<()>| {
                if caller.instance.is_64_bit() {
                    caller.instance.set_reg(Reg::A0, 0x123456789abcdefe);
                    caller.instance.set_reg(Reg::A1, 0x1122334455667788);
                } else {
                    caller.instance.set_reg(Reg::A0, 0x12345678);
                    caller.instance.set_reg(Reg::A1, 0x9abcdefe);
                }

                Ok(())
            })
            .unwrap();

        let instance_pre = linker.instantiate_pre(&module).unwrap();
        let instance = instance_pre.instantiate().unwrap();

        TestInstance { module, instance }
    }

    pub fn call<FnArgs, FnResult>(&mut self, name: &str, args: FnArgs) -> Result<FnResult, crate::CallError>
    where
        FnArgs: crate::linker::FuncArgs,
        FnResult: crate::linker::FuncResult,
    {
        self.instance.call_typed_and_get_result::<FnResult, FnArgs>(&mut (), name, args)
    }
}

struct TestBlobArgs {
    config: Config,
    isa: InstructionSetKind,
    optimize: bool,
    is_64_bit: bool,
    is_cdylib: bool,
}

impl TestBlobArgs {
    fn get_test_program(&self) -> &'static [u8] {
        get_test_program(
            if self.is_cdylib {
                TestProgram::TestBlobSo
            } else {
                TestProgram::TestBlob
            },
            self.is_64_bit,
        )
    }
}

fn test_blob_basic_test(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u32>("push_one_to_global_vec", ()).unwrap(), 1);
    assert_eq!(i.call::<(), u32>("push_one_to_global_vec", ()).unwrap(), 2);
    assert_eq!(i.call::<(), u32>("push_one_to_global_vec", ()).unwrap(), 3);
}

fn test_blob_atomic_fetch_add(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (1,)).unwrap(), 0);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (1,)).unwrap(), 1);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (1,)).unwrap(), 2);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (0,)).unwrap(), 3);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (0,)).unwrap(), 3);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (2,)).unwrap(), 3);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_add", (0,)).unwrap(), 5);
}

fn test_blob_atomic_fetch_swap(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_swap", (10,)).unwrap(), 0);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_swap", (100,)).unwrap(), 10);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_swap", (1000,)).unwrap(), 100);

    assert_eq!(i.call::<(), u32>("atomic_fetch_swap_with_zero", ()).unwrap(), 1000);
    assert_eq!(i.call::<(u32,), u32>("atomic_fetch_swap", (100,)).unwrap(), 0);
}

fn test_blob_atomic_fetch_minmax(args: TestBlobArgs) {
    use core::cmp::{max, min};

    fn maxu(a: i32, b: i32) -> i32 {
        max(a as u32, b as u32) as i32
    }

    fn minu(a: i32, b: i32) -> i32 {
        min(a as u32, b as u32) as i32
    }

    #[allow(clippy::type_complexity)]
    let list: [(&str, fn(i32, i32) -> i32); 4] = [
        ("atomic_fetch_max_signed", max),
        ("atomic_fetch_min_signed", min),
        ("atomic_fetch_max_unsigned", maxu),
        ("atomic_fetch_min_unsigned", minu),
    ];

    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    for (name, cb) in list {
        for a in [-10, 0, 10] {
            for b in [-10, 0, 10] {
                let new_value = cb(a, b);
                i.call::<(i32,), ()>("set_global", (a,)).unwrap();
                assert_eq!(i.call::<(i32,), i32>(name, (b,)).unwrap(), a);
                assert_eq!(i.call::<(), i32>("get_global", ()).unwrap(), new_value);
            }
        }
    }
}

fn test_blob_hostcall(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u32,), u32>("test_multiply_by_6", (10,)).unwrap(), 60);
}

fn test_blob_define_abi(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert!(i.call::<(), ()>("test_define_abi", ()).is_ok());
}

fn test_blob_input_registers(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert!(i.call::<(), ()>("test_input_registers", ()).is_ok());
}

fn test_blob_call_sbrk_from_guest(args: TestBlobArgs) {
    if !args.isa.supports_opcode(Opcode::sbrk) {
        return;
    }

    test_blob_call_sbrk_impl(args, |i, size| i.call::<(u32,), u32>("call_sbrk", (size,)).unwrap())
}

fn test_blob_call_sbrk_from_host_instance(args: TestBlobArgs) {
    if !args.isa.supports_opcode(Opcode::sbrk) {
        return;
    }

    test_blob_call_sbrk_impl(args, |i, size| i.instance.sbrk(size).unwrap().unwrap_or(0))
}

fn test_blob_call_sbrk_from_host_function(args: TestBlobArgs) {
    if !args.isa.supports_opcode(Opcode::sbrk) {
        return;
    }

    test_blob_call_sbrk_impl(args, |i, size| i.call::<(u32,), u32>("call_sbrk_indirectly", (size,)).unwrap())
}

fn test_blob_program_memory_can_be_reused_and_cleared(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let address = i.call::<(), u32>("get_global_address", ()).unwrap();

    assert_eq!(i.instance.read_memory(address, 4).unwrap(), [0x00, 0x00, 0x00, 0x00]);

    i.call::<(), ()>("increment_global", ()).unwrap();
    assert_eq!(i.instance.read_memory(address, 4).unwrap(), [0x01, 0x00, 0x00, 0x00]);

    i.call::<(), ()>("increment_global", ()).unwrap();
    assert_eq!(i.instance.read_memory(address, 4).unwrap(), [0x02, 0x00, 0x00, 0x00]);

    i.instance.reset_memory().unwrap();
    assert_eq!(i.instance.read_memory(address, 4).unwrap(), [0x00, 0x00, 0x00, 0x00]);

    i.call::<(), ()>("increment_global", ()).unwrap();
    assert_eq!(i.instance.read_memory(address, 4).unwrap(), [0x01, 0x00, 0x00, 0x00]);
}

fn test_blob_out_of_bounds_memory_access_generates_a_trap(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let address = i.call::<(), u32>("get_global_address", ()).unwrap();
    assert_eq!(i.call::<(u32,), u32>("read_u32", (address,)).unwrap(), 0);
    i.call::<(), ()>("increment_global", ()).unwrap();
    assert_eq!(i.call::<(u32,), u32>("read_u32", (address,)).unwrap(), 1);
    assert!(matches!(i.call::<(u32,), u32>("read_u32", (4,)), Err(CallError::Trap)));

    assert_eq!(i.call::<(u32,), u32>("read_u32", (address,)).unwrap(), 1);
    i.call::<(), ()>("increment_global", ()).unwrap();
    assert_eq!(i.call::<(u32,), u32>("read_u32", (address,)).unwrap(), 2);
}

fn test_blob_call_sbrk_impl(args: TestBlobArgs, mut call_sbrk: impl FnMut(&mut TestInstance, u32) -> u32) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let memory_map = i.module.memory_map().clone();
    let heap_base = memory_map.heap_base();
    let page_size = memory_map.page_size();

    assert_eq!(i.instance.read_memory(memory_map.rw_data_range().end - 1, 1).unwrap(), vec![0]);
    assert!(i.instance.read_memory(memory_map.rw_data_range().end, 1).is_err());
    assert!(i
        .instance
        .read_memory(heap_base, memory_map.rw_data_range().end - heap_base)
        .unwrap()
        .iter()
        .all(|&byte| byte == 0));
    assert_eq!(i.instance.heap_size(), 0);

    assert_eq!(call_sbrk(&mut i, 0), heap_base);
    assert_eq!(i.instance.heap_size(), 0);
    assert_eq!(call_sbrk(&mut i, 0), heap_base);
    assert_eq!(call_sbrk(&mut i, 1), heap_base + 1);
    assert_eq!(i.instance.heap_size(), 1);
    assert_eq!(call_sbrk(&mut i, 0), heap_base + 1);
    assert_eq!(call_sbrk(&mut i, 0xffffffff), 0);
    assert_eq!(call_sbrk(&mut i, 0), heap_base + 1);

    i.instance.write_memory(heap_base, &[0x33]).unwrap();
    assert_eq!(i.instance.read_memory(heap_base, 1).unwrap(), vec![0x33]);

    let new_origin = align_to_next_page_u32(memory_map.page_size(), heap_base + i.instance.heap_size()).unwrap();
    {
        let until_next_page = new_origin - (heap_base + i.instance.heap_size());
        assert_eq!(call_sbrk(&mut i, until_next_page), new_origin);
    }

    assert_eq!(i.instance.read_memory(new_origin - 1, 1).unwrap(), vec![0]);
    assert!(i.instance.read_memory(new_origin, 1).is_err());
    assert!(i.instance.write_memory(new_origin, &[0x34]).is_err());

    assert_eq!(call_sbrk(&mut i, 1), new_origin + 1);
    assert_eq!(i.instance.read_memory(new_origin, page_size).unwrap().len(), page_size as usize);
    assert!(i.instance.read_memory(new_origin, page_size + 1).is_err());
    assert!(i.instance.write_memory(new_origin, &[0x35]).is_ok());

    assert_eq!(call_sbrk(&mut i, page_size - 1), new_origin + page_size);
    assert!(i.instance.read_memory(new_origin, page_size + 1).is_err());

    i.instance.reset_memory().unwrap();
    assert_eq!(call_sbrk(&mut i, 0), heap_base);
    assert_eq!(i.instance.heap_size(), 0);
    assert!(i.instance.read_memory(memory_map.rw_data_range().end, 1).is_err());

    assert_eq!(call_sbrk(&mut i, 1), heap_base + 1);
    assert_eq!(i.instance.read_memory(heap_base, 1).unwrap(), vec![0]);
}

fn test_blob_add_u32(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u32, u32), u32>("add_u32", (1, 2,)).unwrap(), 3);
    assert_eq!(i.instance.reg(Reg::A0), 3);

    assert_eq!(i.call::<(u32, u32), u32>("add_u32", (0xfffffffa, 2,)).unwrap(), 0xfffffffc);
    assert_eq!(i.instance.reg(Reg::A0), 0xfffffffc);

    assert_eq!(i.call::<(u32, u32), u32>("add_u32", (0xffffffff, 2,)).unwrap(), 1);
    assert_eq!(i.instance.reg(Reg::A0), 1);

    if args.is_64_bit {
        assert_eq!(i.call::<(u32, u32), u32>("add_u32_asm", (0xfffffffa, 2,)).unwrap(), 0xfffffffc);
        assert_eq!(i.instance.reg(Reg::A0), 0xfffffffffffffffc);
    }
}

fn test_blob_add_u64(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u64, u64), u64>("add_u64", (1, 2,)).unwrap(), 3);
    assert_eq!(i.instance.reg(Reg::A0), 3);
    assert_eq!(
        i.call::<(u64, u64), u64>("add_u64", (0xaaaaaaaa, 0xcccccccc,)).unwrap(),
        0xaaaaaaaa + 0xcccccccc
    );
}

fn test_blob_xor_imm_u32(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    for value in [0, 0xaaaaaaaa, 0x55555555, 0x12345678, 0xffffffff] {
        assert_eq!(i.call::<(u32,), u32>("xor_imm_u32", (value,)).unwrap(), value ^ 0xfb8f5c1e);
    }
}

fn test_blob_branch_less_than_zero(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    i.call::<(), ()>("test_branch_less_than_zero", ()).unwrap();
}

fn test_blob_fetch_add_atomic_u64(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u64,), u64>("fetch_add_atomic_u64", (1,)).unwrap(), 0);
    assert_eq!(i.call::<(u64,), u64>("fetch_add_atomic_u64", (0,)).unwrap(), 1);
    assert_eq!(i.call::<(u64,), u64>("fetch_add_atomic_u64", (0,)).unwrap(), 1);
    assert_eq!(i.call::<(u64,), u64>("fetch_add_atomic_u64", (0xffffffff,)).unwrap(), 1);
    assert_eq!(i.call::<(u64,), u64>("fetch_add_atomic_u64", (0,)).unwrap(), 0x100000000);
}

fn test_blob_cmov_if_zero_with_zero_reg(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    i.call::<(), ()>("cmov_if_zero_with_zero_reg", ()).unwrap();
}

fn test_blob_cmov_if_not_zero_with_zero_reg(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    i.call::<(), ()>("cmov_if_not_zero_with_zero_reg", ()).unwrap();
}

fn test_blob_min_stack_size(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let i = TestInstance::new(&args, elf);
    assert_eq!(i.instance.module().memory_map().stack_size(), 65536);
}

fn test_blob_negate_and_add(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    if !args.is_64_bit {
        assert_eq!(i.call::<(u32, u32), u32>("negate_and_add", (123, 1,)).unwrap(), 15);
    } else {
        assert_eq!(i.call::<(u64, u64), u64>("negate_and_add", (123, 1,)).unwrap(), 15);
    }
}

fn test_blob_return_tuple_from_import(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    i.call::<(), ()>("test_return_tuple", ()).unwrap();
}

fn test_blob_return_tuple_from_export(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    if args.is_64_bit {
        let a0 = 0x123456789abcdefe_u64;
        let a1 = 0x1122334455667788_u64;
        i.call::<(), ()>("export_return_tuple_u64", ()).unwrap();
        assert_eq!(i.instance.reg(Reg::A0), a0);
        assert_eq!(i.instance.reg(Reg::A1), a1);

        i.instance.set_reg(Reg::A0, 0);
        i.instance.set_reg(Reg::A1, 0);
        i.call::<(), ()>("export_return_tuple_usize", ()).unwrap();
        assert_eq!(i.instance.reg(Reg::A0), a0);
        assert_eq!(i.instance.reg(Reg::A1), a1);
    } else {
        let a0 = 0x12345678_u64;
        let a1 = 0x9abcdefe_u64;
        i.call::<(), ()>("export_return_tuple_u32", ()).unwrap();
        assert_eq!(i.instance.reg(Reg::A0), a0);
        assert_eq!(i.instance.reg(Reg::A1), a1);

        i.instance.set_reg(Reg::A0, 0);
        i.instance.set_reg(Reg::A1, 0);
        i.call::<(), ()>("export_return_tuple_usize", ()).unwrap();
        assert_eq!(i.instance.reg(Reg::A0), a0);
        assert_eq!(i.instance.reg(Reg::A1), a1);
    }
}

fn test_blob_get_heap_base(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let heap_base = i.call::<(), u32>("get_heap_base", ()).unwrap();
    assert_eq!(heap_base, i.instance.module().memory_map().heap_base());
}

fn test_blob_get_self_address(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let addr = i.call::<(), u32>("get_self_address", ()).unwrap();
    assert_ne!(addr, 0);
}

fn test_blob_get_self_address_naked(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    let addr = i.call::<(), u32>("get_self_address_naked", ()).unwrap();
    assert_ne!(addr, 0);
}

fn test_blob_sub_i32_min_64(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u64,), u64>("sub_i32_min_64", (0,)).unwrap(), 0x80000000);
}

fn test_blob_sub_i32_min_32(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(u64,), u64>("sub_i32_min_32", (0,)).unwrap(), 0xffffffff80000000);
}

fn test_blob_orn_zero_const_64(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u64>("orn_zero_const_64", ()).unwrap(), 0x80000001);
}

fn test_blob_xnor_zero_const_64(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u64>("xnor_zero_const_64", ()).unwrap(), 0x80000001);
}

fn test_blob_min_zero_const_64(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u64>("min_zero_const_64", ()).unwrap(), 0xffffffff7ffffffe);
}

fn test_blob_max_zero_const_64(args: TestBlobArgs) {
    if !args.is_64_bit {
        return;
    }
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u64>("max_zero_const_64", ()).unwrap(), 0);
}

fn test_blob_jump_table_with_constant_index(args: TestBlobArgs) {
    let elf = args.get_test_program();
    let mut i = TestInstance::new(&args, elf);
    assert_eq!(i.call::<(), u32>("jump_table_with_constant_index", ()).unwrap(), 2);
}

fn test_asm_reloc_add_sub(config: Config, isa: InstructionSetKind, optimize: bool) {
    let args = TestBlobArgs {
        config,
        isa,
        optimize,
        is_64_bit: true,
        is_cdylib: false,
    };
    const BLOB_64: &[u8] = include_bytes!("../../../guest-programs/asm-tests/output/reloc_add_sub_64.elf");

    let elf = BLOB_64;
    let mut i = TestInstance::new(&args, elf);

    let address = i.call::<(u32,), u32>("get_string", (0,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0x01010101);

    let address = i.call::<(u32,), u32>("get_string", (1,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0x02020202);

    let address = i.call::<(u32,), u32>("get_string", (2,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0x03030303);
}

fn test_asm_reloc_hi_lo(config: Config, isa: InstructionSetKind, optimize: bool) {
    let args = TestBlobArgs {
        config,
        isa,
        optimize,
        is_64_bit: true,
        is_cdylib: false,
    };
    const BLOB_64: &[u8] = include_bytes!("../../../guest-programs/asm-tests/output/reloc_hi_lo_64.elf");

    let elf = BLOB_64;
    let mut i = TestInstance::new(&args, elf);

    let address = i.call::<(u32,), u32>("get_string", (0,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0xA1010101);

    let address = i.call::<(u32,), u32>("get_string", (1,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0xB2020202);

    let address = i.call::<(u32,), u32>("get_string", (2,)).unwrap();
    assert_eq!(i.instance.read_u32(address).unwrap(), 0xC3030303);
}

fn basic_gas_metering(config: Config, isa: InstructionSetKind, gas_metering_kind: GasMeteringKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::fallthrough(), asm::add_imm_32(A0, A0, 666), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(gas_metering_kind));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    {
        instance.set_gas(3);
        instance.call_typed(&mut (), "main", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 0);
        assert_eq!(instance.program_counter(), None);
        assert_eq!(instance.next_program_counter(), None);
    }

    {
        instance.set_gas(2);
        let result = instance.call_typed(&mut (), "main", ());
        assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
        match gas_metering_kind {
            GasMeteringKind::Sync => {
                assert_eq!(instance.gas(), 1);
                assert_eq!(instance.get_result_typed::<i32>(), 0);
                assert_eq!(instance.program_counter(), Some(ProgramCounter(1)));
                assert_eq!(instance.next_program_counter(), Some(ProgramCounter(1)));

                let result = instance.run().unwrap();
                assert!(matches!(result, InterruptKind::NotEnoughGas), "unexpected result: {result:?}");
                assert_eq!(instance.gas(), 1);
                assert_eq!(instance.program_counter(), Some(ProgramCounter(1)));
                assert_eq!(instance.next_program_counter(), Some(ProgramCounter(1)));

                instance.set_gas(2);
                let result = instance.run().unwrap();
                assert!(matches!(result, InterruptKind::Finished), "unexpected result: {result:?}");
                assert_eq!(instance.get_result_typed::<i32>(), 666);
                assert_eq!(instance.gas(), 0);
                assert_eq!(instance.program_counter(), None);
                assert_eq!(instance.next_program_counter(), None);
            }
            GasMeteringKind::Async => {
                assert!(instance.gas() < 0);
                assert_eq!(instance.program_counter(), None);
                assert_eq!(instance.next_program_counter(), None);
            }
        }
    }

    {
        instance.set_gas(6);
        instance.call_typed(&mut (), "main", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 3);
        assert_eq!(instance.program_counter(), None);
        assert_eq!(instance.next_program_counter(), None);

        instance.call_typed(&mut (), "main", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 0);
        assert_eq!(instance.program_counter(), None);
        assert_eq!(instance.next_program_counter(), None);

        let result = instance.call_typed(&mut (), "main", ());
        assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
        match gas_metering_kind {
            GasMeteringKind::Sync => {
                assert_eq!(instance.gas(), 0);
            }
            GasMeteringKind::Async => {
                assert!(instance.gas() < 0);
            }
        }
    }

    {
        core::mem::drop(instance);
        let mut instance = instance_pre.instantiate().unwrap();
        assert_eq!(instance.gas(), 0);

        let result = instance.call_typed(&mut (), "main", ());
        assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
        match gas_metering_kind {
            GasMeteringKind::Sync => {
                assert_eq!(instance.gas(), 0);
            }
            GasMeteringKind::Async => {
                assert!(instance.gas() < 0);
            }
        }
    }

    // Stress test.
    let mut instance = instance_pre.instantiate().unwrap();
    for _ in 0..100 {
        instance.set_gas(2);
        let result = instance.call_typed(&mut (), "main", ());
        assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
        match gas_metering_kind {
            GasMeteringKind::Sync => {
                assert_eq!(instance.get_result_typed::<i32>(), 0);
                assert_eq!(instance.gas(), 1);
            }
            GasMeteringKind::Async => {
                assert!(instance.gas() < 0);
            }
        }

        instance.set_gas(5);
        instance.call_typed(&mut (), "main", ()).unwrap();
        assert_eq!(instance.gas(), 2);
        assert_eq!(instance.get_result_typed::<i32>(), 666);
    }
}

fn basic_gas_metering_sync(config: Config, isa: InstructionSetKind) {
    basic_gas_metering(config, isa, GasMeteringKind::Sync);
}

fn basic_gas_metering_async(config: Config, isa: InstructionSetKind) {
    basic_gas_metering(config, isa, GasMeteringKind::Async);
}

#[test]
fn per_instruction_gas_metering() {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::add_imm_32(A0, A0, 1),
            asm::add_imm_32(A0, A0, 2),
            asm::add_imm_32(A0, A0, 4),
            asm::add_imm_32(A0, A0, 8),
            asm::add_imm_32(A0, A0, 16),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();

    let mut config = Config::default();
    config.set_backend(Some(crate::BackendKind::Interpreter));

    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_per_instruction_metering(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    instance.set_gas(1);
    let result = instance.call_typed(&mut (), "main", ());
    assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
    assert_eq!(instance.reg(Reg::A0), 1);
    assert_eq!(instance.gas(), 0);
    assert_eq!(instance.program_counter(), Some(offsets[1]));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));

    instance.set_gas(1);
    let result = instance.run().unwrap();
    assert!(matches!(result, InterruptKind::NotEnoughGas), "unexpected result: {result:?}");
    assert_eq!(instance.reg(Reg::A0), 1 + 2);
    assert_eq!(instance.gas(), 0);
    assert_eq!(instance.program_counter(), Some(offsets[2]));
    assert_eq!(instance.next_program_counter(), Some(offsets[2]));

    instance.set_gas(2);
    let result = instance.run().unwrap();
    assert!(matches!(result, InterruptKind::NotEnoughGas), "unexpected result: {result:?}");
    assert_eq!(instance.reg(Reg::A0), 1 + 2 + 4 + 8);
    assert_eq!(instance.gas(), 0);
    assert_eq!(instance.program_counter(), Some(offsets[4]));
    assert_eq!(instance.next_program_counter(), Some(offsets[4]));
}

fn consume_gas_in_host_function(config: Config, isa: InstructionSetKind, gas_metering_kind: GasMeteringKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.add_import(b"hostfn");
    builder.set_code(&[asm::ecalli(0), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(gas_metering_kind));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let mut linker: Linker<i64, core::convert::Infallible> = Linker::new();
    linker
        .define_typed("hostfn", |caller: Caller<i64>| -> u32 {
            assert_eq!(caller.instance.gas(), 1);
            caller.instance.set_gas(1 - *caller.user_data);
            666
        })
        .unwrap();

    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    {
        instance.set_gas(3);
        instance.call_typed(&mut 0, "main", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 1);
    }

    {
        instance.set_gas(3);
        instance.call_typed(&mut 1, "main", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 0);
    }

    {
        instance.set_gas(3);
        let result = instance.call_typed(&mut 2, "main", ());
        assert_eq!(instance.gas(), -1);
        assert!(matches!(result, Err(CallError::NotEnoughGas)), "unexpected result: {result:?}");
    }
}

fn consume_gas_in_host_function_sync(config: Config, isa: InstructionSetKind) {
    consume_gas_in_host_function(config, isa, GasMeteringKind::Sync);
}

fn consume_gas_in_host_function_async(config: Config, isa: InstructionSetKind) {
    consume_gas_in_host_function(config, isa, GasMeteringKind::Async);
}

fn gas_metering_with_more_than_one_basic_block(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"export_1");
    builder.add_export_by_basic_block(1, b"export_2");
    builder.set_code(
        &[
            asm::add_imm_32(A0, A0, 666),
            asm::ret(),
            asm::add_imm_32(A0, A0, 666),
            asm::add_imm_32(A0, A0, 100),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    {
        instance.set_gas(10);
        instance.call_typed(&mut (), "export_1", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 666);
        assert_eq!(instance.gas(), 8);
    }

    {
        instance.set_gas(10);
        instance.call_typed(&mut (), "export_2", ()).unwrap();
        assert_eq!(instance.get_result_typed::<i32>(), 766);
        assert_eq!(instance.gas(), 7);
    }
}

fn gas_metering_with_implicit_trap(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::add_imm_32(A0, A0, 666)], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    instance.set_gas(10);
    assert!(matches!(instance.call_typed(&mut (), "main", ()).unwrap_err(), CallError::Trap));
    assert_eq!(instance.get_result_typed::<i32>(), 666);
    assert_eq!(instance.gas(), 8);
}

fn gas_gets_charged_when_jumping_in_the_middle_of_a_basic_block(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::add_imm_32(A0, A0, 1),
            asm::add_imm_32(A0, A0, 2),
            asm::ecalli(0),
            asm::add_imm_32(A0, A0, 4),
            asm::add_imm_32(A0, A0, 8),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let offsets: Vec<_> = blob.instructions().map(|inst| inst.offset).collect();

    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let initial_gas = 100;
    let expected_gas_cost = 6;

    // Execute from the first instruction.
    let mut instance = module.instantiate().unwrap();
    instance.set_gas(initial_gas);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Ecalli(0)));
    let final_gas = instance.gas();
    assert_eq!(final_gas, initial_gas - expected_gas_cost);
    assert_eq!(instance.reg(Reg::A0), 1 + 2);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 1 + 2 + 4 + 8);

    // Execute from the second instruction.
    instance.set_gas(initial_gas);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[1]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Ecalli(0)));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 2);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 2 + 4 + 8);

    // Execute from the third instruction.
    instance.set_gas(initial_gas);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[2]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Ecalli(0)));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 0);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 4 + 8);

    // Execute from the first instruction, but set the program counter after we've been interrupted.
    instance.set_gas(initial_gas);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[0]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Ecalli(0)));
    assert_eq!(instance.gas(), final_gas);
    assert_eq!(instance.reg(Reg::A0), 1 + 2);
    assert_eq!(instance.next_program_counter(), Some(offsets[3]));
    instance.set_next_program_counter(offsets[3]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.gas(), initial_gas - (initial_gas - final_gas) * 2); // Charged again, since we've reset the PC.
    assert_eq!(instance.reg(Reg::A0), 1 + 2 + 4 + 8);

    // Execute from the second instruction, but without enough gas.
    instance.set_gas(0);
    instance.set_reg(Reg::A0, 0);
    instance.set_next_program_counter(offsets[1]);
    assert!(matches!(instance.run().unwrap(), InterruptKind::NotEnoughGas));
    assert_eq!(instance.next_program_counter(), Some(offsets[1]));

    instance.set_gas(initial_gas - final_gas);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Ecalli(0)));
    assert_eq!(instance.gas(), 0);
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.gas(), 0);
    assert_eq!(instance.reg(Reg::A0), 2 + 4 + 8);
}

fn trapping_preserves_all_registers_normal_trap(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::default(), blob).unwrap();
    let mut instance = module.instantiate().unwrap();
    instance.set_next_program_counter(ProgramCounter(0));
    for (index, reg) in Reg::ALL.into_iter().enumerate() {
        instance.set_reg(reg, index as u64 + 0x100);
    }
    assert_eq!(instance.run().unwrap(), InterruptKind::Trap);
    for (index, reg) in Reg::ALL.into_iter().enumerate() {
        assert_eq!(instance.reg(reg), index as u64 + 0x100);
    }
}

fn trapping_preserves_all_registers_segfault(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_u32(0, 0x12345678), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::default(), blob).unwrap();
    let mut instance = module.instantiate().unwrap();
    instance.set_next_program_counter(ProgramCounter(0));
    for (index, reg) in Reg::ALL.into_iter().enumerate() {
        instance.set_reg(reg, index as u64 + 0x100);
    }
    assert_eq!(instance.run().unwrap(), InterruptKind::Trap);
    for (index, reg) in Reg::ALL.into_iter().enumerate() {
        assert_eq!(instance.reg(reg), index as u64 + 0x100, "mismatch for register {reg}");
    }
}

fn memset_basic(config: Config, isa: InstructionSetKind) {
    if !isa.supports_opcode(Opcode::memset) {
        return;
    }

    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_ro_data_size(1);
    builder.set_rw_data_size(1);
    builder.set_ro_data(vec![0x00]);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::memset(), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let mut module_config = ModuleConfig::default();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let memory_map = module.memory_map();
    let linker: Linker = Linker::new();
    let instance_pre = linker.instantiate_pre(&module).unwrap();
    let mut instance = instance_pre.instantiate().unwrap();

    instance.set_gas(100);

    // Write near the start of RW data.
    instance
        .call_typed(&mut (), "main", (memory_map.rw_data_address() + 1, 0x1234567a, 3))
        .unwrap();
    assert_eq!(
        instance.read_memory(memory_map.rw_data_address(), 5).unwrap(),
        vec![0, 0x7a, 0x7a, 0x7a, 0]
    );
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_address() + 4)); // Pointer is incremented.
    assert_eq!(instance.reg(Reg::A1), 0x1234567a); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 0); // Count is zeroed.

    // Write at the end of RW data.
    instance
        .call_typed(&mut (), "main", (memory_map.rw_data_range().end - 3, 0x1234567b, 3))
        .unwrap();
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().end - 4, 4).unwrap(),
        vec![0, 0x7b, 0x7b, 0x7b]
    );
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().end)); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x1234567b); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 0); // Count is zeroed.

    // Write going out of bounds (partial write).
    instance.set_gas(100);
    assert!(matches!(
        instance
            .call_typed(&mut (), "main", (memory_map.rw_data_range().end - 3, 0x1234567c, 10))
            .unwrap_err(),
        CallError::Trap
    ));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().end)); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x1234567c); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 7); // Count is partially decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().end - 4, 4).unwrap(),
        vec![0, 0x7c, 0x7c, 0x7c]
    );
    assert_eq!(instance.gas(), 95);

    // Write out of bounds (empty write).
    assert!(matches!(
        instance
            .call_typed(&mut (), "main", (memory_map.rw_data_range().end + 2, 0x1234567d, 10))
            .unwrap_err(),
        CallError::Trap
    ));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().end + 2)); // Pointer is unchanged.
    assert_eq!(instance.reg(Reg::A1), 0x1234567d); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 10); // Count is unchanged.

    // Gas-limited write.
    instance.zero_memory(memory_map.rw_data_address(), 10).unwrap();
    instance.set_gas(5);
    assert!(matches!(
        instance
            .call_typed(&mut (), "main", (memory_map.rw_data_address(), 0x1234567e, 100))
            .unwrap_err(),
        CallError::NotEnoughGas
    ));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_address()) + 3); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x1234567e); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 97); // Count is partially decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_address(), 5).unwrap(),
        vec![0x7e, 0x7e, 0x7e, 0, 0]
    );
    assert_eq!(instance.program_counter(), Some(offsets[0]));
    assert_eq!(instance.next_program_counter(), Some(offsets[0]));

    // Continue gas-limited write.
    instance.set_gas(1);
    instance.set_reg(Reg::A1, 0x1234567f);
    assert_eq!(instance.run().unwrap(), InterruptKind::NotEnoughGas);
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_address()) + 4); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x1234567f); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 96); // Count is partially decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_address(), 5).unwrap(),
        vec![0x7e, 0x7e, 0x7e, 0x7f, 0]
    );
    assert_eq!(instance.program_counter(), Some(offsets[0]));
    assert_eq!(instance.next_program_counter(), Some(offsets[0]));

    // Write out of bounds during a gas-limited write.
    instance.set_gas(50);
    assert!(matches!(
        instance
            .call_typed(&mut (), "main", (memory_map.rw_data_range().end - 3, 0x12345680, 100))
            .unwrap_err(),
        CallError::Trap
    ));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().end)); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x12345680); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 97); // Count is partially decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().end - 4, 4).unwrap(),
        vec![0, 0x80, 0x80, 0x80]
    );
    assert_eq!(instance.gas(), 45);
}

fn memset_with_dynamic_paging(mut config: Config, isa: InstructionSetKind) {
    if !isa.supports_opcode(Opcode::memset) {
        return;
    }

    let _ = env_logger::try_init();

    config.set_allow_dynamic_paging(true);

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::memset(), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut module_config = ModuleConfig::new();
    module_config.set_page_size(page_size);
    module_config.set_dynamic_paging(true);
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));

    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let offsets: Vec<_> = module.blob().instructions().map(|inst| inst.offset).collect();

    let memory_map = module.memory_map();

    let mut instance = module.instantiate().unwrap();
    instance.set_gas(100);
    instance.prepare_call_typed(offsets[0], (memory_map.rw_data_range().start, 0x1234567a, 3));
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, memory_map.rw_data_range().start);
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start)); // Pointer is unchanged.
    assert_eq!(instance.reg(Reg::A1), 0x1234567a); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 3); // Count is unchanged.
    assert_eq!(instance.program_counter(), Some(offsets[0]));
    assert_eq!(instance.next_program_counter(), Some(offsets[0]));
    assert_eq!(instance.gas(), 98);
    instance
        .zero_memory_with_memory_protection(segfault.page_address, segfault.page_size, MemoryProtection::ReadWrite)
        .unwrap();
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start + 3)); // Pointer is at the end.
    assert_eq!(instance.reg(Reg::A1), 0x1234567a); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 0); // Count is decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().start, 5).unwrap(),
        vec![0x7a, 0x7a, 0x7a, 0, 0]
    );
    assert_eq!(instance.gas(), 95);

    let mut instance = module.instantiate().unwrap();
    instance
        .zero_memory_with_memory_protection(memory_map.rw_data_range().start, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    instance.set_gas(100);
    instance.prepare_call_typed(offsets[0], (memory_map.rw_data_range().start + page_size - 1, 0x1234567a, 4));
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, memory_map.rw_data_range().start + page_size);
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start + page_size)); // Pointer is incremented.
    assert_eq!(instance.reg(Reg::A1), 0x1234567a); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 3); // Count is decremented.
    assert_eq!(instance.program_counter(), Some(offsets[0]));
    assert_eq!(instance.next_program_counter(), Some(offsets[0]));
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().start + page_size - 2, 2).unwrap(),
        vec![0, 0x7a]
    );
    assert_eq!(instance.gas(), 97);
    // Change everything mid-flight.
    instance.set_reg(Reg::A0, u64::from(memory_map.rw_data_range().start + page_size + 1));
    instance.set_reg(Reg::A1, 0x1234567b);
    instance.set_reg(Reg::A2, 2);
    instance
        .zero_memory_with_memory_protection(memory_map.rw_data_range().start + page_size, page_size, MemoryProtection::ReadWrite)
        .unwrap();
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start + page_size + 3)); // Pointer is incremented.
    assert_eq!(instance.reg(Reg::A1), 0x1234567b); // Value is unchanged.
    assert_eq!(instance.reg(Reg::A2), 0); // Count is decremented.
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().start + page_size - 2, 6).unwrap(),
        vec![0, 0x7a, 0, 0x7b, 0x7b, 0]
    );
    assert_eq!(instance.gas(), 95);

    let mut instance = module.instantiate().unwrap();
    instance.set_gas(100);
    instance.set_reg(Reg::A0, 0xffffffff_00000000 | u64::from(memory_map.rw_data_range().start));
    instance.set_reg(Reg::A1, 0x1234567a);
    instance.set_reg(Reg::A2, 3);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(offsets[0]);
    let segfault = expect_segfault(instance.run().unwrap());
    assert_eq!(segfault.page_address, memory_map.rw_data_range().start);
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start));
    instance
        .zero_memory_with_memory_protection(segfault.page_address, segfault.page_size, MemoryProtection::ReadWrite)
        .unwrap();
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), u64::from(memory_map.rw_data_range().start + 3));
    assert_eq!(instance.reg(Reg::A2), 0);
    assert_eq!(
        instance.read_memory(memory_map.rw_data_range().start, 4).unwrap(),
        vec![0x7a, 0x7a, 0x7a, 0]
    );
}

fn memset_truncates_a0_and_a2(config: Config, isa: InstructionSetKind) {
    if isa != InstructionSetKind::Latest64 {
        return;
    }
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::memset(), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 0x0000000100000000);
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::A2, 0x0000000100000000);
    instance.set_reg(Reg::A3, 0xffffffffff08bdbd);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 0);
    assert_eq!(instance.reg(Reg::A2), 0);
    assert_eq!(instance.reg(Reg::A3), 0xffffffffff08bdbd);
}

fn memset_truncates_the_destination_to_32_bits(config: Config, isa: InstructionSetKind) {
    if !isa.supports_opcode(Opcode::memset) {
        return;
    }

    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.set_rw_data_size(64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::memset(), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
    let address = module.memory_map().rw_data_address();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 0xffffffff_00000000 | u64::from(address));
    instance.set_reg(Reg::A1, 0x7a);
    instance.set_reg(Reg::A2, 3);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    assert_eq!(instance.read_memory(address, 4).unwrap(), vec![0x7a, 0x7a, 0x7a, 0]);
    assert_eq!(instance.reg(Reg::A0), u64::from(address) + 3);
    assert_eq!(instance.reg(Reg::A2), 0);
}

fn memset_truncates_the_count_to_32_bits(config: Config, isa: InstructionSetKind) {
    if !isa.supports_opcode(Opcode::memset) {
        return;
    }

    let _ = env_logger::try_init();

    let memset = |gas: Option<i64>, count: u64| {
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.set_rw_data_size(64);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&[asm::memset(), asm::ret()], &[]);

        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let engine = Engine::new(&config).unwrap();
        let mut module_config = ModuleConfig::new();
        module_config.set_gas_metering(gas.map(|_| GasMeteringKind::Sync));
        let module = Module::from_blob(&engine, &module_config, blob).unwrap();
        let address = module.memory_map().rw_data_address();

        let mut instance = module.instantiate().unwrap();
        if let Some(gas) = gas {
            instance.set_gas(gas);
        }
        instance.set_reg(Reg::A0, u64::from(address));
        instance.set_reg(Reg::A1, 0x7a);
        instance.set_reg(Reg::A2, count);
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        let interrupt = instance.run().unwrap();
        let written = instance.read_memory(address, 8).unwrap().iter().filter(|&&b| b == 0x7a).count();
        (interrupt, written, instance.reg(Reg::A2))
    };

    assert_eq!(memset(None, 0x00000001_00000000), (InterruptKind::Finished, 0, 0));
    assert_eq!(memset(None, 0x00000001_00000003), (InterruptKind::Finished, 3, 0));
    assert_eq!(memset(Some(100), 0x00000001_00000003), (InterruptKind::Finished, 3, 0));

    let (interrupt, written, remaining) = memset(Some(5), 0x00000001_00000008);
    assert_eq!(interrupt, InterruptKind::NotEnoughGas);
    assert_eq!(written + usize::try_from(remaining).unwrap(), 8);
}

fn count_leading_zero_bits_32_with_zero_input(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_leading_zero_bits_32(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 32);
}

fn count_leading_zero_bits_64_with_zero_input(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_leading_zero_bits_64(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 64);
}

fn count_leading_zero_bits_64_with_ffff0000(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_leading_zero_bits_64(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0xffff0000);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 32);
}

fn count_trailing_zero_bits_32_with_zero_input(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_trailing_zero_bits_32(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 32);
}

fn count_trailing_zero_bits_64_with_zero_input(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_trailing_zero_bits_64(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 64);
}

fn count_trailing_zero_bits_64_with_ffff0000(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::count_trailing_zero_bits_64(Reg::A0, Reg::A1), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A1, 0xffff0000);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 16);
}

fn rotate_right_imm_alt_64(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::rotate_right_imm_alt_64(Reg::A0, Reg::A0, cast(0x80000000_u32).bitwise_as_i32()),
            asm::ret(),
        ],
        &[],
    );

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 0xffffffff80000000);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 0xffffffff80000000);
}

fn jam_validate_ok(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(Reg::A0, 0x12345678), asm::ret()], &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    assert!(blob.validate_code_with_isa(isa).is_ok());
    Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
}

fn jam_validate_invalid_opcode(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(Reg::A0, 0x12345678), asm::ret()], &[]);
    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut raw_code = blob.code().to_vec();
    raw_code[0] = INVALID_RAW_OPCODE;
    blob.set_code(raw_code.into());
    assert!(blob.validate_code_with_isa(isa).is_err());
    assert!(matches!(
        Module::from_blob(&engine, &ModuleConfig::new(), blob),
        Err(CompileError::ValidationFailed(..))
    ));
}

fn jam_validate_invalid_fallthrough(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::load_imm(Reg::A0, 0x12345678), asm::fallthrough()], &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    assert!(blob.validate_code_with_isa(isa).is_err());
    assert!(matches!(
        Module::from_blob(&engine, &ModuleConfig::new(), blob),
        Err(CompileError::ValidationFailed(..))
    ));
}

fn jam_validate_invalid_branch(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::branch_eq_imm(Reg::A0, 33, 2),
            asm::load_imm(Reg::A1, 1),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 3),
            asm::trap(),
        ],
        &[],
    );
    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let instructions: Vec<_> = blob.instructions().collect();
    let mut raw_code = blob.code().to_vec();
    raw_code[instructions[0].next_offset.0 as usize - 1] -= 1;
    blob.set_code(raw_code.into());
    assert!(blob.validate_code_with_isa(isa).is_ok());
    Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
}

fn jam_validate_invalid_skip(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(
        &[
            asm::branch_eq_imm(Reg::A0, 33, 2),
            asm::load_imm(Reg::A1, 1),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::trap(),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 2),
            asm::load_imm(Reg::A1, 3),
            asm::trap(),
        ],
        &[],
    );
    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut raw_bitmask = blob.bitmask().to_vec();
    assert!(raw_bitmask.len() >= 5);
    raw_bitmask[0..4].fill(0);
    raw_bitmask[0] = 1;
    blob.set_bitmask(raw_bitmask.into());
    assert!(blob.validate_code_with_isa(isa).is_err());
    assert!(matches!(
        Module::from_blob(&engine, &ModuleConfig::new(), blob),
        Err(CompileError::ValidationFailed(..))
    ));
}

fn jam_branch_target_just_past_the_end_of_code(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::branch_eq_imm(Reg::A0, 33, 1), asm::trap()], &[]);

    // Patch the branch so that its target is exactly `code_length + 2`;
    // compiling this used to panic on the compiler backend. (See #392.)
    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let code_length = cast(blob.code().len()).to_u32_or_panic();
    let instructions: Vec<_> = blob.instructions().collect();
    let mut raw_code = blob.code().to_vec();
    raw_code[instructions[0].next_offset.0 as usize - 1] += (code_length + 2 - instructions[0].next_offset.0) as u8;
    blob.set_code(raw_code.into());
    assert!(matches!(
        blob.instructions().next().unwrap().kind,
        polkavm_common::program::Instruction::branch_eq_imm(_, _, target) if target == code_length + 2
    ));

    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    // Taking the branch must trap instead of crashing the VM.
    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 33);
    instance.set_next_program_counter(ProgramCounter(0));
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
}

fn jam_reg_nibble_clamped_to_a5(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&config).unwrap();

    // Register nibbles greater than 12 are clamped to 12 (see `RawReg::get`), so a `load_imm`
    // whose destination nibble is 13, 14 or 15 must still write to A5 and not to T2.
    // (See https://github.com/paritytech/polkavm/issues/391.)
    for nibble in 13..=15 {
        let (mut blob, position) = {
            let mut builder = ProgramBlobBuilder::new(isa);
            builder.add_export_by_basic_block(0, b"main");
            builder.set_code(&[asm::load_imm(Reg::A4, 0x12345678), asm::ret()], &[]);
            let blob_a = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

            let mut builder = ProgramBlobBuilder::new(isa);
            builder.add_export_by_basic_block(0, b"main");
            builder.set_code(&[asm::load_imm(Reg::A5, 0x12345678), asm::ret()], &[]);
            let blob_b = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

            // Automatically detect where the register is encoded.
            let position = blob_a
                .code()
                .iter()
                .copied()
                .zip(blob_b.code().iter().copied())
                .take_while(|(a, b)| a == b)
                .count();

            (blob_b, position)
        };

        // Rewrite `load_imm`'s destination register nibble from 12 to the out-of-bounds value.
        let mut raw_code = blob.code().to_vec();
        assert_eq!(raw_code[position], 12);
        raw_code[position] = nibble;
        blob.set_code(raw_code.into());
        assert!(blob.validate_code_with_isa(isa).is_ok());

        let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();
        let mut instance = module.instantiate().unwrap();
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        instance.set_next_program_counter(ProgramCounter(0));
        assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
        assert_eq!(instance.reg(Reg::A5), 0x12345678);
        assert_eq!(instance.reg(Reg::T2), 0);
    }
}

fn test_basic_debug_info(raw_blob: &'static [u8], isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let program = get_blob(raw_blob, isa);
    let entry_point = program.exports().find(|export| export == "read_u32").unwrap().program_counter();
    let mut line_program = program.get_debug_line_program_at(entry_point).unwrap().unwrap();
    let info = line_program.run().unwrap().unwrap();

    let line = include_str!("../../../guest-programs/test-blob/src/main.rs")
        .split('\n')
        .enumerate()
        .find(|(_, line)| line.starts_with("extern \"C\" fn read_u32("))
        .unwrap()
        .0
        + 1;
    let frame = info
        .frames()
        .find(|frame| frame.kind() == polkavm_common::program::FrameKind::Line)
        .unwrap();
    assert_eq!(frame.line(), Some(line as u32 + 1));
    assert_eq!(frame.full_name().unwrap().to_string(), "read_u32");
    assert_eq!(frame.path().unwrap().unwrap(), "test-blob/src/main.rs");
}

#[ignore]
#[test]
fn test_basic_debug_info_32() {
    test_basic_debug_info(get_test_program(TestProgram::TestBlob, false), InstructionSetKind::Latest32);
}

#[ignore]
#[test]
fn test_basic_debug_info_64() {
    test_basic_debug_info(get_test_program(TestProgram::TestBlob, true), InstructionSetKind::Latest64);
}

#[test]
fn test_advance_pc_and_const_add_pc_debug_info_64() {
    // This ELF file was generated from Revive's unit test tests::unit::messages::transfer_suppressed
    // To generate it again, either update Revive's source code to dump ELF file after linking and
    // run `cargo t -p resolc -- --test 'tests::unit::messages::transfer_suppressed' --exact`
    // or extract the solidity file from Revive's repo and run `resolc --solc solc test --debug-output-dir /tmp/test`
    let elf = decompress_zstd(include_bytes!("../../../test-data/revive-transfer-example.zst"));

    let mut config = polkavm_linker::Config::default();
    config.set_optimize(true);
    config.set_strip(false);

    // Since the ELF file has been generated from Revive, use ReviveV1 instruction set to test it.
    let bytes = polkavm_linker::program_from_elf(config, TargetInstructionSet::ReviveV1, elf.as_slice());
    assert!(bytes.is_ok());
    let program = ProgramBlob::parse(bytes.unwrap().into()).unwrap();

    let pc = ProgramCounter(0x222);

    let frame = program
        .get_frame_info_for(pc, None)
        .find(|frame| frame.kind() == polkavm_common::program::FrameKind::Line)
        .unwrap();

    assert_eq!(frame.full_name().unwrap().to_string(), "__revive_store_immutable_data");
    assert_eq!(
        frame.path().unwrap().unwrap(),
        "/tmp/bad_debug_info/_tmp_bad_debug_info.sol.TransferExample.yul"
    );
    assert_eq!(frame.line().unwrap(), 35);
    assert_eq!(frame.column().unwrap(), 13);
}

#[test]
fn blob_len_works() {
    const EXAMPLE_BLOB: &[u8] = include_bytes!("../../../guest-programs/output/example-hello-world.polkavm");
    assert_eq!(Some(EXAMPLE_BLOB.len() as BlobLen), ProgramBlob::blob_length(EXAMPLE_BLOB));
}

#[cfg(not(feature = "std"))]
fn spawn_stress_test(_config: Config, _: InstructionSetKind) {}

#[cfg(feature = "std")]
fn spawn_stress_test(mut config: Config, isa: InstructionSetKind) {
    let _lock = StressTestLock::new();
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_ro_data_size(1);
    builder.set_rw_data_size(1);
    builder.set_ro_data(vec![0x00]);
    builder.set_code(&[asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    for worker_count in [0, 1] {
        config.set_worker_count(worker_count);
        let engine = Engine::new(&config).unwrap();

        let module = Module::from_blob(&engine, &ModuleConfig::default(), blob.clone()).unwrap();
        let linker: Linker = Linker::new();
        let instance_pre = linker.instantiate_pre(&module).unwrap();

        const THREAD_COUNT: usize = 24;
        let barrier = alloc::sync::Arc::new(std::sync::Barrier::new(THREAD_COUNT));

        let mut threads = Vec::new();
        for _ in 0..THREAD_COUNT {
            let instance_pre = instance_pre.clone();
            let barrier = alloc::sync::Arc::clone(&barrier);
            let thread = std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..64 {
                    let mut instance = instance_pre.instantiate().unwrap();
                    instance.call_typed(&mut (), "main", ()).unwrap();
                }
            });
            threads.push(thread);
        }

        let mut results = Vec::new();
        for thread in threads {
            results.push(thread.join());
        }

        for result in results {
            result.unwrap();
        }
    }
}

fn spawn_inner_vm(config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();

    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::ret()], &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::default(), blob).unwrap();
    let mut outer = module.instantiate().unwrap();
    let mut inner_1 = module.instantiate_nested(&outer).unwrap();
    let mut inner_2 = module.instantiate_nested(&outer).unwrap();
    for instance in [&mut outer, &mut inner_1, &mut inner_2] {
        instance.set_next_program_counter(ProgramCounter(0));
        instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
        match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);
    }
}

#[cfg(not(feature = "module-cache"))]
fn module_cache(_config: Config, _: InstructionSetKind) {}

#[cfg(feature = "module-cache")]
fn module_cache(mut config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let blob = get_blob_impl(BlobMapKey {
        optimize: true,
        strip: false,
        isa,
        elf: get_test_program(TestProgram::TestBlob, true),
        is_permissive: true,
    });

    config.set_worker_count(0);

    config.set_cache_enabled(true);
    config.set_lru_cache_size(0);
    let engine_with_cache = Engine::new(&config).unwrap();

    config.set_cache_enabled(true);
    config.set_lru_cache_size(1);
    let engine_with_lru_cache = Engine::new(&config).unwrap();

    config.set_cache_enabled(false);
    config.set_lru_cache_size(0);
    let engine_without_cache = Engine::new(&config).unwrap();

    assert!(Module::from_cache(&engine_with_cache, &Default::default(), &blob).is_none());
    let module_with_cache_1 = Module::from_blob(&engine_with_cache, &Default::default(), blob.clone()).unwrap();
    assert!(Module::from_cache(&engine_with_cache, &Default::default(), &blob).is_some());
    let module_with_cache_2 = Module::from_blob(&engine_with_cache, &Default::default(), blob.clone()).unwrap();
    assert!(Module::from_cache(&engine_with_cache, &Default::default(), &blob).is_some());

    let mut module_config_with_cost = ModuleConfig::new();
    module_config_with_cost.set_interpreter_compilation_cost(Some(TEST_COMPILATION_COST));
    assert!(Module::from_cache(&engine_with_cache, &module_config_with_cost, &blob).is_none());

    assert!(Module::from_cache(&engine_without_cache, &Default::default(), &blob).is_none());
    let module_without_cache_1 = Module::from_blob(&engine_without_cache, &Default::default(), blob.clone()).unwrap();
    assert!(Module::from_cache(&engine_without_cache, &Default::default(), &blob).is_none());
    let module_without_cache_2 = Module::from_blob(&engine_without_cache, &Default::default(), blob.clone()).unwrap();

    if engine_with_cache.backend() == BackendKind::Compiler {
        assert_eq!(
            module_with_cache_1.machine_code().unwrap().as_ptr(),
            module_with_cache_2.machine_code().unwrap().as_ptr()
        );
        assert_ne!(
            module_without_cache_1.machine_code().unwrap().as_ptr(),
            module_without_cache_2.machine_code().unwrap().as_ptr()
        );
    }

    core::mem::drop(module_with_cache_2);
    assert!(Module::from_cache(&engine_with_cache, &Default::default(), &blob).is_some());
    core::mem::drop(module_with_cache_1);
    assert!(Module::from_cache(&engine_with_cache, &Default::default(), &blob).is_none());

    assert!(Module::from_cache(&engine_with_lru_cache, &Default::default(), &blob).is_none());
    Module::from_blob(&engine_with_lru_cache, &Default::default(), blob.clone()).unwrap();
    assert!(Module::from_cache(&engine_with_lru_cache, &Default::default(), &blob).is_some());
}

fn run_riscv_test(engine_config: Config, isa: InstructionSetKind, elf: &[u8], testnum_reg: Reg, optimize: bool) {
    let is_64_bit = elf[4] == 2;
    if is_64_bit != isa.is_64_bit() {
        return;
    }

    let _ = env_logger::try_init();
    let mut linker_config = polkavm_linker::Config::default();
    linker_config.set_optimize(optimize);
    linker_config.set_strip(true);
    linker_config.set_min_stack_size(0);
    let raw_blob = polkavm_linker::program_from_elf(linker_config, isa.into(), elf).unwrap();

    let _ = env_logger::try_init();
    let blob = ProgramBlob::parse(raw_blob.into()).unwrap();

    let engine = Engine::new(&engine_config).unwrap();
    let mut module_config = ModuleConfig::new();
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    let entry_point = module.exports().find(|export| export == "main").unwrap().program_counter();

    // Set some gas to prevent infinite loops.
    instance.set_gas(10000);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(entry_point);
    let result = instance.run().unwrap();
    if !matches!(result, InterruptKind::Finished) {
        panic!("test {} failed: unexpected result: {result:?}", instance.reg(testnum_reg));
    }
}

macro_rules! riscv_test {
    ($test_name:ident, $elf_path:expr, $testnum_reg:ident, $is_optimized:expr) => {
        fn $test_name(engine_config: Config, isa: InstructionSetKind) {
            run_riscv_test(
                engine_config,
                isa,
                &include_bytes!($elf_path)[..],
                Reg::$testnum_reg,
                $is_optimized,
            );
        }

        run_tests! { $test_name }
    };
}

include!("tests_riscv.rs");

#[cfg(all(not(miri), target_os = "linux", target_arch = "x86_64", feature = "std"))]
#[test]
fn core_pinning() {
    use crate::config::CorePinning;
    use crate::sandbox::linux::CpuMask;

    let _ = env_logger::try_init();
    let original_affinity = CpuMask::get_affinity(0).unwrap();
    assert!(
        original_affinity.count() > 1,
        "thread pinned to only a single core; can't run the test"
    );

    let blob = basic_test_blob(InstructionSetKind::Latest64);

    let mut engine_config = Config::new();
    engine_config.set_core_pinning(CorePinning::Disabled);
    {
        let engine = Engine::new(&engine_config).unwrap();
        let module = Module::from_blob(&engine, &Default::default(), blob.clone()).unwrap();
        let _instance = module.instantiate().unwrap();
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
    }

    {
        engine_config.set_core_pinning(CorePinning::PinToCore);
        let engine = Engine::new(&engine_config).unwrap();
        let module = Module::from_blob(&engine, &Default::default(), blob.clone()).unwrap();
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
        let instance = module.instantiate().unwrap();
        let current_affinity = CpuMask::get_affinity(0).unwrap();
        assert_ne!(current_affinity, original_affinity);
        assert_eq!(current_affinity.count(), 1);
        core::mem::drop(instance);
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
    }

    assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);

    {
        engine_config.set_core_pinning(CorePinning::PinToCcx);
        let engine = Engine::new(&engine_config).unwrap();
        let module = Module::from_blob(&engine, &Default::default(), blob.clone()).unwrap();
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
        let instance = module.instantiate().unwrap();
        let current_affinity = CpuMask::get_affinity(0).unwrap();
        assert!(current_affinity.count() > 1);
        core::mem::drop(instance);
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
    }

    assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);

    {
        engine_config.set_core_pinning(CorePinning::PinToCore);
        let engine_1 = Engine::new(&engine_config).unwrap();
        let engine_2 = Engine::new(&engine_config).unwrap();
        let module_1 = Module::from_blob(&engine_1, &Default::default(), blob.clone()).unwrap();
        let module_2 = Module::from_blob(&engine_2, &Default::default(), blob).unwrap();

        let instance_1 = module_1.instantiate().unwrap();
        let current_affinity = CpuMask::get_affinity(0).unwrap();
        assert_ne!(current_affinity, original_affinity);
        assert_eq!(current_affinity.count(), 1);

        let instance_2 = module_2.instantiate().unwrap();
        assert_eq!(current_affinity, CpuMask::get_affinity(0).unwrap());

        core::mem::drop(instance_1);
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);

        core::mem::drop(instance_2);
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);

        let instance_2 = module_2.instantiate().unwrap();
        let current_affinity = CpuMask::get_affinity(0).unwrap();
        assert_ne!(current_affinity, original_affinity);
        assert_eq!(current_affinity.count(), 1);
        core::mem::drop(instance_2);
        assert_eq!(CpuMask::get_affinity(0).unwrap(), original_affinity);
    }
}

run_tests! {
    simple_test
    basic_test
    fallback_hostcall_handler_works
    step_tracing_basic
    step_tracing_invalid_store
    step_tracing_invalid_load
    step_tracing_out_of_gas
    dynamic_jump_to_null
    jump_indirect_into_return_to_host_page
    jump_into_middle_of_basic_block_from_outside
    jump_into_middle_of_basic_block_from_within
    entry_into_the_middle_of_an_instruction_is_rejected
    jump_after_invalid_instruction_from_within
    jump_after_an_injected_invalid_instruction
    jump_to_macro_block_boundary_is_always_valid
    unterminated_macro_block_traps_at_its_last_instruction
    instructions_across_macro_block_boundary_are_traps_1
    instructions_across_macro_block_boundary_are_traps_2
    instructions_across_code_blob_boundary_are_traps
    jump_to_an_injected_invalid_instruction
    jump_indirect_simple
    jump_indirect_big_table
    jump_indirect_truncates_the_target_to_32_bits
    dynamic_paging_basic
    dynamic_paging_freeing_pages
    dynamic_paging_protect_memory
    dynamic_paging_stress_test
    dynamic_paging_initialize_multiple_pages
    dynamic_paging_preinitialize_pages
    dynamic_paging_reading_does_not_resolve_segfaults
    dynamic_paging_read_at_page_boundary
    dynamic_paging_read_at_top_of_address_space
    dynamic_paging_read_at_bottom_of_address_space
    dynamic_paging_read_below_the_guard_threshold
    dynamic_paging_read_with_upper_bits_set
    dynamic_paging_read_memory_which_is_not_paged_in
    dynamic_paging_write_at_page_boundary_with_no_pages
    dynamic_paging_write_at_page_boundary_with_first_page
    dynamic_paging_write_at_page_boundary_with_second_page
    dynamic_paging_change_written_value_and_address_during_segfault
    dynamic_paging_cancel_segfault_by_changing_address
    dynamic_paging_worker_recycle_turn_dynamic_paging_on_and_off
    dynamic_paging_worker_recycle_during_segfault
    dynamic_paging_change_program_counter_during_segfault
    dynamic_paging_run_out_of_gas
    dynamic_paging_receive_from_another_thread_and_run
    dynamic_paging_instantiate_on_another_thread
    dynamic_paging_parallel_page_fault_stress_test
    memory_access_with_upper_address_bits_set
    zero_memory
    doom_o3_dwarf5
    doom_o1_dwarf5
    doom_o3_dwarf2
    doom
    pinky_standard
    pinky_dynamic_paging
    dispatch_table
    fallthrough_into_already_compiled_block
    implicit_trap_after_fallthrough
    invalid_instruction_after_fallthrough
    invalid_branch_target
    branch_gas_cost_consistent_across_backends
    aux_data_works
    aux_data_accessible_area
    aux_data_content_on_new_instance
    aux_data_after_memory_reset
    aux_data_content_after_shrink
    access_memory_from_host
    access_memory_from_within
    write_read_memory_from_host
    stack_size_not_power_of_two
    sbrk_knob_works

    basic_gas_metering_sync
    basic_gas_metering_async
    consume_gas_in_host_function_sync
    consume_gas_in_host_function_async
    gas_metering_with_more_than_one_basic_block
    gas_metering_with_implicit_trap
    gas_gets_charged_when_jumping_in_the_middle_of_a_basic_block

    trapping_preserves_all_registers_normal_trap
    trapping_preserves_all_registers_segfault

    out_of_range_execution

    reclaim_cache_memory
    bounded_interpreter_cache
    lazy_branch_target_compilation_on_first_take
    lazy_branch_target_cache_reset
    lazy_branch_target_cache_limit
    lazy_branch_target_steps_and_charges_gas
    cache_limit_with_middle_of_basic_block_entries
    undersized_max_block_size_with_middle_of_basic_block_entries
    compilation_is_charged_when_compiling
    compilation_cost_does_nothing_on_other_backends
    compilation_cost_errors
    large_compilation_cost_is_charged_in_full
    code_length_dependent_state_is_charged
    out_of_gas_after_a_compilation
    compilation_thrown_away_by_a_cache_reset_is_charged
    compilation_thrown_away_on_entry_into_the_middle_of_a_block_is_charged
    compilation_charge_on_entry_into_the_middle_of_a_block
    cache_reset_while_suspended_resumes_at_the_next_program_counter
    compilation_thrown_away_by_a_host_cache_reset_is_charged
    step_interrupt_right_after_a_compilation
    arbitrary_jump_inside_of_an_instruction_does_not_compile_again
    cache_size_limit_disables_debug_handlers
    lowering_the_cache_size_limit_clears_the_cache
    cache_resets_past_the_epoch_range
    cache_clears_past_the_epoch_range_overwrite_the_code_map
    reset_interpreter_cache_restarts_the_epoch_count
    epoch_wrap_without_block_gas_metering_charges_nothing
    host_cache_clear_leaving_negative_gas_stops_the_next_run

    memset_basic
    memset_with_dynamic_paging

    jam_validate_ok
    jam_validate_invalid_branch
    jam_branch_target_just_past_the_end_of_code
    jam_reg_nibble_clamped_to_a5

    spawn_stress_test
    spawn_inner_vm
    module_cache

    rotate_right_imm_alt_64
    count_leading_zero_bits_32_with_zero_input
    count_leading_zero_bits_64_with_zero_input
    count_leading_zero_bits_64_with_ffff0000
    count_trailing_zero_bits_32_with_zero_input
    count_trailing_zero_bits_64_with_zero_input
    count_trailing_zero_bits_64_with_ffff0000
    memset_truncates_a0_and_a2
    memset_truncates_the_count_to_32_bits
    memset_truncates_the_destination_to_32_bits
}

run_tests_on_isa! { jam_v1, InstructionSetKind::JamV1,
    jam_validate_invalid_opcode
    jam_validate_invalid_fallthrough
    jam_validate_invalid_skip
}

run_test_blob_tests! {
    test_blob_basic_test
    test_blob_atomic_fetch_add
    test_blob_atomic_fetch_swap
    test_blob_atomic_fetch_minmax
    test_blob_hostcall
    test_blob_define_abi
    test_blob_input_registers
    test_blob_call_sbrk_from_guest
    test_blob_call_sbrk_from_host_instance
    test_blob_call_sbrk_from_host_function
    test_blob_program_memory_can_be_reused_and_cleared
    test_blob_out_of_bounds_memory_access_generates_a_trap
    test_blob_add_u32
    test_blob_add_u64
    test_blob_xor_imm_u32
    test_blob_branch_less_than_zero
    test_blob_fetch_add_atomic_u64
    test_blob_cmov_if_zero_with_zero_reg
    test_blob_cmov_if_not_zero_with_zero_reg
    test_blob_min_stack_size
    test_blob_negate_and_add
    test_blob_return_tuple_from_import
    test_blob_return_tuple_from_export
    test_blob_get_heap_base
    test_blob_get_self_address
    test_blob_get_self_address_naked
    test_blob_sub_i32_min_64
    test_blob_sub_i32_min_32
    test_blob_orn_zero_const_64
    test_blob_xnor_zero_const_64
    test_blob_min_zero_const_64
    test_blob_max_zero_const_64
    test_blob_jump_table_with_constant_index
}

run_asm_tests! {
    test_asm_reloc_add_sub
    test_asm_reloc_hi_lo
}

macro_rules! assert_impl {
    ($x:ty, $($t:path),+ $(,)*) => {
        const _: fn() -> () = || {
            struct Check where $x: $($t),+;
        };
    };
}

macro_rules! assert_send_sync {
    ($($x: ty,)+) => {
        $(
            assert_impl!($x, Send);
            assert_impl!($x, Sync);
        )+
    }
}

assert_send_sync! {
    crate::Config,
    crate::Engine,
    crate::Error,
    crate::Gas,
    crate::Instance<(), ()>,
    crate::InstancePre<(), ()>,
    crate::Linker<(), ()>,
    crate::Module,
    crate::ModuleConfig,
    crate::ProgramBlob,
}
