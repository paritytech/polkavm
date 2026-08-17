use crate::mutex::Mutex;
use crate::{
    BackendKind, CallError, Caller, CompileError, Config, Engine, GasMeteringKind, InterruptKind, Linker, MemoryAccessError,
    MemoryProtection, Module, ModuleConfig, ProgramBlob, ProgramCounter, Reg, Segfault, SetCacheSizeLimitArgs,
};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use polkavm_common::abi::MemoryMapBuilder;
use polkavm_common::cast::cast;
use polkavm_common::program::{asm, InstructionSet, InstructionSetKind, Opcode};
use polkavm_common::program::{BlobLen, Reg::*, INTERPRETER_CACHE_ENTRY_SIZE};
use polkavm_common::utils::align_to_next_page_u32;
use polkavm_common::writer::ProgramBlobBuilder;
use polkavm_linker::TargetInstructionSet;

use paste::paste;

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

    instance.reset_interpreter_cache();

    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_reg(Reg::A1, 0);
    instance.set_next_program_counter(list[0].offset);
    instance.reset_interpreter_cache();

    // ecalli 0

    match_interrupt!(instance.run().unwrap(), InterruptKind::Ecalli(0));
    assert_eq!(instance.program_counter(), Some(list[2].offset));
    assert_eq!(instance.next_program_counter(), Some(list[2].next_offset));
    assert_eq!(instance.reg(Reg::A0), 0x1234);
    assert_eq!(instance.reg(Reg::A1), 100);

    // ret

    instance.reset_interpreter_cache();
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

fn jump_after_invalid_instruction_from_within(engine_config: Config, isa: InstructionSetKind) {
    let _ = env_logger::try_init();
    let engine = Engine::new(&engine_config).unwrap();
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::trap(), asm::add_imm_32(A0, A0, 100), asm::jump(1)], &[]);

    let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let mut raw_code = blob.code().to_vec();
    raw_code[0] = 255;
    blob.set_code(raw_code.into());
    let instructions: Vec<_> = blob.instructions().collect();
    assert_eq!(
        instructions[0],
        polkavm_common::program::ParsedInstruction {
            kind: crate::program::Instruction::invalid,
            offset: ProgramCounter(0),
            next_offset: ProgramCounter(1),
        }
    );

    let mut module_config: ModuleConfig = ModuleConfig::new();
    module_config.set_page_size(get_native_page_size().try_into().unwrap());
    module_config.set_gas_metering(Some(GasMeteringKind::Sync));
    let module = Module::from_blob(&engine, &module_config, blob.clone()).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_gas(1000);

    instance.set_next_program_counter(instructions[1].offset);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Trap);
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

    let engine = Engine::new(&engine_config).unwrap();
    let page_size = get_native_page_size() as u32;
    let mut builder = ProgramBlobBuilder::new(isa);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::store_imm_indirect_u32(Reg::A0, 0, 0x12345678), asm::ret()], &[]);

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
    assert_eq!(instance.read_memory(0x11000, 4).unwrap(), vec![0x78, 0x56, 0x34, 0x12]);
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
    raw_code[instructions[1].offset.0 as usize] = 255;
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
    {
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

    // A basic block ending in a conditional branch whose fall-through is a `trap`.
    // With the full cost model computing the branch's gas cost requires peeking at the
    // opcode of the instruction following the branch; the interpreter (via
    // `visit_parsing`) and the compiler (via the static dispatch tables) used to derive
    // a different `args_length` for that peek, so they charged a different
    // branch-prediction cost. Under the tracing config the harness runs both backends
    // in lockstep and cross-checks the gas, so any divergence fails the test.
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
}

fn memset_preserves_a0_and_a2(config: Config, isa: InstructionSetKind) {
    if isa != InstructionSetKind::Latest64 {
        return;
    }
    let _ = env_logger::try_init();

    // Memset must not truncate A0 or A2. With count=0, memset is a no-op
    // and both registers must pass through with their upper 32 bits intact.
    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::Latest64);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&[asm::memset(), asm::ret()], &[]);

    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &ModuleConfig::new(), blob).unwrap();

    let mut instance = module.instantiate().unwrap();
    instance.set_reg(Reg::A0, 0x0000000100000000);
    instance.set_reg(Reg::A1, 0);
    instance.set_reg(Reg::A2, 0);
    instance.set_reg(Reg::A3, 0xffffffffff08bdbd);
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(ProgramCounter(0));
    assert!(matches!(instance.run().unwrap(), InterruptKind::Finished));
    assert_eq!(instance.reg(Reg::A0), 0x0000000100000000, "memset truncated A0");
    assert_eq!(instance.reg(Reg::A2), 0);
    assert_eq!(instance.reg(Reg::A3), 0xffffffffff08bdbd);
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
    assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_ok());
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
    raw_code[0] = 255;
    blob.set_code(raw_code.into());
    assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_err());
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
    assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_err());
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
    assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_ok());
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
    assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_err());
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
        let mut builder = ProgramBlobBuilder::new(isa);
        builder.add_export_by_basic_block(0, b"main");
        builder.set_code(&[asm::load_imm(Reg::A5, 0x12345678), asm::ret()], &[]);
        let mut blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

        // Rewrite `load_imm`'s destination register nibble from 12 to the out-of-bounds value.
        let mut raw_code = blob.code().to_vec();
        assert_eq!(raw_code[1], 12);
        raw_code[1] = nibble;
        blob.set_code(raw_code.into());
        assert!(blob.validate_code_with_isa(polkavm_common::program::ISA_JamV1).is_ok());

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
    jump_into_middle_of_basic_block_from_outside
    jump_into_middle_of_basic_block_from_within
    jump_after_invalid_instruction_from_within
    jump_indirect_simple
    jump_indirect_big_table
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
    access_memory_from_host
    access_memory_from_within
    write_read_memory_from_host
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
    memset_preserves_a0_and_a2
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

/// Runs one wide instruction in the interpreter and returns what it left in memory.
///
/// The program reads its two operands from the start of the read-write data, runs `body`,
/// and the caller reads the result back from the third slot.
#[cfg(feature = "std")]
/// Runs every wide and vector instruction once and folds each result into an accumulator,
/// then checks the accumulator against a fresh interpreter run of the same program.
///
/// The point is the recompiler's operand plumbing: each instruction's call site packs its
/// registers and immediate into a descriptor by hand, and a transposed field would produce
/// a program that runs fine and computes something else. The interpreter is the reference,
/// so no result here is hand computed.
#[cfg(feature = "std")]
fn every_wide_and_vector_instruction_matches_the_interpreter(config: Config) {
    use polkavm_common::program::Instruction;
    use polkavm_common::program::Reg::*;
    use polkavm_common::program::VecReg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::vector::{VectorArithmetic, VectorConfig, VectorOperand, VectorOperation};
    use polkavm_common::wide::U256;

    let arithmetic = |operation, dst: u32, src: u32, operand| {
        let packed = VectorArithmetic {
            operation,
            dst,
            src,
            operand,
        }
        .to_packed();
        asm::vector_arithmetic(cast(packed).bitwise_as_i32())
    };

    let memory_map = MemoryMapBuilder::new(0x4000).rw_data_size(0x4000).build().unwrap();
    let absolute_address = cast(memory_map.rw_data_address()).bitwise_as_i32();

    let operands = [
        U256([0x0123_4567_89ab_cdef, 0xfeed_face_0000_0007, 3, 0x8000_0000_0000_0001]),
        U256([29, 0, 0xffff_ffff_ffff_fffb, 1 << 62]),
        (U256::ONE.shift_left(255)).wrapping_add(U256::from_u64(9)),
    ];

    let mut body: Vec<Instruction> = Vec::new();
    body.push(asm::load_imm(A2, 77));
    body.push(asm::wide_load_imm_signed(W3, -19));

    // Everything that writes a wide register, computed into W4 and folded.
    let wide_results = [
        asm::wide_add(W4, W0, W1),
        asm::wide_sub(W4, W0, W1),
        asm::wide_mul(W4, W0, W1),
        asm::wide_and(W4, W0, W1),
        asm::wide_or(W4, W0, W1),
        asm::wide_xor(W4, W0, W1),
        asm::wide_div_unsigned(W4, W0, W1),
        asm::wide_div_signed(W4, W0, W1),
        asm::wide_rem_unsigned(W4, W0, W1),
        asm::wide_rem_signed(W4, W0, W1),
        asm::wide_exp(W4, W0, W1),
        asm::wide_sign_extend_byte(W4, W1, W0),
        asm::wide_add_mod(W4, W0, W1, W2),
        asm::wide_mul_mod(W4, W0, W1, W2),
        asm::wide_shift_logical_left(W4, W0, A2),
        asm::wide_shift_logical_right(W4, W0, A2),
        asm::wide_shift_arithmetic_right(W4, W0, A2),
        asm::wide_shift_logical_left_imm(W4, W0, 13),
        asm::wide_shift_logical_right_imm(W4, W0, 13),
        asm::wide_shift_arithmetic_right_imm(W4, W0, 250),
        asm::wide_move(W4, W0),
        asm::wide_reverse_bytes(W4, W0),
        asm::wide_from_reg_unsigned(W4, A2),
        asm::wide_from_reg_signed(W4, A2),
        asm::wide_load_imm_unsigned(W4, -5),
        asm::wide_load_imm_signed(W4, -5),
        asm::wide_load(W4, A0, 32),
        asm::wide_load_absolute(W4, absolute_address),
    ];
    for instruction in wide_results {
        body.push(instruction);
        body.push(asm::wide_xor(W3, W3, W4));
    }

    // A round trip through memory.
    body.push(asm::wide_store(W3, A0, 128));
    body.push(asm::wide_load(W4, A0, 128));
    body.push(asm::wide_xor(W3, W3, W4));

    // Everything that writes a general purpose register, folded through a widening.
    let register_results = [
        asm::wide_set_equal(A1, W0, W1),
        asm::wide_set_not_equal(A1, W0, W1),
        asm::wide_set_less_than_unsigned(A1, W0, W1),
        asm::wide_set_less_than_signed(A1, W0, W1),
        asm::wide_to_reg(W0, A1),
        asm::wide_count_set_bits(W0, A1),
        asm::wide_count_leading_zero_bits(W0, A1),
        asm::wide_count_trailing_zero_bits(W0, A1),
    ];
    for instruction in register_results {
        body.push(instruction);
        body.push(asm::wide_from_reg_unsigned(W4, A1));
        body.push(asm::wide_xor(W3, W3, W4));
    }

    // Four 64-bit elements across a register pair. The operand data sits in v0 to v3
    // because the harness loaded W0 and W1, and v0 doubles as the mask register, which is
    // what the merge and the masked forms read.
    let configure = VectorConfig::new(0b011_001, 4);
    body.push(asm::vector_config(cast(configure.to_packed()).bitwise_as_i32()));

    // Element-wise operations in every operand shape, plus the ones with special paths.
    let element_wise = [
        arithmetic(VectorOperation::Add, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::Subtract, 12, 0, VectorOperand::Register(A2 as u32)),
        arithmetic(VectorOperation::Xor, 12, 0, VectorOperand::Immediate(-3)),
        arithmetic(VectorOperation::MinimumSigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::MaximumUnsigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::ShiftLeft, 12, 0, VectorOperand::Register(A2 as u32)),
        arithmetic(VectorOperation::ShiftRightSigned, 12, 0, VectorOperand::Immediate(7)),
        arithmetic(VectorOperation::Multiply, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::MultiplyHighSigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::MultiplyHighUnsigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::DivideUnsigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::DivideSigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::RemainderUnsigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::RemainderSigned, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::MultiplyAdd, 12, 0, VectorOperand::Vector(2)),
        arithmetic(VectorOperation::Merge, 12, 2, VectorOperand::Vector(4)),
        arithmetic(VectorOperation::SlideUp, 12, 0, VectorOperand::Immediate(1)),
        arithmetic(VectorOperation::SlideDown, 12, 0, VectorOperand::Immediate(2)),
        asm::vector_move(V12, V0),
        asm::vector_splat(V12, A2),
        asm::vector_splat_imm(V12, -7),
        asm::vector_insert(V12, A2),
        asm::vector_insert_imm(V12, 9),
        asm::vector_element_index(V12),
        asm::vector_set_equal(V12, V0, V2),
        asm::vector_set_not_equal(V12, V0, V2),
        asm::vector_set_equal_imm(V12, V0, -1),
        asm::vector_set_not_equal_imm(V12, V0, 29),
        asm::vector_mask_and(V12, V0, V2),
        asm::vector_mask_and_not(V12, V0, V2),
        asm::vector_mask_or(V12, V0, V2),
        asm::vector_mask_xor(V12, V0, V2),
        asm::vector_mask_nand(V12, V0, V2),
        asm::vector_mask_nor(V12, V0, V2),
        asm::vector_mask_or_not(V12, V0, V2),
        asm::vector_mask_xnor(V12, V0, V2),
    ];
    for instruction in element_wise {
        body.push(instruction);
        body.push(asm::wide_xor(W3, W3, W6));
    }

    // Everything that reads the file into a general purpose register.
    let register_reads = [
        asm::vector_count_mask(A1, V0),
        asm::vector_count_mask_masked(A1, V2),
        asm::vector_first_mask(A1, V2),
        asm::vector_first_mask_masked(A1, V2),
        asm::vector_extract(A1, V2),
    ];
    for instruction in register_reads {
        body.push(instruction);
        body.push(asm::wide_from_reg_unsigned(W4, A1));
        body.push(asm::wide_xor(W3, W3, W4));
    }

    // The memory forms: whole register, and one unit-stride width at a time.
    let memory_forms = [
        [asm::vector_store(V0, A0, 160), asm::vector_load(V12, A0, 160)],
        [asm::vector_store_u8(V0, A0, 192), asm::vector_load_u8(V12, A0, 192)],
        [asm::vector_store_u16(V0, A0, 224), asm::vector_load_u16(V12, A0, 224)],
        [asm::vector_store_u32(V0, A0, 256), asm::vector_load_u32(V12, A0, 256)],
        [asm::vector_store_u64(V0, A0, 288), asm::vector_load_u64(V12, A0, 288)],
    ];
    for [store, load] in memory_forms {
        body.push(store);
        body.push(load);
        body.push(asm::wide_xor(W3, W3, W6));
    }

    // The dynamic configurations, which write the settled element count back.
    body.push(asm::load_imm(A3, 3));
    body.push(asm::vector_config_dynamic(A1, A3, cast(0b000_000_u32).bitwise_as_i32()));
    body.push(asm::wide_from_reg_unsigned(W4, A1));
    body.push(asm::wide_xor(W3, W3, W4));
    body.push(asm::vector_config_dynamic_discard(A3, cast(0b011_001_u32).bitwise_as_i32()));
    body.push(arithmetic(VectorOperation::Add, 12, 0, VectorOperand::Vector(2)));
    body.push(asm::wide_xor(W3, W3, W6));

    body.push(asm::wide_store(W3, A0, 96));

    let expected = {
        let mut reference = Config::default();
        reference.set_backend(Some(BackendKind::Interpreter));
        run_wide_program(&reference, &operands, &body)
    };
    let actual = run_wide_program(&config, &operands, &body);
    assert_eq!(U256::from_le_bytes(actual), U256::from_le_bytes(expected));
}

/// Generates per-backend wrappers for the wide and vector execution tests.
///
/// These always build a `ReviveV1` blob, because no other instruction set holds the wide
/// instructions, so they cannot go through `run_tests!`. The tracing variants run the
/// recompiler and the interpreter in lockstep and compare the registers after every
/// instruction.
#[cfg(feature = "std")]
macro_rules! run_wide_tests {
    ($($test_name:ident)+) => {
        if_compiler_is_supported! {
            $(
                paste! {
                    #[cfg(target_os = "linux")]
                    #[test]
                    fn [<compiler_linux_ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_worker_count(1);
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Linux));
                        $test_name(config);
                    }

                    #[cfg(target_os = "linux")]
                    #[test]
                    fn [<tracing_linux_ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Linux));
                        config.set_allow_experimental(true);
                        config.set_crosscheck(true);
                        $test_name(config);
                    }

                    #[cfg(feature = "generic-sandbox")]
                    #[test]
                    fn [<compiler_generic_ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Generic));
                        config.set_allow_experimental(true);
                        $test_name(config);
                    }

                    #[cfg(feature = "generic-sandbox")]
                    #[test]
                    fn [<tracing_generic_ $test_name>]() {
                        let mut config = crate::Config::default();
                        config.set_backend(Some(crate::BackendKind::Compiler));
                        config.set_sandbox(Some(crate::SandboxKind::Generic));
                        config.set_allow_experimental(true);
                        config.set_crosscheck(true);
                        $test_name(config);
                    }
                }
            )+
        }

        $(
            paste! {
                #[test]
                fn [<interpreter_ $test_name>]() {
                    let mut config = crate::Config::default();
                    config.set_backend(Some(crate::BackendKind::Interpreter));
                    $test_name(config);
                }
            }
        )+
    };
}

#[cfg(feature = "std")]
run_wide_tests! {
    wide_arithmetic_follows_evm_semantics
    wide_signed_division_takes_the_sign_of_the_dividend
    wide_modular_operations_use_the_untruncated_result
    wide_shifts_past_the_width_clear_the_value
    wide_moves_and_conversions_round_trip
    wide_comparisons_write_a_general_purpose_register
    wide_load_imm_widens_like_the_register_form
    wide_load_absolute_reads_without_a_base_register
    wide_shift_imm_matches_the_register_form
    wide_bit_counts_write_a_general_purpose_register
    vector_whole_register_moves_reach_the_halves_of_a_wide_one
    vector_loads_and_stores_reach_one_register
    vector_compares_produce_a_mask_the_population_count_reads
    every_wide_and_vector_instruction_matches_the_interpreter
}

fn run_wide_program(config: &Config, operands: &[polkavm_common::wide::U256], body: &[polkavm_common::program::Instruction]) -> [u8; 32] {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    const DATA_SIZE: u32 = 0x4000;
    const RESULT_OFFSET: i32 = 96;

    let memory_map = MemoryMapBuilder::new(0x4000).rw_data_size(DATA_SIZE).build().unwrap();
    let base = memory_map.rw_data_address();

    let mut code = vec![asm::load_imm(A0, cast(base).bitwise_as_i32())];
    for (index, register) in [W0, W1, W2].iter().enumerate().take(operands.len()) {
        code.push(asm::wide_load(*register, A0, cast(index as u32 * 32).bitwise_as_i32()));
    }
    code.extend_from_slice(body);
    code.push(asm::ret());

    let mut builder = ProgramBlobBuilder::new(InstructionSetKind::ReviveV1);
    builder.set_rw_data_size(DATA_SIZE);
    builder.add_export_by_basic_block(0, b"main");
    builder.set_code(&code, &[]);
    let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();

    let engine = Engine::new(config).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();
    let mut instance = module.instantiate().unwrap();

    for (index, operand) in operands.iter().enumerate() {
        instance.write_memory(base + index as u32 * 32, &operand.to_le_bytes()).unwrap();
    }

    let result_address = base + cast(RESULT_OFFSET).bitwise_as_u32();
    instance.write_memory(result_address, &U256::ZERO.to_le_bytes()).unwrap();

    let entry_point = module.exports().find(|export| export == "main").unwrap().program_counter();
    instance.set_reg(Reg::RA, crate::RETURN_TO_HOST);
    instance.set_next_program_counter(entry_point);
    match_interrupt!(instance.run().unwrap(), InterruptKind::Finished);

    let mut result = [0; 32];
    instance.read_memory_into(result_address, &mut result[..]).unwrap();
    result
}

/// Runs `body` and returns the 256-bit value it stored.
#[cfg(feature = "std")]
fn run_wide(
    config: &Config,
    operands: &[polkavm_common::wide::U256],
    body: &[polkavm_common::program::Instruction],
) -> polkavm_common::wide::U256 {
    polkavm_common::wide::U256::from_le_bytes(run_wide_program(config, operands, body))
}

#[cfg(feature = "std")]
fn wide_arithmetic_follows_evm_semantics(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let seven = U256::from_u64(7);
    let three = U256::from_u64(3);
    let max = U256([u64::MAX; 4]);
    let store = asm::wide_store(W3, A0, 96);

    for (name, body, operands, expected) in [
        ("add", asm::wide_add(W3, W0, W1), [seven, three], U256::from_u64(10)),
        ("sub", asm::wide_sub(W3, W0, W1), [seven, three], U256::from_u64(4)),
        ("mul", asm::wide_mul(W3, W0, W1), [seven, three], U256::from_u64(21)),
        ("and", asm::wide_and(W3, W0, W1), [seven, three], U256::from_u64(3)),
        ("or", asm::wide_or(W3, W0, W1), [seven, three], U256::from_u64(7)),
        ("xor", asm::wide_xor(W3, W0, W1), [seven, three], U256::from_u64(4)),
        ("divu", asm::wide_div_unsigned(W3, W0, W1), [seven, three], U256::from_u64(2)),
        ("remu", asm::wide_rem_unsigned(W3, W0, W1), [seven, three], U256::ONE),
        ("divu by zero", asm::wide_div_unsigned(W3, W0, W1), [seven, U256::ZERO], U256::ZERO),
        ("remu by zero", asm::wide_rem_unsigned(W3, W0, W1), [seven, U256::ZERO], U256::ZERO),
        ("add wraps", asm::wide_add(W3, W0, W1), [max, U256::ONE], U256::ZERO),
        ("exp", asm::wide_exp(W3, W0, W1), [three, U256::from_u64(5)], U256::from_u64(243)),
        (
            "signextend",
            asm::wide_sign_extend_byte(W3, W0, W1),
            [U256::from_u64(0xff), U256::ZERO],
            max,
        ),
    ] {
        assert_eq!(run_wide(&config, &operands, &[body, store]), expected, "{name}");
    }
}

#[cfg(feature = "std")]
fn wide_signed_division_takes_the_sign_of_the_dividend(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let minus_seven = U256::from_u64(7).wrapping_neg();
    let two = U256::from_u64(2);
    let store = asm::wide_store(W3, A0, 96);

    assert_eq!(
        run_wide(&config, &[minus_seven, two], &[asm::wide_div_signed(W3, W0, W1), store]),
        U256::from_u64(3).wrapping_neg()
    );
    assert_eq!(
        run_wide(&config, &[minus_seven, two], &[asm::wide_rem_signed(W3, W0, W1), store]),
        U256::ONE.wrapping_neg()
    );
}

#[cfg(feature = "std")]
fn wide_modular_operations_use_the_untruncated_result(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let max = U256([u64::MAX; 4]);
    let store = asm::wide_store(W3, A0, 96);

    // 2^256 mod 7 is 2, which a truncating add would have lost.
    assert_eq!(
        run_wide(
            &config,
            &[max, U256::ONE, U256::from_u64(7)],
            &[asm::wide_add_mod(W3, W0, W1, W2), store]
        ),
        U256::from_u64(2)
    );
    assert_eq!(
        run_wide(
            &config,
            &[U256::from_u64(5), U256::from_u64(6), U256::from_u64(7)],
            &[asm::wide_mul_mod(W3, W0, W1, W2), store]
        ),
        U256::from_u64(30 % 7)
    );
    assert_eq!(
        run_wide(&config, &[max, U256::ONE, U256::ZERO], &[asm::wide_add_mod(W3, W0, W1, W2), store]),
        U256::ZERO
    );
}

#[cfg(feature = "std")]
fn wide_shifts_past_the_width_clear_the_value(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let max = U256([u64::MAX; 4]);
    let store = asm::wide_store(W3, A0, 96);

    assert_eq!(
        run_wide(
            &config,
            &[U256::ONE],
            &[asm::load_imm(A1, 8), asm::wide_shift_logical_left(W3, W0, A1), store]
        ),
        U256::from_u64(256)
    );
    assert_eq!(
        run_wide(
            &config,
            &[max],
            &[asm::load_imm(A1, 256), asm::wide_shift_logical_left(W3, W0, A1), store]
        ),
        U256::ZERO
    );
    assert_eq!(
        run_wide(
            &config,
            &[max],
            &[asm::load_imm(A1, 256), asm::wide_shift_arithmetic_right(W3, W0, A1), store]
        ),
        max
    );
    assert_eq!(
        run_wide(
            &config,
            &[U256::ONE],
            &[asm::load_imm(A1, 256), asm::wide_shift_arithmetic_right(W3, W0, A1), store]
        ),
        U256::ZERO
    );
}

#[cfg(feature = "std")]
fn wide_moves_and_conversions_round_trip(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let value = U256([0x0123_4567_89ab_cdef, 2, 3, 4]);
    let store = asm::wide_store(W3, A0, 96);

    assert_eq!(run_wide(&config, &[value], &[asm::wide_move(W3, W0), store]), value);
    assert_eq!(
        run_wide(&config, &[value], &[asm::wide_reverse_bytes(W3, W0), store]),
        value.swap_bytes()
    );
    assert_eq!(
        run_wide(
            &config,
            &[value],
            &[asm::wide_to_reg(W0, A1), asm::wide_from_reg_unsigned(W3, A1), store]
        ),
        U256::from_u64(value.low_u64())
    );
    assert_eq!(
        run_wide(
            &config,
            &[],
            &[
                asm::load_imm(A1, -1),
                asm::wide_from_reg_signed(W3, A1),
                asm::wide_store(W3, A0, 96)
            ]
        ),
        U256([u64::MAX; 4])
    );
}

#[cfg(feature = "std")]
fn wide_load_imm_widens_like_the_register_form(config: Config) {
    use polkavm_common::program::Reg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let store = asm::wide_store(W3, A0, 96);

    // The immediate stands for a general purpose register the caller would have loaded, so
    // it is sign extended to the register width before the widening kind applies.
    assert_eq!(
        run_wide(&config, &[], &[asm::wide_load_imm_unsigned(W3, 255), store]),
        U256::from_u64(255)
    );
    assert_eq!(
        run_wide(&config, &[], &[asm::wide_load_imm_signed(W3, 255), store]),
        U256::from_u64(255)
    );
    assert_eq!(
        run_wide(&config, &[], &[asm::wide_load_imm_unsigned(W3, -1), store]),
        U256::from_u64(u64::MAX)
    );
    assert_eq!(
        run_wide(&config, &[], &[asm::wide_load_imm_signed(W3, -1), store]),
        U256([u64::MAX; 4])
    );

    // And it agrees with loading the same value into a register and widening that.
    for value in [0, 1, -1, i32::MIN, i32::MAX] {
        assert_eq!(
            run_wide(&config, &[], &[asm::wide_load_imm_unsigned(W3, value), store]),
            run_wide(
                &config,
                &[],
                &[asm::load_imm(A1, value), asm::wide_from_reg_unsigned(W3, A1), store]
            )
        );
        assert_eq!(
            run_wide(&config, &[], &[asm::wide_load_imm_signed(W3, value), store]),
            run_wide(&config, &[], &[asm::load_imm(A1, value), asm::wide_from_reg_signed(W3, A1), store])
        );
    }
}

#[cfg(feature = "std")]
fn wide_load_absolute_reads_without_a_base_register(config: Config) {
    use polkavm_common::program::Reg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    // The harness writes the first operand at the start of the read-write data, which is
    // where the absolute form reads from here.
    let memory_map = MemoryMapBuilder::new(0x4000).rw_data_size(0x4000).build().unwrap();
    let address = cast(memory_map.rw_data_address()).bitwise_as_i32();
    let value = U256([0x0123_4567_89ab_cdef, 2, 3, 4]);

    assert_eq!(
        run_wide(
            &config,
            &[value],
            &[asm::wide_load_absolute(W3, address), asm::wide_store(W3, A0, 96)]
        ),
        value
    );
}

#[cfg(feature = "std")]
fn wide_shift_imm_matches_the_register_form(config: Config) {
    use polkavm_common::program::Reg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let value = U256([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 7, 1 << 63]);
    let store = asm::wide_store(W3, A0, 96);

    for amount in [0, 1, 63, 64, 65, 127, 255, 256, 1000] {
        for (with_imm, with_reg) in [
            (
                asm::wide_shift_logical_left_imm(W3, W0, amount),
                asm::wide_shift_logical_left(W3, W0, A1),
            ),
            (
                asm::wide_shift_logical_right_imm(W3, W0, amount),
                asm::wide_shift_logical_right(W3, W0, A1),
            ),
            (
                asm::wide_shift_arithmetic_right_imm(W3, W0, amount),
                asm::wide_shift_arithmetic_right(W3, W0, A1),
            ),
        ] {
            assert_eq!(
                run_wide(&config, &[value], &[with_imm, store]),
                run_wide(&config, &[value], &[asm::load_imm(A1, amount), with_reg, store]),
                "shift by {amount}"
            );
        }
    }
}

#[cfg(feature = "std")]
fn wide_bit_counts_write_a_general_purpose_register(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let count = |body: polkavm_common::program::Instruction, operand: U256| -> u64 {
        let bytes = run_wide_program(&config, &[operand], &[body, asm::store_indirect_u64(A1, A0, 96)]);
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    };

    let top = U256([0, 0, 0, 1 << 63]);
    assert_eq!(count(asm::wide_count_set_bits(W0, A1), U256([u64::MAX; 4])), 256);
    assert_eq!(count(asm::wide_count_set_bits(W0, A1), U256::ZERO), 0);
    assert_eq!(count(asm::wide_count_leading_zero_bits(W0, A1), U256::ZERO), 256);
    assert_eq!(count(asm::wide_count_leading_zero_bits(W0, A1), top), 0);
    assert_eq!(count(asm::wide_count_trailing_zero_bits(W0, A1), U256::ZERO), 256);
    assert_eq!(count(asm::wide_count_trailing_zero_bits(W0, A1), top), 255);
}

#[cfg(feature = "std")]
fn wide_comparisons_write_a_general_purpose_register(config: Config) {
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    let minus_one = U256([u64::MAX; 4]);
    let one = U256::ONE;

    let compare = |body: polkavm_common::program::Instruction, operands: [U256; 2]| -> u64 {
        let bytes = run_wide_program(&config, &operands, &[body, asm::store_indirect_u64(A1, A0, 96)]);
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    };

    assert_eq!(compare(asm::wide_set_equal(A1, W0, W1), [one, one]), 1);
    assert_eq!(compare(asm::wide_set_equal(A1, W0, W1), [one, minus_one]), 0);
    assert_eq!(compare(asm::wide_set_not_equal(A1, W0, W1), [one, minus_one]), 1);
    // Unsigned, `minus_one` is the largest value there is; signed, it is the smallest.
    assert_eq!(compare(asm::wide_set_less_than_unsigned(A1, W0, W1), [one, minus_one]), 1);
    assert_eq!(compare(asm::wide_set_less_than_signed(A1, W0, W1), [one, minus_one]), 0);
    assert_eq!(compare(asm::wide_set_less_than_signed(A1, W0, W1), [minus_one, one]), 1);
}

#[cfg(feature = "std")]
fn vector_whole_register_moves_reach_the_halves_of_a_wide_one(config: Config) {
    use polkavm_common::program::VecReg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    // The two files are one, so a wide register is the pair of vector registers whose
    // lower half it starts at: moving both halves of `w0` into those of `w3` one at a
    // time has to leave `w3` holding what `w0` did.
    let value = U256([1, 2, 3, 4]);
    let store = asm::wide_store(W3, A0, 96);
    assert_eq!(
        run_wide(&config, &[value], &[asm::vector_move(V6, V0), asm::vector_move(V7, V1), store]),
        value
    );

    // Only the named register moves, so moving the low half alone leaves the high half of
    // the destination as it was.
    assert_eq!(
        run_wide(
            &config,
            &[value, U256::ZERO],
            &[asm::wide_move(W3, W1), asm::vector_move(V6, V0), store]
        ),
        U256([1, 2, 0, 0])
    );
}

#[cfg(feature = "std")]
fn vector_loads_and_stores_reach_one_register(config: Config) {
    use polkavm_common::program::VecReg::*;
    use polkavm_common::program::WideReg::*;
    use polkavm_common::wide::U256;

    // A vector store writes half of what a wide store does, so storing the high half of
    // `w0` over the low half of the result leaves the rest of the result untouched.
    let value = U256([1, 2, 3, 4]);
    assert_eq!(
        run_wide(
            &config,
            &[value],
            &[
                asm::wide_store(W0, A0, 96),
                asm::vector_store(V1, A0, 96),
                asm::wide_store(W3, A0, 128),
            ]
        ),
        U256([3, 4, 3, 4])
    );

    // And a vector load fills only the register it names.
    assert_eq!(
        run_wide(
            &config,
            &[value, U256::ZERO],
            &[asm::wide_move(W3, W1), asm::vector_load(V7, A0, 0), asm::wide_store(W3, A0, 96),]
        ),
        U256([0, 0, 1, 2])
    );
}

#[cfg(feature = "std")]
fn vector_compares_produce_a_mask_the_population_count_reads(config: Config) {
    use polkavm_common::program::VecReg::*;
    use polkavm_common::vector::VectorConfig;
    use polkavm_common::wide::U256;

    // Thirty-two byte-wide elements across a pair of registers, which is the shape a
    // comparison of two 256-bit values in memory takes.
    let vector_config = VectorConfig::new(0b11_000_001, 32);
    assert_eq!(vector_config.element_bits(), 8);
    assert_eq!(vector_config.max_element_count(), 32);
    let configure = asm::vector_config(cast(vector_config.to_packed()).bitwise_as_i32());

    let count = |operands: [U256; 2], equal: bool| -> u64 {
        let compare = if equal {
            asm::vector_set_equal(V6, V0, V2)
        } else {
            asm::vector_set_not_equal(V6, V0, V2)
        };
        let bytes = run_wide_program(
            &config,
            &operands,
            &[
                configure,
                compare,
                asm::vector_count_mask(A1, V6),
                asm::store_indirect_u64(A1, A0, 96),
            ],
        );
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    };

    let value = U256([0x0807060504030201, 2, 3, 4]);
    assert_eq!(count([value, value], true), 32);
    assert_eq!(count([value, value], false), 0);

    // One differing byte in the low limb leaves thirty-one equal and one not.
    let mut other = value;
    other.0[0] ^= 0xff;
    assert_eq!(count([value, other], true), 31);
    assert_eq!(count([value, other], false), 1);
}
