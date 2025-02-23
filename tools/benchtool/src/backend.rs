use crate::BenchmarkKind;
use std::path::Path;

#[allow(unused_imports)]
mod backend_prelude {
    pub use super::{Backend, CreateArgs};
    pub use std::path::{Path, PathBuf};
}

pub struct CreateArgs {
    pub is_compile_only: bool,
}

pub trait Backend: Copy + Clone {
    type Engine;
    type Blob;
    type Module;
    type Instance;

    fn name(&self) -> &'static str;
    fn create(&self, args: CreateArgs) -> Self::Engine;
    fn load(&self, path: &Path) -> Self::Blob;
    fn compile(&self, engine: &mut Self::Engine, blob: &Self::Blob) -> Self::Module;
    fn spawn(&self, engine: &mut Self::Engine, module: &Self::Module) -> Self::Instance;
    fn initialize(&self, instance: &mut Self::Instance);
    fn run(&self, instance: &mut Self::Instance);
    fn pid(&self, _instance: &Self::Instance) -> Option<u32> {
        None
    }

    fn is_slow(&self) -> bool {
        false
    }

    fn is_compiled(&self) -> bool {
        false
    }
}

mod backend_polkavm;

#[cfg(all(feature = "pvf-executor", target_arch = "x86_64"))]
mod backend_pvfexecutor;

#[cfg(all(feature = "ckb-vm", target_arch = "x86_64"))]
mod backend_ckbvm;

#[cfg(all(feature = "wasm3", target_arch = "x86_64"))]
mod backend_wasm3;

#[cfg(all(feature = "wasmtime", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod backend_wasmtime;

#[cfg(all(feature = "wasmer", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod backend_wasmer;

#[cfg(all(feature = "wazero", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod backend_wazero;

#[cfg(all(feature = "solana_rbpf", target_arch = "x86_64", not(target_os = "windows")))]
mod backend_solana_rbpf;

#[cfg(feature = "native")]
mod backend_native;

#[cfg(feature = "wasmi")]
mod backend_wasmi;

macro_rules! define_backends {
    ($(
        #[cfg($($cfg:tt)*)]
        $backend:ident => $module:ident::$struct:ident($($ctor_args:tt)*)
    ),+) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        pub enum BackendKind {
            $(
                #[cfg($($cfg)*)]
                $backend
            ),+
        }

        #[allow(non_camel_case_types)]
        pub enum AnyEngine {
            $(
                #[cfg($($cfg)*)]
                $backend(<self::$module::$struct as Backend>::Engine)
            ),+
        }

        #[allow(non_camel_case_types)]
        pub enum AnyBlob {
            $(
                #[cfg($($cfg)*)]
                $backend(<self::$module::$struct as Backend>::Blob)
            ),+
        }

        #[allow(non_camel_case_types)]
        #[allow(clippy::large_enum_variant)]
        pub enum AnyModule {
            $(
                #[cfg($($cfg)*)]
                $backend(<self::$module::$struct as Backend>::Module)
            ),+
        }

        #[allow(non_camel_case_types)]
        #[allow(clippy::large_enum_variant)]
        pub enum AnyInstance {
            $(
                #[cfg($($cfg)*)]
                $backend(<self::$module::$struct as Backend>::Instance)
            ),+
        }

        impl Backend for BackendKind {
            type Engine = AnyEngine;
            type Blob = AnyBlob;
            type Module = AnyModule;
            type Instance = AnyInstance;

            fn name(&self) -> &'static str {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => self::$module::$struct($($ctor_args)*).name(),
                    )+
                }
            }

            fn create(&self, args: CreateArgs) -> Self::Engine {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => AnyEngine::$backend(self::$module::$struct($($ctor_args)*).create(args)),
                    )+
                }
            }

            fn load(&self, path: &Path) -> Self::Blob {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => AnyBlob::$backend(self::$module::$struct($($ctor_args)*).load(path)),
                    )+
                }
            }

            fn compile(&self, engine: &mut Self::Engine, blob: &Self::Blob) -> Self::Module {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => {
                            let AnyEngine::$backend(engine) = engine else { unreachable!() };
                            let AnyBlob::$backend(blob) = blob else { unreachable!() };
                            AnyModule::$backend(self::$module::$struct($($ctor_args)*).compile(engine, blob))
                        },
                    )+
                }
            }

            fn spawn(&self, engine: &mut Self::Engine, module: &Self::Module) -> Self::Instance {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => {
                            let AnyEngine::$backend(engine) = engine else { unreachable!() };
                            let AnyModule::$backend(module) = module else { unreachable!() };
                            AnyInstance::$backend(self::$module::$struct($($ctor_args)*).spawn(engine, module))
                        },
                    )+
                }
            }

            fn initialize(&self, instance: &mut Self::Instance) {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => {
                            let AnyInstance::$backend(instance) = instance else { unreachable!() };
                            self::$module::$struct($($ctor_args)*).initialize(instance)
                        },
                    )+
                }
            }

            fn run(&self, instance: &mut Self::Instance) {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => {
                            let AnyInstance::$backend(instance) = instance else { unreachable!() };
                            self::$module::$struct($($ctor_args)*).run(instance)
                        },
                    )+
                }
            }

            fn pid(&self, instance: &Self::Instance) -> Option<u32> {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => {
                            let AnyInstance::$backend(instance) = instance else { unreachable!() };
                            self::$module::$struct($($ctor_args)*).pid(instance)
                        },
                    )+
                }
            }

            fn is_slow(&self) -> bool {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => self::$module::$struct($($ctor_args)*).is_slow(),
                    )+
                }
            }

            fn is_compiled(&self) -> bool {
                match self {
                    $(
                        #[cfg($($cfg)*)]
                        Self::$backend => self::$module::$struct($($ctor_args)*).is_compiled(),
                    )+
                }
            }
        }
    };
}

define_backends! {
    #[cfg(not(dummy))] // A dummy cfg since the macro requires it.
    PolkaVM_Compiler32_NoGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, None, false),
    #[cfg(not(dummy))]
    PolkaVM_Compiler32_AsyncGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, Some(polkavm::GasMeteringKind::Async), false),
    #[cfg(not(dummy))]
    PolkaVM_Compiler32_SyncGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, Some(polkavm::GasMeteringKind::Sync), false),
    #[cfg(not(dummy))]
    PolkaVM_Compiler64_NoGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, None, true),
    #[cfg(not(dummy))]
    PolkaVM_Compiler64_AsyncGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, Some(polkavm::GasMeteringKind::Async), true),
    #[cfg(not(dummy))]
    PolkaVM_Compiler64_SyncGas => backend_polkavm::PolkaVM(polkavm::BackendKind::Compiler, Some(polkavm::GasMeteringKind::Sync), true),
    #[cfg(not(dummy))]
    PolkaVM_Interpreter32 => backend_polkavm::PolkaVM(polkavm::BackendKind::Interpreter, None, false),
    #[cfg(not(dummy))]
    PolkaVM_Interpreter64 => backend_polkavm::PolkaVM(polkavm::BackendKind::Interpreter, None, true),

    #[cfg(all(feature = "wasmtime", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Wasmtime_Cranelift =>
        backend_wasmtime::Wasmtime(wasmtime::Strategy::Cranelift, backend_wasmtime::Metering::None),

    #[cfg(all(feature = "wasmtime", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Wasmtime_CraneliftConsumeFuel =>
        backend_wasmtime::Wasmtime(wasmtime::Strategy::Cranelift, backend_wasmtime::Metering::ConsumeFuel),

    #[cfg(all(feature = "wasmtime", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Wasmtime_CraneliftEpochInterruption =>
        backend_wasmtime::Wasmtime(wasmtime::Strategy::Cranelift, backend_wasmtime::Metering::EpochInterruption),

    #[cfg(all(feature = "wasmtime", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Wasmtime_Winch =>
        backend_wasmtime::Wasmtime(wasmtime::Strategy::Winch, backend_wasmtime::Metering::None),

    #[cfg(all(feature = "wasmer", any(target_arch = "x86_64", target_arch = "aarch64")))]
    Wasmer => backend_wasmer::Wasmer(),

    #[cfg(all(feature = "wazero", any(target_arch = "x86_64", target_arch = "aarch64")))]
    WaZero => backend_wazero::WaZero(),

    #[cfg(all(feature = "pvf-executor", target_arch = "x86_64"))]
    PvfExecutor => backend_pvfexecutor::PvfExecutor(),

    #[cfg(all(feature = "ckb-vm", target_arch = "x86_64"))]
    Ckbvm_Asm => backend_ckbvm::Ckbvm(backend_ckbvm::CkbvmBackend::Asm),

    #[cfg(all(feature = "ckb-vm", target_arch = "x86_64"))]
    Ckbvm_NonAsm => backend_ckbvm::Ckbvm(backend_ckbvm::CkbvmBackend::NonAsm),

    #[cfg(all(feature = "solana_rbpf", target_arch = "x86_64", not(target_os = "windows")))]
    SolanaRbpf => backend_solana_rbpf::SolanaRbpf(),

    #[cfg(all(feature = "wasm3", target_arch = "x86_64"))]
    Wasm3 => backend_wasm3::Wasm3(),

    #[cfg(feature = "wasmi")]
    Wasmi_Eager => backend_wasmi::Wasmi(backend_wasmi::WasmiConfig::Eager),
    #[cfg(feature = "wasmi")]
    Wasmi_Lazy => backend_wasmi::Wasmi(backend_wasmi::WasmiConfig::Lazy),
    #[cfg(feature = "wasmi")]
    Wasmi_Lazy_Unchecked => backend_wasmi::Wasmi(backend_wasmi::WasmiConfig::LazyUnchecked),
    #[cfg(feature = "wasmi")]
    Wasmi_LazyTranslation => backend_wasmi::Wasmi(backend_wasmi::WasmiConfig::LazyTranslation),

    #[cfg(feature = "native")]
    Native => backend_native::Native()
}

impl BenchmarkKind {
    pub fn matching_backends(self) -> Vec<BackendKind> {
        let mut output = Vec::new();
        match self {
            BenchmarkKind::PolkaVM32 => {
                if polkavm::BackendKind::Compiler.is_supported() {
                    output.extend([
                        BackendKind::PolkaVM_Compiler32_NoGas,
                        BackendKind::PolkaVM_Compiler32_AsyncGas,
                        BackendKind::PolkaVM_Compiler32_SyncGas,
                    ]);
                }

                output.push(BackendKind::PolkaVM_Interpreter32);
            }
            BenchmarkKind::PolkaVM64 => {
                if polkavm::BackendKind::Compiler.is_supported() {
                    output.extend([
                        BackendKind::PolkaVM_Compiler64_NoGas,
                        BackendKind::PolkaVM_Compiler64_AsyncGas,
                        BackendKind::PolkaVM_Compiler64_SyncGas,
                    ]);
                }

                output.push(BackendKind::PolkaVM_Interpreter64);
            }
            BenchmarkKind::WebAssembly => {
                #[cfg(feature = "wasmi")]
                {
                    output.push(BackendKind::Wasmi_Eager);
                    output.push(BackendKind::Wasmi_Lazy);
                    output.push(BackendKind::Wasmi_LazyTranslation);
                }

                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                {
                    #[cfg(feature = "wasmtime")]
                    {
                        output.extend([
                            BackendKind::Wasmtime_Cranelift,
                            BackendKind::Wasmtime_CraneliftConsumeFuel,
                            BackendKind::Wasmtime_CraneliftEpochInterruption,
                            BackendKind::Wasmtime_Winch,
                        ]);
                    }

                    #[cfg(feature = "wasmer")]
                    output.push(BackendKind::Wasmer);

                    #[cfg(feature = "wazero")]
                    if backend_wazero::is_available() {
                        output.push(BackendKind::WaZero);
                    }
                }

                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(feature = "pvf-executor")]
                    output.push(BackendKind::PvfExecutor);
                    #[cfg(feature = "wasm3")]
                    output.push(BackendKind::Wasm3);
                }
            }
            BenchmarkKind::Ckbvm => {
                #[cfg(all(feature = "ckb-vm", target_arch = "x86_64"))]
                {
                    output.push(BackendKind::Ckbvm_Asm);
                    output.push(BackendKind::Ckbvm_NonAsm);
                }
            }
            BenchmarkKind::Solana => {
                #[cfg(all(feature = "solana_rbpf", target_arch = "x86_64", not(target_os = "windows")))]
                {
                    output.push(BackendKind::SolanaRbpf);
                }
            }
            BenchmarkKind::Native => {
                #[cfg(feature = "native")]
                output.push(BackendKind::Native);
            }
        }

        output
    }
}
