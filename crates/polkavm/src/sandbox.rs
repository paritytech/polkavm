use alloc::sync::Arc;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use polkavm_common::zygote::AddressTable;

use crate::api::{EngineState, MemoryProtection, Module};
use crate::compiler::CompiledModule;
use crate::config::{Config, SandboxKind};
use crate::error::Error;
use crate::utils::GuestInit;
use crate::{Gas, InterruptKind, MemoryAccessError, ProgramCounter, Reg, RegValue};

macro_rules! get_field_offset {
    ($struct:expr, |$struct_ident:ident| $get_field:expr) => {{
        let $struct_ident = $struct;
        let struct_ref: *const _ = &$struct_ident;
        let field_ptr: *const _ = $get_field;
        let struct_addr = struct_ref as usize;
        let field_addr = field_ptr as usize;
        field_addr - struct_addr
    }};
}

#[cfg(feature = "generic-sandbox")]
pub mod generic;

#[cfg(target_os = "linux")]
pub mod linux;

// This is literally the only thing we need from `libc` on Linux, so instead of including
// the whole crate let's just define these ourselves.
#[cfg(target_os = "linux")]
const _SC_PAGESIZE: core::ffi::c_int = 30;

#[cfg(target_os = "linux")]
extern "C" {
    fn sysconf(name: core::ffi::c_int) -> core::ffi::c_long;
}

#[cfg(not(target_os = "linux"))]
use libc::{sysconf, _SC_PAGESIZE};

static NATIVE_PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn init_native_page_size() {
    if NATIVE_PAGE_SIZE.load(Ordering::Relaxed) != 0 {
        return;
    }

    // SAFETY: This function has no safety invariants and should be always safe to call.
    let page_size = unsafe { sysconf(_SC_PAGESIZE) as usize };
    NATIVE_PAGE_SIZE.store(page_size, Ordering::Relaxed);
}

#[inline(always)]
pub(crate) fn get_native_page_size() -> usize {
    let page_size = NATIVE_PAGE_SIZE.load(Ordering::Relaxed);
    debug_assert_ne!(page_size, 0);

    page_size
}

pub trait SandboxConfig: Default {
    fn enable_logger(&mut self, value: bool);
    fn enable_sandboxing(&mut self, value: bool);
}

pub trait SandboxAddressSpace {
    fn native_code_origin(&self) -> u64;
}

pub trait SandboxProgram: Clone {
    fn machine_code(&self) -> &[u8];
}

pub struct OffsetTable {
    pub arg: usize,
    pub gas: usize,
    pub heap_info: usize,
    pub next_native_program_counter: usize,
    pub next_program_counter: usize,
    pub program_counter: usize,
    pub regs: usize,
    pub futex: usize,
}

pub(crate) trait Sandbox: Sized {
    const KIND: SandboxKind;

    type Config: SandboxConfig;
    type Error: core::fmt::Debug + core::fmt::Display;
    type Program: SandboxProgram;
    type AddressSpace: SandboxAddressSpace;
    type GlobalState;
    type JumpTable: AsRef<[usize]> + AsMut<[usize]>;

    fn downcast_module(module: &Module) -> &CompiledModule<Self>;
    fn downcast_global_state(global: &GlobalStateKind) -> &Self::GlobalState;

    fn allocate_jump_table(global: &Self::GlobalState, count: usize) -> Result<Self::JumpTable, Self::Error>;

    fn reserve_address_space() -> Result<Self::AddressSpace, Self::Error>;
    fn prepare_program(
        global: &Self::GlobalState,
        init: SandboxInit<Self>,
        address_space: Self::AddressSpace,
    ) -> Result<Self::Program, Self::Error>;
    fn spawn(global: &Self::GlobalState, config: &Self::Config, outer_instance: Option<&Self>) -> Result<Box<Self>, Self::Error>;
    fn load_module(&mut self, global: &Self::GlobalState, module: &Module) -> Result<(), Self::Error>;
    fn recycle(sandbox: Box<Self>, global: &Self::GlobalState) -> Result<(), Self::Error>;
    fn address_table() -> AddressTable;
    fn offset_table() -> OffsetTable;
    fn idle_worker_pids(global: &Self::GlobalState) -> Vec<u32>;

    fn run(&mut self) -> Result<InterruptKind, Self::Error>;
    fn reg(&self, reg: Reg) -> RegValue;
    fn set_reg(&mut self, reg: Reg, value: RegValue);
    fn gas(&self) -> Gas;
    fn set_gas(&mut self, gas: Gas);
    fn program_counter(&self) -> Option<ProgramCounter>;
    fn next_program_counter(&self) -> Option<ProgramCounter>;
    fn next_native_program_counter(&self) -> Option<usize>;
    fn set_next_program_counter(&mut self, pc: ProgramCounter);
    fn accessible_aux_size(&self) -> u32;
    fn set_accessible_aux_size(&mut self, size: u32) -> Result<(), Self::Error>;
    fn is_memory_accessible(&self, address: u32, size: u32, minimum_protection: MemoryProtection) -> bool;
    fn reset_memory(&mut self) -> Result<(), Self::Error>;
    fn read_memory_into<'slice>(&self, address: u32, slice: &'slice mut [MaybeUninit<u8>]) -> Result<&'slice mut [u8], MemoryAccessError>;
    fn write_memory(&mut self, address: u32, data: &[u8]) -> Result<(), MemoryAccessError>;
    fn zero_memory(&mut self, address: u32, length: u32, memory_protection: Option<MemoryProtection>) -> Result<(), MemoryAccessError>;
    fn change_memory_protection(&mut self, address: u32, length: u32, protection: MemoryProtection) -> Result<(), MemoryAccessError>;
    fn free_pages(&mut self, address: u32, length: u32) -> Result<(), Self::Error>;
    fn heap_size(&self) -> u32;
    fn sbrk(&mut self, size: u32) -> Result<Option<u32>, Self::Error>;
    fn pid(&self) -> Option<u32>;
}

#[derive(Copy, Clone, Default)]
pub struct SandboxInit<'a, S>
where
    S: Sandbox,
{
    pub guest_init: GuestInit<'a>,
    pub code: &'a [u8],
    pub jump_table: S::JumpTable,
    pub sysenter_address: u64,
    pub sysreturn_address: u64,
}

pub(crate) struct SandboxInstance<S>
where
    S: Sandbox,
{
    engine_state: Arc<EngineState>,
    sandbox: Option<Box<S>>,
}

impl<S> SandboxInstance<S>
where
    S: Sandbox,
{
    pub fn spawn_and_load_module(engine_state: Arc<EngineState>, module: &Module, outer_instance: Option<&Self>) -> Result<Self, Error> {
        use crate::sandbox::SandboxConfig;

        let mut sandbox_config = S::Config::default();
        sandbox_config.enable_logger(is_sandbox_logging_enabled());
        sandbox_config.enable_sandboxing(engine_state.sandboxing_enabled);

        let global = S::downcast_global_state(engine_state.sandbox_global.as_ref().unwrap());
        let mut sandbox = S::spawn(
            global,
            &sandbox_config,
            outer_instance.and_then(|instance| instance.sandbox.as_deref()),
        )
        .map_err(Error::from_display)
        .map_err(|error| error.context("instantiation failed: failed to create a sandbox"))?;

        let result = sandbox
            .load_module(global, module)
            .map_err(Error::from_display)
            .map_err(|error| error.context("instantiation failed: failed to upload the program into the sandbox"));

        if let Err(error) = result {
            if let Err(recycle_error) = S::recycle(sandbox, global) {
                log::warn!("Failed to recycle sandbox: {recycle_error}");
            }

            return Err(error);
        }

        Ok(SandboxInstance {
            sandbox: Some(sandbox),
            engine_state,
        })
    }

    pub fn sandbox(&self) -> &S {
        self.sandbox.as_ref().unwrap()
    }

    pub fn sandbox_mut(&mut self) -> &mut S {
        self.sandbox.as_mut().unwrap()
    }
}

impl<S> Drop for SandboxInstance<S>
where
    S: Sandbox,
{
    fn drop(&mut self) {
        if let Some(sandbox) = self.sandbox.take() {
            let global = S::downcast_global_state(self.engine_state.sandbox_global.as_ref().unwrap());
            if let Err(error) = S::recycle(sandbox, global) {
                log::warn!("Failed to recycle sandbox: {error}");
            }
        }
    }
}

pub(crate) enum GlobalStateKind {
    #[cfg(target_os = "linux")]
    Linux(crate::sandbox::linux::GlobalState),
    #[cfg(feature = "generic-sandbox")]
    Generic(crate::sandbox::generic::GlobalState),
}

impl GlobalStateKind {
    pub(crate) fn new(kind: SandboxKind, config: &Config) -> Result<Self, Error> {
        match kind {
            SandboxKind::Linux => {
                #[cfg(target_os = "linux")]
                {
                    Ok(Self::Linux(
                        crate::sandbox::linux::GlobalState::new(config)
                            .map_err(|error| format!("failed to initialize Linux sandbox: {error}"))?,
                    ))
                }

                #[cfg(not(target_os = "linux"))]
                {
                    unreachable!()
                }
            }
            SandboxKind::Generic => {
                #[cfg(feature = "generic-sandbox")]
                {
                    Ok(Self::Generic(
                        crate::sandbox::generic::GlobalState::new(config)
                            .map_err(|error| format!("failed to initialize generic sandbox: {error}"))?,
                    ))
                }

                #[cfg(not(feature = "generic-sandbox"))]
                {
                    unreachable!()
                }
            }
        }
    }

    pub(crate) fn idle_worker_pids(&self) -> Vec<u32> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(target_os = "linux")]
            GlobalStateKind::Linux(state) => crate::sandbox::linux::Sandbox::idle_worker_pids(state),
            _ => Vec::new(),
        }
    }
}

fn is_sandbox_logging_enabled() -> bool {
    cfg!(test) || log::log_enabled!(target: "polkavm", log::Level::Trace) || log::log_enabled!(target: "polkavm::zygote", log::Level::Trace)
}

// This is the same for both sandboxes.
#[cfg(any(target_os = "linux", feature = "generic-sandbox"))]
pub(crate) fn charge_gas_on_entry<S>(
    module: &Module,
    pc: ProgramCounter,
    native_address: u64,
    compiled_module: &CompiledModule<S>,
    gas: i64,
) -> Option<Result<i64, ()>>
where
    S: Sandbox,
{
    use polkavm_common::cast::cast;

    module.gas_metering()?;

    let Some(origin) = compiled_module.lookup_gas_metering_offset_for_basic_block_if_address_is_in_the_middle(native_address) else {
        log::debug!("Will not charge gas on entry: native address 0x{native_address:x} already points at the start of a basic block");
        return None;
    };

    let origin_address = compiled_module.native_code_offset_to_address(origin);
    let gas_cost = crate::compiler::extract_gas_cost::<S>(compiled_module.machine_code(), cast(origin).to_usize());
    let gas_cost = cast(cast(gas_cost).to_u64()).to_signed();
    if gas_cost > gas {
        log::debug!("Not enough gas to start execution at {pc} (0x{native_address:x}, gas metering stub at 0x{origin_address:x}): required={gas_cost}, got={gas}");
        return Some(Err(()));
    }

    let new_gas = gas - gas_cost;
    log::debug!("Charging gas on entry at {pc} (0x{native_address:x}, gas metering stub at 0x{origin_address:x}): {gas} -> {new_gas}");

    Some(Ok(new_gas))
}
