//! Hardware-virtualization sandbox backend (aarch64-only).
//!
//! Runs guest code in a vCPU with its own stage-1 MMU, giving 4K guest pages on
//! any host page size (Apple Hypervisor.framework on macOS; KVM on Linux later).
//!
//! Status: the micro-VM engine (`Vm`), the guest `VmCtx` layout / `offset_table` /
//! `address_table`, the memory + register plumbing, `prepare_program` and
//! `load_module` are implemented; `run`/`sbrk` are a first cut. NONE of the guest
//! *execution* path is runtime-verified — that needs a binary signed with the
//! `com.apple.security.hypervisor` entitlement. See the report / NOTEs below.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicI64, AtomicU32, AtomicU64};

use polkavm_common::abi::VM_ADDR_RETURN_TO_HOST;
#[cfg(target_os = "macos")]
use polkavm_common::regmap::to_native_reg;
use polkavm_common::zygote::{AddressTable, CacheAligned, VM_ADDR_JUMP_TABLE, VM_ADDR_JUMP_TABLE_RETURN_TO_HOST};

use super::{OffsetTable, SandboxInit, SandboxKind};
use crate::api::{MemoryAccessError, MemoryProtection, Module};
use crate::compiler::CompiledModule;
use crate::config::{Config, GasMeteringKind};
use crate::{Gas, InterruptKind, ProgramCounter, Reg, RegValue, Segfault};

/// Backend-local error type (mirrors the generic sandbox's `Error`).
#[derive(Debug)]
pub struct Error(alloc::string::String);

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str(&self.0)
    }
}

impl From<&'static str> for Error {
    fn from(value: &'static str) -> Self {
        Self(value.into())
    }
}

impl From<alloc::string::String> for Error {
    fn from(value: alloc::string::String) -> Self {
        Self(value)
    }
}

// ---------------------------------------------------------------------------
// Guest address-space layout (identity: guest VA == guest IPA).
//
//   [MAPPED_IPA ................................ MAPPED_IPA + MAPPED_LEN)   data (RWX)
//     +0 .............. +GUEST_MEM_OFFSET        VmCtx page (one 4K page)
//     +GUEST_MEM_OFFSET = GUEST_MEM_BASE ....... PolkaVM address 0 (x14 base), 4 GiB
//   [STUB_IPA ................................. STUB_IPA + CODE_WINDOW)     code (RX)
//     +0 ............. +STUB_REGION             hvc trampoline stubs (one page)
//     +STUB_REGION = NATIVE_CODE_ORIGIN ........ compiled code + inline jump table
//   [PT_IPA ...]                                 stage-1 page tables (walker-only)
//
// The VmCtx sits exactly 4096 bytes below the guest memory base because the shared
// AArch64 codegen addresses it as `x14 + GUEST_MEMORY_TO_VMCTX_OFFSET (-4096) + off`
// (see compiler/aarch64.rs). This is only correct when the `generic-sandbox` feature
// is also enabled (the codegen applies that offset only then); the hypervisor test
// matrix always enables both.
// ---------------------------------------------------------------------------

/// Host (stage-2 / hv_vm_map) page size. Apple Silicon is 16K; all hv_vm_map regions
/// must be aligned to it, even though the guest stage-1 MMU uses a 4K granule.
const HOST_PAGE: usize = 0x4000;
/// Start of the mapped data region in guest IPA space (16K-aligned).
const MAPPED_IPA: u64 = 0x4000_0000;
/// Bytes from the mapped region base to PolkaVM address 0. One host page, so the base
/// stays 16K-aligned; the VmCtx lives in the 4K page just below the base.
const GUEST_MEM_OFFSET: usize = HOST_PAGE;
/// Offset of the VmCtx within the mapped region (exactly 4K below the guest memory base).
const VMCTX_OFFSET: usize = GUEST_MEM_OFFSET - 0x1000;
/// Guest memory base (PolkaVM address 0); held in x14 while the guest runs.
const GUEST_MEM_BASE: u64 = MAPPED_IPA + GUEST_MEM_OFFSET as u64;
/// The 4 GiB PolkaVM address window.
const RAM_SIZE: usize = 0x1_0000_0000;
/// Total host-backed data mapping: VmCtx page + 4 GiB window.
const MAPPED_LEN: usize = GUEST_MEM_OFFSET + RAM_SIZE;
/// Guest page size enforced by the stage-1 MMU (4K granule, TG0 = 0b00).
const GUEST_PAGE_SIZE: usize = 4096;
/// Number of 4K guest pages backing the RAM window.
const NUM_PAGES: usize = RAM_SIZE / GUEST_PAGE_SIZE;

/// Base of the guest code region. First page: EL1 exception vector (VBAR_EL1) then the
/// hvc trampoline stubs. STUB_IPA is 2048-aligned as VBAR requires.
const STUB_IPA: u64 = 0x1_8000_0000;
/// Guest EL1 exception vector table (16 x 0x80). VBAR_EL1 points here.
const VBAR_IPA: u64 = STUB_IPA;
/// hvc trampoline stubs, placed just after the 2 KiB vector table.
const STUBS_IPA: u64 = STUB_IPA + 0x800;
/// Bytes reserved ahead of the code (vector + stubs), one page.
const STUB_REGION: usize = 0x1000;
/// Where the compiler is told the guest's native code lives.
const NATIVE_CODE_ORIGIN: u64 = STUB_IPA + STUB_REGION as u64;
/// Maximum code+jump-table window we identity-map for a module (512 MiB).
const CODE_WINDOW: usize = 0x2000_0000;
/// End (exclusive) of the IPA range covered by the identity page tables.
const COVER_END: u64 = STUB_IPA + CODE_WINDOW as u64;
/// Number of L1 (1 GiB) entries the identity tables span.
const NUM_L1: usize = (COVER_END - MAPPED_IPA).div_ceil(1 << 30) as usize;

/// hvc immediates distinguishing the trampoline that trapped (decoded from ESR ISS).
const HVC_HOSTCALL: u16 = 1;
const HVC_TRAP: u16 = 2;
const HVC_RETURN: u16 = 3;
const HVC_STEP: u16 = 4;
const HVC_SBRK: u16 = 5;
const HVC_NOT_ENOUGH_GAS: u16 = 6;
/// Emitted by the guest EL1 exception vector on an abort, to report it to the host.
const HVC_FAULT: u16 = 7;
/// Emitted by the TLB-maintenance stub once the invalidation has completed.
const HVC_TLBI: u16 = 8;
/// Stride between consecutive stubs (2 x 4-byte instructions).
const STUB_STRIDE: u64 = 8;
/// Number of `br`/`blr`-target stubs laid out at `STUBS_IPA` with `STUB_STRIDE` spacing.
const NUM_BR_STUBS: u64 = 6;
/// The TLB-maintenance stub, placed after them (4 instructions, so it needs its own slot).
const TLBI_STUB_IPA: u64 = STUBS_IPA + NUM_BR_STUBS * STUB_STRIDE;

/// Per-page protection state tracked host-side (mirrors the L3 descriptor AP bits).
const PROT_NONE: u8 = 0;
const PROT_READ: u8 = 1;
const PROT_READ_WRITE: u8 = 2;

// ---------------------------------------------------------------------------
// Guest VmCtx.
//
// NOTE: this MUST stay bit-identical (field order / types / sizes) to
// `generic::VmCtx`, because the shared AArch64 codegen reads these fields at the
// offsets reported by `offset_table()`, and `emit_gas_metering_stub` hard-asserts
// `gas` lands at 0x60. `maps`/`sandbox` are host-only stand-ins kept only so the
// following fields land at the right offsets.
// ---------------------------------------------------------------------------

const REG_COUNT: usize = 13;

#[repr(C)]
struct HeapInfo {
    heap_top: u64,
    heap_threshold: u64,
}

#[repr(C)]
#[allow(dead_code)] // variants selected by value when decoding exits
enum HvExitReason {
    None,
    Error,
    Signal,
    NotEnoughGas,
    Trap,
    Ecalli(u32),
    Segfault(u64),
    Step,
}

#[repr(C)]
struct VmCtx {
    return_address: usize,
    return_stack_pointer: usize,
    arg: AtomicU32,
    heap_info: HeapInfo,
    heap_base: u32,
    heap_initial_threshold: u32,
    heap_max_size: u32,
    heap_map_index: usize,
    page_size: u32,
    maps: Vec<u8>, // layout-compatible stand-in for generic's Vec<ProgramMap>
    gas: AtomicI64,
    program_range: core::ops::Range<u64>,
    exit_reason: HvExitReason,
    regs: CacheAligned<[RegValue; REG_COUNT]>,
    tmp_reg: AtomicU64,
    sandbox: *mut (),
    program_counter: AtomicU32,
    next_program_counter: AtomicU32,
    next_native_program_counter: AtomicU64,
    memset_continuation: AtomicU64,
}

impl VmCtx {
    fn new() -> Self {
        VmCtx {
            return_address: 0,
            return_stack_pointer: 0,
            arg: AtomicU32::new(0),
            heap_info: HeapInfo {
                heap_top: 0,
                heap_threshold: 0,
            },
            heap_base: 0,
            heap_initial_threshold: 0,
            heap_max_size: 0,
            heap_map_index: 0,
            page_size: 0,
            maps: Vec::new(),
            gas: AtomicI64::new(0),
            program_range: 0..0,
            exit_reason: HvExitReason::None,
            regs: CacheAligned([0; REG_COUNT]),
            tmp_reg: AtomicU64::new(0),
            sandbox: core::ptr::null_mut(),
            program_counter: AtomicU32::new(0),
            next_program_counter: AtomicU32::new(0),
            next_native_program_counter: AtomicU64::new(0),
            memset_continuation: AtomicU64::new(0),
        }
    }
}

polkavm_common::static_assert!(core::mem::size_of::<VmCtx>() <= 0x1000);

// ---------------------------------------------------------------------------
// Hypervisor.framework FFI.
//
// Transcribed from the macOS SDK headers under:
//   .../MacOSX.sdk/System/Library/Frameworks/Hypervisor.framework/Versions/A/Headers/
//   (hv_vm.h, hv_vm_types.h, hv_vcpu.h, hv_vcpu_types.h, hv_error.h)
// `hv_return_t` is `mach_error_t` (a C `int`); `HV_SUCCESS == 0`. The ARM memory
// flags live in the (kernel-only) hv_kern_types.h; READ/WRITE/EXEC are 1/2/4.
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
#[allow(non_camel_case_types, dead_code)] // full FFI surface kept for completeness
mod hv {
    use core::ffi::c_void;

    pub type hv_return_t = i32; // mach_error_t
    pub type hv_vcpu_t = u64;
    pub type hv_reg_t = u32;
    pub type hv_sys_reg_t = u16;
    pub type hv_ipa_t = u64;
    pub type hv_memory_flags_t = u64;
    pub type hv_exit_reason_t = u32;

    pub const HV_SUCCESS: hv_return_t = 0;

    // hv_memory_flags_t (hv_vm_map / hv_vm_protect).
    pub const HV_MEMORY_READ: hv_memory_flags_t = 1 << 0;
    pub const HV_MEMORY_WRITE: hv_memory_flags_t = 1 << 1;
    pub const HV_MEMORY_EXEC: hv_memory_flags_t = 1 << 2;

    // hv_reg_t (subset we touch).
    pub const HV_REG_X0: hv_reg_t = 0;
    pub const HV_REG_PC: hv_reg_t = 31;
    pub const HV_REG_CPSR: hv_reg_t = 34;

    // hv_sys_reg_t (from hv_vcpu_types.h).
    pub const HV_SYS_REG_SCTLR_EL1: hv_sys_reg_t = 0xc080;
    pub const HV_SYS_REG_CPACR_EL1: hv_sys_reg_t = 0xc082;
    pub const HV_SYS_REG_TTBR0_EL1: hv_sys_reg_t = 0xc100;
    pub const HV_SYS_REG_TCR_EL1: hv_sys_reg_t = 0xc102;
    pub const HV_SYS_REG_MAIR_EL1: hv_sys_reg_t = 0xc510;
    pub const HV_SYS_REG_VBAR_EL1: hv_sys_reg_t = 0xc600;
    pub const HV_SYS_REG_SP_EL1: hv_sys_reg_t = 0xe208;
    pub const HV_SYS_REG_ESR_EL1: hv_sys_reg_t = 0xc290;
    pub const HV_SYS_REG_FAR_EL1: hv_sys_reg_t = 0xc300;
    pub const HV_SYS_REG_ELR_EL1: hv_sys_reg_t = 0xc201;

    // hv_exit_reason_t (from hv_vcpu_types.h).
    pub const HV_EXIT_REASON_CANCELED: hv_exit_reason_t = 0;
    pub const HV_EXIT_REASON_EXCEPTION: hv_exit_reason_t = 1;
    pub const HV_EXIT_REASON_VTIMER_ACTIVATED: hv_exit_reason_t = 2;
    pub const HV_EXIT_REASON_UNKNOWN: hv_exit_reason_t = 3;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct hv_vcpu_exit_exception_t {
        pub syndrome: u64,
        pub virtual_address: u64,
        pub physical_address: hv_ipa_t,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct hv_vcpu_exit_t {
        pub reason: hv_exit_reason_t,
        pub exception: hv_vcpu_exit_exception_t,
    }

    #[link(name = "Hypervisor", kind = "framework")]
    extern "C" {
        pub fn hv_vm_create(config: *mut c_void) -> hv_return_t;
        pub fn hv_vm_destroy() -> hv_return_t;
        pub fn hv_vm_map(addr: *mut c_void, ipa: hv_ipa_t, size: usize, flags: hv_memory_flags_t) -> hv_return_t;
        pub fn hv_vm_unmap(ipa: hv_ipa_t, size: usize) -> hv_return_t;
        pub fn hv_vm_protect(ipa: hv_ipa_t, size: usize, flags: hv_memory_flags_t) -> hv_return_t;

        pub fn hv_vcpu_create(vcpu: *mut hv_vcpu_t, exit: *mut *const hv_vcpu_exit_t, config: *mut c_void) -> hv_return_t;
        pub fn hv_vcpu_destroy(vcpu: hv_vcpu_t) -> hv_return_t;
        pub fn hv_vcpu_run(vcpu: hv_vcpu_t) -> hv_return_t;
        pub fn hv_vcpus_exit(vcpus: *const hv_vcpu_t, vcpu_count: u32) -> hv_return_t;

        pub fn hv_vcpu_get_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: *mut u64) -> hv_return_t;
        pub fn hv_vcpu_set_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: u64) -> hv_return_t;
        pub fn hv_vcpu_get_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: *mut u64) -> hv_return_t;
        pub fn hv_vcpu_set_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: u64) -> hv_return_t;
    }
}

// ---------------------------------------------------------------------------
// The micro-VM engine.
// ---------------------------------------------------------------------------

/// Reason a `Vm::run` returned control to the host (decoded from ESR EC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum VmExit {
    /// Guest issued `HVC` (EC 0x16): a host-call / return-to-host trampoline; `imm` is the stub id.
    HostCall { imm: u16 },
    /// Data/instruction abort (EC 0x24/0x20): a guest memory fault.
    MemoryFault { address: u64, esr: u64 },
    /// Undefined instruction (EC 0x00): a guest trap.
    Trap { esr: u64 },
    /// Some other synchronous exception.
    Exception { ec: u64, esr: u64 },
    /// `hv_vcpus_exit` was called from another thread.
    Canceled,
    /// The virtual timer fired.
    VTimer,
    /// The framework could not classify the exit.
    Unknown,
}

// Stage-1 descriptor bits (4K granule).
#[cfg(target_os = "macos")]
const DESC_VALID: u64 = 1 << 0;
#[cfg(target_os = "macos")]
const DESC_PAGE: u64 = 0b11; // valid + page/table
#[cfg(target_os = "macos")]
const PTE_ATTRINDX0: u64 = 0 << 2;
#[cfg(target_os = "macos")]
const PTE_AP_RW_EL1: u64 = 0 << 6; // EL1 read/write, no EL0
#[cfg(target_os = "macos")]
const PTE_AP_RO_EL1: u64 = 2 << 6; // EL1 read-only, no EL0
#[cfg(target_os = "macos")]
const PTE_AP_MASK: u64 = 0b11 << 6;
#[cfg(target_os = "macos")]
const PTE_SH_INNER: u64 = 3 << 8;
#[cfg(target_os = "macos")]
const PTE_AF: u64 = 1 << 10;
#[cfg(target_os = "macos")]
const PTE_OUTPUT_MASK: u64 = 0x0000_ffff_ffff_f000; // output address bits [47:12]

/// A live micro-VM: backing RAM + identity 4K page tables + a code region + one vCPU.
pub struct Vm {
    #[cfg(target_os = "macos")]
    ram: *mut u8,
    #[cfg(target_os = "macos")]
    ram_len: usize,
    #[cfg(target_os = "macos")]
    pt: *mut u8,
    #[cfg(target_os = "macos")]
    pt_len: usize,
    /// IPA at which the page-table region is mapped.
    #[cfg(target_os = "macos")]
    pt_ipa: u64,
    /// Next free table page in the PT region (for on-demand L2/L3 chains).
    #[cfg(target_os = "macos")]
    pt_next: usize,
    /// Total table pages available in the PT region.
    #[cfg(target_os = "macos")]
    pt_cap: usize,
    /// Guest code region (stubs + compiled code), mapped once a module is loaded.
    #[cfg(target_os = "macos")]
    code: *mut u8,
    #[cfg(target_os = "macos")]
    code_len: usize,
    /// Host backing for the sparse return-to-host jump-table slot page.
    #[cfg(target_os = "macos")]
    slot: *mut u8,
    #[cfg(target_os = "macos")]
    slot_len: usize,
    #[cfg(target_os = "macos")]
    slot_ipa: u64,
    #[cfg(target_os = "macos")]
    vcpu: hv::hv_vcpu_t,
    /// Whether `vcpu` holds a live vCPU (its id can legitimately be 0, so no sentinel).
    #[cfg(target_os = "macos")]
    has_vcpu: bool,
    #[cfg(target_os = "macos")]
    exit: *const hv::hv_vcpu_exit_t,
    /// Set when the host edits the stage-1 tables; cleared by `flush_tlb` before the next entry.
    #[cfg(target_os = "macos")]
    tlb_dirty: bool,
    /// Held for the VM's whole lifetime: only one HV VM may exist per process, so
    /// concurrent sandboxes serialize here instead of failing. Released on drop.
    #[cfg(target_os = "macos")]
    _guard: std::sync::MutexGuard<'static, ()>,
}

// SAFETY: the raw pointers refer to this VM's own mappings; ownership moves with the box.
unsafe impl Send for Vm {}
// SAFETY: guest RAM/page tables are plain memory; a vCPU is only ever driven from the
// owning instance's thread. Mirrors the generic sandbox's Send+Sync treatment of `Mmap`.
unsafe impl Sync for Vm {}

/// How long a `spawn` waits for a live instance to be dropped before giving up.
#[cfg(target_os = "macos")]
const VM_ACQUIRE_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);

/// Serializes VM creation/teardown: Hypervisor.framework allows only one VM per process.
#[cfg(target_os = "macos")]
static VM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_os = "macos")]
thread_local! {
    /// Whether the current thread owns the live VM (its vCPU is bound to this thread).
    static VM_HELD: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

impl Vm {
    /// Create the VM: mmap RAM, create the HV VM, map RAM + page tables, build the
    /// identity 4K tables, create the vCPU and inject the MMU sysregs.
    pub fn new() -> Result<Self, Error> {
        #[cfg(target_os = "macos")]
        {
            // One VM per process, one vCPU per thread: fail cleanly instead of self-deadlocking.
            let start = std::time::Instant::now();
            if VM_HELD.with(core::cell::Cell::get) {
                return Err("hypervisor sandbox: this thread already owns the single per-process VM; \
                    drop the previous instance before creating another (nested/simultaneous instances are unsupported)"
                    .into());
            }
            // Bounded wait: the owner may be the caller's own live instance, which never yields.
            let guard = loop {
                match VM_LOCK.try_lock() {
                    Ok(guard) => break guard,
                    Err(std::sync::TryLockError::Poisoned(error)) => break error.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => {
                        if start.elapsed() >= VM_ACQUIRE_TIMEOUT {
                            return Err("hypervisor sandbox: timed out waiting for the single per-process VM \
                                (Hypervisor.framework allows one VM per process and one vCPU per thread, so \
                                instances cannot overlap)"
                                .into());
                        }
                        std::thread::sleep(core::time::Duration::from_millis(1));
                    }
                }
            };
            VM_HELD.with(|h| h.set(true));
            let result = Self::new_macos(guard);
            if result.is_err() {
                VM_HELD.with(|h| h.set(false));
            }
            result
        }
        #[cfg(not(target_os = "macos"))]
        {
            // TODO(hypervisor): implement the KVM-backed engine for aarch64-linux.
            todo!("hypervisor sandbox: no Hypervisor.framework on this OS (KVM backend not yet implemented)")
        }
    }

    #[cfg(target_os = "macos")]
    fn new_macos(guard: std::sync::MutexGuard<'static, ()>) -> Result<Self, Error> {
        // Cover [MAPPED_IPA, COVER_END): one L1 table, one L2 per 1 GiB, one L3 per 2 MiB.
        let l1_span = 1u64 << 30;
        let num_l1 = (COVER_END - MAPPED_IPA).div_ceil(l1_span) as usize;
        let num_l3 = num_l1 * 512;
        let identity_tables = 1 + num_l1 + num_l3;
        // A few spare table pages back on-demand chains (e.g. the sparse return slot).
        const SPARE_TABLES: usize = 8;
        let pt_len = align_up((identity_tables + SPARE_TABLES) * GUEST_PAGE_SIZE, HOST_PAGE);
        let pt_cap = pt_len / GUEST_PAGE_SIZE;
        let pt_ipa = COVER_END; // walker-only; needs no stage-1 mapping

        let ram = map_anon(MAPPED_LEN)?;
        let pt = match map_anon(pt_len) {
            Ok(pt) => pt,
            Err(error) => {
                unmap_anon(ram, MAPPED_LEN);
                return Err(error);
            }
        };

        let mut vm = Vm {
            ram,
            ram_len: MAPPED_LEN,
            pt,
            pt_len,
            pt_ipa,
            pt_next: identity_tables,
            pt_cap,
            code: core::ptr::null_mut(),
            code_len: 0,
            slot: core::ptr::null_mut(),
            slot_len: 0,
            slot_ipa: 0,
            vcpu: 0,
            has_vcpu: false,
            exit: core::ptr::null(),
            tlb_dirty: false,
            _guard: guard,
        };

        if let Err(error) = vm.bringup(num_l1) {
            vm.teardown();
            return Err(error);
        }

        Ok(vm)
    }

    #[cfg(target_os = "macos")]
    fn bringup(&mut self, num_l1: usize) -> Result<(), Error> {
        // Create the per-process VM.
        // SAFETY: FFI call; a null config selects the framework default.
        check(unsafe { hv::hv_vm_create(core::ptr::null_mut()) }, "hv_vm_create")?;

        // Map the data region (RW only; Apple's Hypervisor.framework forbids W+X). Code
        // is mapped R+X separately in `load_code`.
        check(
            // SAFETY: `self.ram`/`self.ram_len` describe a live anonymous mapping.
            unsafe { hv::hv_vm_map(self.ram.cast(), MAPPED_IPA, self.ram_len, hv::HV_MEMORY_READ | hv::HV_MEMORY_WRITE) },
            "hv_vm_map(ram)",
        )?;
        // Map the page-table region (RW).
        check(
            // SAFETY: `self.pt`/`self.pt_len` describe a live anonymous mapping.
            unsafe { hv::hv_vm_map(self.pt.cast(), self.pt_ipa, self.pt_len, hv::HV_MEMORY_READ | hv::HV_MEMORY_WRITE) },
            "hv_vm_map(page-tables)",
        )?;

        self.build_identity_4k_tables(num_l1);

        // Create the vCPU on the current thread.
        let mut vcpu: hv::hv_vcpu_t = 0;
        let mut exit: *const hv::hv_vcpu_exit_t = core::ptr::null();
        check(
            // SAFETY: FFI call; `vcpu`/`exit` are valid out-pointers, null config = default.
            unsafe { hv::hv_vcpu_create(&mut vcpu, &mut exit, core::ptr::null_mut()) },
            "hv_vcpu_create",
        )?;
        self.vcpu = vcpu;
        self.has_vcpu = true;
        self.exit = exit;

        self.inject_mmu_sysregs()?;
        Ok(())
    }

    /// Build identity 4K page tables covering `[MAPPED_IPA, MAPPED_IPA + num_l1 GiB)`.
    ///
    /// Layout in the PT region: `[L1][L2 x num_l1][L3 x (num_l1 * 512)]`, contiguous.
    // The PT region is `mmap`-allocated and thus page-aligned, so the `*mut u8` -> `*mut u64`
    // casts and every 8-byte-aligned table offset within it are correctly aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    fn build_identity_4k_tables(&mut self, num_l1: usize) {
        let l1_start_idx = (MAPPED_IPA >> 30) as usize & 0x1ff;
        let table_ipa = |table_index: usize| self.pt_ipa + (table_index as u64) * GUEST_PAGE_SIZE as u64;

        // L1 table is table 0; L2 tables are 1..=num_l1; L3 tables follow.
        let l1 = self.pt.cast::<u64>();
        for i in 0..num_l1 {
            let entry = table_ipa(1 + i) | DESC_PAGE;
            // SAFETY: `l1_start_idx + i` is a valid entry index within the L1 table page.
            unsafe { l1.add(l1_start_idx + i).write(entry) };
        }

        for i in 0..num_l1 {
            // SAFETY: L2 table `i` lives at table offset `1 + i`, inside the PT region.
            let l2 = unsafe { self.pt.add((1 + i) * GUEST_PAGE_SIZE).cast::<u64>() };
            for j in 0..512usize {
                let entry = table_ipa(1 + num_l1 + i * 512 + j) | DESC_PAGE;
                // SAFETY: `j` is a valid entry index (0..512) within this L2 table page.
                unsafe { l2.add(j).write(entry) };
            }
        }

        for t in 0..(num_l1 * 512) {
            // SAFETY: L3 table `t` lives at table offset `1 + num_l1 + t`, inside the PT region.
            let l3 = unsafe { self.pt.add((1 + num_l1 + t) * GUEST_PAGE_SIZE).cast::<u64>() };
            for m in 0..512usize {
                let page_va = MAPPED_IPA + ((t * 512 + m) * GUEST_PAGE_SIZE) as u64;
                let entry = (page_va & PTE_OUTPUT_MASK) | PTE_AF | PTE_SH_INNER | PTE_AP_RW_EL1 | PTE_ATTRINDX0 | DESC_PAGE;
                // SAFETY: `m` is a valid entry index (0..512) within this L3 table page.
                unsafe { l3.add(m).write(entry) };
            }
        }
    }

    /// Invalidate every L3 descriptor covering the PolkaVM data window so unmapped
    /// accesses fault (dynamic paging). Iterates the L3 tables directly for speed.
    // The PT region is page-aligned so the `*mut u8` -> `*mut u64` casts are aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    fn clear_window(&mut self) {
        self.tlb_dirty = true;
        let start = GUEST_MEM_OFFSET / GUEST_PAGE_SIZE; // covered-page index of PolkaVM address 0
        let end = start + NUM_PAGES;
        for p in start..end {
            // SAFETY: covered page `p` maps to L3 table `1 + NUM_L1 + p/512`, inside the PT region.
            let l3 = unsafe { self.pt.add((1 + NUM_L1 + p / 512) * GUEST_PAGE_SIZE).cast::<u64>() };
            // SAFETY: `p % 512` is a valid entry index in that table.
            unsafe {
                let e = l3.add(p % 512).read();
                l3.add(p % 512).write(e & !DESC_VALID);
            }
        }
    }

    /// Grab a spare, zeroed table page from the PT region; returns (host ptr, IPA).
    // The PT region is page-aligned so the `*mut u8` -> `*mut u64` cast is aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    fn alloc_table_page(&mut self) -> Result<(*mut u64, u64), Error> {
        if self.pt_next >= self.pt_cap {
            return Err("out of spare page-table pages".into());
        }
        let idx = self.pt_next;
        self.pt_next += 1;
        let ipa = self.pt_ipa + (idx * GUEST_PAGE_SIZE) as u64;
        // SAFETY: `idx < pt_cap`, so the page lies within the PT mapping; mmap zeroed it.
        let host = unsafe { self.pt.add(idx * GUEST_PAGE_SIZE).cast::<u64>() };
        Ok((host, ipa))
    }

    /// Map a single read-only guest page holding `value` at `slot_va`, wiring an
    /// on-demand L1->L2->L3 chain. Used for the sparse return-to-host jump-table slot.
    // The PT region is page-aligned so the `*mut u8` -> `*mut u64` casts are aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    fn map_return_slot(&mut self, slot_va: u64, value: u64) -> Result<(), Error> {
        self.tlb_dirty = true;
        // Back the containing host page and map it read-only into the guest IPA space.
        let page_base = slot_va & !(HOST_PAGE as u64 - 1);
        if self.slot.is_null() {
            self.slot = map_anon(HOST_PAGE)?;
            self.slot_len = HOST_PAGE;
            self.slot_ipa = page_base;
            check(
                // SAFETY: `self.slot`/HOST_PAGE describe a live anonymous mapping.
                unsafe { hv::hv_vm_map(self.slot.cast(), page_base, HOST_PAGE, hv::HV_MEMORY_READ) },
                "hv_vm_map(return-slot)",
            )?;
        }
        // SAFETY: offset is within the freshly-mapped HOST_PAGE region.
        unsafe {
            self.slot.add((slot_va - page_base) as usize).cast::<u64>().write_unaligned(value);
        }

        // Wire L1 -> L2 -> L3 for the 4K guest page containing `slot_va`.
        let l1 = self.pt.cast::<u64>();
        let l1i = (slot_va >> 30) as usize & 0x1ff;
        // SAFETY: `l1i < 512`; L1 table is the first PT page.
        let l1d = unsafe { l1.add(l1i).read() };
        let l2 = if l1d & DESC_VALID == 0 {
            let (host, ipa) = self.alloc_table_page()?;
            // SAFETY: `l1i` in range; installing the new L2 table descriptor.
            unsafe { l1.add(l1i).write(ipa | DESC_PAGE) };
            host
        } else {
            // SAFETY: descriptor output is an IPA inside the PT region.
            unsafe { self.pt.add(((l1d & PTE_OUTPUT_MASK) - self.pt_ipa) as usize).cast::<u64>() }
        };
        let l2i = (slot_va >> 21) as usize & 0x1ff;
        // SAFETY: `l2i < 512`; `l2` is a valid table page.
        let l2d = unsafe { l2.add(l2i).read() };
        let l3 = if l2d & DESC_VALID == 0 {
            let (host, ipa) = self.alloc_table_page()?;
            // SAFETY: `l2i` in range; installing the new L3 table descriptor.
            unsafe { l2.add(l2i).write(ipa | DESC_PAGE) };
            host
        } else {
            // SAFETY: descriptor output is an IPA inside the PT region.
            unsafe { self.pt.add(((l2d & PTE_OUTPUT_MASK) - self.pt_ipa) as usize).cast::<u64>() }
        };
        let l3i = (slot_va >> 12) as usize & 0x1ff;
        let entry = (slot_va & PTE_OUTPUT_MASK) | PTE_AF | PTE_SH_INNER | PTE_AP_RO_EL1 | PTE_ATTRINDX0 | DESC_PAGE;
        // SAFETY: `l3i < 512`; `l3` is a valid table page.
        unsafe { l3.add(l3i).write(entry) };
        Ok(())
    }

    /// Program the vCPU's stage-1 MMU and drop it into EL1h with the MMU enabled.
    #[cfg(target_os = "macos")]
    fn inject_mmu_sysregs(&mut self) -> Result<(), Error> {
        // TCR_EL1: T0SZ=25 (39-bit VA), IRGN0/ORGN0 = WB-WA, SH0 = inner-shareable,
        // TG0 = 0b00 (4K, so no TG0 bits set), EPD1 = 1 (disable TTBR1), IPS = 2 (40-bit PA).
        let tcr: u64 = 25 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2 << 32);
        // MAIR: attr0 = Normal, inner/outer write-back non-transient (0xFF).
        let mair: u64 = 0xFF;

        self.set_sys_reg(hv::HV_SYS_REG_MAIR_EL1, mair)?;
        self.set_sys_reg(hv::HV_SYS_REG_TCR_EL1, tcr)?;
        self.set_sys_reg(hv::HV_SYS_REG_TTBR0_EL1, self.pt_ipa)?;
        self.set_sys_reg(hv::HV_SYS_REG_VBAR_EL1, VBAR_IPA)?;
        // Hardware SP scratch for the trampolines (e.g. sbrk's push/pop): the RW region
        // just below the VmCtx, growing down. Distinct from the guest's Reg::SP.
        self.set_sys_reg(hv::HV_SYS_REG_SP_EL1, MAPPED_IPA + VMCTX_OFFSET as u64)?;
        // Allow FP/SIMD at EL0/EL1 (FPEN = 0b11).
        self.set_sys_reg(hv::HV_SYS_REG_CPACR_EL1, 3 << 20)?;

        // SCTLR_EL1: keep reset value, add M (MMU), C (data cache), I (instr cache).
        let sctlr = self.get_sys_reg(hv::HV_SYS_REG_SCTLR_EL1)?;
        self.set_sys_reg(hv::HV_SYS_REG_SCTLR_EL1, sctlr | 1 | (1 << 2) | (1 << 12))?;

        // CPSR: EL1h with DAIF masked.
        self.set_reg(hv::HV_REG_CPSR, 0x3c5)?;
        Ok(())
    }

    /// Map the code region (stubs + compiled code) at STUB_IPA as RX and install the
    /// hvc trampoline stubs. `code` is the compiled machine code (+ inline jump table).
    #[cfg(target_os = "macos")]
    fn load_code(&mut self, code: &[u8]) -> Result<(), Error> {
        let total = STUB_REGION + code.len();
        if total > CODE_WINDOW {
            return Err("compiled code exceeds the mapped code window".into());
        }
        let map_len = align_up(total, HOST_PAGE); // hv_vm_map needs 16K multiples

        // Reuse the existing mapping if it is large enough, otherwise (re)allocate.
        if self.code.is_null() || map_len > self.code_len {
            if !self.code.is_null() {
                // SAFETY: unmapping our own live code region before replacing it.
                unsafe { hv::hv_vm_unmap(STUB_IPA, self.code_len) };
                unmap_anon(self.code, self.code_len);
                self.code = core::ptr::null_mut();
                self.code_len = 0;
            }
            let ptr = map_anon(map_len)?;
            check(
                // SAFETY: `ptr`/`map_len` describe a live anonymous mapping.
                unsafe { hv::hv_vm_map(ptr.cast(), STUB_IPA, map_len, hv::HV_MEMORY_READ | hv::HV_MEMORY_EXEC) },
                "hv_vm_map(code)",
            )?;
            self.code = ptr;
            self.code_len = map_len;
        }

        // Install the hvc stubs in the first page and copy the code after it.
        self.install_stubs();
        // SAFETY: `code.len() + STUB_REGION <= map_len`; destination is within the mapping.
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), self.code.add(STUB_REGION), code.len());
            // Keep the guest instruction cache coherent with the writes above.
            sys_dcache_flush(self.code.cast(), map_len);
            sys_icache_invalidate(self.code.cast(), map_len);
        }
        Ok(())
    }

    /// Write the EL1 exception vector and the `hvc`-based trampoline stubs into the
    /// first code page.
    #[cfg(target_os = "macos")]
    fn install_stubs(&mut self) {
        let hvc = |imm: u16| 0xD400_0002u32 | (u32::from(imm) << 5); // hvc #imm

        // Exception vector: every 0x80-aligned entry does `hvc #FAULT` so any guest
        // EL1 exception (esp. a stage-1 data/instruction abort) is reported to the host.
        let vbar_off = (VBAR_IPA - STUB_IPA) as usize;
        for entry in 0..16usize {
            let off = vbar_off + entry * 0x80;
            // SAFETY: 16 * 0x80 = 2 KiB, within the reserved stub page.
            unsafe { self.code.add(off).cast::<u32>().write_unaligned(hvc(HVC_FAULT)) };
        }

        // br-targets (hostcall/trap/return/step/not_enough_gas): a single `hvc #imm`;
        // the host stops the run loop and re-enters later via sysenter. sbrk is a
        // blr-target so it needs `hvc #imm; ret` to return into its trampoline.
        let ret = 0xD65F03C0u32; // ret x30
        let stubs = [
            (HVC_HOSTCALL, false),
            (HVC_TRAP, false),
            (HVC_RETURN, false),
            (HVC_STEP, false),
            (HVC_SBRK, true),
            (HVC_NOT_ENOUGH_GAS, false),
        ];
        let stubs_off = (STUBS_IPA - STUB_IPA) as usize;
        debug_assert_eq!(stubs.len() as u64, NUM_BR_STUBS);
        for (nth, (imm, needs_ret)) in stubs.iter().enumerate() {
            let off = stubs_off + nth * STUB_STRIDE as usize;
            // SAFETY: the stubs occupy STUBS_IPA.. within the reserved stub page.
            unsafe {
                self.code.add(off).cast::<u32>().write_unaligned(hvc(*imm));
                let second = if *needs_ret { ret } else { 0x0000_0000u32 }; // udf #0 filler
                self.code.add(off + 4).cast::<u32>().write_unaligned(second);
            }
        }

        // TLB maintenance; the host enters here after editing the page tables (see `flush_tlb`).
        const TLBI_VMALLE1: u32 = 0xD508_871F;
        const DSB_ISH: u32 = 0xD503_3B9F;
        const ISB: u32 = 0xD503_3FDF;
        let tlbi_off = (TLBI_STUB_IPA - STUB_IPA) as usize;
        // SAFETY: 4 instructions at TLBI_STUB_IPA, still within the reserved stub page.
        unsafe {
            for (nth, word) in [TLBI_VMALLE1, DSB_ISH, ISB, hvc(HVC_TLBI)].into_iter().enumerate() {
                self.code.add(tlbi_off + nth * 4).cast::<u32>().write_unaligned(word);
            }
        }
    }

    /// Locate the L3 descriptor for a guest VA by walking the tables in host memory.
    // The PT region is page-aligned, so the `*mut u8` -> `*mut u64` casts are aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn l3_entry_ptr(&self, va: u64) -> *mut u64 {
        let host_of_ipa = |ipa: u64| self.pt.add((ipa - self.pt_ipa) as usize).cast::<u64>();

        let l1 = self.pt.cast::<u64>();
        let l1d = l1.add((va >> 30) as usize & 0x1ff).read();
        let l2 = host_of_ipa(l1d & PTE_OUTPUT_MASK);
        let l2d = l2.add((va >> 21) as usize & 0x1ff).read();
        let l3 = host_of_ipa(l2d & PTE_OUTPUT_MASK);
        l3.add((va >> 12) as usize & 0x1ff)
    }

    /// Flip the AP bits of the L3 descriptors for `[va, va + len)` (PolkaVM addresses).
    /// This is where the 4K granule matters: protections are per 4K page.
    #[cfg(target_os = "macos")]
    fn set_page_protection(&mut self, va: u32, len: u32, ap_bits: u64) {
        if len == 0 {
            return;
        }
        self.tlb_dirty = true;
        let first = u64::from(va) & !(GUEST_PAGE_SIZE as u64 - 1);
        let last = (u64::from(va) + u64::from(len) - 1) & !(GUEST_PAGE_SIZE as u64 - 1);
        let mut page_va = first;
        while page_va <= last {
            // SAFETY: `page_va` maps into the identity-mapped window, so the walk resolves
            // to a real L3 descriptor in the PT region.
            let entry_ptr = unsafe { self.l3_entry_ptr(GUEST_MEM_BASE + page_va) };
            // SAFETY: `entry_ptr` points at a live, aligned L3 descriptor.
            let entry = unsafe { entry_ptr.read() };
            let entry = (entry & !PTE_AP_MASK) | (ap_bits & PTE_AP_MASK) | DESC_PAGE;
            // SAFETY: same descriptor pointer; writing back the updated entry.
            unsafe { entry_ptr.write(entry) };
            page_va += GUEST_PAGE_SIZE as u64;
        }
    }

    /// Invalidate (make faulting) the L3 descriptors for `[va, va + len)`.
    #[cfg(target_os = "macos")]
    fn invalidate_pages(&mut self, va: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.tlb_dirty = true;
        let first = u64::from(va) & !(GUEST_PAGE_SIZE as u64 - 1);
        let last = (u64::from(va) + u64::from(len) - 1) & !(GUEST_PAGE_SIZE as u64 - 1);
        let mut page_va = first;
        while page_va <= last {
            // SAFETY: `page_va` maps into the identity-mapped window; the walk resolves
            // to a real L3 descriptor in the PT region.
            let entry_ptr = unsafe { self.l3_entry_ptr(GUEST_MEM_BASE + page_va) };
            // SAFETY: `entry_ptr` points at a live, aligned L3 descriptor.
            let entry = unsafe { entry_ptr.read() };
            // SAFETY: same descriptor pointer; clearing the valid bit.
            unsafe { entry_ptr.write(entry & !DESC_VALID) };
            page_va += GUEST_PAGE_SIZE as u64;
        }
    }

    /// Invalidate the guest's stage-1 TLB if the host edited the page tables. Only the guest can do
    /// it, so enter briefly at the TLBI stub: it touches no GPRs and the caller resets PC anyway.
    #[cfg(target_os = "macos")]
    fn flush_tlb(&mut self) -> Result<(), Error> {
        if !self.tlb_dirty {
            return Ok(());
        }
        self.set_reg(hv::HV_REG_PC, TLBI_STUB_IPA)?;
        match self.run_raw()? {
            VmExit::HostCall { imm: HVC_TLBI } => {
                self.tlb_dirty = false;
                Ok(())
            }
            other => Err(alloc::format!("unexpected exit while invalidating the guest TLB: {other:?}").into()),
        }
    }

    /// Host pointer to the guest VmCtx (one 4K page below the guest memory base).
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)] // `ram` is page-aligned from mmap; VMCTX_OFFSET is 4K-aligned
    fn vmctx_ptr(&self) -> *mut VmCtx {
        // SAFETY: `VMCTX_OFFSET` is within the mapped region.
        unsafe { self.ram.add(VMCTX_OFFSET).cast::<VmCtx>() }
    }

    /// Run the vCPU and decode the exit reason.
    #[allow(dead_code)]
    pub fn run_raw(&mut self) -> Result<VmExit, Error> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: FFI call on this thread's own vCPU.
            check(unsafe { hv::hv_vcpu_run(self.vcpu) }, "hv_vcpu_run")?;
            // SAFETY: `exit` points at framework-owned storage updated by hv_vcpu_run.
            let exit = unsafe { &*self.exit };
            let vm_exit = match exit.reason {
                hv::HV_EXIT_REASON_CANCELED => VmExit::Canceled,
                hv::HV_EXIT_REASON_VTIMER_ACTIVATED => VmExit::VTimer,
                hv::HV_EXIT_REASON_UNKNOWN => VmExit::Unknown,
                hv::HV_EXIT_REASON_EXCEPTION => {
                    let esr = exit.exception.syndrome;
                    let ec = (esr >> 26) & 0x3f;
                    match ec {
                        0x16 => VmExit::HostCall {
                            imm: (esr & 0xffff) as u16,
                        }, // HVC
                        0x24 | 0x20 => VmExit::MemoryFault {
                            address: exit.exception.virtual_address,
                            esr,
                        }, // Data / Instruction Abort
                        0x00 => VmExit::Trap { esr }, // Undefined instruction
                        _ => VmExit::Exception { ec, esr },
                    }
                }
                _ => VmExit::Unknown,
            };
            Ok(vm_exit)
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: KVM run loop not yet implemented")
        }
    }

    #[allow(dead_code)]
    pub fn get_reg(&self, reg: u32) -> Result<u64, Error> {
        #[cfg(target_os = "macos")]
        {
            let mut value = 0u64;
            // SAFETY: FFI call; `value` is a valid out-pointer.
            check(unsafe { hv::hv_vcpu_get_reg(self.vcpu, reg, &mut value) }, "hv_vcpu_get_reg")?;
            Ok(value)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = reg;
            todo!("hypervisor sandbox: KVM get_reg not yet implemented")
        }
    }

    #[allow(dead_code)]
    pub fn set_reg(&mut self, reg: u32, value: u64) -> Result<(), Error> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: FFI call on this thread's own vCPU.
            check(unsafe { hv::hv_vcpu_set_reg(self.vcpu, reg, value) }, "hv_vcpu_set_reg")
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (reg, value);
            todo!("hypervisor sandbox: KVM set_reg not yet implemented")
        }
    }

    #[allow(dead_code)]
    pub fn get_sys_reg(&self, reg: u16) -> Result<u64, Error> {
        #[cfg(target_os = "macos")]
        {
            let mut value = 0u64;
            // SAFETY: FFI call; `value` is a valid out-pointer.
            let ret = unsafe { hv::hv_vcpu_get_sys_reg(self.vcpu, reg, &mut value) };
            check(ret, "hv_vcpu_get_sys_reg")?;
            Ok(value)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = reg;
            todo!("hypervisor sandbox: KVM get_sys_reg not yet implemented")
        }
    }

    #[allow(dead_code)]
    pub fn set_sys_reg(&mut self, reg: u16, value: u64) -> Result<(), Error> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: FFI call on this thread's own vCPU.
            check(unsafe { hv::hv_vcpu_set_sys_reg(self.vcpu, reg, value) }, "hv_vcpu_set_sys_reg")
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (reg, value);
            todo!("hypervisor sandbox: KVM set_sys_reg not yet implemented")
        }
    }

    /// Host-side view of guest RAM at PolkaVM `[addr, addr + len)`.
    #[cfg(target_os = "macos")]
    fn ram_slice(&self, addr: u32, len: usize) -> Option<&[u8]> {
        let end = u64::from(addr).checked_add(len as u64)?;
        if end > RAM_SIZE as u64 {
            return None;
        }
        // SAFETY: bounds checked against the 4 GiB window; skip the VmCtx page.
        Some(unsafe { core::slice::from_raw_parts(self.ram.add(GUEST_MEM_OFFSET + addr as usize), len) })
    }

    #[cfg(target_os = "macos")]
    fn ram_slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
        let end = u64::from(addr).checked_add(len as u64)?;
        if end > RAM_SIZE as u64 {
            return None;
        }
        // SAFETY: bounds checked against the 4 GiB window; skip the VmCtx page.
        Some(unsafe { core::slice::from_raw_parts_mut(self.ram.add(GUEST_MEM_OFFSET + addr as usize), len) })
    }

    #[cfg(target_os = "macos")]
    fn teardown(&mut self) {
        // vCPU (if created) must be destroyed before the VM.
        if self.has_vcpu {
            // SAFETY: FFI call; destroying this thread's own vCPU.
            unsafe { hv::hv_vcpu_destroy(self.vcpu) };
            self.has_vcpu = false;
        }
        // SAFETY: FFI call; all vCPUs have been destroyed above.
        unsafe { hv::hv_vm_destroy() };
        if !self.ram.is_null() {
            unmap_anon(self.ram, self.ram_len);
            self.ram = core::ptr::null_mut();
        }
        if !self.pt.is_null() {
            unmap_anon(self.pt, self.pt_len);
            self.pt = core::ptr::null_mut();
        }
        if !self.code.is_null() {
            unmap_anon(self.code, self.code_len);
            self.code = core::ptr::null_mut();
        }
        if !self.slot.is_null() {
            unmap_anon(self.slot, self.slot_len);
            self.slot = core::ptr::null_mut();
        }
        // Release this thread's ownership; the VM_LOCK guard drops after this returns.
        VM_HELD.with(|h| h.set(false));
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if !self.ram.is_null() {
                self.teardown();
            }
        }
    }
}

/// mmap an anonymous, private, read-write host region.
#[cfg(target_os = "macos")]
fn map_anon(len: usize) -> Result<*mut u8, Error> {
    // SAFETY: standard anonymous mmap; result is checked below.
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err("mmap failed".into());
    }
    Ok(ptr.cast())
}

#[cfg(target_os = "macos")]
fn unmap_anon(ptr: *mut u8, len: usize) {
    // SAFETY: `ptr`/`len` come from a matching `map_anon`.
    unsafe { libc::munmap(ptr.cast(), len) };
}

#[cfg(target_os = "macos")]
fn check(ret: hv::hv_return_t, what: &str) -> Result<(), Error> {
    if ret == hv::HV_SUCCESS {
        Ok(())
    } else {
        Err(alloc::format!("{what} failed: hv_return_t=0x{:08x}", ret as u32).into())
    }
}

#[cfg(target_os = "macos")]
fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

// Apple's cache-maintenance helpers (from <libkern/OSCacheControl.h>). Signatures match
// the generic sandbox's declaration to avoid a cross-module redeclaration clash.
#[cfg(target_os = "macos")]
extern "C" {
    fn sys_icache_invalidate(start: *mut core::ffi::c_void, len: libc::size_t);
    fn sys_dcache_flush(start: *mut core::ffi::c_void, len: libc::size_t);
}

// ---------------------------------------------------------------------------
// Sandbox trait plumbing.
// ---------------------------------------------------------------------------

/// Per-engine state shared across sandboxes.
pub struct GlobalState {}

impl GlobalState {
    pub fn new(_config: &Config) -> Result<Self, Error> {
        Ok(GlobalState {})
    }
}

#[derive(Default)]
pub struct SandboxConfig {}

impl super::SandboxConfig for SandboxConfig {
    fn enable_logger(&mut self, _value: bool) {}
    fn enable_sandboxing(&mut self, _value: bool) {}
}

/// One region to materialize into guest RAM when a module is loaded.
#[derive(Clone)]
struct ProgramMap {
    address: u32,
    length: u32,
    is_writable: bool,
    /// Initial bytes; `None` means zero-filled.
    data: Option<Arc<[u8]>>,
}

/// A prepared program: guest machine code + jump table + memory layout.
#[derive(Clone)]
pub struct SandboxProgram(Arc<SandboxProgramInner>);

struct SandboxProgramInner {
    /// Compiled code followed by the inline jump table (as emitted by the compiler).
    code: Vec<u8>,
    /// Raw jump-table bytes (already appended to `code` by the compiler pipeline layout).
    jump_table: Vec<u8>,
    sysenter_address: u64,
    // Written into the return-to-host jump-table slot (see load_module).
    sysreturn_address: u64,
    memory_map: Vec<ProgramMap>,
    heap_map_index: usize,
}

impl super::SandboxProgram for SandboxProgram {
    fn machine_code(&self) -> &[u8] {
        &self.0.code
    }
}

/// Reserved guest address space backing the stage-1 MMU.
pub struct AddressSpace {
    native_code_origin: u64,
}

impl super::SandboxAddressSpace for AddressSpace {
    fn native_code_origin(&self) -> u64 {
        self.native_code_origin
    }
}

/// A live sandbox: a vCPU with its own guest memory.
pub struct Sandbox {
    vm: Vm,
    module: Option<Module>,
    program: Option<SandboxProgram>,
    gas_metering: Option<GasMeteringKind>,
    regs: [RegValue; REG_COUNT],
    gas: Gas,
    program_counter: Option<ProgramCounter>,
    is_program_counter_valid: bool,
    next_program_counter: Option<ProgramCounter>,
    next_program_counter_changed: bool,
    charge_gas_on_entry: bool,
    /// Page size faults are reported at; descriptors stay 4K, so any `GUEST_PAGE_SIZE` multiple works.
    page_size: u32,
    /// Aux data the guest may read, as set by `set_accessible_aux_size`.
    accessible_aux_size: u32,
    aux_data_address: u32,
    aux_data_full_length: u32,
    heap_base: u32,
    heap_top: u32,
    dynamic_paging: bool,
    /// Per-4K-page protection state, indexed by page number within the RAM window.
    page_prot: Vec<u8>,
}

impl Drop for Sandbox {
    fn drop(&mut self) {}
}

/// Bounds-check a `[address, address + length)` access against the 4 GiB RAM window.
fn bounds_ok(address: u32, length: u32) -> bool {
    u64::from(address) + u64::from(length) <= RAM_SIZE as u64
}

impl super::Sandbox for Sandbox {
    const KIND: SandboxKind = SandboxKind::Hypervisor;

    type Config = SandboxConfig;
    type Error = Error;
    type Program = SandboxProgram;
    type AddressSpace = AddressSpace;
    type GlobalState = GlobalState;
    type JumpTable = Vec<usize>;

    fn downcast_module(module: &Module) -> &CompiledModule<Self> {
        #[allow(unreachable_patterns, clippy::match_wildcard_for_single_variants)]
        match module.compiled_module() {
            crate::api::CompiledModuleKind::Hypervisor(ref module) => module,
            _ => unreachable!(),
        }
    }

    fn downcast_global_state(global: &crate::sandbox::GlobalStateKind) -> &Self::GlobalState {
        #[allow(unreachable_patterns, clippy::match_wildcard_for_single_variants)]
        match global {
            crate::sandbox::GlobalStateKind::Hypervisor(global) => global,
            _ => unreachable!(),
        }
    }

    fn allocate_jump_table(_global: &Self::GlobalState, count: usize) -> Result<Self::JumpTable, Self::Error> {
        // Byte size must be a whole number of native pages (see compiler.rs finish_compilation).
        let page = crate::sandbox::get_native_page_size();
        let size = polkavm_common::utils::align_to_next_page_usize(page, count * core::mem::size_of::<usize>()).unwrap();
        Ok(vec![0; size / core::mem::size_of::<usize>()])
    }

    fn reserve_address_space() -> Result<Self::AddressSpace, Self::Error> {
        Ok(AddressSpace {
            native_code_origin: NATIVE_CODE_ORIGIN,
        })
    }

    fn prepare_program(
        _global: &Self::GlobalState,
        init: SandboxInit<Self>,
        _address_space: Self::AddressSpace,
    ) -> Result<Self::Program, Self::Error> {
        let cfg = init.guest_init.memory_map().map_err(Error::from)?;

        // The jump table is a separate byte blob; the compiler places it right after
        // the code, reached via a PC-relative `adr` (see compiler/aarch64.rs).
        let jump_table = as_bytes(init.jump_table.as_ref()).to_vec();

        let mut memory_map = Vec::new();
        let mut push_region = |address: u32, data: &[u8], virtual_size: u32, is_writable: bool| {
            if virtual_size == 0 {
                return;
            }
            let physical = core::cmp::min(data.len() as u32, virtual_size);
            if physical > 0 {
                memory_map.push(ProgramMap {
                    address,
                    length: physical,
                    is_writable,
                    data: Some(Arc::from(&data[..physical as usize])),
                });
            }
            if virtual_size > physical {
                memory_map.push(ProgramMap {
                    address: address + physical,
                    length: virtual_size - physical,
                    is_writable,
                    data: None,
                });
            }
        };

        push_region(cfg.ro_data_address(), init.guest_init.ro_data, cfg.ro_data_size(), false);
        push_region(cfg.rw_data_address(), init.guest_init.rw_data, cfg.rw_data_size(), true);

        // Reserve a (transient) entry for the heap right after rw-data.
        let heap_map_index = memory_map.len();
        memory_map.push(ProgramMap {
            address: cfg.rw_data_range().end,
            length: 0,
            is_writable: true,
            data: None,
        });

        if cfg.stack_size() > 0 {
            memory_map.push(ProgramMap {
                address: cfg.stack_address_low(),
                length: cfg.stack_size(),
                is_writable: true,
                data: None,
            });
        }
        if cfg.aux_data_size() > 0 {
            memory_map.push(ProgramMap {
                address: cfg.aux_data_address(),
                length: cfg.aux_data_size(),
                is_writable: true,
                data: None,
            });
        }

        Ok(SandboxProgram(Arc::new(SandboxProgramInner {
            code: init.code.to_vec(),
            jump_table,
            sysenter_address: init.sysenter_address,
            sysreturn_address: init.sysreturn_address,
            memory_map,
            heap_map_index,
        })))
    }

    fn spawn(_global: &Self::GlobalState, _config: &Self::Config, _outer_instance: Option<&Self>) -> Result<Box<Self>, Self::Error> {
        let vm = Vm::new()?;
        Ok(Box::new(Sandbox {
            vm,
            module: None,
            program: None,
            gas_metering: None,
            regs: [0; REG_COUNT],
            gas: 0,
            program_counter: None,
            is_program_counter_valid: false,
            next_program_counter: None,
            next_program_counter_changed: false,
            charge_gas_on_entry: true,
            page_size: GUEST_PAGE_SIZE as u32,
            accessible_aux_size: 0,
            aux_data_address: 0,
            aux_data_full_length: 0,
            heap_base: 0,
            heap_top: 0,
            dynamic_paging: false,
            page_prot: vec![PROT_READ_WRITE; NUM_PAGES],
        }))
    }

    fn load_module(&mut self, _global: &Self::GlobalState, module: &Module) -> Result<(), Self::Error> {
        #[cfg(target_os = "macos")]
        {
            let compiled = Self::downcast_module(module);
            let program = compiled.sandbox_program.clone();

            // Assemble the code + inline jump table blob and map it RX.
            let mut blob = program.0.code.clone();
            // The compiler pads code to a page and defines the jump-table label at the end;
            // append the jump-table bytes so the PC-relative `adr` lands on real data.
            blob.extend_from_slice(&program.0.jump_table);
            self.vm.load_code(&blob)?;

            // The return-to-host jump-table slot lives far into the (sparse) virtual jump
            // table, at `jump_table_base + RETURN_TO_HOST*8`. `jump_table_base` is the
            // padded end of the code (where the codegen's `adr` points). Map that single
            // page and store the sysreturn stub address so a `ret` reaches the host.
            let jump_table_base = NATIVE_CODE_ORIGIN + program.0.code.len() as u64;
            let slot_va = jump_table_base + (u64::from(VM_ADDR_RETURN_TO_HOST) << 3);
            debug_assert_eq!(slot_va, jump_table_base + (VM_ADDR_JUMP_TABLE_RETURN_TO_HOST - VM_ADDR_JUMP_TABLE));
            self.vm.map_return_slot(slot_va, program.0.sysreturn_address)?;

            self.dynamic_paging = module.is_dynamic_paging();
            // Start with the whole window inaccessible so out-of-bounds guest accesses fault
            // (static -> Trap, dynamic -> Segfault), mirroring the generic sandbox.
            self.vm.clear_window();
            self.page_prot.fill(PROT_NONE);
            if !self.dynamic_paging {
                // Static paging: materialize the memory map into guest RAM and enable it.
                for map in &program.0.memory_map {
                    if map.length == 0 {
                        continue;
                    }
                    if let Some(data) = &map.data {
                        self.vm
                            .ram_slice_mut(map.address, data.len())
                            .ok_or_else(|| Error::from("memory map region out of range"))?
                            .copy_from_slice(data);
                    }
                    let ap = if map.is_writable { PTE_AP_RW_EL1 } else { PTE_AP_RO_EL1 };
                    self.vm.set_page_protection(map.address, map.length, ap);
                    let state = if map.is_writable { PROT_READ_WRITE } else { PROT_READ };
                    mark_pages(&mut self.page_prot, map.address, map.length, state);
                }
            }
            // Dynamic paging leaves everything faulted; the host pages it in on demand.

            // Initialize the guest VmCtx.
            let mmap = module.memory_map();
            // SAFETY: `vmctx_ptr` points at the mapped, zero-initialized VmCtx page.
            unsafe {
                core::ptr::write(self.vm.vmctx_ptr(), VmCtx::new());
                let vmctx = &mut *self.vm.vmctx_ptr();
                vmctx.heap_info.heap_top = u64::from(mmap.heap_base());
                vmctx.heap_info.heap_threshold = u64::from(mmap.rw_data_range().end);
                vmctx.heap_base = mmap.heap_base();
                vmctx.heap_initial_threshold = mmap.rw_data_range().end;
                vmctx.heap_max_size = mmap.max_heap_size();
                vmctx.heap_map_index = program.0.heap_map_index;
                vmctx.page_size = mmap.page_size();
            }

            self.heap_base = mmap.heap_base();
            self.heap_top = mmap.heap_base();
            if mmap.page_size() as usize % GUEST_PAGE_SIZE != 0 {
                return Err(alloc::format!(
                    "module page size ({}) must be a multiple of the guest granule ({GUEST_PAGE_SIZE})",
                    mmap.page_size()
                )
                .into());
            }
            self.page_size = mmap.page_size();
            self.aux_data_address = mmap.aux_data_address();
            self.aux_data_full_length = mmap.aux_data_size();
            // Fully accessible until the host narrows it.
            self.accessible_aux_size = self.aux_data_full_length;
            self.gas_metering = module.gas_metering();
            self.program = Some(program);
            self.module = Some(module.clone());
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = module;
            todo!("hypervisor sandbox: load_module needs the KVM backend")
        }
    }

    fn recycle(_sandbox: Box<Self>, _global: &Self::GlobalState) -> Result<(), Self::Error> {
        // Dropping the box tears down the vCPU + VM.
        Ok(())
    }

    /// Guest addresses of the `hvc` trampoline stubs (see `install_stubs`).
    fn address_table() -> AddressTable {
        AddressTable {
            syscall_hostcall: STUBS_IPA + u64::from(HVC_HOSTCALL - 1) * STUB_STRIDE,
            syscall_trap: STUBS_IPA + u64::from(HVC_TRAP - 1) * STUB_STRIDE,
            syscall_return: STUBS_IPA + u64::from(HVC_RETURN - 1) * STUB_STRIDE,
            syscall_step: STUBS_IPA + u64::from(HVC_STEP - 1) * STUB_STRIDE,
            syscall_sbrk: STUBS_IPA + u64::from(HVC_SBRK - 1) * STUB_STRIDE,
            syscall_not_enough_gas: STUBS_IPA + u64::from(HVC_NOT_ENOUGH_GAS - 1) * STUB_STRIDE,
        }
    }

    /// Offsets into the guest `VmCtx`; must match the shared AArch64 codegen.
    fn offset_table() -> OffsetTable {
        OffsetTable {
            arg: get_field_offset!(VmCtx::new(), |base| base.arg.as_ptr()),
            gas: get_field_offset!(VmCtx::new(), |base| base.gas.as_ptr()),
            heap_info: get_field_offset!(VmCtx::new(), |base| &base.heap_info),
            next_native_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_native_program_counter.as_ptr()),
            next_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_program_counter.as_ptr()),
            program_counter: get_field_offset!(VmCtx::new(), |base| base.program_counter.as_ptr()),
            regs: get_field_offset!(VmCtx::new(), |base| &base.regs),
            memset_continuation: get_field_offset!(VmCtx::new(), |base| base.memset_continuation.as_ptr()),
            futex: usize::MAX,
        }
    }

    fn idle_worker_pids(_global: &Self::GlobalState) -> Vec<u32> {
        Vec::new()
    }

    // First cut: enter via the `sysenter` stub, run the vCPU, and translate the
    // `hvc` / abort exits into `InterruptKind`, mirroring generic.rs:1880-1896.
    // NOTE(hypervisor): unverified — needs the hypervisor entitlement to execute, and
    // full correctness also depends on codegen assumptions documented in the report.
    fn run(&mut self) -> Result<InterruptKind, Self::Error> {
        #[cfg(target_os = "macos")]
        {
            let module = self.module.as_ref().ok_or_else(|| Error::from("no module loaded"))?.clone();
            let compiled = Self::downcast_module(&module);

            if self.next_program_counter_changed {
                let pc = self
                    .next_program_counter
                    .ok_or_else(|| Error::from("next program counter not set"))?;
                let Some(address) = compiled.lookup_native_code_address(pc) else {
                    // `program_counter()` reads this back out of the VmCtx.
                    // SAFETY: the VmCtx page is mapped and initialized.
                    unsafe {
                        (*self.vm.vmctx_ptr())
                            .program_counter
                            .store(pc.0, core::sync::atomic::Ordering::Relaxed);
                    }
                    self.program_counter = Some(pc);
                    self.is_program_counter_valid = true;
                    return Ok(InterruptKind::Trap);
                };
                if self.charge_gas_on_entry {
                    match crate::sandbox::charge_gas_on_entry(&module, pc, address, compiled, self.gas) {
                        Some(Ok(new_gas)) => self.gas = new_gas,
                        Some(Err(())) => return Ok(InterruptKind::NotEnoughGas),
                        None => {}
                    }
                }
                self.next_program_counter_changed = false;
                self.charge_gas_on_entry = false;
                // SAFETY: the VmCtx page is mapped and initialized.
                unsafe {
                    let vmctx = &mut *self.vm.vmctx_ptr();
                    vmctx.next_program_counter.store(pc.0, core::sync::atomic::Ordering::Relaxed);
                    vmctx
                        .next_native_program_counter
                        .store(address, core::sync::atomic::Ordering::Relaxed);
                }
                self.next_program_counter = None;
            }

            let sysenter = self.program.as_ref().unwrap().0.sysenter_address;

            // Push host state into the guest VmCtx before entering.
            // SAFETY: the VmCtx page is mapped and initialized.
            unsafe {
                let vmctx = &mut *self.vm.vmctx_ptr();
                vmctx.exit_reason = HvExitReason::None;
                vmctx.gas.store(self.gas, core::sync::atomic::Ordering::Relaxed);
                for (i, value) in self.regs.iter().enumerate() {
                    vmctx.regs.0[i] = *value;
                }
            }

            self.is_program_counter_valid = true;
            let interrupt = self.execute(sysenter)?;

            // Pull guest state back out.
            // SAFETY: the VmCtx page is mapped and initialized.
            unsafe {
                let vmctx = &*self.vm.vmctx_ptr();
                self.gas = vmctx.gas.load(core::sync::atomic::Ordering::Relaxed);
                for (i, value) in self.regs.iter_mut().enumerate() {
                    *value = vmctx.regs.0[i];
                }
            }

            // Async metering doesn't trap in-guest; the counter is only checked on the way out.
            if self.gas_metering == Some(GasMeteringKind::Async) && self.gas < 0 {
                self.is_program_counter_valid = false;
                // SAFETY: the VmCtx page is mapped and initialized.
                unsafe {
                    (*self.vm.vmctx_ptr())
                        .next_native_program_counter
                        .store(0, core::sync::atomic::Ordering::Relaxed);
                }
                return Ok(InterruptKind::NotEnoughGas);
            }

            Ok(interrupt)
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: run needs the KVM backend")
        }
    }

    fn reg(&self, reg: Reg) -> RegValue {
        self.regs[reg as usize]
    }

    fn set_reg(&mut self, reg: Reg, value: RegValue) {
        self.regs[reg as usize] = value;
    }

    fn gas(&self) -> Gas {
        self.gas
    }

    fn set_gas(&mut self, gas: Gas) {
        self.gas = gas;
    }

    fn program_counter(&self) -> Option<ProgramCounter> {
        if !self.is_program_counter_valid {
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            // Written by the guest stubs and the fault paths.
            // SAFETY: the VmCtx page is mapped and initialized.
            let vmctx = unsafe { &*self.vm.vmctx_ptr() };
            Some(ProgramCounter(vmctx.program_counter.load(core::sync::atomic::Ordering::Relaxed)))
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.program_counter
        }
    }

    fn next_program_counter(&self) -> Option<ProgramCounter> {
        if self.next_program_counter.is_some() {
            return self.next_program_counter;
        }
        // After an in-guest exit (e.g. ecall) the resume PC lives in the VmCtx.
        #[cfg(target_os = "macos")]
        {
            use core::sync::atomic::Ordering;
            // SAFETY: the VmCtx page is mapped and initialized once a module is loaded.
            let vmctx = unsafe { &*self.vm.vmctx_ptr() };
            if vmctx.next_native_program_counter.load(Ordering::Relaxed) == 0 {
                None
            } else {
                Some(ProgramCounter(vmctx.next_program_counter.load(Ordering::Relaxed)))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    fn next_native_program_counter(&self) -> Option<usize> {
        #[cfg(target_os = "macos")]
        {
            // SAFETY: the VmCtx page is mapped and zeroed before a module is loaded.
            let vmctx = unsafe { &*self.vm.vmctx_ptr() };
            match vmctx.next_native_program_counter.load(core::sync::atomic::Ordering::Relaxed) {
                0 => None,
                address => Some(address as usize),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    fn set_next_program_counter(&mut self, pc: ProgramCounter) {
        self.is_program_counter_valid = false;
        self.next_program_counter = Some(pc);
        self.next_program_counter_changed = true;
        self.charge_gas_on_entry = true;
    }

    fn accessible_aux_size(&self) -> u32 {
        self.accessible_aux_size
    }

    fn set_accessible_aux_size(&mut self, size: u32) -> Result<(), Self::Error> {
        if self.aux_data_address == 0 || self.aux_data_full_length == 0 {
            return Err(Error::from("aux data address or length is zero"));
        }
        if size > self.aux_data_full_length {
            return Err(Error::from("size exceeds the full length of aux data"));
        }
        self.accessible_aux_size = size;
        Ok(())
    }

    fn is_memory_accessible(&self, address: u32, size: u32, minimum_protection: MemoryProtection) -> bool {
        if size == 0 {
            return true;
        }
        if !bounds_ok(address, size) {
            return false;
        }
        let required = match minimum_protection {
            MemoryProtection::Read => PROT_READ,
            MemoryProtection::ReadWrite => PROT_READ_WRITE,
        };
        self.accessible_for(address, size, required)
    }

    fn reset_memory(&mut self) -> Result<(), Self::Error> {
        #[cfg(target_os = "macos")]
        {
            if self.module.is_none() {
                return Err("no module loaded into the sandbox".into());
            }
            if self.dynamic_paging {
                return self.free_pages(0x10000, 0xffff_0000);
            }

            // Static paging: the guest can only have written where the map made it writable, so
            // restore those regions instead of scrubbing the whole 4 GiB window.
            let program = self.program.clone().ok_or_else(|| Error::from("no program loaded"))?;

            // Anything `sbrk` grew into lies outside the map; revoke and clear it.
            // SAFETY: the VmCtx page is mapped and initialized.
            let (threshold, initial) = unsafe {
                let vmctx = &*self.vm.vmctx_ptr();
                (vmctx.heap_info.heap_threshold as u32, vmctx.heap_initial_threshold)
            };
            if threshold > initial {
                let length = threshold - initial;
                if let Some(slice) = self.vm.ram_slice_mut(initial, length as usize) {
                    slice.fill(0);
                }
                self.vm.invalidate_pages(initial, length);
                mark_pages(&mut self.page_prot, initial, length, PROT_NONE);
            }

            for map in &program.0.memory_map {
                if !map.is_writable || map.length == 0 {
                    continue;
                }
                let Some(slice) = self.vm.ram_slice_mut(map.address, map.length as usize) else {
                    return Err("memory map region out of range".into());
                };
                match &map.data {
                    Some(data) => slice.copy_from_slice(data),
                    None => slice.fill(0),
                }
            }

            self.heap_top = self.heap_base;
            // SAFETY: the VmCtx page is mapped and initialized.
            unsafe {
                let vmctx = &mut *self.vm.vmctx_ptr();
                vmctx.heap_info.heap_top = u64::from(self.heap_base);
                vmctx.heap_info.heap_threshold = u64::from(initial);
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: reset_memory needs the KVM backend")
        }
    }

    fn read_memory_into<'slice>(&self, address: u32, slice: &'slice mut [MaybeUninit<u8>]) -> Result<&'slice mut [u8], MemoryAccessError> {
        if slice.is_empty() {
            // SAFETY: empty slice; nothing is read.
            return Ok(unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), 0) });
        }
        if !bounds_ok(address, slice.len() as u32) || !self.accessible_for(address, slice.len() as u32, PROT_READ) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: slice.len() as u64,
            });
        }
        #[cfg(target_os = "macos")]
        {
            let src = self.vm.ram_slice(address, slice.len()).ok_or(MemoryAccessError::OutOfRangeAccess {
                address,
                length: slice.len() as u64,
            })?;
            for (dst, byte) in slice.iter_mut().zip(src.iter()) {
                dst.write(*byte);
            }
            // SAFETY: every element was just initialized.
            Ok(unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) })
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: guest RAM access needs the KVM backend")
        }
    }

    fn write_memory(&mut self, address: u32, data: &[u8]) -> Result<(), MemoryAccessError> {
        if data.is_empty() {
            return Ok(());
        }
        if !bounds_ok(address, data.len() as u32) || !self.accessible_for(address, data.len() as u32, PROT_READ_WRITE) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: data.len() as u64,
            });
        }
        #[cfg(target_os = "macos")]
        {
            let dst = self
                .vm
                .ram_slice_mut(address, data.len())
                .ok_or(MemoryAccessError::OutOfRangeAccess {
                    address,
                    length: data.len() as u64,
                })?;
            dst.copy_from_slice(data);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: guest RAM access needs the KVM backend")
        }
    }

    fn zero_memory(&mut self, address: u32, length: u32, memory_protection: Option<MemoryProtection>) -> Result<(), MemoryAccessError> {
        if length == 0 {
            return Ok(());
        }
        if !bounds_ok(address, length) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: u64::from(length),
            });
        }
        // `Some(protection)` = dynamic-paging page-in: make the pages present with that
        // protection, then zero them. `None` requires the pages already be writable.
        if let Some(protection) = memory_protection {
            #[cfg(target_os = "macos")]
            {
                let (ap, state) = match protection {
                    MemoryProtection::Read => (PTE_AP_RO_EL1, PROT_READ),
                    MemoryProtection::ReadWrite => (PTE_AP_RW_EL1, PROT_READ_WRITE),
                };
                if let Some(dst) = self.vm.ram_slice_mut(address, length as usize) {
                    dst.fill(0);
                }
                self.vm.set_page_protection(address, length, ap);
                mark_pages(&mut self.page_prot, address, length, state);
                return Ok(());
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = protection;
                todo!("hypervisor sandbox: guest RAM access needs the KVM backend")
            }
        }
        if !self.accessible_for(address, length, PROT_READ_WRITE) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: u64::from(length),
            });
        }
        #[cfg(target_os = "macos")]
        {
            let dst = self
                .vm
                .ram_slice_mut(address, length as usize)
                .ok_or(MemoryAccessError::OutOfRangeAccess {
                    address,
                    length: u64::from(length),
                })?;
            dst.fill(0);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: guest RAM access needs the KVM backend")
        }
    }

    fn change_memory_protection(&mut self, address: u32, length: u32, protection: MemoryProtection) -> Result<(), MemoryAccessError> {
        if length == 0 {
            return Ok(());
        }
        if !bounds_ok(address, length) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: u64::from(length),
            });
        }
        #[cfg(target_os = "macos")]
        {
            // Only present pages can be re-protected.
            if !self.accessible_for(address, length, PROT_READ) {
                return Err(MemoryAccessError::OutOfRangeAccess {
                    address,
                    length: u64::from(length),
                });
            }
            let (ap, state) = match protection {
                MemoryProtection::Read => (PTE_AP_RO_EL1, PROT_READ),
                MemoryProtection::ReadWrite => (PTE_AP_RW_EL1, PROT_READ_WRITE),
            };
            self.vm.set_page_protection(address, length, ap);
            mark_pages(&mut self.page_prot, address, length, state);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = protection;
            todo!("hypervisor sandbox: page protection needs the KVM backend")
        }
    }

    fn free_pages(&mut self, address: u32, length: u32) -> Result<(), Self::Error> {
        if length == 0 {
            return Ok(());
        }
        if !bounds_ok(address, length) {
            return Err("free_pages: out of range".into());
        }
        #[cfg(target_os = "macos")]
        {
            // Walk the page tables only for pages that are actually present: freeing the whole
            // window is the common case (`reset_memory`) and is nearly always sparse.
            let first = address as usize / GUEST_PAGE_SIZE;
            let last = (address as usize + length as usize - 1) / GUEST_PAGE_SIZE;
            let mut page = first;
            while page <= last {
                if self.page_prot[page] == PROT_NONE {
                    page += 1;
                    continue;
                }
                let start = page;
                // Cap a run at 1 GiB so the byte length always fits a `u32`.
                let limit = core::cmp::min(last, start + (1 << 18) - 1);
                while page <= limit && self.page_prot[page] != PROT_NONE {
                    page += 1;
                }
                let va = (start * GUEST_PAGE_SIZE) as u32;
                let len = ((page - start) * GUEST_PAGE_SIZE) as u32;
                self.vm.invalidate_pages(va, len);
                self.page_prot[start..page].fill(PROT_NONE);
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: free_pages needs the KVM backend")
        }
    }

    fn heap_size(&self) -> u32 {
        self.heap_top - self.heap_base
    }

    // First cut: grow the heap top; the pages already exist in the identity mapping.
    // NOTE(hypervisor): a full implementation would map fresh 4K guest pages on demand.
    fn sbrk(&mut self, size: u32) -> Result<Option<u32>, Self::Error> {
        let module = self.module.as_ref().ok_or_else(|| Error::from("no module loaded"))?;
        let max_heap_size = module.memory_map().max_heap_size();
        let new_top = self.heap_top.checked_add(size).ok_or_else(|| Error::from("sbrk overflow"))?;
        if new_top - self.heap_base > max_heap_size {
            return Ok(None);
        }
        self.heap_top = new_top;
        #[cfg(target_os = "macos")]
        // SAFETY: the VmCtx page is mapped and initialized.
        unsafe {
            (*self.vm.vmctx_ptr()).heap_info.heap_top = u64::from(new_top);
        }
        Ok(Some(new_top))
    }

    fn pid(&self) -> Option<u32> {
        None
    }
}

impl Sandbox {
    /// Host memory-access permission check. Only restricts under dynamic paging (where pages
    /// must be explicitly mapped in); static paging keeps the whole window accessible.
    fn accessible_for(&self, address: u32, length: u32, required: u8) -> bool {
        if length == 0 {
            return true;
        }
        let Some(address_end) = address.checked_add(length) else {
            return false;
        };

        // Aux data is bounded by what the host made accessible, not the region's full size.
        if address >= self.aux_data_address
            && self.aux_data_full_length != 0
            && address_end < self.aux_data_address + self.aux_data_full_length
        {
            return address_end <= self.aux_data_address + self.accessible_aux_size;
        }

        // `page_prot` tracks what is mapped under either paging mode, including `sbrk` growth.
        let first = address as usize / GUEST_PAGE_SIZE;
        let last = (address_end as usize - 1) / GUEST_PAGE_SIZE;
        self.page_prot[first..=last].iter().all(|&p| p >= required)
    }

    /// Drive the vCPU from `entry` (the sysenter stub) until it returns to the host,
    /// translating exits to `InterruptKind`. First cut; see the `run` NOTE.
    #[cfg(target_os = "macos")]
    fn execute(&mut self, entry: u64) -> Result<InterruptKind, Error> {
        use core::sync::atomic::Ordering;

        self.apply_aux_data_protection();

        // Any page-table edits since the last entry (paging in, protection changes, `free_pages`,
        // aux data) must be visible to the guest before it runs again.
        self.vm.flush_tlb()?;

        // The guest expects x14 = memory base and x13 = tmp_reg on entry (set by run).
        let tmp_reg = {
            // SAFETY: VmCtx is mapped.
            let vmctx = unsafe { &*self.vm.vmctx_ptr() };
            vmctx.tmp_reg.load(Ordering::Relaxed)
        };
        self.vm.set_reg(14, GUEST_MEM_BASE)?; // AUX_TMP_REG / memory base
        self.vm.set_reg(13, tmp_reg)?; // TMP_REG
        self.vm.set_reg(hv::HV_REG_PC, entry)?;

        loop {
            match self.vm.run_raw()? {
                VmExit::HostCall { imm } => match imm {
                    HVC_HOSTCALL => {
                        // SAFETY: VmCtx is mapped.
                        let arg = unsafe { (*self.vm.vmctx_ptr()).arg.load(Ordering::Relaxed) };
                        return Ok(InterruptKind::Ecalli(arg));
                    }
                    HVC_TRAP => return Ok(InterruptKind::Trap),
                    HVC_RETURN => {
                        self.is_program_counter_valid = false;
                        return Ok(InterruptKind::Finished);
                    }
                    HVC_STEP => return Ok(InterruptKind::Step),
                    HVC_NOT_ENOUGH_GAS => return Ok(InterruptKind::NotEnoughGas),
                    HVC_SBRK => {
                        // sbrk is resumable: compute the new heap top and hand it back in x0.
                        // The vCPU PC already points past the `hvc` (at the stub's `ret`), so
                        // just continue the loop to run it.
                        let pending = self.vm.get_reg(hv::HV_REG_X0)?;
                        let result = self.sbrk_pending(pending);
                        self.vm.set_reg(hv::HV_REG_X0, u64::from(result))?;
                    }
                    HVC_FAULT => {
                        // The guest EL1 vector reports a stage-1 abort; details are in EL1 sysregs.
                        let far = self.vm.get_sys_reg(hv::HV_SYS_REG_FAR_EL1)?;
                        let esr = self.vm.get_sys_reg(hv::HV_SYS_REG_ESR_EL1)?;
                        let elr = self.vm.get_sys_reg(hv::HV_SYS_REG_ELR_EL1)?;
                        return self.handle_fault(far, esr, elr);
                    }
                    _ => return Ok(InterruptKind::Trap),
                },
                // A stage-2 abort (from hv itself) surfaces directly; PC is the faulting instr.
                VmExit::MemoryFault { address, esr } => {
                    let pc = self.vm.get_reg(hv::HV_REG_PC)?;
                    return self.handle_fault(address, esr, pc);
                }
                VmExit::Trap { .. } | VmExit::Exception { .. } | VmExit::Unknown => return Ok(InterruptKind::Trap),
                VmExit::Canceled | VmExit::VTimer => return Err("vCPU run was canceled".into()),
            }
        }
    }

    /// Translate a guest fault (`FAR`/`ESR`, faulting instr at `elr`) into an `InterruptKind`.
    /// Mirrors generic: a dynamic-paging abort above the null guard becomes a `Segfault` whose
    /// faulting instruction can be re-executed on the next `run` (registers snapshotted into the
    /// VmCtx, resume address recomputed); anything else is a `Trap`.
    #[cfg(target_os = "macos")]
    fn handle_fault(&mut self, far: u64, esr: u64, elr: u64) -> Result<InterruptKind, Error> {
        use core::sync::atomic::Ordering;

        let ec = (esr >> 26) & 0x3f;
        // EC 0x00 ("unknown") is an undefined instruction: either the sync gas-metering `udf` or
        // executing garbage. Only the former has a negative gas counter.
        if ec == 0x00 {
            if let Some(interrupt) = self.handle_gas_trap(elr)? {
                return Ok(interrupt);
            }
            return self.trap_at(elr);
        }
        // Data abort (0x24/0x25) or instruction abort (0x20/0x21).
        if !matches!(ec, 0x20 | 0x21 | 0x24 | 0x25) {
            return self.trap_at(elr);
        }
        let guest_addr = far.wrapping_sub(GUEST_MEM_BASE);
        let page_address = (guest_addr as u32) & !(self.page_size - 1);
        if !(self.dynamic_paging && guest_addr < RAM_SIZE as u64 && page_address >= 0x10000) {
            return self.trap_at(elr);
        }
        // Like generic: `is_write_protected` means the page exists but is read-only (vs absent).
        let page_idx = page_address as usize / GUEST_PAGE_SIZE;
        let is_write_protected = self.page_prot[page_idx] == PROT_READ;

        let module = self.module.clone().ok_or_else(|| Error::from("fault: no module"))?;
        let compiled = <Self as super::Sandbox>::downcast_module(&module);

        self.snapshot_regs()?;

        // Map the faulting native PC to a guest program counter and the native address to resume
        // at (start of that instruction, so re-running retries the access).
        let offset = elr.wrapping_sub(compiled.native_code_origin);
        let pc = compiled
            .program_counter_by_native_code_offset(offset, false)
            .ok_or_else(|| Error::from("fault: no program counter for faulting address"))?;
        let resume = compiled.resume_native_address_for_pagefault(pc, elr, self.gas_metering);
        // SAFETY: the VmCtx page is mapped and initialized.
        unsafe {
            let vmctx = &mut *self.vm.vmctx_ptr();
            vmctx.program_counter.store(pc.0, Ordering::Relaxed);
            vmctx.next_program_counter.store(pc.0, Ordering::Relaxed);
            vmctx.next_native_program_counter.store(resume, Ordering::Relaxed);
        }
        self.program_counter = Some(pc);
        self.is_program_counter_valid = true;
        self.next_program_counter = None;
        self.next_program_counter_changed = false;
        self.charge_gas_on_entry = false;

        Ok(InterruptKind::Segfault(Segfault {
            page_address,
            page_size: self.page_size,
            is_write_protected,
        }))
    }

    /// Guest may read only the first `accessible_aux_size` bytes of aux data; the rest faults. Like
    /// generic's `set_aux_data_permission_for_guest`; the host is unaffected (it uses its own map).
    #[cfg(target_os = "macos")]
    fn apply_aux_data_protection(&mut self) {
        if self.aux_data_full_length == 0 || self.dynamic_paging {
            return;
        }
        self.vm.invalidate_pages(self.aux_data_address, self.aux_data_full_length);
        if self.accessible_aux_size > 0 {
            self.vm
                .set_page_protection(self.aux_data_address, self.accessible_aux_size, PTE_AP_RO_EL1);
        }
    }

    /// Copy the live guest registers into the VmCtx so the host can read/edit them and `sysenter`
    /// restores them. The `hvc` trampolines do this themselves; the EL1 vector does not.
    #[cfg(target_os = "macos")]
    fn snapshot_regs(&mut self) -> Result<(), Error> {
        for reg in Reg::ALL {
            let value = self.vm.get_reg(to_native_reg(reg) as u32)?;
            // SAFETY: the VmCtx page is mapped and initialized.
            unsafe { (*self.vm.vmctx_ptr()).regs.0[reg as usize] = value };
        }
        Ok(())
    }

    /// Plain `Trap`: the EL1 vector saved nothing, so recover regs and the PC from `elr`, else from
    /// the jump site the indirect-jump codegen stashed.
    #[cfg(target_os = "macos")]
    fn trap_at(&mut self, elr: u64) -> Result<InterruptKind, Error> {
        use core::sync::atomic::Ordering;

        self.snapshot_regs()?;
        let Some(module) = self.module.clone() else {
            return Ok(InterruptKind::Trap);
        };
        let compiled = <Self as super::Sandbox>::downcast_module(&module);
        // SAFETY: the VmCtx page is mapped and initialized.
        let stashed = unsafe { (*self.vm.vmctx_ptr()).next_native_program_counter.load(Ordering::Relaxed) };
        let code_len = compiled.machine_code().len() as u64;

        for address in [elr, stashed] {
            let Some(offset) = address.checked_sub(compiled.native_code_origin) else {
                continue;
            };
            if offset >= code_len {
                continue;
            }
            let Some(pc) = compiled.program_counter_by_native_code_offset(offset, false) else {
                continue;
            };
            // SAFETY: the VmCtx page is mapped and initialized.
            unsafe {
                let vmctx = &mut *self.vm.vmctx_ptr();
                vmctx.program_counter.store(pc.0, Ordering::Relaxed);
                vmctx.next_native_program_counter.store(0, Ordering::Relaxed);
            }
            self.program_counter = Some(pc);
            self.is_program_counter_valid = true;
            return Ok(InterruptKind::Trap);
        }

        Ok(InterruptKind::Trap)
    }

    /// The sync metering stub's `udf` fired: refund the charged gas, snapshot registers and re-enter
    /// at the stub, as generic's `handle_guest_signal` does. `None` if it wasn't a gas trap.
    #[cfg(target_os = "macos")]
    fn handle_gas_trap(&mut self, elr: u64) -> Result<Option<InterruptKind>, Error> {
        use core::sync::atomic::Ordering;

        // SAFETY: the VmCtx page is mapped and initialized.
        let gas = unsafe { (*self.vm.vmctx_ptr()).gas.load(Ordering::Relaxed) };
        if self.gas_metering.is_none() || gas >= 0 {
            return Ok(None);
        }

        let module = self.module.clone().ok_or_else(|| Error::from("gas trap: no module"))?;
        let compiled = <Self as super::Sandbox>::downcast_module(&module);
        // The `udf` sits a fixed distance into the stub; back up to its first instruction so
        // re-entering recharges the refunded gas.
        let machine_code_offset = elr.wrapping_sub(compiled.native_code_origin);
        let Some(offset) = machine_code_offset.checked_sub(crate::compiler::GAS_METERING_TRAP_OFFSET) else {
            return Ok(None);
        };
        let Some(pc) = compiled.program_counter_by_native_code_offset(offset, false) else {
            return Ok(None);
        };

        let cost = crate::compiler::extract_gas_cost::<Self>(compiled.machine_code(), offset as usize);
        self.snapshot_regs()?;
        // SAFETY: the VmCtx page is mapped and initialized.
        unsafe {
            let vmctx = &mut *self.vm.vmctx_ptr();
            vmctx.gas.fetch_add(i64::from(cost), Ordering::Relaxed);
            vmctx.program_counter.store(pc.0, Ordering::Relaxed);
            vmctx.next_program_counter.store(pc.0, Ordering::Relaxed);
            vmctx
                .next_native_program_counter
                .store(compiled.native_code_origin + offset, Ordering::Relaxed);
        }
        self.program_counter = Some(pc);
        self.is_program_counter_valid = true;
        self.next_program_counter = None;
        self.next_program_counter_changed = false;
        self.charge_gas_on_entry = false;

        Ok(Some(InterruptKind::NotEnoughGas))
    }

    /// Heap-growth helper shared by the `sbrk` hvc path: returns the new heap top or 0.
    /// Mirrors generic's `sbrk`: enable the newly-crossed 4K pages RW and bump the
    /// heap top/threshold in the VmCtx.
    #[cfg(target_os = "macos")]
    fn sbrk_pending(&mut self, pending_top: u64) -> u32 {
        let Some(module) = self.module.as_ref() else { return 0 };
        let mmap = module.memory_map();
        let page = u64::from(mmap.page_size());
        if pending_top > u64::from(mmap.heap_base()) + u64::from(mmap.max_heap_size()) {
            return 0;
        }
        // SAFETY: VmCtx is mapped.
        let old_top = unsafe { (*self.vm.vmctx_ptr()).heap_info.heap_top };
        let start = old_top.next_multiple_of(page);
        let end = pending_top.next_multiple_of(page);
        if end > start {
            let len = (end - start) as u32;
            self.vm.set_page_protection(start as u32, len, PTE_AP_RW_EL1);
            mark_pages(&mut self.page_prot, start as u32, len, PROT_READ_WRITE);
        }
        self.heap_top = pending_top as u32;
        // SAFETY: VmCtx is mapped.
        unsafe {
            let vmctx = &mut *self.vm.vmctx_ptr();
            vmctx.heap_info.heap_top = pending_top;
            vmctx.heap_info.heap_threshold = end;
        }
        pending_top as u32
    }
}

/// Mark `[address, address + length)` (PolkaVM addresses) with a protection state.
fn mark_pages(page_prot: &mut [u8], address: u32, length: u32, state: u8) {
    let first = address as usize / GUEST_PAGE_SIZE;
    let last = (address as usize + length as usize - 1) / GUEST_PAGE_SIZE;
    page_prot[first..=last].fill(state);
}

/// Reinterpret a `&[usize]` (the jump table) as bytes.
fn as_bytes(slice: &[usize]) -> &[u8] {
    // SAFETY: `u8` has no alignment requirement and any bit pattern is valid.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), core::mem::size_of_val(slice)) }
}
