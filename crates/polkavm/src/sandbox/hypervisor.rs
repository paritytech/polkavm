//! Hardware-virtualization sandbox backend (aarch64-only).
//!
//! Runs guest code in a vCPU with its own stage-1 MMU, giving 4K guest pages on
//! any host page size (Apple Hypervisor.framework on macOS; KVM on Linux later).
//! The micro-VM engine (`vm` module) and the memory/register-backed trait methods
//! are real; the methods needing PolkaVM's `VmCtx`-in-guest-memory execution
//! contract are left as documented `todo!()`s.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use polkavm_common::zygote::AddressTable;

use super::{OffsetTable, SandboxInit, SandboxKind};
use crate::api::{MemoryAccessError, MemoryProtection, Module};
use crate::compiler::CompiledModule;
use crate::config::Config;
use crate::{Gas, InterruptKind, ProgramCounter, Reg, RegValue};

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

/// Guest RAM base in the guest's intermediate-physical (== virtual, identity) space.
const GUEST_RAM_IPA: u64 = 0x4000_0000;
/// Size of the guest RAM window: the full 4 GiB PolkaVM address space.
const RAM_SIZE: usize = 0x1_0000_0000;
/// Guest page size enforced by the stage-1 MMU (4K granule, TG0 = 0b00).
const GUEST_PAGE_SIZE: usize = 4096;
/// Number of 4K guest pages backing the RAM window.
const NUM_PAGES: usize = RAM_SIZE / GUEST_PAGE_SIZE;
/// Where the compiler is told the guest's native code lives.
const NATIVE_CODE_ORIGIN: u64 = GUEST_RAM_IPA;

/// Per-page protection state tracked host-side (mirrors the L3 descriptor AP bits).
const PROT_NONE: u8 = 0;
const PROT_READ: u8 = 1;
const PROT_READ_WRITE: u8 = 2;

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
#[allow(non_camel_case_types, dead_code)] // full FFI surface kept for the future `run` path
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
    /// Guest issued `HVC` (EC 0x16): our host-call / return-to-host trampoline.
    HostCall { imm: u64 },
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

/// A live micro-VM: backing RAM + identity 4K page tables + one vCPU.
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
    #[cfg(target_os = "macos")]
    vcpu: hv::hv_vcpu_t,
    #[cfg(target_os = "macos")]
    exit: *const hv::hv_vcpu_exit_t,
}

// SAFETY: the raw pointers refer to this VM's own mappings; ownership moves with the box.
unsafe impl Send for Vm {}
// SAFETY: guest RAM/page tables are plain memory; a vCPU is only ever driven from the
// owning instance's thread. Mirrors the generic sandbox's Send+Sync treatment of `Mmap`.
unsafe impl Sync for Vm {}

/// Only one Hypervisor.framework VM may exist per process, so guard creation.
#[cfg(target_os = "macos")]
static VM_EXISTS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

impl Vm {
    /// Create the VM: mmap RAM, create the HV VM, map RAM + page tables, build the
    /// identity 4K tables, create the vCPU and inject the MMU sysregs.
    pub fn new() -> Result<Self, Error> {
        #[cfg(target_os = "macos")]
        {
            use core::sync::atomic::Ordering;

            if VM_EXISTS.swap(true, Ordering::SeqCst) {
                return Err("a hypervisor VM already exists in this process (Hypervisor.framework allows only one)".into());
            }

            let result = Self::new_macos();
            if result.is_err() {
                VM_EXISTS.store(false, Ordering::SeqCst);
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
    fn new_macos() -> Result<Self, Error> {
        // Page-table region: one L1 table, one L2 table per used 1 GiB, and one L3
        // table per 2 MiB. For a 1-GiB-aligned RAM window starting at GUEST_RAM_IPA.
        let l1_span = 1u64 << 30;
        assert_eq!(RAM_SIZE as u64 % l1_span, 0, "RAM must be a multiple of 1 GiB");
        let num_l1 = (RAM_SIZE as u64 / l1_span) as usize; // L1 entries used
        let num_l3 = num_l1 * 512; // one L3 table per 2 MiB
        let num_tables = 1 + num_l1 + num_l3;
        let pt_len = num_tables * GUEST_PAGE_SIZE;
        let pt_ipa = GUEST_RAM_IPA + RAM_SIZE as u64;

        // Back the guest RAM and page tables with anonymous host memory.
        let ram = map_anon(RAM_SIZE)?;
        let pt = match map_anon(pt_len) {
            Ok(pt) => pt,
            Err(error) => {
                unmap_anon(ram, RAM_SIZE);
                return Err(error);
            }
        };

        let mut vm = Vm {
            ram,
            ram_len: RAM_SIZE,
            pt,
            pt_len,
            pt_ipa,
            vcpu: 0,
            exit: core::ptr::null(),
        };

        // Everything below is fallible; tear down on error.
        if let Err(error) = vm.bringup(num_l1) {
            // `bringup` created nothing that Drop cannot clean up except the VM itself,
            // which it undoes as needed; fall through to explicit teardown.
            vm.teardown_partial();
            return Err(error);
        }

        Ok(vm)
    }

    #[cfg(target_os = "macos")]
    fn bringup(&mut self, num_l1: usize) -> Result<(), Error> {
        // Create the per-process VM.
        // SAFETY: FFI call; a null config selects the framework default.
        check(unsafe { hv::hv_vm_create(core::ptr::null_mut()) }, "hv_vm_create")?;

        // Map the guest RAM (RWX) and the page-table region (RW) into the IPA space.
        check(
            // SAFETY: `self.ram`/`self.ram_len` describe a live anonymous mapping.
            unsafe {
                hv::hv_vm_map(
                    self.ram.cast(),
                    GUEST_RAM_IPA,
                    self.ram_len,
                    hv::HV_MEMORY_READ | hv::HV_MEMORY_WRITE | hv::HV_MEMORY_EXEC,
                )
            },
            "hv_vm_map(ram)",
        )?;
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
        self.exit = exit;

        self.inject_mmu_sysregs()?;
        Ok(())
    }

    /// Build identity 4K page tables covering `[GUEST_RAM_IPA, GUEST_RAM_IPA + RAM_SIZE)`.
    ///
    /// Layout in the PT region: `[L1][L2 x num_l1][L3 x (num_l1 * 512)]`, contiguous.
    // The PT region is `mmap`-allocated and thus page-aligned, so the `*mut u8` -> `*mut u64`
    // casts and every 8-byte-aligned table offset within it are correctly aligned.
    #[cfg(target_os = "macos")]
    #[allow(clippy::cast_ptr_alignment)]
    fn build_identity_4k_tables(&mut self, num_l1: usize) {
        let l1_start_idx = (GUEST_RAM_IPA >> 30) as usize & 0x1ff;
        let table_ipa = |table_index: usize| self.pt_ipa + (table_index as u64) * GUEST_PAGE_SIZE as u64;

        // L1 table is table 0; L2 tables are 1..=num_l1; L3 tables follow.
        let l1 = self.pt.cast::<u64>();
        for i in 0..num_l1 {
            let l2_table_index = 1 + i;
            let entry = table_ipa(l2_table_index) | DESC_PAGE;
            // SAFETY: `l1_start_idx + i` is a valid entry index within the L1 table page.
            unsafe { l1.add(l1_start_idx + i).write(entry) };
        }

        for i in 0..num_l1 {
            let l2_table_index = 1 + i;
            // SAFETY: L2 table `i` lives at table offset `1 + i`, inside the PT region.
            let l2 = unsafe { self.pt.add(l2_table_index * GUEST_PAGE_SIZE).cast::<u64>() };
            for j in 0..512usize {
                let l3_table_index = 1 + num_l1 + i * 512 + j;
                let entry = table_ipa(l3_table_index) | DESC_PAGE;
                // SAFETY: `j` is a valid entry index (0..512) within this L2 table page.
                unsafe { l2.add(j).write(entry) };
            }
        }

        for t in 0..(num_l1 * 512) {
            let l3_table_index = 1 + num_l1 + t;
            // SAFETY: L3 table `t` lives at table offset `1 + num_l1 + t`, inside the PT region.
            let l3 = unsafe { self.pt.add(l3_table_index * GUEST_PAGE_SIZE).cast::<u64>() };
            for m in 0..512usize {
                let page_va = GUEST_RAM_IPA + ((t * 512 + m) * GUEST_PAGE_SIZE) as u64;
                let entry = (page_va & PTE_OUTPUT_MASK) | PTE_AF | PTE_SH_INNER | PTE_AP_RW_EL1 | PTE_ATTRINDX0 | DESC_PAGE;
                // SAFETY: `m` is a valid entry index (0..512) within this L3 table page.
                unsafe { l3.add(m).write(entry) };
            }
        }
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
        // VBAR is required for exception handling once the guest runs.
        self.set_sys_reg(hv::HV_SYS_REG_VBAR_EL1, GUEST_RAM_IPA)?;
        self.set_sys_reg(hv::HV_SYS_REG_SP_EL1, GUEST_RAM_IPA)?;
        // Allow FP/SIMD at EL0/EL1 (FPEN = 0b11).
        self.set_sys_reg(hv::HV_SYS_REG_CPACR_EL1, 3 << 20)?;

        // SCTLR_EL1: keep reset value, add M (MMU), C (data cache), I (instr cache).
        let sctlr = self.get_sys_reg(hv::HV_SYS_REG_SCTLR_EL1)?;
        self.set_sys_reg(hv::HV_SYS_REG_SCTLR_EL1, sctlr | 1 | (1 << 2) | (1 << 12))?;

        // CPSR: EL1h with DAIF masked.
        self.set_reg(hv::HV_REG_CPSR, 0x3c5)?;
        Ok(())
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

    /// Flip the AP bits of the L3 descriptors for `[va, va + len)` (guest addresses).
    /// This is where the 4K granule matters: protections are per 4K page.
    #[cfg(target_os = "macos")]
    fn set_page_protection(&mut self, va: u32, len: u32, ap_bits: u64) {
        if len == 0 {
            return;
        }
        let first = u64::from(va) & !(GUEST_PAGE_SIZE as u64 - 1);
        let last = (u64::from(va) + u64::from(len) - 1) & !(GUEST_PAGE_SIZE as u64 - 1);
        let mut page_va = first;
        while page_va <= last {
            // SAFETY: `page_va` is within the identity-mapped RAM window, so the walk
            // resolves to a real L3 descriptor in the PT region.
            let entry_ptr = unsafe { self.l3_entry_ptr(GUEST_RAM_IPA + page_va) };
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
        let first = u64::from(va) & !(GUEST_PAGE_SIZE as u64 - 1);
        let last = (u64::from(va) + u64::from(len) - 1) & !(GUEST_PAGE_SIZE as u64 - 1);
        let mut page_va = first;
        while page_va <= last {
            // SAFETY: `page_va` is within the identity-mapped RAM window; the walk
            // resolves to a real L3 descriptor in the PT region.
            let entry_ptr = unsafe { self.l3_entry_ptr(GUEST_RAM_IPA + page_va) };
            // SAFETY: `entry_ptr` points at a live, aligned L3 descriptor.
            let entry = unsafe { entry_ptr.read() };
            // SAFETY: same descriptor pointer; clearing the valid bit.
            unsafe { entry_ptr.write(entry & !DESC_VALID) };
            page_va += GUEST_PAGE_SIZE as u64;
        }
        // NOTE: when `run` is implemented this needs a guest TLBI; the guest is not
        // executing while the host mutates the tables in the current scaffold.
    }

    /// Run the vCPU and decode the exit reason.
    #[allow(dead_code)]
    pub fn run(&mut self) -> Result<VmExit, Error> {
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
                        0x16 => VmExit::HostCall { imm: esr & 0xffff }, // HVC
                        0x24 | 0x20 => VmExit::MemoryFault {
                            address: exit.exception.virtual_address,
                            esr,
                        }, // Data / Instruction Abort
                        0x00 => VmExit::Trap { esr },                   // Undefined instruction
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
            check(unsafe { hv::hv_vcpu_get_sys_reg(self.vcpu, reg, &mut value) }, "hv_vcpu_get_sys_reg")?;
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

    /// Host-side view of guest RAM at `[addr, addr + len)` (PolkaVM address space).
    #[cfg(target_os = "macos")]
    fn ram_slice(&self, addr: u32, len: usize) -> Option<&[u8]> {
        let end = u64::from(addr).checked_add(len as u64)?;
        if end > self.ram_len as u64 {
            return None;
        }
        // SAFETY: bounds checked against the mapped RAM window.
        Some(unsafe { core::slice::from_raw_parts(self.ram.add(addr as usize), len) })
    }

    #[cfg(target_os = "macos")]
    fn ram_slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
        let end = u64::from(addr).checked_add(len as u64)?;
        if end > self.ram_len as u64 {
            return None;
        }
        // SAFETY: bounds checked against the mapped RAM window.
        Some(unsafe { core::slice::from_raw_parts_mut(self.ram.add(addr as usize), len) })
    }

    #[cfg(target_os = "macos")]
    fn teardown_partial(&mut self) {
        // vCPU (if created) must be destroyed before the VM.
        if self.vcpu != 0 {
            // SAFETY: FFI call; destroying this thread's own vCPU.
            unsafe { hv::hv_vcpu_destroy(self.vcpu) };
            self.vcpu = 0;
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
        VM_EXISTS.store(false, core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if !self.ram.is_null() {
                self.teardown_partial();
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

// ---------------------------------------------------------------------------
// Sandbox trait plumbing.
// ---------------------------------------------------------------------------

/// Per-engine state shared across sandboxes (vCPU factory handles, etc.).
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

/// A prepared program: guest machine code plus MMU layout, ready to map into a vCPU.
#[derive(Clone)]
pub struct SandboxProgram(alloc::sync::Arc<Vec<u8>>);

impl super::SandboxProgram for SandboxProgram {
    fn machine_code(&self) -> &[u8] {
        &self.0
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
    #[allow(dead_code)] // recorded by load_module (still a todo!)
    module: Option<Module>,
    regs: [RegValue; 13],
    gas: Gas,
    program_counter: Option<ProgramCounter>,
    next_program_counter: Option<ProgramCounter>,
    accessible_aux_size: u32,
    heap_base: u32,
    heap_top: u32,
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
        Ok(vec![0; count])
    }

    fn reserve_address_space() -> Result<Self::AddressSpace, Self::Error> {
        Ok(AddressSpace {
            native_code_origin: NATIVE_CODE_ORIGIN,
        })
    }

    // TODO(hypervisor): copy the compiled code into guest RAM, map the code pages RX,
    // and install the `hvc`-based trampoline stubs at the address-table slots. Requires
    // the compiler to emit `hvc`-based host-call/return trampolines.
    fn prepare_program(
        _global: &Self::GlobalState,
        _init: SandboxInit<Self>,
        _address_space: Self::AddressSpace,
    ) -> Result<Self::Program, Self::Error> {
        todo!("hypervisor sandbox: prepare_program needs guest code mapping + hvc trampoline stubs")
    }

    fn spawn(_global: &Self::GlobalState, _config: &Self::Config, _outer_instance: Option<&Self>) -> Result<Box<Self>, Self::Error> {
        let vm = Vm::new()?;
        Ok(Box::new(Sandbox {
            vm,
            module: None,
            regs: [0; 13],
            gas: 0,
            program_counter: None,
            next_program_counter: None,
            accessible_aux_size: 0,
            heap_base: 0,
            heap_top: 0,
            page_prot: vec![PROT_READ_WRITE; NUM_PAGES],
        }))
    }

    // TODO(hypervisor): copy compiled code into guest RAM, map code pages RX, wire the
    // heap/stack layout, and record the module. Needs the shared VmCtx contract.
    fn load_module(&mut self, _global: &Self::GlobalState, _module: &Module) -> Result<(), Self::Error> {
        todo!("hypervisor sandbox: load_module needs guest code upload + VmCtx layout")
    }

    fn recycle(_sandbox: Box<Self>, _global: &Self::GlobalState) -> Result<(), Self::Error> {
        // Dropping the box tears down the vCPU + VM.
        Ok(())
    }

    // TODO(hypervisor): must match the compiler's expectations and point at the guest
    // `hvc` stub addresses; depends on the trampoline design in prepare_program.
    fn address_table() -> AddressTable {
        todo!("hypervisor sandbox: address_table needs the guest hvc trampoline addresses")
    }

    // TODO(hypervisor): offsets into the guest-memory VmCtx, which is currently private
    // to the generic sandbox; requires sharing that layout.
    fn offset_table() -> OffsetTable {
        todo!("hypervisor sandbox: offset_table needs the shared VmCtx layout")
    }

    fn idle_worker_pids(_global: &Self::GlobalState) -> Vec<u32> {
        Vec::new()
    }

    // TODO(hypervisor): the execution loop needs the VmCtx-in-guest-memory exit contract
    // and guest entry via a sysenter stub, plus VmExit -> InterruptKind translation.
    fn run(&mut self) -> Result<InterruptKind, Self::Error> {
        todo!("hypervisor sandbox: run needs the VmCtx exit contract + guest entry stub")
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
        self.program_counter
    }

    fn next_program_counter(&self) -> Option<ProgramCounter> {
        self.next_program_counter
    }

    fn next_native_program_counter(&self) -> Option<usize> {
        None
    }

    fn set_next_program_counter(&mut self, pc: ProgramCounter) {
        self.next_program_counter = Some(pc);
    }

    fn accessible_aux_size(&self) -> u32 {
        self.accessible_aux_size
    }

    fn set_accessible_aux_size(&mut self, size: u32) -> Result<(), Self::Error> {
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
        let first = address as usize / GUEST_PAGE_SIZE;
        let last = (address as usize + size as usize - 1) / GUEST_PAGE_SIZE;
        (first..=last).all(|page| self.page_prot[page] >= required)
    }

    fn reset_memory(&mut self) -> Result<(), Self::Error> {
        // Zero the backing RAM and drop all pages back to inaccessible.
        #[cfg(target_os = "macos")]
        {
            if let Some(slice) = self.vm.ram_slice_mut(0, RAM_SIZE) {
                slice.fill(0);
            }
        }
        self.free_pages(0, 0xffff_f000)?;
        Ok(())
    }

    fn read_memory_into<'slice>(&self, address: u32, slice: &'slice mut [MaybeUninit<u8>]) -> Result<&'slice mut [u8], MemoryAccessError> {
        if slice.is_empty() {
            // SAFETY: empty slice; nothing is read.
            return Ok(unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), 0) });
        }
        if !bounds_ok(address, slice.len() as u32) {
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
            // Initialize the MaybeUninit slice from guest RAM.
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
        if !bounds_ok(address, data.len() as u32) {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: data.len() as u64,
            });
        }
        #[cfg(target_os = "macos")]
        {
            let dst = self.vm.ram_slice_mut(address, data.len()).ok_or(MemoryAccessError::OutOfRangeAccess {
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

    fn zero_memory(&mut self, address: u32, length: u32, _memory_protection: Option<MemoryProtection>) -> Result<(), MemoryAccessError> {
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
            let dst = self.vm.ram_slice_mut(address, length as usize).ok_or(MemoryAccessError::OutOfRangeAccess {
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
            let (ap, state) = match protection {
                MemoryProtection::Read => (PTE_AP_RO_EL1, PROT_READ),
                MemoryProtection::ReadWrite => (PTE_AP_RW_EL1, PROT_READ_WRITE),
            };
            self.vm.set_page_protection(address, length, ap);
            let first = address as usize / GUEST_PAGE_SIZE;
            let last = (address as usize + length as usize - 1) / GUEST_PAGE_SIZE;
            for page in first..=last {
                self.page_prot[page] = state;
            }
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
            self.vm.invalidate_pages(address, length);
        }
        #[cfg(not(target_os = "macos"))]
        {
            todo!("hypervisor sandbox: free_pages needs the KVM backend")
        }
        #[cfg(target_os = "macos")]
        {
            let first = address as usize / GUEST_PAGE_SIZE;
            let last = (address as usize + length as usize - 1) / GUEST_PAGE_SIZE;
            for page in first..=last {
                self.page_prot[page] = PROT_NONE;
            }
            Ok(())
        }
    }

    fn heap_size(&self) -> u32 {
        self.heap_top - self.heap_base
    }

    // TODO(hypervisor): grow the heap by mapping new 4K guest pages and updating the
    // guest-memory VmCtx heap info; depends on the load_module layout.
    fn sbrk(&mut self, _size: u32) -> Result<Option<u32>, Self::Error> {
        todo!("hypervisor sandbox: sbrk needs guest heap page mapping + VmCtx heap info")
    }

    fn pid(&self) -> Option<u32> {
        None
    }
}
