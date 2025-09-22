#![allow(clippy::manual_range_contains)]

use polkavm_common::{
    cast::cast,
    program::Reg,
    utils::{align_to_next_page_usize, byte_slice_init},
    zygote::{
        AddressTable, AddressTableRaw, CacheAligned, VM_ADDR_JUMP_TABLE, VM_ADDR_JUMP_TABLE_RETURN_TO_HOST,
        VM_SANDBOX_MAXIMUM_JUMP_TABLE_VIRTUAL_SIZE, VM_SANDBOX_MAXIMUM_NATIVE_CODE_SIZE,
    },
};

use core::mem::MaybeUninit;

use core::cell::UnsafeCell;
use core::ops::Range;
use core::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::{get_native_page_size, OffsetTable, SandboxInit, SandboxKind, WorkerCache, WorkerCacheKind};
use crate::api::{CompiledModuleKind, MemoryAccessError, Module};
use crate::compiler::CompiledModule;
use crate::config::Config;
use crate::config::GasMeteringKind;
use crate::page_set::PageSet;
use crate::{Gas, InterruptKind, ProgramCounter, RegValue, Segfault};

#[inline(always)]
pub(crate) fn as_bytes(slice: &[usize]) -> &[u8] {
    // SAFETY: Casting a &[usize] into a &[u8] is always safe as `u8` doesn't have any alignment requirements.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), core::mem::size_of_val(slice)) }
}

#[repr(transparent)]
struct Cast<T, U>(T, core::marker::PhantomData<U>);

impl Cast<u32, usize> {
    #[inline]
    fn get(self) -> usize {
        const _: () = {
            assert!(core::mem::size_of::<usize>() >= core::mem::size_of::<u32>());
        };

        self.0 as usize
    }
}

impl core::ops::Add<Cast<u32, usize>> for usize {
    type Output = usize;

    #[inline]
    fn add(self, rhs: Cast<u32, usize>) -> usize {
        self + rhs.get()
    }
}

#[inline]
fn to_usize<T>(value: T) -> Cast<T, usize> {
    Cast(value, core::marker::PhantomData)
}

// On Linux don't depend on the `libc` crate to lower the number of dependencies.
#[cfg(target_os = "linux")]
#[allow(non_camel_case_types)]
mod sys {
    pub use polkavm_linux_raw::{c_int, c_void, siginfo_t, size_t, ucontext as ucontext_t, SIG_DFL, SIG_IGN};
    pub const SIGSEGV: c_int = polkavm_linux_raw::SIGSEGV as c_int;
    pub const SIGILL: c_int = polkavm_linux_raw::SIGILL as c_int;
    pub const PROT_READ: c_int = polkavm_linux_raw::PROT_READ as c_int;
    pub const PROT_WRITE: c_int = polkavm_linux_raw::PROT_WRITE as c_int;
    pub const PROT_EXEC: c_int = polkavm_linux_raw::PROT_EXEC as c_int;
    pub const MAP_ANONYMOUS: c_int = polkavm_linux_raw::MAP_ANONYMOUS as c_int;
    pub const MAP_PRIVATE: c_int = polkavm_linux_raw::MAP_PRIVATE as c_int;
    pub const MAP_FIXED: c_int = polkavm_linux_raw::MAP_FIXED as c_int;
    pub const MAP_FAILED: *mut c_void = !0 as *mut c_void;
    pub const SA_SIGINFO: c_int = polkavm_linux_raw::SA_SIGINFO as c_int;
    pub const SA_NODEFER: c_int = polkavm_linux_raw::SA_NODEFER as c_int;
    pub const MADV_DONTNEED: c_int = polkavm_linux_raw::MADV_DONTNEED as c_int;
    pub const MADV_FREE: c_int = polkavm_linux_raw::MADV_FREE as c_int;

    pub type sighandler_t = size_t;

    #[repr(C)]
    pub struct sigset_t {
        #[cfg(target_pointer_width = "32")]
        __val: [u32; 32],
        #[cfg(target_pointer_width = "64")]
        __val: [u64; 16],
    }

    #[repr(C)]
    pub struct sigaction {
        pub sa_sigaction: sighandler_t,
        pub sa_mask: sigset_t,
        pub sa_flags: c_int,
        pub sa_restorer: Option<extern "C" fn()>,
    }

    extern "C" {
        pub fn mmap(addr: *mut c_void, len: size_t, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;

        pub fn munmap(addr: *mut c_void, len: size_t) -> c_int;

        pub fn mprotect(addr: *mut c_void, len: size_t, prot: c_int) -> c_int;

        pub fn madvise(addr: *mut c_void, len: size_t, advice: c_int) -> c_int;

        pub fn sigaction(signum: c_int, act: *const sigaction, oldact: *mut sigaction) -> c_int;

        pub fn sigemptyset(set: *mut sigset_t) -> c_int;
    }
}

#[cfg(not(target_os = "linux"))]
use libc as sys;

use core::ffi::c_void;
use sys::{c_int, size_t, MADV_DONTNEED, MADV_FREE, MAP_ANONYMOUS, MAP_FIXED, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

pub(crate) const GUEST_MEMORY_TO_VMCTX_OFFSET: isize = -4096;

const GAS_METERING_TRAP_OFFSET: u64 = 3;
const GAS_COST_GENERIC_SANDBOX_OFFSET: usize = 7;

fn get_guest_memory_offset() -> usize {
    get_native_page_size()
}

#[derive(Debug)]
pub struct Error(std::io::Error);

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0.fmt(fmt)
    }
}

impl From<&'static str> for Error {
    fn from(value: &'static str) -> Self {
        Self(std::io::Error::new(std::io::ErrorKind::Other, value))
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error)
    }
}

pub struct Mmap {
    pointer: *mut c_void,
    length: usize,
}

// SAFETY: The ownership of an mmapped piece of memory can be safely transferred to other threads.
unsafe impl Send for Mmap {}

// SAFETY: An mmaped piece of memory can be safely accessed from multiple threads.
unsafe impl Sync for Mmap {}

impl Mmap {
    unsafe fn raw_mmap(address: *mut c_void, length: usize, protection: c_int, flags: c_int) -> Result<Self, Error> {
        let pointer = {
            let pointer = sys::mmap(address, length, protection, flags, -1, 0);
            if pointer == sys::MAP_FAILED {
                return Err(Error(std::io::Error::last_os_error()));
            }
            pointer
        };

        Ok(Self { pointer, length })
    }

    fn mmap_within(&mut self, offset: usize, length: usize, protection: c_int) -> Result<(), Error> {
        if !offset.checked_add(length).map_or(false, |end| end <= self.length) {
            return Err("out of bounds mmap".into());
        }

        // SAFETY: The mapping is always within the bounds of the original map.
        unsafe {
            let pointer = self.pointer.cast::<u8>().add(offset).cast();
            core::mem::forget(Self::raw_mmap(
                pointer,
                length,
                protection,
                MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE,
            )?);
        }

        Ok(())
    }

    fn unmap_inplace(&mut self) -> Result<(), Error> {
        if self.length > 0 {
            // SAFETY: The map is always valid here, so it can be safely unmapped.
            unsafe {
                if sys::munmap(self.pointer, self.length) < 0 {
                    return Err(Error(std::io::Error::last_os_error()));
                }
            }

            self.length = 0;
            self.pointer = core::ptr::NonNull::<u8>::dangling().as_ptr().cast::<c_void>();
        }

        Ok(())
    }

    pub fn unmap(mut self) -> Result<(), Error> {
        self.unmap_inplace()
    }

    pub fn reserve_address_space(length: size_t) -> Result<Self, Error> {
        // SAFETY: `MAP_FIXED` is not specified, so this is always safe.
        unsafe { Mmap::raw_mmap(core::ptr::null_mut(), length, 0, MAP_ANONYMOUS | MAP_PRIVATE) }
    }

    pub fn madvise(&mut self, offset: usize, length: usize, advice: c_int) -> Result<(), Error> {
        if !offset.checked_add(length).map_or(false, |end| end <= self.length) {
            return Err("out of bounds madvise".into());
        }

        // SAFETY: The bounds are always within the range of this map.
        unsafe {
            if sys::madvise(self.pointer.add(offset), length, advice) < 0 {
                return Err(Error(std::io::Error::last_os_error()));
            }
        }

        Ok(())
    }

    pub fn mprotect(&mut self, offset: usize, length: usize, protection: c_int) -> Result<(), Error> {
        if !offset.checked_add(length).map_or(false, |end| end <= self.length) {
            return Err("out of bounds mprotect".into());
        }

        // SAFETY: The bounds are always within the range of this map.
        unsafe {
            if sys::mprotect(self.pointer.add(offset), length, protection) < 0 {
                return Err(Error(std::io::Error::last_os_error()));
            }
        }

        Ok(())
    }

    pub fn modify_and_protect(
        &mut self,
        offset: usize,
        length: usize,
        protection: c_int,
        callback: impl FnOnce(&mut [u8]),
    ) -> Result<(), Error> {
        self.mprotect(offset, length, PROT_READ | PROT_WRITE)?;
        callback(&mut self.as_slice_mut()[offset..offset + length]);
        if protection != PROT_READ | PROT_WRITE {
            self.mprotect(offset, length, protection)?;
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.pointer
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.pointer
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: The pointer is either always valid, or is dangling and the length is zero.
        //
        // The memory might not be mapped as readable, so accessing this slice can still produce
        // a segfault, but this is expected due to the low level nature of this helper, and assuming
        // the signal handler is correct it cannot result in unsoundness, for the same reason as to why
        // the `std::process::abort` is also safe.
        unsafe { core::slice::from_raw_parts(self.as_ptr().cast::<u8>(), self.length) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: The pointer is either always valid, or is dangling and the length is zero.
        //
        // The memory might not be mapped as readable or writable, so accessing this slice can still produce
        // a segfault, but this is expected due to the low level nature of this helper, and assuming
        // the signal handler is correct it cannot result in unsoundness, for the same reason as to why
        // the `std::process::abort` is also safe.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr().cast::<u8>(), self.length) }
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Default for Mmap {
    fn default() -> Self {
        Self {
            pointer: core::ptr::NonNull::<u8>::dangling().as_ptr().cast::<c_void>(),
            length: 0,
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        let _ = self.unmap_inplace();
    }
}

static mut OLD_SIGSEGV: sys::sigaction = unsafe { core::mem::zeroed() };
static mut OLD_SIGILL: sys::sigaction = unsafe { core::mem::zeroed() };

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
static mut OLD_SIGBUS: sys::sigaction = unsafe { core::mem::zeroed() };

unsafe extern "C" fn signal_handler(signal: c_int, info: &sys::siginfo_t, context: &sys::ucontext_t) {
    let old = match signal {
        sys::SIGSEGV => core::ptr::addr_of!(OLD_SIGSEGV),
        sys::SIGILL => core::ptr::addr_of!(OLD_SIGILL),
        #[cfg(any(target_os = "macos", target_os = "freebsd"))]
        sys::SIGBUS => core::ptr::addr_of!(OLD_SIGBUS),
        _ => unreachable!("received unknown signal"),
    };

    let vmctx = THREAD_VMCTX.with(|thread_ctx| *thread_ctx.get());
    if !vmctx.is_null() {
        #[cfg(target_os = "macos")]
        macro_rules! macos_reg_field {
            (rax) => {
                (*context.uc_mcontext).__ss.__rax
            };
            (rbx) => {
                (*context.uc_mcontext).__ss.__rbx
            };
            (rcx) => {
                (*context.uc_mcontext).__ss.__rcx
            };
            (rdx) => {
                (*context.uc_mcontext).__ss.__rdx
            };
            (rdi) => {
                (*context.uc_mcontext).__ss.__rdi
            };
            (rsi) => {
                (*context.uc_mcontext).__ss.__rsi
            };
            (rbp) => {
                (*context.uc_mcontext).__ss.__rbp
            };
            (rsp) => {
                (*context.uc_mcontext).__ss.__rsp
            };
            (r8) => {
                (*context.uc_mcontext).__ss.__r8
            };
            (r9) => {
                (*context.uc_mcontext).__ss.__r9
            };
            (r10) => {
                (*context.uc_mcontext).__ss.__r10
            };
            (r11) => {
                (*context.uc_mcontext).__ss.__r11
            };
            (r12) => {
                (*context.uc_mcontext).__ss.__r12
            };
            (r13) => {
                (*context.uc_mcontext).__ss.__r13
            };
            (r14) => {
                (*context.uc_mcontext).__ss.__r14
            };
            (r15) => {
                (*context.uc_mcontext).__ss.__r15
            };
            (rip) => {
                (*context.uc_mcontext).__ss.__rip
            };
        }

        macro_rules! fetch_reg {
            ($reg:ident) => {{
                #[cfg(target_os = "linux")]
                {
                    context.uc_mcontext.$reg as u64
                }
                #[cfg(target_os = "macos")]
                {
                    macos_reg_field!($reg) as u64
                }
                #[cfg(target_os = "freebsd")]
                {
                    context.uc_mcontext.mc_($reg) as u64
                }
            }};
        }

        const X86_TRAP_PF: u64 = 14;
        let is_page_fault = {
            #[cfg(target_os = "linux")]
            {
                signal == sys::SIGSEGV && context.uc_mcontext.trapno == X86_TRAP_PF
            }
            #[cfg(target_os = "macos")]
            {
                signal == sys::SIGBUS && (*context.uc_mcontext).__es.__trapno as u64 == X86_TRAP_PF
            }
            #[cfg(target_os = "freebsd")]
            {
                signal == sys::SIGBUS && context.uc_mcontext.mc_trapno == X86_TRAP_PF
            }
        };

        let rip = fetch_reg!(rip);
        let vmctx = &mut *vmctx;
        if vmctx.program_range.contains(&rip) {
            use polkavm_common::regmap::NativeReg;
            for reg in polkavm_common::program::Reg::ALL {
                let value = match polkavm_common::regmap::to_native_reg(reg) {
                    NativeReg::rax => fetch_reg!(rax),
                    NativeReg::rcx => fetch_reg!(rcx),
                    NativeReg::rdx => fetch_reg!(rdx),
                    NativeReg::rbx => fetch_reg!(rbx),
                    NativeReg::rbp => fetch_reg!(rbp),
                    NativeReg::rsi => fetch_reg!(rsi),
                    NativeReg::rdi => fetch_reg!(rdi),
                    NativeReg::r8 => fetch_reg!(r8),
                    NativeReg::r9 => fetch_reg!(r9),
                    NativeReg::r10 => fetch_reg!(r10),
                    NativeReg::r11 => fetch_reg!(r11),
                    NativeReg::r12 => fetch_reg!(r12),
                    NativeReg::r13 => fetch_reg!(r13),
                    NativeReg::r14 => fetch_reg!(r14),
                    NativeReg::r15 => fetch_reg!(r15),
                };
                vmctx.regs[reg as usize] = value;
            }

            //
            // RCX is TMP_REG, which is not mapped to any guest register.
            // therefore we store it separately here.
            //
            // N.B Even though it's a volatile register, we still need to
            // restore it to handle page faults correctly.
            //
            polkavm_common::static_assert!(polkavm_common::regmap::TMP_REG.equals(NativeReg::rcx));
            vmctx.tmp_reg.store(fetch_reg!(rcx), Ordering::Relaxed);

            vmctx.next_native_program_counter.store(rip, Ordering::Relaxed);

            if is_page_fault {
                let fault_address = {
                    #[cfg(target_os = "linux")]
                    {
                        info.__bindgen_anon_1.__bindgen_anon_1._sifields._sigfault._addr as u64
                    }
                    #[cfg(target_os = "macos")]
                    {
                        info.si_addr as u64
                    }
                    #[cfg(target_os = "freebsd")]
                    {
                        info.si_addr as u64
                    }
                };

                log::trace!("Page fault at 0x{fault_address:x} (rip: 0x{rip:x})");
                trigger_exit(vmctx, ExitReason::Segfault(fault_address));
            } else {
                log::trace!("Signal received at 0x{rip:x}");
                trigger_exit(vmctx, ExitReason::Signal);
            }
        }
    }

    // This signal is unrelated to anything the guest program did; proceed normally.

    let old = core::ptr::read(old);
    if old.sa_sigaction == sys::SIG_IGN || old.sa_sigaction == sys::SIG_DFL {
        sys::sigaction(signal, &old, core::ptr::null_mut());
        return;
    }

    if old.sa_flags & sys::SA_SIGINFO != 0 {
        let old_handler = core::mem::transmute::<usize, extern "C" fn(c_int, &sys::siginfo_t, &sys::ucontext_t)>(old.sa_sigaction);
        old_handler(signal, info, context);
    } else {
        let old_handler = core::mem::transmute::<usize, extern "C" fn(c_int)>(old.sa_sigaction);
        old_handler(signal);
    }
}

#[allow(clippy::fn_to_numeric_cast_any)]
unsafe fn register_signal_handler_for_signal(signal: c_int, old_sa: *mut sys::sigaction) -> Result<(), Error> {
    let mut sa: sys::sigaction = core::mem::zeroed();
    sa.sa_flags = sys::SA_SIGINFO | sys::SA_NODEFER;
    sa.sa_sigaction = signal_handler as usize;

    sys::sigemptyset(&mut sa.sa_mask);
    if sys::sigaction(signal, &sa, old_sa) < 0 {
        return Err(Error(std::io::Error::last_os_error()));
    }

    Ok(())
}

unsafe fn register_signal_handlers() -> Result<(), Error> {
    register_signal_handler_for_signal(sys::SIGSEGV, core::ptr::addr_of_mut!(OLD_SIGSEGV))?;
    register_signal_handler_for_signal(sys::SIGILL, core::ptr::addr_of_mut!(OLD_SIGILL))?;
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    register_signal_handler_for_signal(sys::SIGBUS, core::ptr::addr_of_mut!(OLD_SIGBUS))?;
    Ok(())
}

fn register_signal_handlers_if_necessary() -> Result<(), Error> {
    const STATE_UNINITIALIZED: usize = 0;
    const STATE_INITIALIZING: usize = 1;
    const STATE_FINISHED: usize = 2;
    const STATE_ERROR: usize = 3;

    static FLAG: AtomicUsize = AtomicUsize::new(STATE_UNINITIALIZED);
    if FLAG.load(Ordering::Relaxed) == STATE_FINISHED {
        return Ok(());
    }

    match FLAG.compare_exchange(STATE_UNINITIALIZED, STATE_INITIALIZING, Ordering::Acquire, Ordering::Relaxed) {
        Ok(_) => {
            // SAFETY: This can only run once and any parallel invocation will
            // wait for the first one that was triggered, so calling this is safe.
            let result = unsafe { register_signal_handlers() };
            if let Err(error) = result {
                FLAG.store(STATE_ERROR, Ordering::Release);
                Err(error)
            } else {
                FLAG.store(STATE_FINISHED, Ordering::Release);
                Ok(())
            }
        }
        Err(_) => loop {
            match FLAG.load(Ordering::Relaxed) {
                STATE_INITIALIZING => continue,
                STATE_FINISHED => return Ok(()),
                _ => return Err("failed to set up signal handlers".into()),
            }
        },
    }
}

thread_local! {
    static THREAD_VMCTX: UnsafeCell<*mut VmCtx> = const { UnsafeCell::new(core::ptr::null_mut()) };
}

unsafe fn sysreturn(vmctx: &mut VmCtx) -> ! {
    debug_assert_ne!(vmctx.return_address, 0);
    debug_assert_ne!(vmctx.return_stack_pointer, 0);

    // SAFETY: This function can only be called while we're executing guest code.
    unsafe {
        core::arch::asm!(r#"
            // Restore the stack pointer to its original value.
            mov rsp, [{vmctx} + 8]

            // Jump back
            jmp [{vmctx}]
        "#,
            vmctx = in(reg) vmctx,
            options(noreturn)
        );
    }
}

#[repr(C)]
enum ExitReason {
    None,
    Error,
    Signal,
    NotEnoughGas,
    Trap,
    Ecalli(u32),
    Segfault(u64),
    Step,
}

unsafe fn trigger_exit(vmctx: &mut VmCtx, exit_reason: ExitReason) -> ! {
    vmctx.exit_reason = exit_reason;
    sysreturn(vmctx);
}

const REG_COUNT: usize = polkavm_common::program::Reg::ALL.len();

#[repr(C)]
struct HeapInfo {
    heap_top: u64,
    heap_threshold: u64,
}

#[repr(C)]
struct VmCtx {
    // NOTE: These two fields are accessed from inline assembly so they shouldn't be moved!
    return_address: usize,
    return_stack_pointer: usize,

    arg: AtomicU32,

    heap_info: HeapInfo,
    heap_base: u32,
    heap_initial_threshold: u32,
    heap_max_size: u32,
    heap_map_index: usize,
    page_size: u32,
    maps: Vec<ProgramMap>,

    gas: AtomicI64,

    program_range: Range<u64>,
    exit_reason: ExitReason,

    regs: CacheAligned<[RegValue; REG_COUNT]>,
    tmp_reg: AtomicU64,
    sandbox: *mut Sandbox,
    program_counter: AtomicU32,
    next_program_counter: AtomicU32,
    next_native_program_counter: AtomicU64,
}

impl VmCtx {
    /// Creates a fresh VM context.
    pub fn new() -> Self {
        VmCtx {
            return_address: 0,
            return_stack_pointer: 0,
            exit_reason: ExitReason::None,
            program_range: 0..0,
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
            arg: AtomicU32::new(0),
            gas: AtomicI64::new(0),
            regs: CacheAligned([0; REG_COUNT]),
            tmp_reg: AtomicU64::new(0),
            sandbox: core::ptr::null_mut(),
            program_counter: AtomicU32::new(0),
            next_program_counter: AtomicU32::new(0),
            next_native_program_counter: AtomicU64::new(0),
        }
    }
}

// Make sure it fits within a single page on amd64.
polkavm_common::static_assert!(core::mem::size_of::<VmCtx>() <= 4096);

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

unsafe fn vmctx_ptr(memory: &Mmap) -> *const VmCtx {
    memory
        .as_ptr()
        .cast::<u8>()
        .offset(get_guest_memory_offset() as isize + GUEST_MEMORY_TO_VMCTX_OFFSET)
        .cast()
}

#[allow(clippy::needless_pass_by_ref_mut)]
unsafe fn vmctx_mut_ptr(memory: &mut Mmap) -> *mut VmCtx {
    memory
        .as_mut_ptr()
        .cast::<u8>()
        .offset(get_guest_memory_offset() as isize + GUEST_MEMORY_TO_VMCTX_OFFSET)
        .cast()
}

unsafe fn conjure_vmctx<'a>() -> &'a mut VmCtx {
    &mut *THREAD_VMCTX.with(|thread_ctx| *thread_ctx.get())
}

unsafe extern "C" fn syscall_hostcall() -> ! {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };
    let hostcall = vmctx.arg.load(Ordering::Relaxed);

    trigger_exit(vmctx, ExitReason::Ecalli(hostcall));
}

unsafe extern "C" fn syscall_step() -> ! {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };

    //
    // vmctx.next_program_counter is set by the guest code
    //

    trigger_exit(vmctx, ExitReason::Step);
}

unsafe extern "C" fn syscall_trap() -> ! {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };

    trigger_exit(vmctx, ExitReason::Trap);
}

unsafe extern "C" fn syscall_return() -> ! {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };

    sysreturn(vmctx);
}

unsafe extern "C" fn syscall_sbrk(pending_heap_top: u64) -> u32 {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };

    match sbrk(vmctx, pending_heap_top) {
        Ok(Some(new_heap_top)) => new_heap_top,
        Ok(None) => 0,
        Err(()) => {
            trigger_exit(vmctx, ExitReason::Error);
        }
    }
}

unsafe extern "C" fn syscall_not_enough_gas() -> ! {
    // SAFETY: We were called from the inside of the guest program, so vmctx must be valid.
    let vmctx = unsafe { conjure_vmctx() };

    trigger_exit(vmctx, ExitReason::NotEnoughGas);
}

unsafe fn sbrk(vmctx: &mut VmCtx, pending_heap_top: u64) -> Result<Option<u32>, ()> {
    if pending_heap_top > u64::from(vmctx.heap_base) + u64::from(vmctx.heap_max_size) {
        return Ok(None);
    }

    let Some(start) = align_to_next_page_usize(vmctx.page_size as usize, vmctx.heap_info.heap_top as usize) else {
        return Err(());
    };
    let Some(end) = align_to_next_page_usize(vmctx.page_size as usize, pending_heap_top as usize) else {
        return Err(());
    };

    let size = end - start;
    if size > 0 {
        let guest_memory_base = (vmctx as *mut VmCtx).cast::<u8>().offset(-GUEST_MEMORY_TO_VMCTX_OFFSET);
        let pointer = guest_memory_base.add(start);
        log::trace!(
            "sbrk: mapping 0x{:x}-0x{:x} (0x{:x}-0x{:x}) (0x{:x})",
            pointer as usize,
            pointer as usize + size,
            start,
            end,
            size
        );

        let result = sys::mmap(
            pointer.cast::<core::ffi::c_void>(),
            size,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_FIXED | sys::MAP_PRIVATE | sys::MAP_ANONYMOUS,
            -1,
            0,
        );

        if result == sys::MAP_FAILED {
            log::error!("sbrk mmap failed!");
            return Err(());
        }
    }

    debug_assert!(matches!(vmctx.maps[vmctx.heap_map_index].kind, MapKind::Transient));
    vmctx.maps[vmctx.heap_map_index].length = ((end as u64) - u64::from(vmctx.heap_initial_threshold)) as u32;

    log::trace!(
        "sbrk: heap memory range: 0x{:x}-0x{:x}",
        vmctx.maps[vmctx.heap_map_index].address,
        vmctx.maps[vmctx.heap_map_index].address + vmctx.maps[vmctx.heap_map_index].length
    );

    vmctx.heap_info.heap_top = pending_heap_top;
    vmctx.heap_info.heap_threshold = end as u64;

    Ok(Some(pending_heap_top as u32))
}

#[derive(Clone)]
pub struct SandboxProgram(Arc<SandboxProgramInner>);

#[derive(Clone)]
enum MapKind {
    Zeroed,
    Transient,
    Initialized(Arc<[u8]>),
}

#[derive(Clone)]
struct ProgramMap {
    address: u32,
    length: u32,
    is_writable: bool,
    kind: MapKind,
}

struct SandboxProgramInner {
    memory_map: Vec<ProgramMap>,
    heap_map_index: usize,
    code_memory: Mmap,
    code_length: usize,
    sysenter_address: u64,
}

impl super::SandboxProgram for SandboxProgram {
    fn machine_code(&self) -> &[u8] {
        &self.0.code_memory.as_slice()[..self.0.code_length]
    }
}

enum Poison {
    None,
    Executing,
    Poisoned,
}

pub struct Sandbox {
    poison: Poison,
    program: Option<SandboxProgram>,
    memory: Mmap,
    guest_memory_offset: usize,
    module: Option<Module>,

    gas_metering: Option<GasMeteringKind>,

    is_program_counter_valid: bool,
    next_program_counter: Option<ProgramCounter>,
    next_program_counter_changed: bool,

    page_set: PageSet,
    dynamic_paging_enabled: bool,

    aux_data_address: u32,
    aux_data_full_length: u32,
    aux_data_length: u32,
}

impl Drop for Sandbox {
    fn drop(&mut self) {}
}

impl Sandbox {
    #[inline]
    fn vmctx(&self) -> &VmCtx {
        // SAFETY: `memory` is always valid and contains a valid `VmCtx`.
        unsafe { &*vmctx_ptr(&self.memory) }
    }

    #[inline]
    fn vmctx_mut(&mut self) -> &mut VmCtx {
        // SAFETY: `memory` is always valid and contains a valid `VmCtx`.
        unsafe { &mut *vmctx_mut_ptr(&mut self.memory) }
    }

    fn clear_program(&mut self) -> Result<(), Error> {
        let length = self.memory.len() - self.guest_memory_offset;
        let program = self.program.take();

        self.memory.mmap_within(self.guest_memory_offset, length, 0)?;

        self.vmctx_mut().arg.store(0, Ordering::Relaxed);
        self.vmctx_mut().gas.store(0, Ordering::Relaxed);
        self.vmctx_mut().maps.clear();
        self.vmctx_mut().heap_info.heap_top = 0;
        self.vmctx_mut().heap_info.heap_threshold = 0;
        self.vmctx_mut().heap_base = 0;
        self.vmctx_mut().heap_initial_threshold = 0;
        self.vmctx_mut().heap_max_size = 0;
        self.vmctx_mut().heap_map_index = 0;
        self.vmctx_mut().page_size = 0;
        self.vmctx_mut().regs.fill(0);
        self.vmctx_mut().tmp_reg.store(0, Ordering::Relaxed);
        self.vmctx_mut().program_counter.store(0, Ordering::Relaxed);
        self.vmctx_mut().next_program_counter.store(0, Ordering::Relaxed);
        self.vmctx_mut().next_native_program_counter.store(0, Ordering::Relaxed);

        if let Some(program) = program {
            if let Some(program) = Arc::into_inner(program.0) {
                program.code_memory.unmap()?;
            }
        }

        Ok(())
    }

    fn force_reset_memory(&mut self) -> Result<(), Error> {
        log::trace!("Resetting memory");

        {
            // SAFETY: `memory` is always valid and contains a valid `VmCtx`.
            let maps = unsafe { &mut (*vmctx_mut_ptr(&mut self.memory)).maps };

            for map in maps {
                if map.length == 0 {
                    continue;
                }

                let offset = self.guest_memory_offset + to_usize(map.address);
                let length = to_usize(map.length).get();
                match map.kind {
                    MapKind::Initialized(ref initialize_with) => {
                        if !map.is_writable {
                            continue;
                        }

                        log::trace!("  Initializing: 0x{:x}..0x{:x}", offset, offset + length);

                        // SAFETY: Both pointers are valid.
                        // NOTE: We can't use `as_slice_mut` as that would return a slice to the whole chunk of memory,
                        //       which would include the vmctx to which we're currently holding a reference.
                        unsafe {
                            core::ptr::copy(
                                initialize_with.as_ptr(),
                                self.memory.as_mut_ptr().cast::<u8>().add(offset),
                                initialize_with.len(),
                            );
                        }
                    }
                    MapKind::Zeroed => {
                        if !map.is_writable {
                            continue;
                        }

                        log::trace!("  Clearing: 0x{:x}..0x{:x}", offset, offset + length);
                        self.memory.mmap_within(offset, length, PROT_READ | PROT_WRITE)?;
                    }
                    MapKind::Transient => {
                        log::trace!("  Clearing transient: 0x{:x}..0x{:x}", offset, offset + length);
                        self.memory.mmap_within(offset, length, 0)?;

                        map.length = 0;
                    }
                }
            }
        }

        let heap_base = self.vmctx().heap_base;
        let heap_initial_threshold = self.vmctx().heap_initial_threshold;
        self.vmctx_mut().heap_info.heap_top = heap_base.into();
        self.vmctx_mut().heap_info.heap_threshold = heap_initial_threshold.into();

        Ok(())
    }

    fn bound_check_access(&self, mut address: u32, mut length: u32, is_writable: bool) -> Result<(), ()> {
        if self.dynamic_paging_enabled {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + length - 1));
            if self.page_set.contains((page_start, page_end)) {
                return Ok(());
            } else {
                return Err(());
            }
        }

        let Some(address_end) = address.checked_add(length) else {
            return Err(());
        };

        if address >= self.aux_data_address && address_end < self.aux_data_address + self.aux_data_full_length {
            if address_end > self.aux_data_address + self.aux_data_length {
                return Err(());
            }
            return Ok(());
        }

        for map in &self.vmctx().maps {
            if address < map.address {
                return Err(());
            }

            let map_end = map.address + map.length;
            if address >= map_end {
                continue;
            } else if is_writable && !map.is_writable {
                return Err(());
            } else if address_end <= map_end {
                return Ok(());
            }

            length -= map_end - address;
            address = map_end;
        }

        if length == 0 {
            Ok(())
        } else {
            Err(())
        }
    }

    fn get_memory_slice(&self, address: u32, length: u32) -> Option<&[u8]> {
        self.bound_check_access(address, length, false).ok()?;
        let range = self.guest_memory_offset + address as usize..self.guest_memory_offset + address as usize + length as usize;
        Some(&self.memory.as_slice()[range])
    }

    fn get_memory_slice_mut(&mut self, address: u32, length: u32) -> Option<&mut [u8]> {
        if self.bound_check_access(address, length, true).is_err() {
            return None;
        }
        let range = self.guest_memory_offset + address as usize..self.guest_memory_offset + address as usize + length as usize;
        Some(&mut self.memory.as_slice_mut()[range])
    }

    fn handle_guest_signal(&mut self) -> Result<InterruptKind, Error> {
        use crate::sandbox::Sandbox;

        let machine_code_address = self.vmctx().next_native_program_counter.load(Ordering::Relaxed);

        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        let Some(machine_code_offset) = machine_code_address.checked_sub(compiled_module.native_code_origin) else {
            return Err(Error::from("internal error: address underflow after a trap"));
        };

        let Some(program_counter) = compiled_module.program_counter_by_native_code_offset(machine_code_offset, false) else {
            return Err(Error::from(
                "internal error: program counter not found for the given machine code address",
            ));
        };

        self.vmctx().program_counter.store(program_counter.0, Ordering::Relaxed);
        self.vmctx().next_program_counter.store(program_counter.0, Ordering::Relaxed);
        self.is_program_counter_valid = true;

        if self.gas_metering.is_some() && self.gas() < 0 {
            let Some(offset) = machine_code_offset.checked_sub(GAS_METERING_TRAP_OFFSET) else {
                return Err(Error::from("internal error: address underflow after a guest signal"));
            };

            self.vmctx()
                .next_native_program_counter
                .store(compiled_module.native_code_origin + offset as u64, Ordering::Relaxed);

            let offset = offset as usize + GAS_COST_GENERIC_SANDBOX_OFFSET;
            let Some(gas_cost) = &compiled_module.machine_code().get(offset..offset + 4) else {
                return Err(Error::from(
                    "internal error: failed to read back the gas cost from the machine code",
                ));
            };

            let gas_cost = u32::from_le_bytes([gas_cost[0], gas_cost[1], gas_cost[2], gas_cost[3]]);
            let gas = self.vmctx().gas.fetch_add(i64::from(gas_cost), Ordering::Relaxed);
            log::trace!(
                "Out of gas; program counter = {program_counter}, reverting gas: {gas} -> {new_gas} (gas cost: {gas_cost})",
                new_gas = gas + i64::from(gas_cost)
            );

            Ok(InterruptKind::NotEnoughGas)
        } else {
            self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
            Ok(InterruptKind::Trap)
        }
    }

    fn handle_guest_pagefault(&mut self, fault_address: u64) -> Result<InterruptKind, Error> {
        use crate::sandbox::Sandbox;

        let machine_code_address = self.vmctx().next_native_program_counter.load(Ordering::Relaxed);

        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        let Some(machine_code_offset) = machine_code_address.checked_sub(compiled_module.native_code_origin) else {
            return Err(Error::from("internal error: address underflow after a trap"));
        };

        let Some(program_counter) = compiled_module.program_counter_by_native_code_offset(machine_code_offset, false) else {
            return Err(Error::from(
                "internal error: program counter not found for the given machine code address",
            ));
        };

        self.vmctx().program_counter.store(program_counter.0, Ordering::Relaxed);
        self.vmctx().next_program_counter.store(program_counter.0, Ordering::Relaxed);
        self.is_program_counter_valid = true;

        let page_size = get_native_page_size() as u32;
        let page_address = fault_address.wrapping_sub(self.memory.as_ptr() as u64 + self.guest_memory_offset as u64) as u32;
        let page_address = page_address & !(page_size - 1);

        let is_trap = if self.dynamic_paging_enabled && page_address >= 0x10000 {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(page_address));
            self.page_set.contains((page_start, page_start))
        } else {
            true
        };

        if is_trap {
            self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
            Ok(InterruptKind::Trap)
        } else {
            self.vmctx()
                .next_native_program_counter
                .store(machine_code_address, Ordering::Relaxed);
            Ok(InterruptKind::Segfault(Segfault { page_address, page_size }))
        }
    }

    fn set_aux_data_permission_for_host(&mut self) -> Result<(), Error> {
        if self.aux_data_full_length != 0 {
            let offset = self.guest_memory_offset + self.aux_data_address as usize;
            let full_length = self.aux_data_full_length as usize;

            self.memory.mprotect(offset, full_length, PROT_READ | PROT_WRITE)?;
        }

        Ok(())
    }

    fn set_aux_data_permission_for_guest(&mut self) -> Result<(), Error> {
        if self.aux_data_full_length != 0 {
            let offset = self.guest_memory_offset + self.aux_data_address as usize;
            let length = self.aux_data_length as usize;
            let full_length = self.aux_data_full_length as usize;

            self.memory.mprotect(offset, full_length, 0)?;
            self.memory.mprotect(offset, length, PROT_READ)?;
        }

        Ok(())
    }
}

impl super::SandboxAddressSpace for Mmap {
    fn native_code_origin(&self) -> u64 {
        self.as_ptr() as u64
    }
}

impl super::Sandbox for Sandbox {
    const KIND: SandboxKind = SandboxKind::Generic;

    type Config = SandboxConfig;
    type Error = Error;
    type Program = SandboxProgram;
    type AddressSpace = Mmap;
    type GlobalState = GlobalState;
    type JumpTable = Vec<usize>;

    fn downcast_module(module: &Module) -> &CompiledModule<Self> {
        match module.compiled_module() {
            CompiledModuleKind::Generic(ref module) => module,
            _ => unreachable!(),
        }
    }

    fn downcast_global_state(global: &crate::sandbox::GlobalStateKind) -> &Self::GlobalState {
        #[allow(clippy::match_wildcard_for_single_variants)]
        match global {
            crate::sandbox::GlobalStateKind::Generic(global) => global,
            _ => unreachable!(),
        }
    }

    fn downcast_worker_cache(cache: &WorkerCacheKind) -> &WorkerCache<Self> {
        #[allow(clippy::match_wildcard_for_single_variants)]
        match cache {
            crate::sandbox::WorkerCacheKind::Generic(ref cache) => cache,
            _ => unreachable!(),
        }
    }

    fn allocate_jump_table(_global: &Self::GlobalState, count: usize) -> Result<Self::JumpTable, Self::Error> {
        // TODO: Cache this and don't unnecessarily double-initialize it.
        let native_page_size = get_native_page_size();
        let size = align_to_next_page_usize(native_page_size, count * core::mem::size_of::<usize>()).unwrap();
        Ok(vec![0; size / core::mem::size_of::<usize>()])
    }

    fn reserve_address_space() -> Result<Self::AddressSpace, Self::Error> {
        Mmap::reserve_address_space(VM_SANDBOX_MAXIMUM_NATIVE_CODE_SIZE as usize + VM_SANDBOX_MAXIMUM_JUMP_TABLE_VIRTUAL_SIZE as usize)
    }

    fn prepare_program(
        _global: &Self::GlobalState,
        init: SandboxInit<Self>,
        mut map: Self::AddressSpace,
    ) -> Result<Self::Program, Self::Error> {
        let native_page_size = get_native_page_size();
        let cfg = init.guest_init.memory_map()?;
        let jump_table = as_bytes(&init.jump_table);

        let code_size = align_to_next_page_usize(native_page_size, init.code.len()).unwrap();
        let jump_table_size = align_to_next_page_usize(native_page_size, jump_table.len()).unwrap();

        let jump_table_offset = code_size as usize;
        let sysreturn_offset = jump_table_offset + (VM_ADDR_JUMP_TABLE_RETURN_TO_HOST - VM_ADDR_JUMP_TABLE) as usize;

        map.modify_and_protect(0, code_size as usize, PROT_EXEC | PROT_READ, |slice| {
            slice[..init.code.len()].copy_from_slice(init.code);
        })?;

        map.modify_and_protect(jump_table_offset, jump_table_size as usize, PROT_READ, |slice| {
            slice[..jump_table.len()].copy_from_slice(jump_table);
        })?;

        map.modify_and_protect(sysreturn_offset, native_page_size, PROT_READ, |slice| {
            slice[..8].copy_from_slice(&init.sysreturn_address.to_le_bytes());
        })?;

        log::trace!(
            "New code range: 0x{:x}-0x{:x} (0x{:x})",
            map.as_ptr() as usize,
            map.as_ptr() as usize + code_size,
            code_size
        );

        log::trace!(
            "New jump table range: 0x{:x}-0x{:x} (0x{:x})",
            map.as_ptr() as usize + jump_table_offset,
            map.as_ptr() as usize + jump_table_offset + jump_table_size,
            jump_table_size
        );

        log::trace!(
            "New sysreturn address: 0x{:x} (set at 0x{:x})",
            init.sysreturn_address,
            map.as_ptr() as usize + sysreturn_offset
        );

        let mut memory_map = Vec::new();
        if cfg.ro_data_size() > 0 {
            let mut ro_data = init.guest_init.ro_data.to_vec();
            let physical_size = align_to_next_page_usize(native_page_size, ro_data.len()).unwrap();
            ro_data.resize(physical_size, 0);
            let physical_size = physical_size.try_into().expect("overflow");

            let virtual_size = cfg.ro_data_size();
            if physical_size > 0 {
                memory_map.push(ProgramMap {
                    address: cfg.ro_data_address(),
                    length: physical_size,
                    is_writable: false,
                    kind: MapKind::Initialized(ro_data.into()),
                });
            }

            let padding = virtual_size - physical_size;
            if padding > 0 {
                memory_map.push(ProgramMap {
                    address: cfg.ro_data_address() + physical_size,
                    length: padding,
                    is_writable: false,
                    kind: MapKind::Zeroed,
                });
            }
        }

        if cfg.rw_data_size() > 0 {
            let mut rw_data = init.guest_init.rw_data.to_vec();
            let physical_size = align_to_next_page_usize(native_page_size, rw_data.len()).unwrap();
            rw_data.resize(physical_size, 0);
            let physical_size = physical_size.try_into().expect("overflow");

            let virtual_size = cfg.rw_data_size();
            if physical_size > 0 {
                memory_map.push(ProgramMap {
                    address: cfg.rw_data_address(),
                    length: physical_size,
                    is_writable: true,
                    kind: MapKind::Initialized(rw_data.into()),
                });
            }

            let padding = virtual_size - physical_size;
            if padding > 0 {
                memory_map.push(ProgramMap {
                    address: cfg.rw_data_address() + physical_size,
                    length: padding,
                    is_writable: true,
                    kind: MapKind::Zeroed,
                });
            }
        }

        let heap_map_index = memory_map.len();

        // Reserve entry for the heap.
        memory_map.push(ProgramMap {
            address: cfg.rw_data_range().end,
            length: 0,
            is_writable: true,
            kind: MapKind::Transient,
        });

        if cfg.stack_size() > 0 {
            memory_map.push(ProgramMap {
                address: cfg.stack_address_low(),
                length: cfg.stack_size(),
                is_writable: true,
                kind: MapKind::Zeroed,
            });
        }

        if cfg.aux_data_size() > 0 {
            memory_map.push(ProgramMap {
                address: cfg.aux_data_address(),
                length: cfg.aux_data_size(),
                is_writable: true,
                kind: MapKind::Transient,
            });
        }

        // Make sure the map is sorted.
        assert!(memory_map.windows(2).all(|pair| {
            matches!(
                (pair[0].address + pair[0].length).cmp(&pair[1].address),
                core::cmp::Ordering::Less | core::cmp::Ordering::Equal
            )
        }));

        Ok(SandboxProgram(Arc::new(SandboxProgramInner {
            memory_map,
            heap_map_index,
            code_memory: map,
            code_length: init.code.len(),
            sysenter_address: init.sysenter_address,
        })))
    }

    fn spawn(_global: &Self::GlobalState, _config: &SandboxConfig) -> Result<Self, Error> {
        register_signal_handlers_if_necessary()?;

        let guest_memory_offset = get_guest_memory_offset();
        let mut memory = Mmap::reserve_address_space(guest_memory_offset + 0x100000000)?;

        // Make the space for VmCtx read-write.
        polkavm_common::static_assert!(GUEST_MEMORY_TO_VMCTX_OFFSET < 0);
        memory.mprotect(0, guest_memory_offset, PROT_READ | PROT_WRITE)?;

        // SAFETY: We just mmaped this and made it read-write.
        unsafe {
            core::ptr::write(vmctx_mut_ptr(&mut memory), VmCtx::new());
        }

        Ok(Sandbox {
            poison: Poison::None,
            program: None,
            memory,
            guest_memory_offset,
            module: None,
            gas_metering: None,
            is_program_counter_valid: false,
            next_program_counter: None,
            next_program_counter_changed: true,
            page_set: PageSet::new(),
            dynamic_paging_enabled: false,
            aux_data_address: 0,
            aux_data_full_length: 0,
            aux_data_length: 0,
        })
    }

    fn load_module(&mut self, _global: &Self::GlobalState, module: &Module) -> Result<(), Self::Error> {
        if self.module.is_some() {
            return Err(Error::from("module already loaded"));
        }

        if module.is_dynamic_paging() && get_native_page_size() != module.memory_map().page_size() as usize {
            return Err(Error::from(
                "dynamic paging is currently unsupported if the module's page size doesn't match the native page size",
            ));
        }

        let compiled_module = <Self as crate::sandbox::Sandbox>::downcast_module(module);
        let program = &compiled_module.sandbox_program.0;

        log::trace!("Loading module into sandbox... (dynamic_paging={})", module.is_dynamic_paging());
        self.clear_program()?;

        if !module.is_dynamic_paging() {
            for map in &program.memory_map {
                if map.length > 0 {
                    let mut protection = PROT_READ;
                    if map.is_writable {
                        protection |= PROT_WRITE;
                    }

                    let offset = self.guest_memory_offset + to_usize(map.address);
                    let length = to_usize(map.length).get();
                    self.memory.modify_and_protect(offset, length, protection, |slice| match map.kind {
                        MapKind::Initialized(ref initialize_with) => {
                            slice.copy_from_slice(initialize_with);
                        }
                        MapKind::Zeroed | MapKind::Transient => {
                            slice.fill(0);
                        }
                    })?;

                    let memory_address = self.memory.as_ptr() as usize + offset;
                    log::trace!(
                        "  New accessible range: 0x{:x}-0x{:x} (0x{:x}-0x{:x}) (0x{:x}){}{}",
                        memory_address,
                        memory_address + length,
                        to_usize(map.address).get(),
                        to_usize(map.address).get() + length,
                        length,
                        if !map.is_writable { " [RO]" } else { "" },
                        if matches!(map.kind, MapKind::Initialized(..)) {
                            " [INIT]"
                        } else {
                            ""
                        },
                    );
                }

                self.vmctx_mut().maps.push(map.clone());
            }
        }

        self.vmctx_mut().heap_info.heap_top = u64::from(module.memory_map().heap_base());
        self.vmctx_mut().heap_info.heap_threshold = u64::from(module.memory_map().rw_data_range().end);
        self.vmctx_mut().heap_base = module.memory_map().heap_base();
        self.vmctx_mut().heap_initial_threshold = module.memory_map().rw_data_range().end;
        self.vmctx_mut().heap_max_size = module.memory_map().max_heap_size();
        self.vmctx_mut().heap_map_index = program.heap_map_index;
        self.vmctx_mut().page_size = module.memory_map().page_size();

        let code = &program.code_memory;
        let address = code.as_ptr() as u64;
        self.vmctx_mut().program_range = address..address + code.len() as u64;
        log::trace!(
            "Program range: 0x{:x}-0x{:x} (0x{:x})",
            address,
            address + code.len() as u64,
            code.len()
        );

        self.program = Some(SandboxProgram(Arc::clone(program)));
        self.gas_metering = module.gas_metering();
        self.module = Some(module.clone());
        self.aux_data_address = module.memory_map().aux_data_address();
        self.aux_data_full_length = module.memory_map().aux_data_size();
        self.aux_data_length = self.aux_data_full_length;
        self.dynamic_paging_enabled = module.is_dynamic_paging();

        Ok(())
    }

    fn recycle(&mut self, _global: &Self::GlobalState) -> Result<(), Self::Error> {
        log::trace!("Recycling sandbox");
        if self.dynamic_paging_enabled {
            self.free_pages(0x10000, 0xffff0000)?;
        }

        self.module = None;
        self.page_set.clear();

        Ok(())
    }

    fn run(&mut self) -> Result<InterruptKind, Self::Error> {
        assert!(!matches!(self.poison, Poison::Poisoned), "sandbox has been poisoned");

        if self.module.is_none() {
            return Err(Error::from("no module loaded into the sandbox"));
        };

        if self.next_program_counter_changed {
            let Some(pc) = self.next_program_counter.take() else {
                panic!("failed to run: next program counter is not set");
            };

            self.next_program_counter_changed = false;

            let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
            let Some(address) = compiled_module.lookup_native_code_address(pc) else {
                log::debug!("Tried to call into {pc} which doesn't have any native code associated with it");
                self.is_program_counter_valid = true;
                self.vmctx().program_counter.store(pc.0, Ordering::Relaxed);
                if self.module.as_ref().unwrap().is_step_tracing() {
                    self.vmctx().next_program_counter.store(pc.0, Ordering::Relaxed);
                    self.vmctx()
                        .next_native_program_counter
                        .store(compiled_module.invalid_code_offset_address, Ordering::Relaxed);
                    return Ok(InterruptKind::Step);
                } else {
                    self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
                    return Ok(InterruptKind::Trap);
                }
            };

            log::trace!("Jumping into: {pc} (0x{address:x})");
            self.vmctx_mut().next_program_counter.store(pc.0, Ordering::Relaxed);
            self.vmctx_mut().next_native_program_counter.store(address, Ordering::Relaxed);
        } else {
            log::trace!(
                "Resuming into: {} (0x{:x})",
                self.vmctx().next_program_counter.load(Ordering::Relaxed),
                self.vmctx().next_native_program_counter.load(Ordering::Relaxed)
            );
        }

        self.vmctx_mut().sandbox = self;
        self.vmctx_mut().exit_reason = ExitReason::None;

        self.poison = Poison::Executing;
        self.is_program_counter_valid = true;

        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        let entry_point = compiled_module.sandbox_program.0.sysenter_address;

        log::trace!("Jumping into guest program: 0x{:x}", entry_point);
        self.set_aux_data_permission_for_guest().map_err(Error::from)?;

        #[allow(clippy::undocumented_unsafe_blocks)]
        unsafe {
            let vmctx = vmctx_mut_ptr(&mut self.memory);
            THREAD_VMCTX.with(|thread_ctx| core::ptr::write(thread_ctx.get(), vmctx));

            let guest_memory = self.memory.as_ptr().cast::<u8>().add(self.guest_memory_offset);
            let tmp_reg = self.vmctx().tmp_reg.load(Ordering::Relaxed);

            core::arch::asm!(r#"
                push rbp
                push rbx

                // Fill in the return address.
                lea rbx, [rip+2f]
                mov [r14], rbx

                // Fill in the return stack pointer.
                mov [r14 + 8], rsp

                // Align the stack.
                sub rsp, 8

                // Call into the guest program.
                jmp {entry_point}

                // We will jump here on exit.
                2:

                pop rbx
                pop rbp
            "#,
                entry_point = in(reg) entry_point,
                // Mark all of the clobbered registers.
                //
                // We need to save and restore rbp and rbx manually since
                // the inline assembly doesn't support using them as operands.
                clobber_abi("C"),
                lateout("rax") _,
                lateout("rdx") _,
                lateout("rsi") _,
                lateout("rdi") _,
                lateout("r8") _,
                lateout("r9") _,
                lateout("r10") _,
                lateout("r11") _,
                lateout("r12") _,
                lateout("r13") _,
                inlateout("rcx") tmp_reg => _,
                inlateout("r14") vmctx => _,
                in("r15") guest_memory,
            );

            THREAD_VMCTX.with(|thread_ctx| core::ptr::write(thread_ctx.get(), core::ptr::null_mut()));
        };

        log::trace!("Returned from guest program");
        self.set_aux_data_permission_for_host().map_err(Error::from)?;
        self.poison = Poison::None;

        if self.module.as_ref().unwrap().gas_metering() == Some(GasMeteringKind::Async) && self.gas() < 0 {
            self.is_program_counter_valid = false;
            self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
            return Ok(InterruptKind::NotEnoughGas);
        }

        Ok(match self.vmctx().exit_reason {
            ExitReason::None => {
                self.is_program_counter_valid = false;
                InterruptKind::Finished
            }
            ExitReason::Error => {
                self.poison = Poison::Poisoned;
                log::error!("Sandbox poisoned");
                InterruptKind::Trap
            }
            ExitReason::Signal => self.handle_guest_signal().map_err(Error::from)?,
            ExitReason::NotEnoughGas => InterruptKind::NotEnoughGas,
            ExitReason::Trap => InterruptKind::Trap,
            ExitReason::Step => InterruptKind::Step,
            ExitReason::Ecalli(num) => InterruptKind::Ecalli(num),
            ExitReason::Segfault(address) => self.handle_guest_pagefault(address).map_err(Error::from)?,
        })
    }

    fn reg(&self, reg: Reg) -> RegValue {
        assert!(!matches!(self.poison, Poison::Poisoned), "sandbox has been poisoned");
        self.vmctx().regs[reg as usize]
    }

    fn set_reg(&mut self, reg: Reg, value: RegValue) {
        assert!(!matches!(self.poison, Poison::Poisoned), "sandbox has been poisoned");
        self.vmctx_mut().regs[reg as usize] = value;
    }

    fn gas(&self) -> Gas {
        self.vmctx().gas.load(Ordering::Relaxed)
    }

    fn set_gas(&mut self, gas: Gas) {
        self.vmctx_mut().gas.store(gas, Ordering::Relaxed);
    }

    fn program_counter(&self) -> Option<ProgramCounter> {
        if !self.is_program_counter_valid {
            return None;
        }
        let program_counter = self.vmctx().program_counter.load(Ordering::Relaxed);
        Some(ProgramCounter(program_counter))
    }

    fn next_program_counter(&self) -> Option<ProgramCounter> {
        if self.next_program_counter.is_some() {
            return self.next_program_counter;
        }

        if self.vmctx().next_native_program_counter.load(Ordering::Relaxed) == 0 {
            None
        } else {
            Some(ProgramCounter(self.vmctx().next_program_counter.load(Ordering::Relaxed)))
        }
    }

    fn set_next_program_counter(&mut self, pc: ProgramCounter) {
        self.is_program_counter_valid = false;
        self.next_program_counter = Some(pc);
        self.next_program_counter_changed = true;
    }

    fn next_native_program_counter(&self) -> Option<usize> {
        let next_native_program_counter = self.vmctx().next_native_program_counter.load(Ordering::Relaxed);
        if next_native_program_counter == 0 {
            return None;
        }
        Some(next_native_program_counter as usize)
    }

    fn accessible_aux_size(&self) -> u32 {
        assert!(!self.dynamic_paging_enabled);
        self.aux_data_length
    }

    fn set_accessible_aux_size(&mut self, size: u32) -> Result<(), Error> {
        assert!(!self.dynamic_paging_enabled);

        if self.aux_data_address == 0 || self.aux_data_full_length == 0 {
            return Err(Error::from("aux data address or length is zero"));
        }
        if size > self.aux_data_full_length {
            return Err(Error::from("size exceeds the full length of aux data"));
        }
        self.aux_data_length = size;
        Ok(())
    }

    fn is_memory_accessible(&self, address: u32, size: u32, is_writable: bool) -> bool {
        assert!(self.dynamic_paging_enabled);
        self.bound_check_access(address, size, is_writable).is_ok()
    }

    fn reset_memory(&mut self) -> Result<(), Error> {
        if self.module.is_none() {
            return Err(Error::from("no module loaded into the sandbox"));
        };

        if !self.dynamic_paging_enabled {
            self.force_reset_memory()
        } else {
            self.free_pages(0x10000, 0xffff0000)
        }
    }

    fn read_memory_into<'slice>(&self, address: u32, slice: &'slice mut [MaybeUninit<u8>]) -> Result<&'slice mut [u8], MemoryAccessError> {
        log::trace!(
            "Reading memory: 0x{:x}-0x{:x} ({} bytes)",
            address,
            cast(address).to_usize() + slice.len(),
            slice.len()
        );

        if matches!(self.poison, Poison::Poisoned) {
            return Err(MemoryAccessError::Error("read failed: sandbox has been poisoned".into()));
        }

        if self.dynamic_paging_enabled {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + slice.len() as u32 - 1));
            if !self.page_set.contains((page_start, page_end)) {
                return Err(MemoryAccessError::Error("incomplete read".into()));
            }
        }

        let Some(memory_slice) = self.get_memory_slice(address, slice.len() as u32) else {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: slice.len() as u64,
            });
        };

        Ok(byte_slice_init(slice, memory_slice))
    }

    fn write_memory(&mut self, address: u32, data: &[u8]) -> Result<(), MemoryAccessError> {
        log::trace!(
            "Writing memory: 0x{:x}-0x{:x} ({} bytes)",
            address,
            address as usize + data.len(),
            data.len()
        );

        if data.is_empty() {
            return Ok(());
        }

        if matches!(self.poison, Poison::Poisoned) {
            return Err(MemoryAccessError::Error("write failed: sandbox has been poisoned".into()));
        }

        if self.dynamic_paging_enabled {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + data.len() as u32 - 1));
            let page_size = get_native_page_size() as u32;
            let page_count = page_end - page_start + 1;

            if !self.page_set.contains((page_start, page_end)) {
                self.memory
                    .mprotect(
                        self.guest_memory_offset + (page_start * page_size) as usize,
                        (page_count * page_size) as usize,
                        PROT_READ | PROT_WRITE,
                    )
                    .map_err(|e| MemoryAccessError::Error(e.into()))?;
            }

            self.page_set.insert((page_start, page_end));
        }

        let Some(slice) = self.get_memory_slice_mut(address, data.len() as u32) else {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: data.len() as u64,
            });
        };

        slice.copy_from_slice(data);
        Ok(())
    }

    fn zero_memory(&mut self, address: u32, length: u32) -> Result<(), MemoryAccessError> {
        log::trace!("Zeroing memory: 0x{:x}-0x{:x} ({} bytes)", address, address + length, length);

        if length == 0 {
            return Ok(());
        }

        if matches!(self.poison, Poison::Poisoned) {
            return Err(MemoryAccessError::Error("zero failed: sandbox has been poisoned".into()));
        }

        if self.dynamic_paging_enabled {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + length - 1));
            let page_size = get_native_page_size() as u32;
            let page_count = page_end - page_start + 1;

            if !self.page_set.contains((page_start, page_end)) {
                self.memory
                    .mprotect(
                        self.guest_memory_offset + (page_start * page_size) as usize,
                        (page_count * page_size) as usize,
                        PROT_READ | PROT_WRITE,
                    )
                    .map_err(|e| MemoryAccessError::Error(e.into()))?;
            }

            self.page_set.insert((page_start, page_end));
        }

        let Some(slice) = self.get_memory_slice_mut(address, length as u32) else {
            return Err(MemoryAccessError::OutOfRangeAccess {
                address,
                length: length as u64,
            });
        };

        slice.fill(0);
        Ok(())
    }

    fn protect_memory(&mut self, address: u32, length: u32) -> Result<(), MemoryAccessError> {
        assert!(self.dynamic_paging_enabled);

        log::trace!(
            "Protecting memory: 0x{:x}-0x{:x} ({} bytes)",
            address,
            address as usize + length as usize,
            length
        );

        self.memory
            .mprotect(self.guest_memory_offset + address as usize, length as usize, PROT_READ)
            .map_err(|e| MemoryAccessError::Error(e.into()))?;

        let module = self.module.as_ref().unwrap();
        let page_start = module.address_to_page(module.round_to_page_size_down(address));
        let page_end = module.address_to_page(module.round_to_page_size_down(address + length - 1));
        self.page_set.insert((page_start, page_end));

        Ok(())
    }

    fn free_pages(&mut self, address: u32, length: u32) -> Result<(), Error> {
        assert!(self.dynamic_paging_enabled);

        self.memory
            .madvise(self.guest_memory_offset + address as usize, length as usize, MADV_DONTNEED)?;

        self.memory
            .madvise(self.guest_memory_offset + address as usize, length as usize, MADV_FREE)?;

        if address <= 0x10000 && length >= 0xffff0000 {
            self.page_set.clear();
            self.memory.mprotect(self.guest_memory_offset, 0x100000000 as usize, 0)?;
        } else {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + length - 1));
            let page_size = get_native_page_size() as u32;
            let page_count = page_end - page_start + 1;
            self.page_set.remove((page_start, page_end));
            self.memory.mprotect(
                self.guest_memory_offset + (page_start * page_size) as usize,
                (page_count * page_size) as usize,
                0,
            )?;
        }

        Ok(())
    }

    fn heap_size(&self) -> u32 {
        let heap_base = self.vmctx().heap_base;
        let heap_top = self.vmctx().heap_info.heap_top;
        (heap_top - u64::from(heap_base)) as u32
    }

    fn sbrk(&mut self, size: u32) -> Result<Option<u32>, Error> {
        let new_heap_top = self.vmctx().heap_info.heap_top + u64::from(size);

        // SAFETY: `vmctx` is valid and was allocated along with the guest memory.
        match unsafe { sbrk(self.vmctx_mut(), new_heap_top) } {
            Ok(result) => Ok(result),
            Err(()) => panic!("sbrk failed"),
        }
    }

    fn pid(&self) -> Option<u32> {
        None
    }

    fn address_table() -> AddressTable {
        AddressTable::from_raw(AddressTableRaw {
            syscall_hostcall,
            syscall_trap,
            syscall_return,
            syscall_step,
            syscall_sbrk,
            syscall_not_enough_gas,
        })
    }

    fn offset_table() -> OffsetTable {
        OffsetTable {
            arg: get_field_offset!(VmCtx::new(), |base| base.arg.as_ptr()),
            gas: get_field_offset!(VmCtx::new(), |base| base.gas.as_ptr()),
            heap_info: get_field_offset!(VmCtx::new(), |base| &base.heap_info),
            next_native_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_native_program_counter.as_ptr()),
            next_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_program_counter.as_ptr()),
            program_counter: get_field_offset!(VmCtx::new(), |base| base.program_counter.as_ptr()),
            regs: get_field_offset!(VmCtx::new(), |base| &base.regs),
        }
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
