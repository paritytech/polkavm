#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::manual_range_contains)]

extern crate polkavm_linux_raw as linux_raw;

use polkavm_common::{
    cast::cast,
    program::Reg,
    utils::{align_to_next_page_usize, slice_assume_init_mut},
    zygote::{
        AddressTable, AddressTablePacked, ExtTable, ExtTablePacked, VmCtx, VmFd, VmMap, VMCTX_FUTEX_BUSY, VMCTX_FUTEX_GUEST_ECALLI,
        VMCTX_FUTEX_GUEST_NOT_ENOUGH_GAS, VMCTX_FUTEX_GUEST_SIGNAL, VMCTX_FUTEX_GUEST_STEP, VMCTX_FUTEX_GUEST_TRAP, VMCTX_FUTEX_IDLE,
        VM_ADDR_NATIVE_CODE,
    },
};

pub use linux_raw::Error;

use core::cell::UnsafeCell;
use core::ffi::{c_int, c_uint};
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;
use core::time::Duration;
use linux_raw::{abort, cstr, syscall_readonly, Fd, Mmap};
use std::sync::Arc;
use std::time::Instant;

use super::{get_native_page_size, OffsetTable, SandboxInit, SandboxKind, WorkerCache, WorkerCacheKind};
use crate::api::{CompiledModuleKind, MemoryAccessError, Module};
use crate::compiler::{Bitness, CompiledModule, B32, B64};
use crate::config::Config;
use crate::config::GasMeteringKind;
use crate::page_set::PageSet;
use crate::shm_allocator::{ShmAllocation, ShmAllocator};
use crate::{Gas, InterruptKind, ProgramCounter, RegValue, Segfault};

pub struct GlobalState {
    shared_memory: ShmAllocator,
    uffd_available: bool,
    zygote_memfd: Fd,
}

const UFFD_REQUIRED_FEATURES: u64 =
    (linux_raw::UFFD_FEATURE_MISSING_SHMEM | linux_raw::UFFD_FEATURE_MINOR_SHMEM | linux_raw::UFFD_FEATURE_WP_HUGETLBFS_SHMEM) as u64;

const SANDBOX_FLAGS: u64 = (linux_raw::CLONE_NEWCGROUP
    | linux_raw::CLONE_NEWIPC
    | linux_raw::CLONE_NEWNET
    | linux_raw::CLONE_NEWNS
    | linux_raw::CLONE_NEWPID
    | linux_raw::CLONE_NEWUSER
    | linux_raw::CLONE_NEWUTS) as u64;

enum Fork {
    Child,
    Host(ChildProcess),
}

fn clone(flags: u64) -> Result<Fork, Error> {
    let mut pidfd: c_int = -1;
    let args = CloneArgs {
        flags: linux_raw::CLONE_CLEAR_SIGHAND | u64::from(linux_raw::CLONE_PIDFD) | flags,
        pidfd: &mut pidfd,
        child_tid: 0,
        parent_tid: 0,
        exit_signal: 0,
        stack: 0,
        stack_size: 0,
        tls: 0,
    };

    let mut child_pid = unsafe { linux_raw::syscall!(linux_raw::SYS_clone3, core::ptr::addr_of!(args), core::mem::size_of::<CloneArgs>()) };

    if child_pid < 0 {
        // Fallback for Linux versions older than 5.5.
        let error = Error::from_syscall("failed to clone the process", child_pid);
        child_pid = unsafe { linux_raw::syscall!(linux_raw::SYS_clone, flags, 0, 0, 0, 0) };

        if child_pid < 0 {
            return Err(error.unwrap_err());
        }
    }

    if child_pid == 0 {
        Ok(Fork::Child)
    } else {
        Ok(Fork::Host(ChildProcess {
            pid: child_pid as c_int,
            pidfd: if pidfd < 0 { None } else { Some(Fd::from_raw_unchecked(pidfd)) },
        }))
    }
}

impl GlobalState {
    pub fn new(config: &Config) -> Result<Self, Error> {
        let uffd_available = config.allow_dynamic_paging;
        if uffd_available {
            let userfaultfd = linux_raw::sys_userfaultfd(linux_raw::O_CLOEXEC).map_err(|error| {
                if error.errno() == linux_raw::EPERM
                    && std::fs::read("/proc/sys/vm/unprivileged_userfaultfd")
                        .map(|blob| blob == b"0\n")
                        .unwrap_or(false)
                {
                    Error::from(
                        "failed to create an userfaultfd: permission denied; run 'sysctl -w vm.unprivileged_userfaultfd=1' to enable it",
                    )
                } else {
                    Error::from(format!("failed to create an userfaultfd: {error}"))
                }
            })?;

            let mut api: linux_raw::uffdio_api = linux_raw::uffdio_api {
                api: linux_raw::UFFD_API,
                ..linux_raw::uffdio_api::default()
            };

            linux_raw::sys_uffdio_api(userfaultfd.borrow(), &mut api)
                .map_err(|error| Error::from(format!("failed to fetch the available userfaultfd features: {error}")))?;

            if (api.features & UFFD_REQUIRED_FEATURES) != UFFD_REQUIRED_FEATURES {
                return Err(Error::from(
                    "not all required userfaultfd features are available; you need to update your Linux kernel to version 6.8 or newer",
                ));
            }

            userfaultfd.close()?;

            let utsname = linux_raw::sys_uname()?;
            fn kernel_version(utsname: &linux_raw::new_utsname) -> Option<(u32, u32)> {
                let release: &[core::ffi::c_char] = &utsname.release;
                let release: &[u8] = unsafe { core::slice::from_raw_parts(release.as_ptr().cast(), release.len()) };
                let mut release = core::ffi::CStr::from_bytes_until_nul(release).ok()?.to_str().ok()?.split('.');
                let major: u32 = release.next()?.parse().ok()?;
                let minor: u32 = release.next()?.parse().ok()?;
                Some((major, minor))
            }

            if let Some((kernel_major, kernel_minor)) = kernel_version(&utsname) {
                log::debug!("Detected Linux kernel: {kernel_major}.{kernel_minor}");
                if kernel_major < 6 || (kernel_major == 6 && kernel_minor < 8) {
                    return Err(Error::from(
                        format!("too old Linux kernel detected: {kernel_major}.{kernel_minor}; you need to update your Linux kernel to version 6.8 or newer")
                    ));
                }
            } else {
                log::warn!("Failed to parse the kernel version; this is a bug, please report it!");
            }
        }

        match clone(SANDBOX_FLAGS)? {
            Fork::Child => {
                let exit_code = if linux_raw::sys_sethostname("localhost").is_err() { 1 } else { 0 };
                let _ = linux_raw::sys_exit(exit_code);
                linux_raw::abort();
            }
            Fork::Host(mut child) => match child.check_status(false)? {
                ChildStatus::Exited(0) => {}
                ChildStatus::Exited(1) => {
                    if std::fs::read("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
                        .map(|blob| blob == b"1\n")
                        .unwrap_or(false)
                    {
                        return Err(Error::from("failed to create a sandboxed worker process; run 'sysctl -w kernel.apparmor_restrict_unprivileged_userns=0' to enable unprivileged user namespaces"));
                    }
                }
                status => {
                    return Err(Error::from(format!("unexpected sandbox child status: {status:?}")));
                }
            },
        }

        let zygote_memfd = prepare_zygote()?;
        Ok(GlobalState {
            shared_memory: ShmAllocator::new()?,
            uffd_available,
            zygote_memfd,
        })
    }
}

pub struct SandboxConfig {
    enable_logger: bool,
}

impl SandboxConfig {
    pub fn new() -> Self {
        SandboxConfig { enable_logger: false }
    }
}

impl super::SandboxConfig for SandboxConfig {
    fn enable_logger(&mut self, value: bool) {
        self.enable_logger = value;
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
struct CloneArgs {
    /// Flags.
    flags: u64,
    /// Where to store PID file descriptor. (int *)
    pidfd: *mut c_int,
    /// Where to store child TID in child's memory. (pid_t *)
    child_tid: u64,
    /// Where to store child TID in parent's memory. (pid_t *)
    parent_tid: u64,
    /// Signal to deliver to parent on child termination.
    exit_signal: u64,
    /// Pointer to lowest byte of stack.
    stack: u64,
    /// Size of the stack.
    stack_size: u64,
    /// Location of the new TLS.
    tls: u64,
}

/// Closes all file descriptors in a given range.
fn close_fd_range(first_fd: c_int, last_fd: c_int) -> Result<(), Error> {
    // Fast path for new kernels.
    if linux_raw::sys_close_range(first_fd, last_fd, 0).is_ok() {
        return Ok(());
    }

    // Slow path for old kernels.
    let dirfd = linux_raw::sys_open(
        cstr!("/proc/self/fd"),
        linux_raw::O_RDONLY | linux_raw::O_DIRECTORY | linux_raw::O_CLOEXEC,
    )?;

    for dirent in linux_raw::readdir(dirfd.borrow()) {
        let dirent = dirent?;
        let name = dirent.d_name();
        if !name.iter().all(|&byte| byte >= b'0' && byte <= b'9') {
            continue;
        }

        let name = core::str::from_utf8(name)
            .ok()
            .ok_or_else(|| Error::from_str("entry in '/proc/self/fd' is not valid utf-8"))?;

        let fd: c_int = name
            .parse()
            .ok()
            .ok_or_else(|| Error::from_str("entry in '/proc/self/fd' is not a number"))?;

        if fd != dirfd.raw() && fd >= first_fd && fd <= last_fd {
            Fd::from_raw_unchecked(fd).close()?;
        }
    }

    dirfd.close()?;
    Ok(())
}

struct Sigmask {
    sigset_original: linux_raw::kernel_sigset_t,
}

impl Sigmask {
    /// Temporarily blocks all signals from being delivered.
    fn block_all_signals() -> Result<Self, Error> {
        let sigset_all: linux_raw::kernel_sigset_t = !0;
        let mut sigset_original: linux_raw::kernel_sigset_t = 0;
        unsafe { linux_raw::sys_rt_sigprocmask(linux_raw::SIG_SETMASK, &sigset_all, Some(&mut sigset_original))? };

        Ok(Sigmask { sigset_original })
    }

    /// Unblocks signal delivery.
    fn unblock(mut self) -> Result<(), Error> {
        let result = self.unblock_inplace();
        core::mem::forget(self);
        result
    }

    /// Unblocks signal delivery.
    fn unblock_inplace(&mut self) -> Result<(), Error> {
        unsafe { linux_raw::sys_rt_sigprocmask(linux_raw::SIG_SETMASK, &self.sigset_original, None) }
    }
}

impl Drop for Sigmask {
    fn drop(&mut self) {
        let _ = self.unblock_inplace();
    }
}

#[derive(Debug)]
struct ChildProcess {
    pid: c_int,
    pidfd: Option<Fd>,
}

#[derive(Debug)]
enum ChildStatus {
    Running,
    NotRunning,
    Exited(c_int),
    ExitedDueToSignal(c_int),
    Trapped,
}

impl ChildStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    pub fn is_trapped(&self) -> bool {
        matches!(self, Self::Trapped)
    }
}

struct Signal(c_int);
impl core::fmt::Display for Signal {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        let name = match self.0 as u32 {
            linux_raw::SIGABRT => "SIGABRT",
            linux_raw::SIGBUS => "SIGBUS",
            linux_raw::SIGCHLD => "SIGCHLD",
            linux_raw::SIGCONT => "SIGCONT",
            linux_raw::SIGFPE => "SIGFPE",
            linux_raw::SIGHUP => "SIGHUP",
            linux_raw::SIGILL => "SIGILL",
            linux_raw::SIGINT => "SIGINT",
            linux_raw::SIGKILL => "SIGKILL",
            linux_raw::SIGPIPE => "SIGPIPE",
            linux_raw::SIGSEGV => "SIGSEGV",
            linux_raw::SIGSTOP => "SIGSTOP",
            linux_raw::SIGSYS => "SIGSYS",
            linux_raw::SIGTERM => "SIGTERM",
            linux_raw::SIGTRAP => "SIGTRAP",
            _ => return write!(fmt, "{}", self.0),
        };

        fmt.write_str(name)
    }
}

impl core::fmt::Display for ChildStatus {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            ChildStatus::Running => fmt.write_str("running"),
            ChildStatus::NotRunning => fmt.write_str("not running"),
            ChildStatus::Exited(code) => write!(fmt, "exited (status = {code})"),
            ChildStatus::ExitedDueToSignal(signum) => write!(fmt, "exited due to signal (signal = {})", Signal(*signum)),
            ChildStatus::Trapped => fmt.write_str("trapped"),
        }
    }
}

impl ChildProcess {
    fn waitid(&mut self, flags: u32) -> Result<linux_raw::siginfo_t, Error> {
        let mut siginfo: linux_raw::siginfo_t = unsafe { core::mem::zeroed() };
        let mut result;
        loop {
            result = if let Some(ref pidfd) = self.pidfd {
                linux_raw::sys_waitid(linux_raw::P_PIDFD, pidfd.raw(), &mut siginfo, flags, None)
            } else {
                linux_raw::sys_waitid(linux_raw::P_PID, self.pid, &mut siginfo, flags, None)
            };

            if let Err(error) = result {
                if error.errno() == linux_raw::EINTR {
                    // Should not happen since we should be blocking all signals while this is called, but just in case.
                    continue;
                }

                return Err(error);
            }

            return Ok(siginfo);
        }
    }

    fn extract_status(result: Result<linux_raw::siginfo_t, Error>) -> Result<ChildStatus, Error> {
        match result {
            Ok(ok) => unsafe {
                if ok.si_signo() == 0 && ok.si_pid() == 0 {
                    Ok(ChildStatus::Running)
                } else if ok.si_signo() as u32 == linux_raw::SIGCHLD && ok.si_code() as u32 == linux_raw::CLD_EXITED {
                    Ok(ChildStatus::Exited(ok.si_status()))
                } else if ok.si_signo() as u32 == linux_raw::SIGCHLD
                    && (ok.si_code() as u32 == linux_raw::CLD_KILLED || ok.si_code() as u32 == linux_raw::CLD_DUMPED)
                {
                    Ok(ChildStatus::ExitedDueToSignal(linux_raw::WTERMSIG(ok.si_status())))
                } else if ok.si_signo() as u32 == linux_raw::SIGCHLD && ok.si_code() as u32 == linux_raw::CLD_STOPPED {
                    Err(Error::from_last_os_error("waitid failed: unexpected CLD_STOPPED status"))
                } else if ok.si_signo() as u32 == linux_raw::SIGCHLD && ok.si_code() as u32 == linux_raw::CLD_TRAPPED {
                    Ok(ChildStatus::Trapped)
                } else if ok.si_signo() as u32 == linux_raw::SIGCHLD && ok.si_code() as u32 == linux_raw::CLD_CONTINUED {
                    Err(Error::from_last_os_error("waitid failed: unexpected CLD_CONTINUED status"))
                } else if ok.si_signo() != 0 {
                    Ok(ChildStatus::ExitedDueToSignal(ok.si_signo()))
                } else {
                    Err(Error::from_last_os_error("waitid failed: internal error: unexpected state"))
                }
            },
            Err(error) => {
                if error.errno() == linux_raw::ECHILD {
                    Ok(ChildStatus::NotRunning)
                } else {
                    Err(error)
                }
            }
        }
    }

    fn check_status(&mut self, non_blocking: bool) -> Result<ChildStatus, Error> {
        // The __WALL here is needed since we're not specifying an exit signal
        // when cloning the child process, so we'd get an ECHILD error without this flag.
        //
        // (And we're not using __WCLONE since that doesn't work for children which ran execve.)
        let mut flags = linux_raw::WEXITED | linux_raw::__WALL;
        if non_blocking {
            flags |= linux_raw::WNOHANG;
        }

        Self::extract_status(self.waitid(flags))
    }

    fn send_signal(&mut self, signal: c_uint) -> Result<(), Error> {
        unsafe {
            if let Some(ref pidfd) = self.pidfd {
                let errcode = syscall_readonly!(linux_raw::SYS_pidfd_send_signal, pidfd, signal, 0, 0);
                Error::from_syscall("pidfd_send_signal", errcode)
            } else {
                linux_raw::sys_kill(self.pid, signal)
            }
        }
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        #[cfg(polkavm_dev_debug_zygote)]
        let _ = self.send_signal(linux_raw::SIGINT);

        #[cfg(not(polkavm_dev_debug_zygote))]
        if self.send_signal(linux_raw::SIGKILL).is_ok() {
            // Reap the zombie process.
            let _ = self.check_status(false);
        }
    }
}

const ZYGOTE_BLOB_CONST: &[u8] = include_bytes!("./polkavm-zygote");
static ZYGOTE_BLOB: &[u8] = ZYGOTE_BLOB_CONST;

// Here we extract the necessary addresses directly from the zygote binary at compile time.
const ZYGOTE_TABLES: (AddressTable, ExtTable) = {
    const fn starts_with(haystack: &[u8], needle: &[u8]) -> bool {
        if haystack.len() < needle.len() {
            return false;
        }

        let mut index = 0;
        while index < needle.len() {
            if haystack[index] != needle[index] {
                return false;
            }
            index += 1;
        }

        true
    }

    const fn cast_slice<T>(slice: &[u8]) -> &T
    where
        T: Copy,
    {
        assert!(slice.len() >= core::mem::size_of::<T>());
        assert!(core::mem::align_of::<T>() == 1);

        // SAFETY: The size and alignment requirements of `T` were `assert`ed,
        //         and it's `Copy` so it's guaranteed not to drop, so this is always safe.
        unsafe { &*slice.as_ptr().cast::<T>() }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct U16([u8; 2]);

    impl U16 {
        const fn get(self) -> u16 {
            u16::from_ne_bytes(self.0)
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct U32([u8; 4]);

    impl U32 {
        const fn get(self) -> u32 {
            u32::from_ne_bytes(self.0)
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct U64([u8; 8]);

    impl U64 {
        const fn get(self) -> u64 {
            u64::from_ne_bytes(self.0)
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ElfIdent {
        magic: [u8; 4],
        class: u8,
        data: u8,
        version: u8,
        os_abi: u8,
        abi_version: u8,
        padding: [u8; 7],
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ElfHeader {
        e_ident: ElfIdent,
        e_type: U16,
        e_machine: U16,
        e_version: U32,
        e_entry: U64,
        e_phoff: U64,
        e_shoff: U64,
        e_flags: U32,
        e_ehsize: U16,
        e_phentsize: U16,
        e_phnum: U16,
        e_shentsize: U16,
        e_shnum: U16,
        e_shstrndx: U16,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ElfSectionHeader {
        sh_name: U32,
        sh_type: U32,
        sh_flags: U64,
        sh_addr: U64,
        sh_offset: U64,
        sh_size: U64,
        sh_link: U32,
        sh_info: U32,
        sh_addralign: U64,
        sh_entsize: U64,
    }

    impl ElfHeader {
        const fn section_header<'a>(&self, blob: &'a [u8], nth_section: u16) -> &'a ElfSectionHeader {
            let size = self.e_shentsize.get() as usize;
            assert!(size == core::mem::size_of::<ElfSectionHeader>());

            let offset = self.e_shoff.get() as usize + nth_section as usize * size;
            cast_slice(blob.split_at(offset).1)
        }
    }

    impl ElfSectionHeader {
        const fn data<'a>(&self, blob: &'a [u8]) -> &'a [u8] {
            blob.split_at(self.sh_offset.get() as usize)
                .1
                .split_at(self.sh_size.get() as usize)
                .0
        }
    }

    let header: &ElfHeader = cast_slice(ZYGOTE_BLOB_CONST);
    let shstr = header
        .section_header(ZYGOTE_BLOB_CONST, header.e_shstrndx.get())
        .data(ZYGOTE_BLOB_CONST);

    let mut address_table = None;
    let mut ext_table = None;

    let mut nth_section = 0;
    while nth_section < header.e_shnum.get() {
        let section_header = header.section_header(ZYGOTE_BLOB_CONST, nth_section);
        if starts_with(shstr.split_at(section_header.sh_name.get() as usize).1, b".address_table") {
            let data = section_header.data(ZYGOTE_BLOB_CONST);
            assert!(data.len() == core::mem::size_of::<AddressTablePacked>());
            address_table = Some(AddressTable::from_packed(cast_slice::<AddressTablePacked>(data)));
        } else if starts_with(shstr.split_at(section_header.sh_name.get() as usize).1, b".ext_table") {
            let data = section_header.data(ZYGOTE_BLOB_CONST);
            assert!(data.len() == core::mem::size_of::<ExtTablePacked>());
            ext_table = Some(ExtTable::from_packed(cast_slice::<ExtTablePacked>(data)));
        }
        nth_section += 1;
    }

    let Some(address_table) = address_table else {
        panic!("broken zygote binary")
    };
    let Some(ext_table) = ext_table else {
        panic!("broken zygote binary")
    };
    (address_table, ext_table)
};

fn create_empty_memfd(name: &core::ffi::CStr) -> Result<Fd, Error> {
    linux_raw::sys_memfd_create(name, linux_raw::MFD_CLOEXEC | linux_raw::MFD_ALLOW_SEALING)
}

fn prepare_sealed_memfd<const N: usize>(memfd: Fd, length: usize, data: [&[u8]; N]) -> Result<Fd, Error> {
    let native_page_size = get_native_page_size();
    if length % native_page_size != 0 {
        return Err(Error::from_str("memfd size doesn't end on a page boundary"));
    }

    linux_raw::sys_ftruncate(memfd.borrow(), length as linux_raw::c_ulong)?;

    let expected_bytes_written = data.iter().map(|slice| slice.len()).sum::<usize>();
    let bytes_written = linux_raw::writev(memfd.borrow(), data)?;
    if bytes_written != expected_bytes_written {
        return Err(Error::from_str("failed to prepare memfd: incomplete write"));
    }

    linux_raw::sys_fcntl(
        memfd.borrow(),
        linux_raw::F_ADD_SEALS,
        linux_raw::F_SEAL_SEAL | linux_raw::F_SEAL_SHRINK | linux_raw::F_SEAL_GROW | linux_raw::F_SEAL_WRITE,
    )?;

    Ok(memfd)
}

fn prepare_zygote() -> Result<Fd, Error> {
    #[cfg(debug_assertions)]
    if cfg!(polkavm_dev_debug_zygote) {
        let paths = [
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../polkavm-zygote/target/x86_64-unknown-linux-gnu/debug/polkavm-zygote"),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../polkavm-zygote/target/x86_64-unknown-linux-gnu/release/polkavm-zygote"),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/sandbox/polkavm-zygote"),
            std::path::PathBuf::from("./polkavm-zygote"),
        ];

        let Some(path) = paths
            .into_iter()
            .find(|path| path.exists() && std::fs::read(path).map(|data| data == ZYGOTE_BLOB).unwrap_or(false))
        else {
            panic!("no matching zygote binary found for debugging");
        };

        let path = std::ffi::CString::new(path.to_str().expect("invalid path to zygote")).expect("invalid path to zygote");
        return Ok(linux_raw::sys_open(&path, linux_raw::O_CLOEXEC | linux_raw::O_PATH).unwrap());
    }

    let native_page_size = get_native_page_size();

    #[allow(clippy::unwrap_used)]
    // The size of the zygote blob is always going to be much less than the size of usize, so this never fails.
    let length_aligned = align_to_next_page_usize(native_page_size, ZYGOTE_BLOB.len()).unwrap();
    prepare_sealed_memfd(create_empty_memfd(cstr!("polkavm_zygote"))?, length_aligned, [ZYGOTE_BLOB])
}

fn prepare_vmctx() -> Result<(Fd, Mmap), Error> {
    let native_page_size = get_native_page_size();

    #[allow(clippy::unwrap_used)] // The size of VmCtx is always going to be much less than the size of usize, so this never fails.
    let length_aligned = align_to_next_page_usize(native_page_size, core::mem::size_of::<VmCtx>()).unwrap();

    let memfd = create_empty_memfd(cstr!("polkavm_vmctx"))?;
    linux_raw::sys_ftruncate(memfd.borrow(), length_aligned as linux_raw::c_ulong)?;
    linux_raw::sys_fcntl(
        memfd.borrow(),
        linux_raw::F_ADD_SEALS,
        linux_raw::F_SEAL_SEAL | linux_raw::F_SEAL_SHRINK | linux_raw::F_SEAL_GROW,
    )?;

    let vmctx = unsafe {
        linux_raw::Mmap::map(
            core::ptr::null_mut(),
            length_aligned,
            linux_raw::PROT_READ | linux_raw::PROT_WRITE,
            linux_raw::MAP_SHARED,
            Some(memfd.borrow()),
            0,
        )?
    };

    unsafe {
        *vmctx.as_mut_ptr().cast::<VmCtx>() = VmCtx::new();
    }

    Ok((memfd, vmctx))
}

fn prepare_memory() -> Result<(Fd, Mmap), Error> {
    let memfd = create_empty_memfd(cstr!("polkavm_memory"))?;
    linux_raw::sys_ftruncate(memfd.borrow(), linux_raw::c_ulong::from(u32::MAX))?;
    linux_raw::sys_fcntl(
        memfd.borrow(),
        linux_raw::F_ADD_SEALS,
        linux_raw::F_SEAL_SEAL | linux_raw::F_SEAL_SHRINK | linux_raw::F_SEAL_GROW,
    )?;

    let vmctx = unsafe {
        linux_raw::Mmap::map(
            core::ptr::null_mut(),
            u32::MAX as usize,
            linux_raw::PROT_READ | linux_raw::PROT_WRITE,
            linux_raw::MAP_SHARED,
            Some(memfd.borrow()),
            0,
        )?
    };

    Ok((memfd, vmctx))
}

struct ChildFds {
    zygote: Fd,
    socket: Fd,
    vmctx: Fd,
    shm: Fd,
    mem: Fd,
    lifetime_pipe: Fd,
    logging_pipe: Option<Fd>,
}

unsafe fn child_main(uid_map: &str, gid_map: &str, fds: ChildFds) -> Result<(), Error> {
    // Change the name of the process.
    linux_raw::sys_prctl_set_name(b"polkavm-sandbox\0")?;

    if !cfg!(polkavm_dev_debug_zygote) {
        // Overwrite the hostname and domainname.
        linux_raw::sys_sethostname("localhost")?;
        linux_raw::sys_setdomainname("localhost")?;

        // Disable the 'setgroups' syscall. Probably unnecessary since we'll do it though seccomp anyway, but just in case.
        // (See CVE-2014-8989 for more details.)
        let proc_self = linux_raw::sys_open(cstr!("/proc/self"), linux_raw::O_CLOEXEC | linux_raw::O_PATH)?;
        let fd = linux_raw::sys_openat(proc_self.borrow(), cstr!("setgroups"), linux_raw::O_CLOEXEC | linux_raw::O_WRONLY)?;
        linux_raw::sys_write(fd.borrow(), b"deny")?;
        fd.close()?;

        // Set up UID and GID maps. This can only be done once, so if we do it here we'll block the possibility of doing it later.
        let fd = linux_raw::sys_openat(proc_self.borrow(), cstr!("gid_map"), linux_raw::O_CLOEXEC | linux_raw::O_RDWR)?;
        linux_raw::sys_write(fd.borrow(), gid_map.as_bytes())?;
        fd.close()?;

        let fd = linux_raw::sys_openat(proc_self.borrow(), cstr!("uid_map"), linux_raw::O_CLOEXEC | linux_raw::O_RDWR)?;
        linux_raw::sys_write(fd.borrow(), uid_map.as_bytes())?;
        fd.close()?;
        proc_self.close()?;
    }

    fn move_fd_after(fd: linux_raw::Fd, min: i32) -> Result<Fd, Error> {
        let out_fd = linux_raw::sys_fcntl_dupfd(fd.borrow(), min)?;
        fd.close()?;

        Ok(out_fd)
    }

    fn move_fd(fd: linux_raw::Fd, new_fd: i32, flags: u32) -> Result<Fd, Error> {
        linux_raw::sys_dup3(fd.borrow().raw(), new_fd, flags)?;
        fd.close()?;

        Ok(linux_raw::Fd::from_raw_unchecked(new_fd))
    }

    fn move_fd_and_leak(fd: linux_raw::Fd, new_fd: i32) -> Result<(), Error> {
        move_fd(fd, new_fd, 0)?.leak();
        Ok(())
    }

    fn copy_fd_and_leak(fd: linux_raw::FdRef, new_fd: i32) -> Result<(), Error> {
        linux_raw::sys_dup3(fd.raw(), new_fd, 0)
    }

    // Create a dummy FD.
    //
    // WARNING: This CANNOT be a pipe!
    //
    // If the logger is disabled then the child process will unconditionally write
    // to this FD. Unfortunately writing to a closed pipe generates a SIGPIPE signal,
    // and normally that'd be fine since we ignore SIGPIPE in the child, however when
    // running under ptrace this will also stop the process and return a "trapped"
    // status to the host, even if the SIGPIPE would normally be ignored!
    //
    // So we just make a unix socket here instead to side step this problem.
    let (_, fd_dummy) = linux_raw::sys_socketpair(linux_raw::AF_UNIX, linux_raw::SOCK_SEQPACKET | linux_raw::SOCK_CLOEXEC, 0)?;

    pub use polkavm_common::zygote;

    const FD_ZYGOTE: i32 = zygote::LAST_USED_FD + 1;
    const LAST_USED_FD: i32 = FD_ZYGOTE;
    const NEXT_FREE_FD: i32 = FD_ZYGOTE + 1;

    // Make sure no FD we need uses any FD number in range 0..LAST_USED_FD.
    let fd_zygote = move_fd_after(fds.zygote, NEXT_FREE_FD)?;
    let fd_socket = move_fd_after(fds.socket, NEXT_FREE_FD)?;
    let fd_vmctx = move_fd_after(fds.vmctx, NEXT_FREE_FD)?;
    let fd_shm = move_fd_after(fds.shm, NEXT_FREE_FD)?;
    let fd_mem = move_fd_after(fds.mem, NEXT_FREE_FD)?;
    let fd_lifetime_pipe = move_fd_after(fds.lifetime_pipe, NEXT_FREE_FD)?;
    let fd_dummy = move_fd_after(fd_dummy, NEXT_FREE_FD)?;
    let fd_logging_pipe = if let Some(fd_logging_pipe) = fds.logging_pipe {
        Some(move_fd_after(fd_logging_pipe, NEXT_FREE_FD)?)
    } else {
        None
    };

    close_fd_range(0, LAST_USED_FD)?;

    // Move all of the FDs to their hardcoded numbers.
    move_fd_and_leak(fd_socket, zygote::FD_SOCKET)?;
    move_fd_and_leak(fd_vmctx, zygote::FD_VMCTX)?;
    move_fd_and_leak(fd_shm, zygote::FD_SHM)?;
    move_fd_and_leak(fd_mem, zygote::FD_MEM)?;
    move_fd_and_leak(fd_lifetime_pipe, zygote::FD_LIFETIME_PIPE)?;
    if let Some(fd_logging_pipe) = fd_logging_pipe {
        move_fd_and_leak(fd_dummy, zygote::FD_DUMMY_STDIN)?;
        copy_fd_and_leak(fd_logging_pipe.borrow(), zygote::FD_LOGGER_STDOUT)?;
        move_fd_and_leak(fd_logging_pipe, zygote::FD_LOGGER_STDERR)?;
    } else {
        copy_fd_and_leak(fd_dummy.borrow(), zygote::FD_DUMMY_STDIN)?;
        copy_fd_and_leak(fd_dummy.borrow(), zygote::FD_LOGGER_STDOUT)?;
        move_fd_and_leak(fd_dummy, zygote::FD_LOGGER_STDERR)?;
    }

    let fd_zygote = move_fd(fd_zygote, FD_ZYGOTE, linux_raw::O_CLOEXEC)?;
    close_fd_range(NEXT_FREE_FD, c_int::MAX)?;

    if !cfg!(polkavm_dev_debug_zygote) {
        // Hide the host filesystem.
        let mount_flags = linux_raw::MS_REC | linux_raw::MS_NODEV | linux_raw::MS_NOEXEC | linux_raw::MS_NOSUID | linux_raw::MS_RDONLY;
        linux_raw::sys_mount(cstr!("none"), cstr!("/tmp"), cstr!("tmpfs"), mount_flags, Some(cstr!("size=0")))?;
        linux_raw::sys_chdir(cstr!("/tmp"))?;
    }

    // Clear all of our ambient capabilities.
    linux_raw::sys_prctl_cap_ambient_clear_all()?;

    // Flag ourselves that we won't ever want to acquire any new privileges.
    linux_raw::sys_prctl_set_no_new_privs()?;

    // Set resource limits.
    let max_memory = 8 * 1024 * 1024 * 1024;
    linux_raw::sys_setrlimit(
        linux_raw::RLIMIT_DATA,
        &linux_raw::rlimit {
            rlim_cur: max_memory,
            rlim_max: max_memory,
        },
    )?;
    linux_raw::sys_setrlimit(
        linux_raw::RLIMIT_STACK,
        &linux_raw::rlimit {
            rlim_cur: 16 * 1024,
            rlim_max: 16 * 1024,
        },
    )?;

    linux_raw::sys_setrlimit(linux_raw::RLIMIT_NPROC, &linux_raw::rlimit { rlim_cur: 1, rlim_max: 1 })?;
    linux_raw::sys_setrlimit(linux_raw::RLIMIT_FSIZE, &linux_raw::rlimit { rlim_cur: 0, rlim_max: 0 })?;
    linux_raw::sys_setrlimit(linux_raw::RLIMIT_LOCKS, &linux_raw::rlimit { rlim_cur: 0, rlim_max: 0 })?;
    linux_raw::sys_setrlimit(linux_raw::RLIMIT_MEMLOCK, &linux_raw::rlimit { rlim_cur: 0, rlim_max: 0 })?;
    linux_raw::sys_setrlimit(linux_raw::RLIMIT_MSGQUEUE, &linux_raw::rlimit { rlim_cur: 0, rlim_max: 0 })?;

    if cfg!(polkavm_dev_debug_zygote) {
        let pid = linux_raw::sys_getpid()?;
        linux_raw::sys_kill(pid, linux_raw::SIGSTOP)?;
    }

    let child_argv: [*const u8; 2] = [b"polkavm-zygote\0".as_ptr(), core::ptr::null()];
    let child_envp: [*const u8; 1] = [core::ptr::null()];
    linux_raw::sys_execveat(
        Some(fd_zygote.borrow()),
        cstr!(""),
        &child_argv,
        &child_envp,
        linux_raw::AT_EMPTY_PATH,
    )?;

    // This should never happen, but since the never type is still unstable let's return normally.
    Ok(())
}

#[derive(Clone)]
pub struct SandboxProgram(Arc<SandboxProgramInner>);

enum InitializeWith {
    None,
    Shm(ShmAllocation),
    Mem(u32),
}

struct ProgramMap {
    address: u64,
    length: u64,
    is_writable: bool,
    initialize_with: InitializeWith,
}

struct SandboxProgramInner {
    memory_map: Vec<ProgramMap>,
    shm_code: ShmAllocation,
    shm_jump_table: ShmAllocation,
    code_length: usize,
    sysenter_address: u64,
    sysreturn_address: u64,
}

impl super::SandboxProgram for SandboxProgram {
    fn machine_code(&self) -> &[u8] {
        &(unsafe { self.0.shm_code.as_slice() })[..self.0.code_length]
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Map<'a> {
    pub start: u64,
    pub end: u64,
    pub is_readable: bool,
    pub is_writable: bool,
    pub is_executable: bool,
    pub is_shared: bool,
    pub file_offset: u64,
    pub major: u64,
    pub minor: u64,
    pub inode: u64,
    pub name: &'a [u8],
}

fn parse_u64_radix(input: &[u8], radix: u32) -> Option<u64> {
    u64::from_str_radix(core::str::from_utf8(input).ok()?, radix).ok()
}

fn get_until<'a>(p: &mut &'a [u8], delimiter: u8) -> &'a [u8] {
    let mut found = None;
    for (index, ch) in p.iter().enumerate() {
        if *ch == delimiter {
            found = Some(index);
            break;
        }
    }

    if let Some(index) = found {
        let (before, after) = p.split_at(index);
        *p = &after[1..];
        before
    } else {
        let before = *p;
        *p = b"";
        before
    }
}

fn get_char(p: &mut &[u8]) -> Option<u8> {
    let ch = p.first()?;
    *p = &p[1..];
    Some(*ch)
}

fn skip_whitespace(p: &mut &[u8]) {
    while let Some(ch) = p.first() {
        if *ch == b' ' {
            *p = &p[1..];
        } else {
            break;
        }
    }
}

impl<'a> Map<'a> {
    fn parse(mut line: &'a [u8]) -> Option<Self> {
        let start = parse_u64_radix(get_until(&mut line, b'-'), 16)?;
        let end = parse_u64_radix(get_until(&mut line, b' '), 16)?;
        let is_readable = get_char(&mut line)? == b'r';
        let is_writable = get_char(&mut line)? == b'w';
        let is_executable = get_char(&mut line)? == b'x';
        let is_shared = get_char(&mut line)? == b's';
        get_char(&mut line);

        let file_offset = parse_u64_radix(get_until(&mut line, b' '), 16)?;
        let major = parse_u64_radix(get_until(&mut line, b':'), 16)?;
        let minor = parse_u64_radix(get_until(&mut line, b' '), 16)?;
        let inode = parse_u64_radix(get_until(&mut line, b' '), 10)?;
        skip_whitespace(&mut line);
        let name = line;

        Some(Map {
            start,
            end,
            is_readable,
            is_writable,
            is_executable,
            is_shared,
            file_offset,
            major,
            minor,
            inode,
            name,
        })
    }
}

fn get_message(vmctx: &VmCtx) -> Option<String> {
    let message = unsafe {
        let message_length = *vmctx.message_length.get() as usize;
        let message = &*vmctx.message_buffer.get();
        &message[..core::cmp::min(message_length, message.len())]
    };

    if message.is_empty() {
        return None;
    }

    // The message is in shared memory, so clone it first to make sure
    // it doesn't change under us and violate string's invariants.
    let message = message.to_vec();
    match String::from_utf8(message) {
        Ok(message) => Some(message),
        Err(error) => {
            let message = error.into_bytes();
            Some(String::from_utf8_lossy(&message).into_owned())
        }
    }
}

unsafe fn set_message(vmctx: &VmCtx, message: core::fmt::Arguments) {
    struct Adapter<'a>(std::io::Cursor<&'a mut [u8]>);
    impl<'a> core::fmt::Write for Adapter<'a> {
        fn write_str(&mut self, string: &str) -> Result<(), core::fmt::Error> {
            use std::io::Write;
            self.0.write_all(string.as_bytes()).map_err(|_| core::fmt::Error)
        }
    }

    let buffer: &mut [u8] = &mut *vmctx.message_buffer.get();
    let mut cursor = Adapter(std::io::Cursor::new(buffer));
    let _ = core::fmt::write(&mut cursor, message);
    let length = cursor.0.position() as usize;

    *vmctx.message_length.get() = length as u32;
}

struct UffdBuffer(Arc<UnsafeCell<linux_raw::uffd_msg>>);

unsafe impl Send for UffdBuffer {}
unsafe impl Sync for UffdBuffer {}

struct SiginfoBuffer(Box<UnsafeCell<linux_raw::siginfo_t>>);

unsafe impl Sync for SiginfoBuffer {}
unsafe impl Send for SiginfoBuffer {}

impl Default for SiginfoBuffer {
    fn default() -> Self {
        Self(Box::new(UnsafeCell::new(unsafe { core::mem::zeroed() })))
    }
}

struct Pagefault {
    address: u64,
    registers_modified: bool,
}

#[derive(Copy, Clone)]
enum SandboxState {
    Idle,
    Pagefault,
    Hostcall,
}

pub struct Sandbox {
    _lifetime_pipe: Fd,
    vmctx_mmap: Mmap,
    memory_mmap: Mmap,
    iouring: Option<linux_raw::IoUring>,
    iouring_futex_wait_queued: bool,
    iouring_uffd_read_queued: bool,
    iouring_waitid_queued: bool,
    iouring_siginfo: SiginfoBuffer,
    userfaultfd: Fd,
    uffd_msg: UffdBuffer,
    child: ChildProcess,

    count_wait_loop_start: u64,
    count_futex_wait: u64,

    module: Option<Module>,
    gas_metering: Option<GasMeteringKind>,

    state: SandboxState,
    is_program_counter_valid: bool,
    next_program_counter: Option<ProgramCounter>,
    next_program_counter_changed: bool,
    pending_pagefault: Option<Pagefault>,
    page_set: PageSet,
    dynamic_paging_enabled: bool,
    idle_regs: linux_raw::user_regs_struct,
    aux_data_address: u32,
    aux_data_length: u32,
    is_borked: bool,
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if let Some(mut iouring) = self.iouring.take() {
            if let Err(error) = iouring.cancel_all_sync() {
                log::error!("Failed to cancel io_uring requests: {error}");
            }
        }

        let vmctx = self.vmctx();
        let child_futex_wait = unsafe { *vmctx.counters.syscall_futex_wait.get() };
        let child_loop_start = unsafe { *vmctx.counters.syscall_wait_loop_start.get() };
        log::debug!(
            "Host futex wait count: {}/{} ({:.02}%)",
            self.count_futex_wait,
            self.count_wait_loop_start,
            self.count_futex_wait as f64 / self.count_wait_loop_start as f64 * 100.0
        );
        log::debug!(
            "Child futex wait count: {}/{} ({:.02}%)",
            child_futex_wait,
            child_loop_start,
            child_futex_wait as f64 / child_loop_start as f64 * 100.0
        );
    }
}

impl super::SandboxAddressSpace for () {
    fn native_code_origin(&self) -> u64 {
        VM_ADDR_NATIVE_CODE
    }
}

#[repr(transparent)]
pub struct JumpTableAllocation(ShmAllocation);

impl AsRef<[usize]> for JumpTableAllocation {
    fn as_ref(&self) -> &[usize] {
        #[allow(clippy::cast_ptr_alignment)] // Allocation is guaranteed to be page aligned.
        unsafe {
            core::slice::from_raw_parts(self.0.as_ptr().cast::<usize>(), self.0.len() / core::mem::size_of::<usize>())
        }
    }
}

impl AsMut<[usize]> for JumpTableAllocation {
    fn as_mut(&mut self) -> &mut [usize] {
        #[allow(clippy::cast_ptr_alignment)] // Allocation is guaranteed to be page aligned.
        unsafe {
            core::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast::<usize>(), self.0.len() / core::mem::size_of::<usize>())
        }
    }
}

impl super::Sandbox for Sandbox {
    const KIND: SandboxKind = SandboxKind::Linux;

    type Config = SandboxConfig;
    type Error = Error;
    type Program = SandboxProgram;
    type AddressSpace = ();
    type GlobalState = GlobalState;
    type JumpTable = JumpTableAllocation;

    fn downcast_module(module: &Module) -> &CompiledModule<Self> {
        let CompiledModuleKind::Linux(ref module) = module.compiled_module() else {
            unreachable!()
        };
        module
    }

    fn downcast_global_state(global: &crate::sandbox::GlobalStateKind) -> &Self::GlobalState {
        #[allow(irrefutable_let_patterns)]
        let crate::sandbox::GlobalStateKind::Linux(ref global) = global
        else {
            unreachable!()
        };
        global
    }

    fn downcast_worker_cache(cache: &WorkerCacheKind) -> &WorkerCache<Self> {
        #[allow(irrefutable_let_patterns)]
        let crate::sandbox::WorkerCacheKind::Linux(ref cache) = cache
        else {
            unreachable!()
        };
        cache
    }

    fn allocate_jump_table(global: &Self::GlobalState, count: usize) -> Result<Self::JumpTable, Self::Error> {
        let Some(alloc) = global.shared_memory.alloc(count * core::mem::size_of::<usize>()) else {
            return Err(Error::from_str("failed to allocate the jump table: out of shared memory"));
        };

        Ok(JumpTableAllocation(alloc))
    }

    fn reserve_address_space() -> Result<Self::AddressSpace, Self::Error> {
        Ok(())
    }

    fn prepare_program(global: &Self::GlobalState, init: SandboxInit<Self>, (): Self::AddressSpace) -> Result<Self::Program, Self::Error> {
        let cfg = init.guest_init.memory_map()?;

        let Some(shm_ro_data) = global.shared_memory.alloc(init.guest_init.ro_data.len()) else {
            return Err(Error::from_str(
                "failed to prepare the program for the sandbox: out of shared memory",
            ));
        };

        let Some(shm_rw_data) = global.shared_memory.alloc(init.guest_init.rw_data.len()) else {
            return Err(Error::from_str(
                "failed to prepare the program for the sandbox: out of shared memory",
            ));
        };

        let Some(shm_code) = global.shared_memory.alloc(init.code.len()) else {
            return Err(Error::from_str(
                "failed to prepare the program for the sandbox: out of shared memory",
            ));
        };

        unsafe {
            let shm_ro_data = shm_ro_data.as_slice_mut();
            shm_ro_data[..init.guest_init.ro_data.len()].copy_from_slice(init.guest_init.ro_data);
            shm_ro_data[init.guest_init.ro_data.len()..].fill(0);
        }

        unsafe {
            let shm_rw_data = shm_rw_data.as_slice_mut();
            shm_rw_data[..init.guest_init.rw_data.len()].copy_from_slice(init.guest_init.rw_data);
            shm_rw_data[init.guest_init.rw_data.len()..].fill(0);
        }

        let code_length = init.code.len();
        unsafe {
            let shm_code = shm_code.as_slice_mut();
            shm_code[..code_length].copy_from_slice(init.code);
        }

        let mut memory_map = Vec::new();
        if cfg.ro_data_size() > 0 {
            let physical_size = shm_ro_data.len() as u64;
            let virtual_size = u64::from(cfg.ro_data_size());
            if physical_size > 0 {
                memory_map.push(ProgramMap {
                    address: u64::from(cfg.ro_data_address()),
                    length: physical_size,
                    is_writable: false,
                    initialize_with: InitializeWith::Shm(shm_ro_data),
                });
            }

            let padding = virtual_size - physical_size;
            if padding > 0 {
                memory_map.push(ProgramMap {
                    address: u64::from(cfg.ro_data_address()) + physical_size,
                    length: padding,
                    is_writable: false,
                    initialize_with: InitializeWith::None,
                });
            }
        }

        if cfg.rw_data_size() > 0 {
            let physical_size = shm_rw_data.len() as u64;
            let virtual_size = u64::from(cfg.rw_data_size());
            if physical_size > 0 {
                memory_map.push(ProgramMap {
                    address: u64::from(cfg.rw_data_address()),
                    length: physical_size,
                    is_writable: true,
                    initialize_with: InitializeWith::Shm(shm_rw_data),
                });
            }

            let padding = virtual_size - physical_size;
            if padding > 0 {
                memory_map.push(ProgramMap {
                    address: u64::from(cfg.rw_data_address()) + physical_size,
                    length: padding,
                    is_writable: true,
                    initialize_with: InitializeWith::None,
                });
            }
        }

        if cfg.stack_size() > 0 {
            memory_map.push(ProgramMap {
                address: u64::from(cfg.stack_address_low()),
                length: u64::from(cfg.stack_size()),
                is_writable: true,
                initialize_with: InitializeWith::None,
            });
        }

        if cfg.aux_data_size() > 0 {
            memory_map.push(ProgramMap {
                address: u64::from(cfg.aux_data_address()),
                length: u64::from(cfg.aux_data_size()),
                is_writable: false,
                initialize_with: InitializeWith::Mem(cfg.aux_data_address()),
            });
        }

        Ok(SandboxProgram(Arc::new(SandboxProgramInner {
            memory_map,
            shm_code,
            shm_jump_table: init.jump_table.0,
            code_length,
            sysenter_address: init.sysenter_address,
            sysreturn_address: init.sysreturn_address,
        })))
    }

    fn spawn(global: &Self::GlobalState, config: &SandboxConfig) -> Result<Self, Error> {
        let sigset = Sigmask::block_all_signals()?;
        let (socket, child_socket) = linux_raw::sys_socketpair(linux_raw::AF_UNIX, linux_raw::SOCK_SEQPACKET | linux_raw::SOCK_CLOEXEC, 0)?;
        let (lifetime_pipe_host, lifetime_pipe_child) = linux_raw::sys_pipe2(linux_raw::O_CLOEXEC)?;
        let (logger_rx, logger_tx) = if config.enable_logger {
            let (rx, tx) = linux_raw::sys_pipe2(linux_raw::O_CLOEXEC)?;
            (Some(rx), Some(tx))
        } else {
            (None, None)
        };
        // TODO: If not using userfaultfd then don't mmap all of this immediately.
        let (memory_memfd, memory_mmap) = prepare_memory()?;

        let (vmctx_memfd, vmctx_mmap) = prepare_vmctx()?;
        let vmctx = unsafe { &*vmctx_mmap.as_ptr().cast::<VmCtx>() };
        vmctx.init.logging_enabled.store(config.enable_logger, Ordering::Relaxed);
        vmctx.init.uffd_available.store(global.uffd_available, Ordering::Relaxed);
        vmctx.init.sandbox_disabled.store(cfg!(polkavm_dev_debug_zygote), Ordering::Relaxed);

        let sandbox_flags = if !cfg!(polkavm_dev_debug_zygote) { SANDBOX_FLAGS } else { 0 };

        let uid = linux_raw::sys_getuid()?;
        let gid = linux_raw::sys_getgid()?;

        let uid_map = format!("0 {} 1\n", uid);
        let gid_map = format!("0 {} 1\n", gid);

        let mut child = match clone(sandbox_flags)? {
            Fork::Child => {
                // We're in the child.
                //
                // Calling into libc from here risks a deadlock as other threads might have
                // been holding onto internal libc locks while we were cloning ourselves,
                // so from now on we can't use anything from libc anymore.
                core::mem::forget(sigset);

                unsafe {
                    match child_main(
                        &uid_map,
                        &gid_map,
                        ChildFds {
                            zygote: linux_raw::Fd::from_raw_unchecked(global.zygote_memfd.raw()),
                            socket: child_socket,
                            vmctx: vmctx_memfd,
                            shm: linux_raw::Fd::from_raw_unchecked(global.shared_memory.fd().raw()),
                            mem: memory_memfd,
                            lifetime_pipe: lifetime_pipe_child,
                            logging_pipe: logger_tx,
                        },
                    ) {
                        Ok(()) => {
                            // This is impossible.
                            abort();
                        }
                        Err(error) => {
                            let vmctx = &*vmctx_mmap.as_ptr().cast::<VmCtx>();
                            set_message(vmctx, format_args!("fatal error while spawning child: {error}"));

                            abort();
                        }
                    }
                }
            }
            Fork::Host(child) => child,
        };

        let child_pid = child.pid;

        child_socket.close()?;
        vmctx_memfd.close()?;
        memory_memfd.close()?;
        lifetime_pipe_child.close()?;
        if let Some(logger_tx) = logger_tx {
            logger_tx.close()?;
        }

        if let Some(logger_rx) = logger_rx {
            // Hook up the child process' STDERR to our logger.
            std::thread::Builder::new()
                .name("polkavm-logger".into())
                .spawn(move || {
                    let mut tmp = [0; 4096];
                    let mut buffer = Vec::new();
                    loop {
                        if buffer.len() > 8192 {
                            // Make sure the child can't exhaust our memory by spamming logs.
                            buffer.clear();
                        }

                        match linux_raw::sys_read(logger_rx.borrow(), &mut tmp) {
                            Err(error) if error.errno() == linux_raw::EINTR => continue,
                            Err(error) => {
                                log::warn!("Failed to read from logger: {}", error);
                                break;
                            }
                            Ok(0) => break,
                            Ok(count) => {
                                let mut tmp = &tmp[..count];
                                while !tmp.is_empty() {
                                    if let Some(index) = tmp.iter().position(|&byte| byte == b'\n') {
                                        buffer.extend_from_slice(&tmp[..index]);
                                        tmp = &tmp[index + 1..];

                                        log::trace!(target: "polkavm::zygote", "Child #{}: {}", child_pid, String::from_utf8_lossy(&buffer));
                                        buffer.clear();
                                    } else {
                                        buffer.extend_from_slice(tmp);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                })
                .map_err(|error| Error::from_os_error("failed to spawn logger thread", error))?;
        }

        // We're in the parent. Restore the signal mask.
        sigset.unblock()?;

        fn wait_for_futex(vmctx: &VmCtx, child: &mut ChildProcess, current_state: u32, target_state: u32) -> Result<(), Error> {
            let instant = Instant::now();
            loop {
                let state = vmctx.futex.load(Ordering::Relaxed);
                if state == target_state {
                    return Ok(());
                }

                if state != current_state {
                    return Err(Error::from_str("failed to initialize sandbox process: unexpected futex state"));
                }

                let status = child.check_status(true)?;
                if !status.is_running() {
                    let message = get_message(vmctx);
                    if let Some(message) = message {
                        let error = Error::from(format!("failed to initialize sandbox process: {status}: {message}"));
                        return Err(error);
                    } else {
                        return Err(Error::from(format!(
                            "failed to initialize sandbox process: child process unexpectedly quit: {status}",
                        )));
                    }
                }

                if !cfg!(polkavm_dev_debug_zygote) && instant.elapsed() > Duration::from_secs(10) {
                    // This should never happen, but just in case.
                    return Err(Error::from_str("failed to initialize sandbox process: initialization timeout"));
                }

                match linux_raw::sys_futex_wait(&vmctx.futex, state, Some(Duration::from_millis(100))) {
                    Ok(()) => continue,
                    Err(error)
                        if error.errno() == linux_raw::EAGAIN
                            || error.errno() == linux_raw::EINTR
                            || error.errno() == linux_raw::ETIMEDOUT =>
                    {
                        continue
                    }
                    Err(error) => return Err(error),
                }
            }
        }

        #[cfg(debug_assertions)]
        if cfg!(polkavm_dev_debug_zygote) {
            use core::fmt::Write;
            std::thread::sleep(Duration::from_millis(200));

            let mut command = String::new();
            // Make sure gdb can actually attach to the worker process.
            if std::fs::read_to_string("/proc/sys/kernel/yama/ptrace_scope")
                .map(|value| value.trim() == "1")
                .unwrap_or(false)
            {
                command.push_str("echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope ;");
            }

            command.push_str(concat!(
                "gdb",
                " -ex 'set pagination off'",
                " -ex 'layout split'",
                " -ex 'set print asm-demangle on'",
                " -ex 'set debuginfod enabled off'",
                " -ex 'tcatch exec'",
                " -ex 'handle SIGSTOP nostop'",
            ));

            let _ = write!(&mut command, " -ex 'attach {}' -ex 'continue'", child.pid);

            let mut cmd = if std::env::var_os("DISPLAY").is_some() {
                // Running X11; open gdb in a terminal.
                let mut cmd = std::process::Command::new("urxvt");
                cmd.args(["-fg", "rgb:ffff/ffff/ffff"])
                    .args(["-bg", "rgba:0000/0000/0000/7777"])
                    .arg("-e")
                    .arg("sh")
                    .arg("-c")
                    .arg(&command);
                cmd
            } else {
                // Not running under X11; just run it as-is.
                let mut cmd = std::process::Command::new("sh");
                cmd.arg("-c").arg(&command);
                cmd
            };

            let mut gdb = match cmd.spawn() {
                Ok(child) => child,
                Err(error) => {
                    panic!("failed to launch: '{cmd:?}': {error}");
                }
            };

            let pid = child.pid;
            std::thread::spawn(move || {
                let _ = gdb.wait();
                let _ = linux_raw::sys_kill(pid, linux_raw::SIGKILL);
            });
        }

        // Wait until the child process receives the vmctx memfd.
        wait_for_futex(vmctx, &mut child, VMCTX_FUTEX_BUSY, VMCTX_FUTEX_IDLE)?;

        // Grab the child process' maps and see what we can unmap.
        //
        // The child process can't do it itself as it's too sandboxed.
        let maps = std::fs::read(format!("/proc/{}/maps", child_pid))
            .map_err(|error| Error::from_errno("failed to read child's maps", error.raw_os_error().unwrap_or(0)))?;

        for line in maps.split(|&byte| byte == b'\n') {
            if line.is_empty() {
                continue;
            }

            let map = Map::parse(line).ok_or_else(|| Error::from_str("failed to parse the maps of the child process"))?;
            match map.name {
                b"[stack]" => {
                    vmctx.init.stack_address.store(map.start, Ordering::Relaxed);
                    vmctx.init.stack_length.store(map.end - map.start, Ordering::Relaxed);
                }
                b"[vdso]" => {
                    vmctx.init.vdso_address.store(map.start, Ordering::Relaxed);
                    vmctx.init.vdso_length.store(map.end - map.start, Ordering::Relaxed);
                }
                b"[vvar]" => {
                    vmctx.init.vvar_address.store(map.start, Ordering::Relaxed);
                    vmctx.init.vvar_length.store(map.end - map.start, Ordering::Relaxed);
                }
                b"[vsyscall]" => {
                    if map.is_readable {
                        return Err(Error::from_str("failed to initialize sandbox process: vsyscall region is readable"));
                    }
                }
                _ => {}
            }
        }

        // Wake the child so that it finishes initialization.
        vmctx.futex.store(VMCTX_FUTEX_BUSY, Ordering::Release);
        linux_raw::sys_futex_wake_one(&vmctx.futex)?;

        let (iouring, userfaultfd) = if global.uffd_available {
            let iouring = linux_raw::IoUring::new(3)?;

            let userfaultfd = linux_raw::recvfd(socket.borrow()).map_err(|error| {
                let mut error = format!("failed to fetch the userfaultfd from the child process: {error}");
                if let Some(message) = get_message(vmctx) {
                    use core::fmt::Write;
                    write!(&mut error, " (root cause: {message})").unwrap();
                }
                Error::from(error)
            })?;

            let mut api: linux_raw::uffdio_api = linux_raw::uffdio_api {
                api: linux_raw::UFFD_API,
                features: UFFD_REQUIRED_FEATURES,
                ..Default::default()
            };

            linux_raw::sys_uffdio_api(userfaultfd.borrow(), &mut api)
                .map_err(|error| Error::from(format!("failed to initialize the userfaultfd API: {error}")))?;

            linux_raw::sys_ptrace_seize(child.pid)?;

            (Some(iouring), userfaultfd)
        } else {
            (None, Fd::from_raw_unchecked(-1))
        };

        // Close the socket; we don't need it anymore.
        socket.close()?;

        // Wait for the child to finish initialization.
        wait_for_futex(vmctx, &mut child, VMCTX_FUTEX_BUSY, VMCTX_FUTEX_IDLE)?;

        let mut idle_regs = linux_raw::user_regs_struct::default();
        if global.uffd_available {
            // We need to be able to return to idle from a pending segfault,
            // so let's grab the registers which will allow us to do that.

            // First grab all of the general-purpose registers.
            linux_raw::sys_ptrace_interrupt(child.pid)?;
            let status = child.check_status(false)?;
            if !status.is_trapped() {
                log::error!("Child #{}: expected child to trap, found: {status}", child.pid);
                return Err(Error::from_str("internal error: unexpected child status"));
            }

            idle_regs = linux_raw::sys_ptrace_getregs(child.pid)?;
            linux_raw::sys_ptrace_continue(child.pid, None)?;

            // Then grab the worker's idle longjmp registers.
            vmctx.jump_into.store(ZYGOTE_TABLES.1.ext_fetch_idle_regs, Ordering::Relaxed);
            vmctx.futex.store(VMCTX_FUTEX_BUSY, Ordering::Release);
            linux_raw::sys_futex_wake_one(&vmctx.futex)?;
            wait_for_futex(vmctx, &mut child, VMCTX_FUTEX_BUSY, VMCTX_FUTEX_IDLE)?;

            idle_regs.rax = 1;
            idle_regs.rip = vmctx.init.idle_regs.rip.load(Ordering::Relaxed);
            idle_regs.rbx = vmctx.init.idle_regs.rbx.load(Ordering::Relaxed);
            idle_regs.sp = vmctx.init.idle_regs.rsp.load(Ordering::Relaxed);
            idle_regs.rbp = vmctx.init.idle_regs.rbp.load(Ordering::Relaxed);
            idle_regs.r12 = vmctx.init.idle_regs.r12.load(Ordering::Relaxed);
            idle_regs.r13 = vmctx.init.idle_regs.r13.load(Ordering::Relaxed);
            idle_regs.r14 = vmctx.init.idle_regs.r14.load(Ordering::Relaxed);
            idle_regs.r15 = vmctx.init.idle_regs.r15.load(Ordering::Relaxed);
        }

        Ok(Sandbox {
            _lifetime_pipe: lifetime_pipe_host,
            vmctx_mmap,
            memory_mmap,
            iouring,
            iouring_futex_wait_queued: false,
            iouring_uffd_read_queued: false,
            iouring_waitid_queued: false,
            iouring_siginfo: Default::default(),
            userfaultfd,
            #[allow(clippy::arc_with_non_send_sync)]
            uffd_msg: UffdBuffer(Arc::new(UnsafeCell::new(linux_raw::uffd_msg::default()))),
            child,

            count_wait_loop_start: 0,
            count_futex_wait: 0,

            module: None,
            gas_metering: None,

            state: SandboxState::Idle,
            is_program_counter_valid: false,
            next_program_counter: None,
            next_program_counter_changed: true,
            pending_pagefault: None,
            page_set: PageSet::new(),
            dynamic_paging_enabled: false,
            idle_regs,
            aux_data_address: 0,
            aux_data_length: 0,
            is_borked: false,
        })
    }

    fn load_module(&mut self, global: &Self::GlobalState, module: &Module) -> Result<(), Self::Error> {
        if self.module.is_some() {
            return Err(Error::from("module already loaded"));
        }

        if module.is_dynamic_paging() && get_native_page_size() != module.memory_map().page_size() as usize {
            return Err(Error::from(
                "dynamic paging is currently unsupported if the module's page size doesn't match the native page size",
            ));
        }

        log::debug!(
            "Loading module into sandbox #{}... (dynamic paging = {})",
            self.child.pid,
            module.is_dynamic_paging()
        );

        let compiled_module = Self::downcast_module(module);
        let program = &compiled_module.sandbox_program.0;

        let memory_map = if !module.is_dynamic_paging() {
            let Some(memory_map) = global.shared_memory.alloc(core::mem::size_of::<VmMap>() * program.memory_map.len()) else {
                return Err(Error::from_str("out of shared memory"));
            };

            let vm_maps = unsafe { memory_map.as_typed_slice_mut::<VmMap>() };
            for (chunk, vm_map) in program.memory_map.iter().zip(vm_maps.iter_mut()) {
                let (fd, fd_offset) = match chunk.initialize_with {
                    InitializeWith::None => (VmFd::None, 0),
                    InitializeWith::Shm(ref alloc) => (VmFd::Shm, alloc.offset() as u64),
                    InitializeWith::Mem(offset) => (VmFd::Mem, u64::from(offset)),
                };

                *vm_map = VmMap {
                    address: chunk.address,
                    length: chunk.length,
                    protection: linux_raw::PROT_READ | if chunk.is_writable { linux_raw::PROT_WRITE } else { 0 },
                    flags: if !matches!(chunk.initialize_with, InitializeWith::None) {
                        linux_raw::MAP_FIXED | linux_raw::MAP_PRIVATE
                    } else {
                        linux_raw::MAP_FIXED | linux_raw::MAP_PRIVATE | linux_raw::MAP_ANONYMOUS
                    },
                    fd,
                    fd_offset,
                };
            }

            self.vmctx()
                .shm_memory_map_count
                .store(program.memory_map.len() as u64, Ordering::Relaxed);
            memory_map
        } else {
            let Some(memory_map) = global.shared_memory.alloc(core::mem::size_of::<VmMap>()) else {
                return Err(Error::from_str("out of shared memory"));
            };

            let vm_maps = unsafe { memory_map.as_typed_slice_mut::<VmMap>() };
            vm_maps[0] = VmMap {
                address: 0x10000,
                length: u64::from(u32::MAX) + 1 - 0x10000,
                protection: linux_raw::PROT_READ | linux_raw::PROT_WRITE,
                flags: linux_raw::MAP_FIXED | linux_raw::MAP_SHARED,
                fd: VmFd::Mem,
                fd_offset: 0x10000,
            };

            self.vmctx().shm_memory_map_count.store(1, Ordering::Relaxed);
            memory_map
        };

        self.vmctx()
            .shm_memory_map_offset
            .store(memory_map.offset() as u64, Ordering::Relaxed);

        unsafe {
            *self.vmctx().heap_info.heap_top.get() = u64::from(module.memory_map().heap_base());
            *self.vmctx().heap_info.heap_threshold.get() = u64::from(module.memory_map().rw_data_range().end);
            *self.vmctx().heap_base.get() = module.memory_map().heap_base();
            *self.vmctx().heap_initial_threshold.get() = module.memory_map().rw_data_range().end;
            *self.vmctx().heap_max_size.get() = module.memory_map().max_heap_size();
            *self.vmctx().page_size.get() = module.memory_map().page_size();
        }

        self.vmctx()
            .shm_code_offset
            .store(program.shm_code.offset() as u64, Ordering::Relaxed);
        self.vmctx().shm_code_length.store(program.shm_code.len() as u64, Ordering::Relaxed);
        self.vmctx()
            .shm_jump_table_offset
            .store(program.shm_jump_table.offset() as u64, Ordering::Relaxed);
        self.vmctx()
            .shm_jump_table_length
            .store(program.shm_jump_table.len() as u64, Ordering::Relaxed);
        self.vmctx().sysreturn_address.store(program.sysreturn_address, Ordering::Relaxed);

        self.vmctx().program_counter.store(0, Ordering::Relaxed);
        self.vmctx().next_program_counter.store(0, Ordering::Relaxed);
        self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
        self.vmctx().jump_into.store(ZYGOTE_TABLES.1.ext_load_program, Ordering::Relaxed);
        self.vmctx().gas.store(0, Ordering::Relaxed);
        for reg in &self.vmctx().regs {
            reg.store(0, Ordering::Relaxed);
        }

        self.aux_data_address = module.memory_map().aux_data_address();
        self.aux_data_length = module.memory_map().aux_data_size();
        self.dynamic_paging_enabled = module.is_dynamic_paging();
        self.is_program_counter_valid = false;
        self.gas_metering = module.gas_metering();
        self.module = Some(module.clone());
        self.wake_oneshot_and_expect_idle()?;
        core::mem::drop(memory_map);

        if module.is_dynamic_paging() {
            linux_raw::sys_uffdio_register(
                self.userfaultfd.borrow(),
                &mut linux_raw::uffdio_register {
                    range: linux_raw::uffdio_range {
                        start: 0x10000,
                        len: u64::from(u32::MAX) + 1 - 0x10000,
                    },
                    mode: linux_raw::UFFDIO_REGISTER_MODE_MISSING | linux_raw::UFFDIO_REGISTER_MODE_WP,
                    ..linux_raw::uffdio_register::default()
                },
            )
            .map_err(|error| Error::from(format!("failed to register the guest memory with userfaultfd: {error}")))?;
        }

        Ok(())
    }

    fn recycle(&mut self, _global: &Self::GlobalState) -> Result<(), Self::Error> {
        if self.is_borked {
            return Err(Error::from_str("broken sandbox"));
        }

        log::trace!("Recycling sandbox #{}", self.child.pid);
        if self.dynamic_paging_enabled {
            self.free_pages(0x10000, 0xffff0000)?;
        }

        self.module = None;
        self.page_set.clear();

        if self.pending_pagefault.take().is_some() {
            self.resume_child()?;
            Ok(())
        } else {
            self.vmctx().jump_into.store(ZYGOTE_TABLES.1.ext_recycle, Ordering::Relaxed);
            self.wake_oneshot_and_expect_idle()
        }
    }

    fn run(&mut self) -> Result<InterruptKind, Self::Error> {
        if self.module.is_none() {
            return Err(Error::from_str("no module loaded into the sandbox"));
        };

        if self.pending_pagefault.take().is_some() {
            self.resume_child()?;
        }

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
            self.vmctx().next_program_counter.store(pc.0, Ordering::Relaxed);
            self.vmctx().next_native_program_counter.store(address, Ordering::Relaxed);
        } else {
            log::trace!(
                "Resuming into: {} (0x{:x})",
                self.vmctx().next_program_counter.load(Ordering::Relaxed),
                self.vmctx().next_native_program_counter.load(Ordering::Relaxed)
            );
        };

        if let Some(pagefault) = self.pending_pagefault.take() {
            if pagefault.registers_modified {
                self.upload_registers()?;
            }

            // This acts exactly the same as `UFFDIO_WAKE`.
            log::trace!(
                "Child #{}: sys_ptrace_continue after page fault at 0x{:x}",
                self.child.pid,
                pagefault.address
            );
            linux_raw::sys_ptrace_continue(self.child.pid, None)?;
        } else {
            let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
            debug_assert_eq!(self.vmctx().futex.load(Ordering::Relaxed) & 1, VMCTX_FUTEX_IDLE);
            self.vmctx()
                .jump_into
                .store(compiled_module.sandbox_program.0.sysenter_address, Ordering::Relaxed);
            self.wake_worker()?;
            self.is_program_counter_valid = true;
        }

        let result = self.wait()?;
        if self.module.as_ref().unwrap().gas_metering() == Some(GasMeteringKind::Async) && self.gas() < 0 {
            self.is_program_counter_valid = false;
            self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
            return Ok(InterruptKind::NotEnoughGas);
        }

        Ok(match result {
            Interrupt::Idle => {
                self.is_program_counter_valid = false;
                InterruptKind::Finished
            }
            Interrupt::NotEnoughGas => InterruptKind::NotEnoughGas,
            Interrupt::Trap => InterruptKind::Trap,
            Interrupt::Ecalli(num) => {
                self.state = SandboxState::Hostcall;
                InterruptKind::Ecalli(num)
            }
            Interrupt::Segfault(segfault) => {
                self.state = SandboxState::Pagefault;
                InterruptKind::Segfault(segfault)
            }
            Interrupt::Step => InterruptKind::Step,
        })
    }

    fn reg(&self, reg: Reg) -> RegValue {
        let mut value = self.vmctx().regs[reg as usize].load(Ordering::Relaxed);
        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        if compiled_module.bitness == Bitness::B32 {
            value &= 0xffffffff;
        }

        value
    }

    fn set_reg(&mut self, reg: Reg, mut value: RegValue) {
        if let Some(ref mut pagefault) = self.pending_pagefault {
            pagefault.registers_modified = true;
        }

        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        if compiled_module.bitness == Bitness::B32 {
            value &= 0xffffffff;
        }

        self.vmctx().regs[reg as usize].store(value, Ordering::Relaxed)
    }

    fn gas(&self) -> Gas {
        self.vmctx().gas.load(Ordering::Relaxed)
    }

    fn set_gas(&mut self, gas: Gas) {
        self.vmctx().gas.store(gas, Ordering::Relaxed)
    }

    fn program_counter(&self) -> Option<ProgramCounter> {
        if !self.is_program_counter_valid {
            return None;
        }

        Some(ProgramCounter(self.vmctx().program_counter.load(Ordering::Relaxed)))
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
        let compiled_module = Self::downcast_module(self.module.as_ref()?);
        if let Some(pc) = self.next_program_counter {
            return compiled_module.lookup_native_code_address(pc).map(|value| value as usize);
        }

        let value = self.vmctx().next_native_program_counter.load(Ordering::Relaxed);
        if value == 0 {
            None
        } else {
            Some(value as usize)
        }
    }

    fn accessible_aux_size(&self) -> u32 {
        assert!(!self.dynamic_paging_enabled);
        self.aux_data_length
    }

    fn set_accessible_aux_size(&mut self, size: u32) -> Result<(), Error> {
        assert!(!self.dynamic_paging_enabled);

        let module = self.module.as_ref().unwrap();
        self.aux_data_length = size;
        self.vmctx().arg.store(self.aux_data_address, Ordering::Relaxed);
        self.vmctx().arg2.store(size, Ordering::Relaxed);
        self.vmctx().arg3.store(module.memory_map().aux_data_size(), Ordering::Relaxed);
        self.vmctx()
            .jump_into
            .store(ZYGOTE_TABLES.1.ext_set_accessible_aux_size, Ordering::Relaxed);
        self.wake_oneshot_and_expect_idle()
    }

    fn is_memory_accessible(&self, address: u32, size: u32, _is_writable: bool) -> bool {
        assert!(self.dynamic_paging_enabled);

        let module = self.module.as_ref().unwrap();
        let page_start = module.address_to_page(module.round_to_page_size_down(address));
        let page_end = module.address_to_page(module.round_to_page_size_down(address + size));
        self.page_set.contains((page_start, page_end))
    }

    fn reset_memory(&mut self) -> Result<(), Error> {
        if self.module.is_none() {
            return Err(Error::from_str("no module loaded into the sandbox"));
        };

        if !self.dynamic_paging_enabled {
            self.vmctx().jump_into.store(ZYGOTE_TABLES.1.ext_reset_memory, Ordering::Relaxed);
            self.wake_oneshot_and_expect_idle()
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

        if !self.dynamic_paging_enabled {
            let length = slice.len();
            match linux_raw::vm_read_memory(self.child.pid, [slice], [(address as usize, length)]) {
                Ok(actual_length) if actual_length == length => unsafe { Ok(slice_assume_init_mut(slice)) },
                Ok(_) => Err(MemoryAccessError::Error("incomplete read".into())),
                Err(error) => Err(MemoryAccessError::Error(error.into())),
            }
        } else {
            let module = self.module.as_ref().unwrap();
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + slice.len() as u32));
            if !self.page_set.contains((page_start, page_end)) {
                return Err(MemoryAccessError::Error("incomplete read".into()));
            } else {
                let memory: &[core::mem::MaybeUninit<u8>] =
                    unsafe { core::slice::from_raw_parts(self.memory_mmap.as_ptr().cast(), self.memory_mmap.len()) };

                slice.copy_from_slice(&memory[address as usize..address as usize + slice.len()]);
            }

            unsafe { Ok(slice_assume_init_mut(slice)) }
        }
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

        let module = self.module.as_ref().unwrap();
        if !self.dynamic_paging_enabled {
            let memory_map = module.memory_map();
            let is_ok = if address >= memory_map.aux_data_address() {
                let aux_data_size = module.memory_map().aux_data_size();
                let aux_data_end = module.memory_map().aux_data_address() + aux_data_size;
                let address_end = address as usize + data.len();
                if address_end <= aux_data_end as usize {
                    self.memory_mmap.as_slice_mut()[address as usize..address as usize + data.len()].copy_from_slice(data);
                    return Ok(());
                } else {
                    false
                }
            } else if address >= memory_map.stack_address_low() {
                u64::from(address) + data.len() as u64 <= u64::from(memory_map.stack_range().end)
            } else if address >= memory_map.rw_data_address() {
                let end = unsafe { *self.vmctx().heap_info.heap_threshold.get() };
                u64::from(address) + data.len() as u64 <= end
            } else {
                false
            };

            if !is_ok {
                return Err(MemoryAccessError::OutOfRangeAccess {
                    address,
                    length: data.len() as u64,
                });
            }

            let length = data.len();
            match linux_raw::vm_write_memory(self.child.pid, [data], [(address as usize, length)]) {
                Ok(actual_length) if actual_length == length => Ok(()),
                Ok(_) => Err(MemoryAccessError::Error("incomplete write".into())),
                Err(error) => Err(MemoryAccessError::Error(error.into())),
            }
        } else {
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + data.len() as u32));
            self.page_set.insert((page_start, page_end));
            self.memory_mmap.as_slice_mut()[address as usize..address as usize + data.len()].copy_from_slice(data);
            Ok(())
        }
    }

    fn zero_memory(&mut self, address: u32, length: u32) -> Result<(), MemoryAccessError> {
        log::trace!(
            "Zeroing memory: 0x{:x}-0x{:x} ({} bytes)",
            address,
            address as usize + length as usize,
            length
        );

        let module = self.module.as_ref().unwrap();
        if !self.dynamic_paging_enabled {
            let memory_map = module.memory_map();
            let is_ok = if address >= memory_map.aux_data_address() {
                if u64::from(address) + u64::from(length) <= u64::from(memory_map.aux_data_range().end) {
                    self.memory_mmap.as_slice_mut()[address as usize..address as usize + length as usize].fill(0);
                    return Ok(());
                } else {
                    false
                }
            } else if address >= memory_map.stack_address_low() {
                u64::from(address) + u64::from(length) <= u64::from(memory_map.stack_range().end)
            } else if address >= memory_map.rw_data_address() {
                let end = unsafe { *self.vmctx().heap_info.heap_threshold.get() };
                u64::from(address) + u64::from(length) <= end
            } else {
                false
            };

            if !is_ok {
                return Err(MemoryAccessError::OutOfRangeAccess {
                    address,
                    length: u64::from(length),
                });
            }

            self.vmctx().arg.store(address, Ordering::Relaxed);
            self.vmctx().arg2.store(length, Ordering::Relaxed);
            self.vmctx()
                .jump_into
                .store(ZYGOTE_TABLES.1.ext_zero_memory_chunk, Ordering::Relaxed);
            if let Err(error) = self.wake_oneshot_and_expect_idle() {
                return Err(MemoryAccessError::Error(error.into()));
            }
        } else {
            let page_start = module.address_to_page(module.round_to_page_size_down(address));
            let page_end = module.address_to_page(module.round_to_page_size_down(address + length));
            if module.is_multiple_of_page_size(address)
                && module.is_multiple_of_page_size(length)
                && self.page_set.is_whole_region_empty((page_start, page_end))
            {
                let mut arg: linux_raw::uffdio_zeropage = Default::default();
                arg.range.start = u64::from(address);
                arg.range.len = u64::from(length);
                arg.mode = linux_raw::UFFDIO_ZEROPAGE_MODE_DONTWAKE;

                log::trace!(
                    "sys_uffdio_zeropage: 0x{:x}..0x{:x}",
                    arg.range.start,
                    arg.range.start + arg.range.len
                );

                if let Err(error) = linux_raw::sys_uffdio_zeropage(self.userfaultfd.borrow(), &mut arg) {
                    return Err(MemoryAccessError::Error(error.into()));
                }
            } else {
                self.memory_mmap.as_slice_mut()[address as usize..address as usize + length as usize].fill(0);
            }

            self.page_set.insert((page_start, page_end));
        }

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

        let mut arg: linux_raw::uffdio_writeprotect = Default::default();
        arg.range.start = u64::from(address);
        arg.range.len = u64::from(length);
        arg.mode = linux_raw::UFFDIO_WRITEPROTECT_MODE_WP;

        if let Err(error) = linux_raw::sys_uffdio_writeprotect(self.userfaultfd.borrow(), &mut arg) {
            return Err(MemoryAccessError::Error(error.into()));
        }

        Ok(())
    }

    fn free_pages(&mut self, address: u32, length: u32) -> Result<(), Self::Error> {
        if !self.dynamic_paging_enabled {
            todo!();
        } else {
            unsafe {
                linux_raw::sys_madvise(
                    self.memory_mmap.as_mut_ptr().add(address as usize),
                    length as usize,
                    linux_raw::MADV_REMOVE,
                )?;
            }

            if address <= 0x10000 && length >= 0xffff0000 {
                self.page_set.clear();
            } else {
                let module = self.module.as_ref().unwrap();
                let page_start = module.address_to_page(module.round_to_page_size_down(address));
                let page_end = module.address_to_page(module.round_to_page_size_down(address + length));
                self.page_set.remove((page_start, page_end));
            }

            Ok(())
        }
    }

    fn heap_size(&self) -> u32 {
        let heap_base = unsafe { *self.vmctx().heap_base.get() };
        let heap_top = unsafe { *self.vmctx().heap_info.heap_top.get() };
        (heap_top - u64::from(heap_base)) as u32
    }

    fn sbrk(&mut self, size: u32) -> Result<Option<u32>, Error> {
        if size == 0 {
            return Ok(Some(unsafe { *self.vmctx().heap_info.heap_top.get() as u32 }));
        }

        self.vmctx().jump_into.store(ZYGOTE_TABLES.1.ext_sbrk, Ordering::Relaxed);
        self.vmctx().arg.store(size, Ordering::Relaxed);
        self.wake_worker()?;
        self.wait()?.expect_idle()?;

        let result = self.vmctx().arg.load(Ordering::Relaxed);
        if result == 0 {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }

    fn pid(&self) -> Option<u32> {
        Some(self.child.pid as u32)
    }

    fn address_table() -> AddressTable {
        ZYGOTE_TABLES.0
    }

    #[inline]
    fn offset_table() -> OffsetTable {
        OffsetTable {
            arg: get_field_offset!(VmCtx::new(), |base| base.arg.as_ptr()),
            gas: get_field_offset!(VmCtx::new(), |base| base.gas.as_ptr()),
            heap_info: get_field_offset!(VmCtx::new(), |base| &base.heap_info),
            next_native_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_native_program_counter.as_ptr()),
            next_program_counter: get_field_offset!(VmCtx::new(), |base| base.next_program_counter.as_ptr()),
            program_counter: get_field_offset!(VmCtx::new(), |base| base.program_counter.as_ptr()),
            regs: get_field_offset!(VmCtx::new(), |base| base.regs.as_ptr()),
        }
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        self.wait()?.expect_idle()
    }
}

#[must_use]
enum Interrupt {
    Idle,
    Trap,
    NotEnoughGas,
    Ecalli(u32),
    Segfault(Segfault),
    Step,
}

impl Interrupt {
    fn expect_idle(self) -> Result<(), Error> {
        match self {
            Interrupt::Idle => Ok(()),
            Interrupt::Trap => Err(Error::from_str("unexpected trap")),
            Interrupt::NotEnoughGas => Err(Error::from_str("unexpected not enough gas")),
            Interrupt::Ecalli(_) => Err(Error::from_str("unexpected ecalli")),
            Interrupt::Segfault(_) => Err(Error::from_str("unexpected segfault")),
            Interrupt::Step => Err(Error::from_str("unexpected step")),
        }
    }
}

impl Sandbox {
    #[inline]
    fn vmctx(&self) -> &VmCtx {
        unsafe { &*self.vmctx_mmap.as_ptr().cast::<VmCtx>() }
    }

    fn wake_worker(&self) -> Result<(), Error> {
        self.vmctx().futex.store(VMCTX_FUTEX_BUSY, Ordering::Release);
        linux_raw::sys_futex_wake_one(&self.vmctx().futex).map(|_| ())
    }

    fn wake_oneshot_and_expect_idle(&mut self) -> Result<(), Error> {
        self.wake_worker()?;
        self.wait()?.expect_idle()
    }

    fn handle_guest_signal(&mut self, machine_code_address: u64) -> Result<Interrupt, Error> {
        use crate::sandbox::Sandbox;

        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        let Some(machine_code_offset) = machine_code_address.checked_sub(compiled_module.native_code_origin) else {
            return Err(Error::from_str("internal error: address underflow after a trap"));
        };

        let is_out_of_gas = match compiled_module.bitness {
            Bitness::B32 => crate::compiler::ArchVisitor::<Self, B32>::on_signal_trap(
                compiled_module,
                self.gas_metering.is_some(),
                machine_code_offset,
                self.vmctx(),
            ),
            Bitness::B64 => crate::compiler::ArchVisitor::<Self, B64>::on_signal_trap(
                compiled_module,
                self.gas_metering.is_some(),
                machine_code_offset,
                self.vmctx(),
            ),
        }
        .map_err(Error::from_str)?;

        self.is_program_counter_valid = true;
        if is_out_of_gas {
            Ok(Interrupt::NotEnoughGas)
        } else {
            Ok(Interrupt::Trap)
        }
    }

    #[inline(never)]
    #[cold]
    fn wait(&mut self) -> Result<Interrupt, Error> {
        use crate::sandbox::Sandbox;

        'outer: loop {
            self.count_wait_loop_start += 1;

            let state = self.vmctx().futex.load(Ordering::Relaxed);
            if state == VMCTX_FUTEX_IDLE {
                core::sync::atomic::fence(Ordering::Acquire);
                return Ok(Interrupt::Idle);
            }

            if state == VMCTX_FUTEX_GUEST_SIGNAL {
                core::sync::atomic::fence(Ordering::Acquire);

                let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
                if compiled_module.bitness == Bitness::B32 {
                    for reg_value in &self.vmctx().regs {
                        reg_value.fetch_and(0xffffffff, Ordering::Relaxed);
                    }
                }

                let machine_code_address = self.vmctx().rip.load(Ordering::Relaxed);
                return self.handle_guest_signal(machine_code_address);
            }

            if state == VMCTX_FUTEX_GUEST_ECALLI {
                core::sync::atomic::fence(Ordering::Acquire);
                let hostcall = self.vmctx().arg.load(Ordering::Relaxed);
                return Ok(Interrupt::Ecalli(hostcall));
            }

            if state == VMCTX_FUTEX_GUEST_TRAP {
                core::sync::atomic::fence(Ordering::Acquire);
                return Ok(Interrupt::Trap);
            }

            if state == VMCTX_FUTEX_GUEST_NOT_ENOUGH_GAS {
                core::sync::atomic::fence(Ordering::Acquire);
                return Ok(Interrupt::NotEnoughGas);
            }

            if state == VMCTX_FUTEX_GUEST_STEP {
                core::sync::atomic::fence(Ordering::Acquire);
                return Ok(Interrupt::Step);
            }

            if state != VMCTX_FUTEX_BUSY {
                log::error!("Unexpected worker process state: {state}");
                return Err(Error::from_str("internal error: unexpected worker process state"));
            }

            if self.dynamic_paging_enabled {
                let iouring = self.iouring.as_mut().unwrap();

                const IO_URING_JOB_FUTEX_WAIT: u64 = 1;
                const IO_URING_JOB_USERFAULTFD_READ: u64 = 2;
                const IO_URING_JOB_WAITID: u64 = 3;

                if !self.iouring_futex_wait_queued {
                    self.count_futex_wait += 1;
                    let vmctx = unsafe { &*self.vmctx_mmap.as_ptr().cast::<VmCtx>() };
                    iouring
                        .queue_futex_wait(IO_URING_JOB_FUTEX_WAIT, &vmctx.futex, VMCTX_FUTEX_BUSY)
                        .expect("internal error: io_uring queue overflow");
                    self.iouring_futex_wait_queued = true;
                }

                if !self.iouring_uffd_read_queued {
                    iouring
                        .queue_read(
                            IO_URING_JOB_USERFAULTFD_READ,
                            self.userfaultfd.borrow(),
                            self.uffd_msg.0.get().cast(),
                            core::mem::size_of::<linux_raw::uffd_msg>() as u32,
                        )
                        .expect("internal error: io_uring queue overflow");
                    self.iouring_uffd_read_queued = true;
                }

                if !self.iouring_waitid_queued {
                    iouring
                        .queue_waitid(
                            IO_URING_JOB_WAITID,
                            linux_raw::P_PIDFD,
                            self.child.pidfd.as_ref().expect("internal error: no pidfd handle").raw() as u32,
                            self.iouring_siginfo.0.get(),
                            linux_raw::WEXITED | linux_raw::__WALL,
                        )
                        .expect("internal error: io_uring queue overflow");
                    self.iouring_waitid_queued = true;
                }

                unsafe {
                    iouring.submit_and_wait(1).expect("internal error: io_uring failed");
                }

                while let Some(job) = self.iouring.as_mut().unwrap().pop_finished() {
                    // Fetch again to appease the borrow checker.
                    if job.user_data == IO_URING_JOB_FUTEX_WAIT {
                        self.iouring_futex_wait_queued = false;
                    } else if job.user_data == IO_URING_JOB_USERFAULTFD_READ {
                        self.iouring_uffd_read_queued = false;
                        if job.res == -(linux_raw::ERESTARTSYS as i32) {
                            log::trace!("Child #{}: ERESTARTSYS", self.child.pid);
                            continue;
                        }
                        job.to_result()?;

                        let msg = unsafe { &mut *self.uffd_msg.0.get() };
                        let event = u32::from(core::mem::replace(&mut msg.event, 0));
                        if event == linux_raw::UFFD_EVENT_PAGEFAULT {
                            let pagefault = unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(msg.arg.pagefault)) };
                            let address = pagefault.address;
                            let is_minor = (pagefault.flags & u64::from(linux_raw::UFFD_PAGEFAULT_FLAG_MINOR)) != 0;
                            let is_write = (pagefault.flags & u64::from(linux_raw::UFFD_PAGEFAULT_FLAG_WRITE)) != 0;
                            let is_wp = (pagefault.flags & u64::from(linux_raw::UFFD_PAGEFAULT_FLAG_WP)) != 0;

                            log::trace!(
                                "Child #{}: pagefault: address=0x{address:x}, minor={is_minor}, write={is_write}, wp={is_wp}",
                                self.child.pid
                            );

                            self.pending_pagefault = Some(Pagefault {
                                address,
                                registers_modified: false,
                            });

                            debug_assert!(address <= u64::from(u32::MAX));

                            linux_raw::sys_ptrace_interrupt(self.child.pid)?;
                            let status = if !self.iouring_waitid_queued {
                                self.child.check_status(false)?
                            } else {
                                'wait_loop: loop {
                                    unsafe {
                                        self.iouring
                                            .as_mut()
                                            .unwrap()
                                            .submit_and_wait(1)
                                            .expect("internal error: io_uring failed");
                                    }

                                    while let Some(job) = self.iouring.as_mut().unwrap().pop_finished() {
                                        if job.user_data == IO_URING_JOB_FUTEX_WAIT {
                                            self.iouring_futex_wait_queued = false;
                                        } else if job.user_data == IO_URING_JOB_WAITID {
                                            self.iouring_waitid_queued = false;
                                            let result = job
                                                .to_result()
                                                .map(|_| unsafe { core::ptr::read_volatile(self.iouring_siginfo.0.get()) });
                                            break 'wait_loop ChildProcess::extract_status(result)?;
                                        } else {
                                            unreachable!("internal error: unknown io_uring job");
                                        }
                                    }
                                }
                            };

                            if !status.is_trapped() {
                                log::error!("Child #{}: expected child to trap, found: {status}", self.child.pid);
                                return Err(Error::from_str("internal error: unexpected child status"));
                            }

                            let machine_code_address = self.download_registers()?;
                            log::trace!("Child #{}: pagefault: rip=0x{machine_code_address:x}", self.child.pid);

                            let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
                            let Some(machine_code_offset) = machine_code_address.checked_sub(compiled_module.native_code_origin) else {
                                return Err(Error::from_str("internal error: address underflow after a segfault"));
                            };

                            match compiled_module.bitness {
                                Bitness::B32 => crate::compiler::ArchVisitor::<Self, B32>::on_page_fault(
                                    compiled_module,
                                    self.gas_metering.is_some(),
                                    machine_code_address,
                                    machine_code_offset,
                                    self.vmctx(),
                                ),
                                Bitness::B64 => crate::compiler::ArchVisitor::<Self, B64>::on_page_fault(
                                    compiled_module,
                                    self.gas_metering.is_some(),
                                    machine_code_address,
                                    machine_code_offset,
                                    self.vmctx(),
                                ),
                            }
                            .map_err(Error::from_str)?;

                            self.is_program_counter_valid = true;
                            if is_write && is_wp {
                                self.vmctx().next_native_program_counter.store(0, Ordering::Relaxed);
                                return Ok(Interrupt::Trap);
                            }

                            return Ok(Interrupt::Segfault(Segfault {
                                page_address: address as u32,
                                page_size: get_native_page_size() as u32,
                            }));
                        }
                    } else if job.user_data == IO_URING_JOB_WAITID {
                        self.iouring_waitid_queued = false;

                        log::trace!("Child #{}: waitid triggered", self.child.pid);
                        let result = job
                            .to_result()
                            .map(|_| unsafe { core::ptr::read_volatile(self.iouring_siginfo.0.get()) });
                        let status = ChildProcess::extract_status(result)?;
                        if let Some(interrupt) = self.handle_child_status(status)? {
                            return Ok(interrupt);
                        }
                    } else {
                        unreachable!("internal error: unknown io_uring job");
                    }
                }
            } else {
                let spin_target = if self.module.as_ref().map_or(false, |module| module.is_step_tracing()) {
                    128
                } else {
                    0
                };

                let yield_target = 16;

                for _ in 0..spin_target {
                    core::hint::spin_loop();
                    if self.vmctx().futex.load(Ordering::Relaxed) != VMCTX_FUTEX_BUSY {
                        continue 'outer;
                    }
                }

                for _ in 0..yield_target {
                    let _ = linux_raw::sys_sched_yield();
                    if self.vmctx().futex.load(Ordering::Relaxed) != VMCTX_FUTEX_BUSY {
                        continue 'outer;
                    }
                }

                self.count_futex_wait += 1;
                match linux_raw::sys_futex_wait(&self.vmctx().futex, VMCTX_FUTEX_BUSY, Some(Duration::from_millis(100))) {
                    Ok(()) => continue,
                    Err(error) if error.errno() == linux_raw::EAGAIN || error.errno() == linux_raw::EINTR => continue,
                    Err(error) if error.errno() == linux_raw::ETIMEDOUT => {
                        log::trace!("Timeout expired while waiting for child #{}...", self.child.pid);
                        let status = self.child.check_status(true)?;
                        if let Some(interrupt) = self.handle_child_status(status)? {
                            return Ok(interrupt);
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }

    fn handle_child_status(&mut self, status: ChildStatus) -> Result<Option<Interrupt>, Error> {
        self.is_borked = true;

        if self.dynamic_paging_enabled && status.is_trapped() {
            let siginfo = linux_raw::sys_ptrace_get_siginfo(self.child.pid)?;
            let machine_code_address = self.download_registers()?;
            let signal = unsafe { siginfo.si_signo() as u32 };
            log::trace!(
                "Child #{}: trapped with signal {signal} at rip=0x{machine_code_address:x}",
                self.child.pid
            );

            let result = self.handle_guest_signal(machine_code_address)?;
            self.resume_child()?;

            self.is_borked = false;
            return Ok(Some(result));
        }

        if status.is_running() {
            self.is_borked = false;
            return Ok(None);
        }

        log::trace!("Child #{} is not running anymore: {status}", self.child.pid);
        let message = get_message(self.vmctx());
        if let Some(message) = message {
            Err(Error::from(format!("{status}: {message}")))
        } else {
            Err(Error::from(format!("worker process unexpectedly quit: {status}")))
        }
    }

    #[inline]
    fn get_register(reg: polkavm_common::regmap::NativeReg, regs: &mut linux_raw::user_regs_struct) -> &mut u64 {
        use polkavm_common::regmap::NativeReg::*;

        match reg {
            rax => &mut regs.rax,
            rcx => &mut regs.rcx,
            rdx => &mut regs.rdx,
            rbx => &mut regs.rbx,
            rbp => &mut regs.rbp,
            rsi => &mut regs.rsi,
            rdi => &mut regs.rdi,
            r8 => &mut regs.r8,
            r9 => &mut regs.r9,
            r10 => &mut regs.r10,
            r11 => &mut regs.r11,
            r12 => &mut regs.r12,
            r13 => &mut regs.r13,
            r14 => &mut regs.r14,
            r15 => &mut regs.r15,
        }
    }

    fn download_registers(&mut self) -> Result<u64, Error> {
        use crate::sandbox::Sandbox;

        let mut regs = linux_raw::sys_ptrace_getregs(self.child.pid)?;
        let compiled_module = Self::downcast_module(self.module.as_ref().unwrap());
        for reg in Reg::ALL {
            let mut value = *Self::get_register(polkavm_common::regmap::to_native_reg(reg), &mut regs);
            if compiled_module.bitness == Bitness::B32 {
                value &= 0xffffffff;
            }

            self.vmctx().regs[reg as usize].store(value, Ordering::Relaxed);
        }

        self.vmctx()
            .tmp_reg
            .store(*Self::get_register(polkavm_common::regmap::TMP_REG, &mut regs), Ordering::Relaxed);

        Ok(regs.rip)
    }

    fn upload_registers(&mut self) -> Result<(), Error> {
        let mut regs = linux_raw::sys_ptrace_getregs(self.child.pid)?;
        for reg in Reg::ALL {
            *Self::get_register(polkavm_common::regmap::to_native_reg(reg), &mut regs) =
                self.vmctx().regs[reg as usize].load(Ordering::Relaxed);
        }

        regs.rip = self.vmctx().next_native_program_counter.load(Ordering::Relaxed);
        *Self::get_register(polkavm_common::regmap::TMP_REG, &mut regs) = self.vmctx().tmp_reg.load(Ordering::Relaxed);

        linux_raw::sys_ptrace_setregs(self.child.pid, &regs)?;

        Ok(())
    }

    fn resume_child(&mut self) -> Result<(), Error> {
        log::trace!("Child #{}: resuming...", self.child.pid);

        // This will cancel *our own* `futex_wait` which we've queued up with iouring.
        linux_raw::sys_futex_wake_one(&self.vmctx().futex)?;

        // Forcibly return the worker to the idle state.
        //
        // The worker's currently stuck in a page fault or a trap somewhere inside guest code,
        // so it can't do this by itself.
        self.vmctx().futex.store(VMCTX_FUTEX_IDLE, Ordering::Release);
        linux_raw::sys_ptrace_setregs(self.child.pid, &self.idle_regs)?;
        linux_raw::sys_ptrace_continue(self.child.pid, None)
    }
}
