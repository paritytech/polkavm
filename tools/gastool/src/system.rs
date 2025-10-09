use polkavm_linux_raw as linux_raw;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn cpu_mask_to_string(mask: u128) -> String {
    let mut mask = format!("{:x}", mask);
    mask = mask.chars().rev().collect();
    let mut tmp = String::new();
    for (index, ch) in mask.chars().enumerate() {
        if index > 0 && index % 8 == 0 {
            tmp.push(',');
        }
        tmp.push(ch);
    }
    tmp.chars().rev().collect()
}

#[test]
fn test_cpu_mask_to_string() {
    assert_eq!(cpu_mask_to_string(0), "0");
    assert_eq!(cpu_mask_to_string(1), "1");
    assert_eq!(cpu_mask_to_string(0xa), "a");
    assert_eq!(cpu_mask_to_string(0xf), "f");
    assert_eq!(cpu_mask_to_string(0xff), "ff");
    assert_eq!(cpu_mask_to_string(0x12345678), "12345678");
    assert_eq!(cpu_mask_to_string(0xffffffff), "ffffffff");
    assert_eq!(cpu_mask_to_string(0x1_ffffffff), "1,ffffffff");
    assert_eq!(cpu_mask_to_string(0xf_ffffffff), "f,ffffffff");
    assert_eq!(cpu_mask_to_string(0x88_99abcdef_12345678), "88,99abcdef,12345678");
}

fn set_realtime_priority(pid: libc::c_int) -> Result<(), String> {
    #[repr(C)]
    struct SchedAttr {
        size: u32,
        sched_policy: u32,
        sched_flags: u64,
        sched_nice: i32,
        sched_priority: u32,
        sched_runtime: u64,
        sched_deadline: u64,
        sched_period: u64,
    }

    let attr = SchedAttr {
        size: core::mem::size_of::<SchedAttr>() as u32,
        sched_policy: libc::SCHED_FIFO as _,
        sched_flags: 0,
        sched_nice: 0,
        sched_priority: 99,
        sched_runtime: 0,
        sched_deadline: 0,
        sched_period: 0,
    };

    let result = unsafe { libc::syscall(libc::SYS_sched_setattr, pid, &attr, 0) };
    if result < 0 {
        Err(format!(
            "failed to set realtime priority for PID {pid}: {}",
            std::io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

fn set_affinity(pid: libc::c_int, cpu: usize) -> Result<(), String> {
    unsafe {
        let mut cpu_set: libc::cpu_set_t = core::mem::zeroed();
        libc::CPU_SET(cpu, &mut cpu_set);
        if libc::sched_setaffinity(pid, core::mem::size_of::<libc::cpu_set_t>(), &cpu_set) < 0 {
            Err(format!(
                "failed to set cpu affinity for PID {pid}: {}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(())
        }
    }
}

fn mkdir(path: &str) -> Result<(), String> {
    if let Err(error) = std::fs::create_dir(path) {
        return Err(format!("failed to create directory '{path}': {error}"));
    }
    Ok(())
}

fn write(path: impl AsRef<Path>, value: impl AsRef<[u8]>) -> Result<(), String> {
    let path = path.as_ref();
    if let Err(error) = std::fs::write(path, value) {
        return Err(format!("failed to write to '{path:?}': {error}"));
    }
    Ok(())
}

pub(crate) fn read_string(path: impl AsRef<Path>) -> Result<String, String> {
    let path = path.as_ref();
    match std::fs::read_to_string(path) {
        Ok(value) => Ok(value),
        Err(error) => Err(format!("failed to read {path:?}: '{error}'")),
    }
}

fn cleanup_warn<T>(result: Result<T, String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            log::warn!("Failed to clean up: {error}");
            None
        }
    }
}

pub fn list_cpus() -> Result<BTreeMap<usize, Vec<usize>>, String> {
    let mut cores = BTreeMap::new();
    for chunk in read_string("/proc/cpuinfo")?.trim().split("\n\n") {
        let mut nth_logical = None;
        let mut nth_physical = None;
        for line in chunk.split('\n') {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let mut it = line.split(':');
            let Some(key) = it.next() else {
                return Err("failed to parse '/proc/cpuinfo': missing key".to_owned());
            };
            let Some(value) = it.next() else {
                return Err("failed to parse '/proc/cpuinfo': missing value".to_owned());
            };

            let key = key.trim();
            let value = value.trim();
            if key == "processor" {
                let Ok(value) = value.parse::<usize>() else {
                    return Err("failed to parse '/proc/cpuinfo': 'processor' is not a valid integer".to_owned());
                };
                nth_logical = Some(value);
            } else if key == "core id" {
                let Ok(value) = value.parse::<usize>() else {
                    return Err("failed to parse '/proc/cpuinfo': 'core id' is not a valid integer".to_owned());
                };
                nth_physical = Some(value);
            }
        }

        let (Some(nth_physical), Some(nth_logical)) = (nth_physical, nth_logical) else {
            return Err("failed to parse '/proc/cpuinfo': a pair of 'processor' and 'core id' fields was not found".to_owned());
        };

        cores.entry(nth_physical).or_insert_with(Vec::new).push(nth_logical);
    }

    Ok(cores)
}

fn running_as_root() -> bool {
    unsafe { libc::getuid() == 0 }
}

#[allow(clippy::vec_init_then_push)]
fn restart_with_sudo() -> Result<(), String> {
    if running_as_root() {
        return Ok(());
    }

    use std::os::unix::ffi::OsStringExt;

    let path = std::env::var("PATH").map_err(|error| format!("failed to restart with sudo: failed to read PATH: {error}"))?;
    let sudo_path = path.split(':').find_map(|dir| {
        let dir: &Path = dir.as_ref();
        let path = dir.join("sudo");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    });

    let Some(sudo_path) = sudo_path else {
        return Err("failed to restart with sudo: couldn't find 'sudo' anywhere in your $PATH".into());
    };

    let sudo_path: std::ffi::OsString = sudo_path.into();
    let mut sudo_path: Vec<u8> = sudo_path.into_vec();
    sudo_path.push(0);

    extern "C" {
        static environ: *const *const libc::c_char;
    }

    unsafe {
        let exe = match std::fs::read_link("/proc/self/exe") {
            Ok(exe) => exe,
            Err(error) => {
                return Err(format!("failed to restart with sudo: failed to read '/proc/self/exe': {}", error));
            }
        };

        let exe: std::ffi::OsString = exe.into();
        let mut exe: Vec<u8> = exe.into_vec();
        exe.push(0);

        let cmdline = match std::fs::read("/proc/self/cmdline") {
            Ok(cmdline) => cmdline,
            Err(error) => {
                return Err(format!(
                    "failed to restart with sudo: failed to read '/proc/self/cmdline': {}",
                    error
                ));
            }
        };

        if cmdline.is_empty() {
            return Err("failed to restart with sudo: '/proc/self/cmdline' is empty".into());
        }

        let mut argv = Vec::new();
        argv.push(sudo_path.as_ptr().cast());
        argv.push(c"--preserve-env=RUST_LOG".as_ptr().cast());
        argv.push(c"--".as_ptr().cast());
        argv.push(exe.as_ptr());
        argv.extend(
            cmdline[..cmdline.len() - 1]
                .split(|&byte| byte == 0)
                .skip(1)
                .map(|slice| slice.as_ptr()),
        );
        argv.push(core::ptr::null());

        libc::execve(sudo_path.as_ptr().cast(), argv.as_ptr().cast(), environ);

        Err("failed to restart with sudo: execve failed".into())
    }
}

pub fn restart_with_sudo_or_exit() {
    if !running_as_root() {
        log::info!("Not running as root; trying sudo...");
        if let Err(error) = restart_with_sudo() {
            log::error!("Failed to sudo: {error}");
            std::process::exit(1);
        }
    }
}

fn rmdir_if_exists(path: &str) -> Result<(), String> {
    if let Err(error) = std::fs::remove_dir(path) {
        if error.kind() == std::io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(format!("failed to delete '{path}': {error}"));
    }
    Ok(())
}

#[derive(Default)]
pub struct Tweaks {
    backup: Vec<(PathBuf, String)>,
}

impl Drop for Tweaks {
    fn drop(&mut self) {
        self.backup.reverse();
        for (path, value) in self.backup.drain(..) {
            cleanup_warn(write(&path, value));
        }
    }
}

impl Tweaks {
    pub fn write(&mut self, path: impl AsRef<Path>, value: impl AsRef<[u8]>) -> Result<(), String> {
        let path = path.as_ref();
        let value = value.as_ref();
        let old_value = read_string(path)?;
        let old_value = old_value.trim();
        if old_value.as_bytes() == value {
            log::trace!(
                "{path:?} is already equal to '{value}'; skipping...",
                value = String::from_utf8_lossy(value)
            );
            return Ok(());
        }

        log::trace!("Writing '{value}' to {path:?}...", value = String::from_utf8_lossy(value));
        write(path, value)?;
        self.backup.push((path.to_owned(), old_value.to_owned()));
        Ok(())
    }
}

struct Cpuset {
    name: String,
    original_cgroup_for_pid: BTreeMap<i32, String>,
}

impl Cpuset {
    fn create(tweaks: &mut Tweaks, name: &str, cpus: &[usize], max_cpu: usize) -> Result<Cpuset, String> {
        rmdir_if_exists(&format!("/sys/fs/cgroup/{name}"))?;

        let mut other_cpusets = Vec::new();
        let entries = std::fs::read_dir("/sys/fs/cgroup").map_err(|error| format!("failed to read '/sys/fs/cgroup': {error}"))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("failed to read file entry in '/sys/fs/cgroup': {error}"))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let path_cpus = path.join("cpuset.cpus");
            let path_cpus_effective = path.join("cpuset.cpus.effective");
            if !path_cpus.exists() {
                continue;
            }

            let cpuset_cpus = std::fs::read_to_string(&path_cpus_effective)
                .map_err(|error| format!("failed to read {path_cpus_effective:?}: {error}"))?;
            other_cpusets.push((path_cpus, cpuset_cpus));
        }

        mkdir(&format!("/sys/fs/cgroup/{name}"))?;
        let cpus_s: Vec<_> = cpus.iter().map(|cpu| cpu.to_string()).collect();
        let cpus_s = cpus_s.join(",");
        write(format!("/sys/fs/cgroup/{name}/cpuset.cpus"), cpus_s)?;

        let cpuset = Cpuset {
            name: name.to_owned(),
            original_cgroup_for_pid: Default::default(),
        };

        let mut other_cpus_s = Vec::new();
        for cpu in 0..=max_cpu {
            if cpus.contains(&cpu) {
                continue;
            }

            other_cpus_s.push(cpu.to_string());
        }
        let other_cpus_s = other_cpus_s.join(",");
        for (path, _) in other_cpusets {
            tweaks.write(path, &other_cpus_s)?;
        }

        Ok(cpuset)
    }

    fn add_process(&mut self, pid: i32) -> Result<(), String> {
        let cgroup = read_string(format!("/proc/{pid}/cgroup"))?;
        let cgroup = cgroup.trim().split(':').nth(2).unwrap().to_owned();
        write(format!("/sys/fs/cgroup/{}/cgroup.procs", self.name), pid.to_string())?;
        self.original_cgroup_for_pid.insert(pid, cgroup);
        Ok(())
    }
}

impl Drop for Cpuset {
    fn drop(&mut self) {
        if let Some(list) = cleanup_warn(read_string(format!("/sys/fs/cgroup/{}/cgroup.procs", self.name))) {
            for pid in list.trim().lines() {
                if let Ok(pid) = pid.parse::<i32>() {
                    if let Some(cgroup) = self.original_cgroup_for_pid.get(&pid) {
                        cleanup_warn(write(format!("/sys/fs/cgroup{cgroup}/cgroup.procs"), pid.to_string()));
                    }
                }
            }
        }

        cleanup_warn(write(format!("/sys/fs/cgroup/{}/cgroup.procs", self.name), ""));
        cleanup_warn(rmdir_if_exists(&format!("/sys/fs/cgroup/{}", self.name)));
    }
}

pub struct SystemSetup {
    cpus: Vec<usize>,
    _tweaks: Tweaks,
    cpuset: Option<Cpuset>,
    pub(crate) amd_l3_type: u32,
}

impl SystemSetup {
    pub fn initialize(cpus: &[usize], minimize_noise: bool) -> Result<Self, String> {
        let mut tweaks = Tweaks::default();

        let amd_l3_path = PathBuf::from("/sys/bus/event_source/devices/amd_l3/type");
        if !amd_l3_path.exists() {
            let status = std::process::Command::new("modprobe")
                .arg("amd_uncore")
                .status()
                .map_err(|error| format!("failed to execute 'modprobe amd_uncore': {error}"))?;

            if status.code() != Some(0) {
                return Err(String::from("failed to execute 'modprobe amd_uncore'"));
            }
        }

        let amd_l3_type = read_string(&amd_l3_path)?;
        let amd_l3_type: u32 = amd_l3_type.trim().parse().map_err(|_| format!("failed to parse {amd_l3_path:?}"))?;
        log::debug!("Event type for 'amd_l3': {amd_l3_type}");

        log::info!("Enabling 'rdpmc'...");
        tweaks.write("/sys/devices/cpu/rdpmc", "2")?;

        // Disabling the watchdog is important as otherwise it wastes the first performance counter slot.
        log::info!("Disabling watchdog...");
        tweaks.write("/proc/sys/kernel/watchdog", "0")?;

        let cpuset = if minimize_noise {
            let all_cpus = list_cpus()?;
            let mut cores: Vec<usize> = Vec::new();
            for &cpu in cpus {
                let Some(local_cores) = all_cpus.get(&cpu) else {
                    return Err(format!("CPU #{cpu} not found"));
                };
                cores.extend(local_cores);
            }

            let mut max_cpu = 0;
            for threads in all_cpus.values() {
                for &thread in threads {
                    max_cpu = max_cpu.max(thread);
                }
            }

            if max_cpu >= 128 {
                return Err("only machines with at most 128 hardware threads are supported".to_owned());
            }

            let all_cpu_mask: u128 = if max_cpu == 128 { u128::MAX } else { (1 << (max_cpu + 1)) - 1 };

            let mut target_cpu_mask: u128 = 0;
            for &cpu in cpus {
                target_cpu_mask |= 1 << (cpu as u128);
            }

            log::info!("Allowing realtime tasks to monopolize CPU...");
            tweaks.write("/proc/sys/kernel/sched_rt_runtime_us", "-1")?;

            let other_cpu_mask = cpu_mask_to_string(all_cpu_mask & (!target_cpu_mask));

            log::info!("Setting workqueue masks...");
            tweaks.write("/sys/devices/virtual/workqueue/cpumask", &other_cpu_mask)?;
            tweaks.write("/sys/bus/workqueue/devices/writeback/cpumask", &other_cpu_mask)?;

            log::info!("Setting up cpuset...");
            let cpuset = Cpuset::create(&mut tweaks, "gastool", cpus, max_cpu)?;

            // This can also be done by writing "off" to "/sys/devices/system/cpu/smt/control", but that's *much* slower as it turns it off for every core.
            log::info!("Disabling hyperthreading...");
            for core in cores {
                if cpus.contains(&core) {
                    continue;
                }

                log::info!("  Disabling CPU #{}...", core);
                tweaks.write(format!("/sys/devices/system/cpu/cpu{core}/online"), "0")?;
            }

            Some(cpuset)
        } else {
            None
        };

        log::info!("Disabling turbo boost...");
        tweaks.write("/sys/devices/system/cpu/cpufreq/boost", "0")?;

        log::info!("Setting the frequency governor to 'performance'...");
        for &cpu in cpus {
            tweaks.write(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"), "performance")?;
        }

        std::thread::sleep(core::time::Duration::from_millis(100));

        Ok(SystemSetup {
            cpus: cpus.to_owned(),
            _tweaks: tweaks,
            cpuset,
            amd_l3_type,
        })
    }

    pub fn add_process(&mut self, pid: i32) -> Result<(), String> {
        if let Some(ref mut cpuset) = self.cpuset {
            log::info!("Adding PID {pid} to cpuset...");
            cpuset.add_process(pid)?;
        }

        log::info!("Setting affinity for PID {pid}...");
        set_affinity(pid, self.cpus[0])?;
        log::info!("Setting realtime priority for PID {pid}...");
        set_realtime_priority(pid)?;

        Ok(())
    }
}

pub const PERF_TYPE_RAW: u32 = 4;

pub fn configure_perf_counter(pid: u32, cpu: usize, kind: u32, event: u64) -> Result<linux_raw::Fd, linux_raw::Error> {
    #[repr(C)]
    pub struct PerfEventAttr {
        pub kind: u32,
        pub size: u32,
        pub config: u64,
        pub sample_period_or_freq: u64,
        pub sample_type: u64,
        pub read_format: u64,
        pub flags: u64,
        pub wakeup_events_or_watermark: u32,
        pub bp_type: u32,
        pub bp_addr_or_config: u64,
        pub bp_len_or_config: u64,
        pub branch_sample_type: u64,
        pub sample_regs_user: u64,
        pub sample_stack_user: u32,
        pub clock_id: i32,
        pub sample_regs_intr: u64,
        pub aux_watermark: u32,
        pub sample_max_stack: u16,
        pub _reserved_1: u16,
        pub aux_sample_size: u32,
        pub aux_action: u32,
        pub sig_data: u64,
        pub config3: u64,
    }

    const PERF_FLAG_FD_CLOEXEC: usize = 1 << 3;

    let mut attr: PerfEventAttr = unsafe { core::mem::zeroed() };
    attr.size = core::mem::size_of::<PerfEventAttr>() as u32;
    attr.kind = kind;
    attr.config = event;
    let fd = unsafe {
        linux_raw::syscall!(
            linux_raw::SYS_perf_event_open,
            core::ptr::addr_of!(attr),
            pid,
            cpu,
            -1,
            PERF_FLAG_FD_CLOEXEC
        )
    };

    linux_raw::Error::from_syscall("perf_event_open", fd)?;
    let fd = linux_raw::Fd::from_raw_unchecked(fd as core::ffi::c_int);
    Ok(fd)
}
