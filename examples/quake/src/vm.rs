#![allow(non_upper_case_globals)]

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::mem::MaybeUninit;
use polkavm::{
    Config, Engine, GasMeteringKind, InterruptKind, MemoryAccessError, Module, ModuleConfig, ProgramBlob, ProgramCounter, RawInstance, Reg,
};

struct File {
    blob: Cow<'static, [u8]>,
}

struct Fd {
    file: Arc<File>,
    position: u64,
}

pub struct Vm {
    start: ProgramCounter,
    instance: RawInstance,
    filesystem: BTreeMap<Vec<u8>, Arc<File>>,
    fds: BTreeMap<u64, Fd>,
    next_fd: u64,
    input_events: Vec<InputEvent>,
    input_events_head: usize,
    input_events_count: usize,
    audio_channels: u32,

    import_syscall: Option<u32>,
    import_set_palette: Option<u32>,
    import_display: Option<u32>,
    import_fetch_inputs: Option<u32>,
    import_init_audio: Option<u32>,
    import_output_audio: Option<u32>,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct InputEvent {
    key: u8,
    value: u8,
}

const SYS_read: u64 = 63;
const SYS_readv: u64 = 65;
const SYS_writev: u64 = 66;
const SYS_exit: u64 = 93;
const SYS_openat: u64 = 56;
const SYS_lseek: u64 = 62;
const SYS_close: u64 = 57;
const SEEK_SET: u64 = 0;
const SEEK_CUR: u64 = 1;
const SEEK_END: u64 = 2;
const FILENO_STDOUT: u64 = 1;
const FILENO_STDERR: u64 = 2;
const ENOSYS: u64 = 38;
const EFAULT: u64 = 14;
const ENOENT: u64 = 2;
const EBADF: u64 = 9;
const EIO: u64 = 5;
const EACCES: u64 = 13;
const EINVAL: u64 = 22;
const AT_FDCWD: u64 = (-100_i64) as u64;
const IOV_MAX: u64 = 1024;
const O_WRONLY: u64 = 1;
const O_RDWR: u64 = 2;
const AT_PAGESZ: u64 = 6;

fn errno(error: u64) -> u64 {
    (-(error as i64)) as u64
}

pub enum Interruption {
    Exit,
    SetPalette { palette: Vec<u8> },
    Display { width: u64, height: u64, framebuffer: Vec<u8> },
    AudioInit { channels: u32, sample_rate: u32 },
    AudioFrame { buffer: Vec<i16> },
}

impl Vm {
    pub fn from_blob(blob: ProgramBlob) -> Result<Self, polkavm::Error> {
        let config = Config::from_env()?;
        let engine = Engine::new(&config)?;
        let mut module_config = ModuleConfig::new();
        module_config.set_gas_metering(Some(GasMeteringKind::Sync));
        let module = Module::from_blob(&engine, &module_config, blob)?;

        let start = module
            .exports()
            .find(|export| export.symbol() == "_pvm_start")
            .ok_or_else(|| "'_pvm_start' export not found".to_string())?
            .program_counter();

        let mut import_syscall = None;
        let mut import_set_palette = None;
        let mut import_display = None;
        let mut import_fetch_inputs = None;
        let mut import_init_audio = None;
        let mut import_output_audio = None;

        for (import_index, import) in module.imports().into_iter().enumerate() {
            let Some(import) = import else {
                continue;
            };

            let import_index = import_index as u32;
            match import.as_bytes() {
                b"pvm_syscall" => import_syscall = Some(import_index),
                b"pvm_set_palette" => import_set_palette = Some(import_index),
                b"pvm_display" => import_display = Some(import_index),
                b"pvm_fetch_inputs" => import_fetch_inputs = Some(import_index),
                b"pvm_init_audio" => import_init_audio = Some(import_index),
                b"pvm_output_audio" => import_output_audio = Some(import_index),
                _ => return Err(format!("unsupported import: {}", import).into()),
            }
        }

        let instance = module.instantiate()?;

        Ok(Self {
            start,
            instance,
            filesystem: BTreeMap::new(),
            fds: BTreeMap::new(),
            next_fd: 3,
            input_events: vec![InputEvent { key: 0, value: 0 }; 256],
            input_events_head: 0,
            input_events_count: 0,
            audio_channels: 0,
            import_syscall,
            import_set_palette,
            import_display,
            import_fetch_inputs,
            import_init_audio,
            import_output_audio,
        })
    }

    fn send_input_event(&mut self, key: u8, value: u8) {
        if key == crate::keys::MOUSE_X || key == crate::keys::MOUSE_Y {
            for nth in 0..self.input_events_count {
                let mut index = self.input_events_head + nth;
                if index >= self.input_events.len() {
                    index -= self.input_events.len();
                }

                if self.input_events[index].key == key {
                    self.input_events[index].value = (self.input_events[index].value as i8).saturating_add(value as i8) as u8;
                    return;
                }
            }
        }

        if self.input_events_count == self.input_events.len() {
            // Overflow.
            self.input_events_count -= 1;
            self.input_events_head += 1;
            if self.input_events_head == self.input_events.len() {
                self.input_events_head = 0;
            }
        }

        let mut index = self.input_events_head + self.input_events_count;
        if index >= self.input_events.len() {
            index -= self.input_events.len();
        }

        self.input_events[index] = InputEvent { key, value };
        self.input_events_count += 1;
    }

    pub fn send_key(&mut self, key: u8, is_pressed: bool) {
        self.send_input_event(key, if is_pressed { 1 } else { 0 });
    }

    pub fn send_mouse_move(&mut self, delta_x: i8, delta_y: i8) {
        if delta_x != 0 {
            self.send_input_event(crate::keys::MOUSE_X, delta_x as u8);
        }

        if delta_y != 0 {
            self.send_input_event(crate::keys::MOUSE_Y, delta_y as u8);
        }
    }

    fn read_cstr(&mut self, address: u64) -> Result<Option<Vec<u8>>, String> {
        // FIXME: This is slow.
        let mut buffer = Vec::new();
        for offset in 0..255 {
            match self.instance.read_u8((address + offset) as u32) {
                Ok(byte) => {
                    if byte == 0 {
                        return Ok(Some(buffer));
                    }

                    buffer.push(byte)
                }
                Err(MemoryAccessError::Error(error)) => return Err(error.into()),
                Err(MemoryAccessError::OutOfRangeAccess { .. }) => return Ok(None),
            }
        }

        Ok(None)
    }

    pub fn register_file(&mut self, path: &str, blob: Cow<'static, [u8]>) {
        let file = File { blob };

        // TODO: Support proper file trees.
        self.filesystem.insert(path.as_bytes().to_owned(), Arc::new(file));
    }

    fn handle_open(&mut self, path: &[u8], flags: u64) -> u64 {
        log::debug!("Open: path={:?}, flags=0x{:x}", String::from_utf8_lossy(path), flags);

        if let Some(file) = self.filesystem.get(path) {
            if (flags & (O_WRONLY | O_RDWR)) != 0 {
                log::trace!("  -> EACCES");
                return errno(EACCES);
            }

            let fd = self.next_fd;
            log::trace!("  -> fd={fd}");

            self.next_fd += 1;
            self.fds.insert(
                fd,
                Fd {
                    file: Arc::clone(file),
                    position: 0,
                },
            );

            return fd;
        }

        log::trace!("  -> ENOENT");
        errno(ENOENT)
    }

    fn handle_lseek(&mut self, fd: u64, offset: i64, whence: u64) -> u64 {
        log::trace!("Seek: fd={fd}, offset={offset}, whence={whence}");

        let Some(fd) = self.fds.get_mut(&fd) else {
            log::trace!("  -> BADF");
            return errno(EBADF);
        };

        match whence {
            SEEK_SET => {
                fd.position = offset as u64;
            }
            SEEK_CUR => {
                fd.position = core::cmp::min((fd.position as i64).wrapping_add(offset) as u64, fd.file.blob.len() as u64);
            }
            SEEK_END => {
                fd.position = core::cmp::min((fd.file.blob.len() as i64).wrapping_add(offset) as u64, fd.file.blob.len() as u64);
            }
            _ => {
                log::trace!("  -> EINVAL");
                return errno(EINVAL);
            }
        }

        log::trace!("  -> offset={}", fd.position);
        fd.position
    }

    fn handle_read(&mut self, fd: u64, address: u64, length: u64) -> Result<u64, String> {
        log::trace!("Read: fd={fd}, address=0x{address:x}, length={length}");

        let Some(fd) = self.fds.get_mut(&fd) else {
            log::trace!("  -> EBADF");
            return Ok(errno(EBADF));
        };

        if address.checked_add(length).is_none() || u32::try_from(address + length).is_err() {
            log::trace!("  -> EFAULT");
            return Ok(errno(EFAULT));
        }

        let Ok(address) = u32::try_from(address) else {
            log::trace!("  -> EFAULT");
            return Ok(errno(EFAULT));
        };

        let end = core::cmp::min(fd.position.wrapping_add(length), fd.file.blob.len() as u64);
        if fd.position >= end || fd.position >= fd.file.blob.len() as u64 {
            log::trace!("  -> offset={}, length=0", fd.position);
            return Ok(0);
        }

        let blob = &fd.file.blob[fd.position as usize..end as usize];
        match self.instance.write_memory(address, blob) {
            Ok(()) => {}
            Err(MemoryAccessError::Error(error)) => return Err(error.into()),
            Err(MemoryAccessError::OutOfRangeAccess { .. }) => {
                log::trace!("  -> EFAULT");
                return Ok(errno(EFAULT));
            }
        }

        let length_out = blob.len() as u64;
        log::trace!(
            "  -> offset={}, length={}, new offset={}",
            fd.position,
            length_out,
            fd.position + length_out
        );

        fd.position += length_out;
        Ok(length_out)
    }

    fn handle_write(&mut self, fd: u64, address: u64, length: u64) -> Result<u64, String> {
        if fd != FILENO_STDOUT && fd != FILENO_STDERR {
            return Ok(errno(EBADF));
        }

        if address.checked_add(length).is_none() || u32::try_from(address + length).is_err() {
            return Ok(errno(EFAULT));
        }

        let Ok(address) = u32::try_from(address) else {
            return Ok(errno(EFAULT));
        };

        let data = match self.instance.read_memory(address, length as u32) {
            Ok(data) => data,
            Err(MemoryAccessError::Error(error)) => return Err(error.into()),
            Err(MemoryAccessError::OutOfRangeAccess { .. }) => return Ok(errno(EFAULT)),
        };

        use std::io::Write;

        let result = if fd == FILENO_STDOUT {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout.write_all(&data)
        } else {
            let stderr = std::io::stderr();
            let mut stderr = stderr.lock();
            stderr.write_all(&data)
        };

        if let Err(error) = result {
            log::debug!("Error writing to stdout/stderr: {error}");
            return Ok(errno(EIO));
        };

        Ok(0)
    }

    fn handle_close(&mut self, fd: u64) -> u64 {
        log::debug!("Close: fd = {fd}");
        let Some(_fd) = self.fds.remove(&fd) else {
            log::trace!("  -> EBADF");
            return errno(EBADF);
        };

        0
    }

    #[allow(non_upper_case_globals)]
    pub fn setup<'a, I>(&mut self, argv: I) -> Result<(), String>
    where
        I: IntoIterator<Item = &'a str>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let argv = argv.into_iter();
        let argc = argv.len() as u64;
        let envp: &[&str] = &[];
        let envp_len = envp.len() as u64;
        let auxv: &[(u64, u64)] = &[(AT_PAGESZ, 4096)];
        let auxv_len = auxv.len() as u64;

        let mut sp = self.instance.module().default_sp();

        sp -= (1 + argc + 1 + envp_len + 1 + (auxv_len + 1) * 2) * 8;
        let address_init = sp;

        let mut p = sp;
        self.instance.write_u64(p as u32, argc)?;
        p += 8;

        for arg in argv {
            sp -= arg.len() as u64 + 1;
            self.instance.write_memory(sp as u32, arg.as_bytes())?;
            self.instance.write_u64(p as u32, sp)?;
            p += 8;
        }
        p += 8; // Null pointer.

        for arg in envp {
            sp -= arg.len() as u64 + 1;
            self.instance.write_memory(sp as u32, arg.as_bytes())?;
            self.instance.write_u64(p as u32, sp)?;
            p += 8;
        }
        p += 8; // Null pointer.

        for &(key, value) in auxv {
            self.instance.write_u64(p as u32, key)?;
            p += 8;
            self.instance.write_u64(p as u32, value)?;
            p += 8;
        }

        self.instance.set_reg(Reg::SP, sp);
        self.instance.set_reg(Reg::A0, address_init);
        self.instance.set_reg(Reg::RA, polkavm::RETURN_TO_HOST);
        self.instance.set_next_program_counter(self.start);
        Ok(())
    }

    pub fn run(&mut self) -> Result<Interruption, String> {
        self.instance.set_gas(200000000);
        'outer_loop: loop {
            #[allow(clippy::redundant_guards)] // Disable buggy lint.
            match self.instance.run()? {
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_set_palette => {
                    let address = self.instance.reg(Reg::A0);
                    log::debug!("Set palette called: 0x{:x}", address);
                    let palette = self.instance.read_memory(address as u32, 256 * 3)?;
                    return Ok(Interruption::SetPalette { palette });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_display => {
                    let width = self.instance.reg(Reg::A0);
                    let height = self.instance.reg(Reg::A1);
                    let address = self.instance.reg(Reg::A2);
                    log::trace!("Display called: {}x{}, 0x{:x}", width, height, address);
                    let framebuffer = self.instance.read_memory(address as u32, (width * height) as u32)?;
                    return Ok(Interruption::Display {
                        width,
                        height,
                        framebuffer,
                    });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_fetch_inputs => {
                    let address = self.instance.reg(Reg::A0);
                    let mut remaining = self.instance.reg(Reg::A1) as usize;
                    let mut written = 0;
                    let range_1 = self.input_events_head
                        ..self.input_events_head
                            + remaining
                                .min(self.input_events_count)
                                .min(self.input_events.len() - self.input_events_head);
                    self.instance.write_memory(address as u32, unsafe {
                        core::slice::from_raw_parts(
                            self.input_events[range_1.clone()].as_ptr().cast::<u8>(),
                            range_1.len() * core::mem::size_of::<InputEvent>(),
                        )
                    })?;
                    self.input_events_head += range_1.len();
                    self.input_events_count -= range_1.len();
                    remaining -= range_1.len();
                    written += range_1.len();
                    if self.input_events_head == self.input_events.len() {
                        self.input_events_head = 0;
                    }

                    let range_2 = self.input_events_head..self.input_events_head + self.input_events_count.min(remaining);
                    self.instance.write_memory(address as u32, unsafe {
                        core::slice::from_raw_parts(
                            self.input_events[range_2.clone()].as_ptr().cast::<u8>(),
                            range_2.len() * core::mem::size_of::<InputEvent>(),
                        )
                    })?;
                    self.input_events_head += range_2.len();
                    written += range_2.len();

                    self.instance.set_reg(Reg::A0, written as u64);
                    continue;
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_init_audio => {
                    let channels = self.instance.reg(Reg::A0) as u32;
                    let bits_per_sample = self.instance.reg(Reg::A1);
                    let sample_rate = self.instance.reg(Reg::A2) as u32;
                    if bits_per_sample != 16 {
                        self.instance.set_reg(Reg::A0, 0);
                        continue;
                    }

                    self.audio_channels = channels;
                    self.instance.set_reg(Reg::A0, 1);
                    return Ok(Interruption::AudioInit { channels, sample_rate });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_output_audio => {
                    let address = self.instance.reg(Reg::A0);
                    let samples = self.instance.reg(Reg::A1) as usize;
                    let channels = self.audio_channels as usize;
                    let length = (samples * channels).min(1024 * 64); // Protect against huge sizes.
                    let mut buffer: Vec<i16> = Vec::with_capacity(length);
                    unsafe {
                        self.instance.read_memory_into(
                            address as u32,
                            core::slice::from_raw_parts_mut(
                                buffer.spare_capacity_mut().as_mut_ptr().cast::<MaybeUninit<u8>>(),
                                length * core::mem::size_of::<i16>(),
                            ),
                        )?;
                        buffer.set_len(length);
                    }

                    return Ok(Interruption::AudioFrame { buffer });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_syscall => {
                    let syscall = self.instance.reg(Reg::A0);
                    let a1 = self.instance.reg(Reg::A1);
                    let a2 = self.instance.reg(Reg::A2);
                    let a3 = self.instance.reg(Reg::A3);
                    let a4 = self.instance.reg(Reg::A4);
                    let a5 = self.instance.reg(Reg::A5);
                    let pc = self.instance.program_counter().unwrap();
                    log::trace!(
                        "Syscall at pc={pc}: {syscall:>3}, args = [0x{a1:>016x}, 0x{a2:>016x}, 0x{a3:>016x}, 0x{a4:>016x}, 0x{a5:>016x}]"
                    );

                    match syscall {
                        SYS_read => {
                            let result = self.handle_read(a1, a2, a3)?;
                            self.instance.set_reg(Reg::A0, result);
                            continue;
                        }
                        SYS_readv => {
                            if a3 == 0 || a3 > IOV_MAX {
                                self.instance.set_reg(Reg::A0, errno(EINVAL));
                                continue;
                            }

                            let mut total_length = 0;
                            for n in 0..a3 {
                                let address = self.instance.read_u64(a2.wrapping_add(n * 16) as u32)?;
                                let length = self.instance.read_u64(a2.wrapping_add(n * 16).wrapping_add(8) as u32)?;
                                let errcode = self.handle_read(a1, address, length)?;
                                if (errcode as i64) < 0 {
                                    self.instance.set_reg(Reg::A0, errcode);
                                    continue 'outer_loop;
                                }

                                total_length += length;
                            }

                            self.instance.set_reg(Reg::A0, total_length);
                            continue;
                        }
                        SYS_writev => {
                            if a3 == 0 || a3 > IOV_MAX {
                                self.instance.set_reg(Reg::A0, errno(EINVAL));
                                continue;
                            }

                            let mut total_length = 0;
                            for n in 0..a3 {
                                let address = self.instance.read_u64(a2.wrapping_add(n * 16) as u32)?;
                                let length = self.instance.read_u64(a2.wrapping_add(n * 16).wrapping_add(8) as u32)?;
                                let errcode = self.handle_write(a1, address, length)?;
                                if (errcode as i64) < 0 {
                                    self.instance.set_reg(Reg::A0, errcode);
                                    continue 'outer_loop;
                                }

                                total_length += length;
                            }

                            self.instance.set_reg(Reg::A0, total_length);
                            continue;
                        }
                        SYS_exit => {
                            log::info!("Exit called: status={}", a1);
                            if a1 == 0 {
                                return Ok(Interruption::Exit);
                            } else {
                                return Err(format!("exit called with status: {a1}"));
                            }
                        }
                        SYS_openat => {
                            if a1 == AT_FDCWD {
                                let Some(path) = self.read_cstr(a2)? else {
                                    self.instance.set_reg(Reg::A0, errno(EFAULT));
                                    continue;
                                };

                                let result = self.handle_open(&path, a3);
                                self.instance.set_reg(Reg::A0, result);
                                continue;
                            }
                        }
                        SYS_lseek => {
                            let result = self.handle_lseek(a1, a2 as i64, a3);
                            self.instance.set_reg(Reg::A0, result);
                            continue;
                        }
                        SYS_close => {
                            let result = self.handle_close(a1);
                            self.instance.set_reg(Reg::A0, result);
                            continue;
                        }
                        _ => {
                            log::debug!("Unimplemented syscall at pc={pc}: {syscall:>3}, args = [0x{a1:>016x}, 0x{a2:>016x}, 0x{a3:>016x}, 0x{a4:>016x}, 0x{a5:>016x}]");
                        }
                    }

                    self.instance.set_reg(Reg::A0, errno(ENOSYS));
                }
                InterruptKind::Finished => {
                    return Ok(Interruption::Exit);
                }
                InterruptKind::Ecalli(hostcall) => {
                    return Err(format!("unsupported host call: {hostcall}"));
                }
                InterruptKind::Trap => {
                    return Err("execution trapped".into());
                }
                InterruptKind::NotEnoughGas => {
                    return Err("ran out of gas".into());
                }
                InterruptKind::Segfault(_) | InterruptKind::Step => unreachable!(),
            }
        }
    }
}
