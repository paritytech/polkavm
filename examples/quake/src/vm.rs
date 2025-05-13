use crate::machine::PolkaMachine;
use alloc::borrow::Cow;
use core::ffi::CStr;
use core::mem::MaybeUninit;
use polkakernel::FileSystem;
use polkakernel::InMemoryFileSystem;
use polkakernel::Kernel;
use polkakernel::Machine;
use polkakernel::StdEnv;
use polkakernel::SyscallOutcome;
use polkavm::{Config, Engine, GasMeteringKind, InterruptKind, Module, ModuleConfig, ProgramBlob, ProgramCounter, RawInstance, Reg};

pub struct Vm {
    start: ProgramCounter,
    kernel: Kernel<PolkaMachine, StdEnv, InMemoryFileSystem>,
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
            kernel: Kernel::new(PolkaMachine(instance), StdEnv, Default::default()),
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

    fn instance(&self) -> &RawInstance {
        &self.kernel.machine().0
    }

    fn instance_mut(&mut self) -> &mut RawInstance {
        &mut self.kernel.machine_mut().0
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

    pub fn register_file(&mut self, path: &CStr, blob: Cow<'static, [u8]>) {
        self.kernel.fs_mut().write_file(path, blob);
    }

    pub fn setup<'a, I>(&mut self, argv: I) -> Result<(), String>
    where
        I: IntoIterator<Item = &'a CStr>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let default_sp = self.instance().module().default_sp();
        if let Err(e) = self.kernel.machine_mut().init(default_sp, polkavm::RETURN_TO_HOST, argv, []) {
            return Err(e.to_string());
        }
        let start = self.start;
        self.instance_mut().set_next_program_counter(start);
        Ok(())
    }

    pub fn run(&mut self) -> Result<Interruption, String> {
        self.instance_mut().set_gas(200000000);
        loop {
            eprintln!("Heap size {}", self.instance().heap_size());
            #[allow(clippy::redundant_guards)] // Disable buggy lint.
            match self.instance_mut().run()? {
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_set_palette => {
                    let address = self.instance().reg(Reg::A0);
                    log::debug!("Set palette called: 0x{:x}", address);
                    let palette = self.instance().read_memory(address as u32, 256 * 3)?;
                    return Ok(Interruption::SetPalette { palette });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_display => {
                    let width = self.instance().reg(Reg::A0);
                    let height = self.instance().reg(Reg::A1);
                    let address = self.instance().reg(Reg::A2);
                    log::trace!("Display called: {}x{}, 0x{:x}", width, height, address);
                    let framebuffer = self.instance().read_memory(address as u32, (width * height) as u32)?;
                    return Ok(Interruption::Display {
                        width,
                        height,
                        framebuffer,
                    });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_fetch_inputs => {
                    let address = self.instance().reg(Reg::A0);
                    let mut remaining = self.instance().reg(Reg::A1) as usize;
                    let mut written = 0;
                    let range_1 = self.input_events_head
                        ..self.input_events_head
                            + remaining
                                .min(self.input_events_count)
                                .min(self.input_events.len() - self.input_events_head);
                    let slice = unsafe {
                        core::slice::from_raw_parts(
                            self.input_events[range_1.clone()].as_ptr().cast::<u8>(),
                            range_1.len() * core::mem::size_of::<InputEvent>(),
                        )
                    };
                    self.instance_mut().write_memory(address as u32, slice)?;
                    self.input_events_head += range_1.len();
                    self.input_events_count -= range_1.len();
                    remaining -= range_1.len();
                    written += range_1.len();
                    if self.input_events_head == self.input_events.len() {
                        self.input_events_head = 0;
                    }

                    let range_2 = self.input_events_head..self.input_events_head + self.input_events_count.min(remaining);
                    let slice = unsafe {
                        core::slice::from_raw_parts(
                            self.input_events[range_2.clone()].as_ptr().cast::<u8>(),
                            range_2.len() * core::mem::size_of::<InputEvent>(),
                        )
                    };
                    self.instance_mut().write_memory(address as u32, slice)?;
                    self.input_events_head += range_2.len();
                    written += range_2.len();

                    self.instance_mut().set_reg(Reg::A0, written as u64);
                    continue;
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_init_audio => {
                    let channels = self.instance().reg(Reg::A0) as u32;
                    let bits_per_sample = self.instance().reg(Reg::A1);
                    let sample_rate = self.instance().reg(Reg::A2) as u32;
                    if bits_per_sample != 16 {
                        self.instance_mut().set_reg(Reg::A0, 0);
                        continue;
                    }

                    self.audio_channels = channels;
                    self.instance_mut().set_reg(Reg::A0, 1);
                    return Ok(Interruption::AudioInit { channels, sample_rate });
                }
                InterruptKind::Ecalli(hostcall) if Some(hostcall) == self.import_output_audio => {
                    let address = self.instance().reg(Reg::A0);
                    let samples = self.instance().reg(Reg::A1) as usize;
                    let channels = self.audio_channels as usize;
                    let length = (samples * channels).min(1024 * 64); // Protect against huge sizes.
                    let mut buffer: Vec<i16> = Vec::with_capacity(length);
                    unsafe {
                        self.instance().read_memory_into(
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
                    match self.kernel.handle_syscall().expect("Failed to handle syscall") {
                        SyscallOutcome::Continue => {}
                        SyscallOutcome::Exit(code) => {
                            return if code == 0 {
                                Ok(Interruption::Exit)
                            } else {
                                Err(format!("Exited with code {code}"))
                            }
                        }
                    }
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
