#![deny(unreachable_patterns)]

extern crate alloc;

use crate::vm::{Interruption, Vm};
use core::time::Duration;
use polkavm::ProgramBlob;
use sdl2::event::Event;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rect::Rect;

mod keys;
mod vm;

fn get_queued_audio_frames(queue: &sdl2::audio::AudioQueue<i16>) -> f32 {
    let spec = queue.spec();
    let samples_per_frame = (spec.freq / 60) as u32;
    let samples_queued = queue.size() / (2 * u32::from(spec.channels));
    samples_queued as f32 / samples_per_frame as f32
}

fn main() {
    env_logger::init();

    let mut program_override = None;
    for arg in std::env::args().skip(1) {
        let bytes = std::fs::read(arg).unwrap();
        program_override = Some(bytes);
    }

    const DEFAULT_PROGRAM: &[u8] = include_bytes!("../code/quake.polkavm");

    let blob = ProgramBlob::parse(program_override.as_deref().unwrap_or(DEFAULT_PROGRAM).into()).unwrap();
    let mut vm = Vm::from_blob(blob).unwrap();

    vm.register_file("./pak0.pak", include_bytes!("../data/pak0.pak").as_slice().into());
    vm.register_file("./autoexec.cfg", include_bytes!("../data/autoexec.cfg").as_slice().into());

    vm.setup(["./quake"]).unwrap();

    let sdl_context = sdl2::init().unwrap();
    let video_context = sdl_context.video().unwrap();
    let audio_context = sdl_context.audio().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let window = video_context
        .window("polkaquake", 640, 400)
        .position_centered()
        .resizable()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();
    let mut texture = None;

    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut audio_queue: Option<sdl2::audio::AudioQueue<i16>> = None;
    let mut current_palette: Vec<u32> = Vec::new();
    current_palette.resize(256, 0xffffffff);

    const ONE_FRAME: Duration = Duration::from_nanos((Duration::from_secs(1).as_nanos() / 60) as u64);
    let mut next_frame = std::time::Instant::now() + ONE_FRAME;
    let mut framebuffer = Vec::new();

    canvas.window_mut().set_mouse_grab(true);
    'outer: loop {
        loop {
            while let Some(event) = event_pump.poll_event() {
                match event {
                    Event::Quit { .. } => {
                        std::process::exit(0);
                    }
                    Event::KeyDown {
                        scancode: Some(keycode),
                        repeat: false,
                        ..
                    } => crate::keys::from_sdl2_scancode(keycode).map(|key| vm.send_key(key, true)),
                    Event::KeyUp {
                        scancode: Some(keycode),
                        repeat: false,
                        ..
                    } => crate::keys::from_sdl2_scancode(keycode).map(|key| vm.send_key(key, false)),
                    Event::MouseButtonDown { mouse_btn, .. } => crate::keys::from_sdl2_mouse(mouse_btn).map(|key| vm.send_key(key, true)),
                    Event::MouseButtonUp { mouse_btn, .. } => crate::keys::from_sdl2_mouse(mouse_btn).map(|key| vm.send_key(key, false)),
                    Event::MouseWheel { direction, .. } => crate::keys::from_sdl2_wheel(direction).map(|key| {
                        vm.send_key(key, true);
                        vm.send_key(key, false);
                    }),
                    Event::MouseMotion { xrel, yrel, .. } => {
                        let window = canvas.window();
                        let size = window.size();
                        sdl_context
                            .mouse()
                            .warp_mouse_in_window(window, size.0 as i32 / 2, size.1 as i32 / 2);
                        sdl_context.mouse().show_cursor(false);
                        sdl_context.mouse().set_relative_mouse_mode(true);
                        let xrel = if xrel > i32::from(i8::MAX) {
                            i8::MAX
                        } else if xrel < i32::from(i8::MIN) {
                            i8::MIN
                        } else {
                            xrel as i8
                        };

                        let yrel = if yrel > i32::from(i8::MAX) {
                            i8::MAX
                        } else if yrel < i32::from(i8::MIN) {
                            i8::MIN
                        } else {
                            yrel as i8
                        };

                        vm.send_mouse_move(xrel, yrel);
                        Some(())
                    }
                    _ => None,
                };
            }

            if let Some(ref queue) = audio_queue {
                if get_queued_audio_frames(queue) < 2.0 {
                    break;
                }

                std::thread::sleep(core::time::Duration::from_millis(1));
            } else {
                break;
            }
        }

        let (width, height) = loop {
            match vm.run() {
                Ok(Interruption::Exit) => {
                    break 'outer;
                }
                Ok(Interruption::SetPalette { palette }) => {
                    for (color, entry) in palette.chunks_exact(3).zip(current_palette.iter_mut()) {
                        *entry = (color[2] as u32) | ((color[1] as u32) << 8) | ((color[0] as u32) << 16);
                    }
                    continue;
                }
                Ok(Interruption::Display {
                    width,
                    height,
                    framebuffer: raw_framebuffer,
                }) => {
                    assert_eq!(current_palette.len(), 256);

                    framebuffer.clear();
                    framebuffer.reserve(width as usize * height as usize);
                    for byte in raw_framebuffer {
                        framebuffer.push(current_palette[byte as usize]);
                    }
                    break (width as u32, height as u32);
                }
                Ok(Interruption::AudioInit { channels, sample_rate }) => {
                    let buffer_size = ((1.0 / 60.0 * sample_rate as f32 * 2.0) as u32).next_power_of_two() as u16;
                    eprintln!("Sample rate: {sample_rate}");
                    eprintln!("Channels: {channels}");
                    eprintln!("Audio buffer size: {buffer_size}");

                    let queue = audio_context
                        .open_queue::<i16, Option<&str>>(
                            None,
                            &sdl2::audio::AudioSpecDesired {
                                freq: Some(sample_rate as i32),
                                channels: Some(channels as u8),
                                samples: Some(256),
                            },
                        )
                        .unwrap();

                    queue.resume();
                    audio_queue = Some(queue);
                }
                Ok(Interruption::AudioFrame { buffer }) => {
                    if let Some(ref mut queue) = audio_queue {
                        if get_queued_audio_frames(queue) < 300.0 {
                            let _ = queue.queue_audio(&buffer);
                        }
                    }
                }
                Err(error) => {
                    eprintln!("ERROR: {error}");
                    break 'outer;
                }
            }
        };

        canvas.clear();
        if !framebuffer.is_empty() {
            if let Some((_, texture_width, texture_height)) = texture {
                if width != texture_width || height != texture_height {
                    texture = None;
                }
            }

            let (texture, tex_width, tex_height) = if let Some((ref mut texture, width, height)) = texture {
                (texture, width, height)
            } else {
                let tex = texture_creator
                    .create_texture_streaming(PixelFormatEnum::ARGB8888, width, height)
                    .unwrap();

                texture = Some((tex, width, height));
                (&mut texture.as_mut().unwrap().0, width, height)
            };

            let (display_width, display_height) = canvas.output_size().unwrap();
            let aspect = tex_width as f32 / tex_height as f32;
            let out_width = core::cmp::min(display_width, (display_height as f32 * aspect) as u32);

            let framebuffer = unsafe { core::slice::from_raw_parts(framebuffer.as_ptr().cast::<u8>(), framebuffer.len() * 4) };
            texture.update(None, framebuffer, width as usize * 4).unwrap();
            canvas
                .copy(
                    texture,
                    None,
                    Some(Rect::new(((display_width - out_width) / 2) as i32, 0, out_width, display_height)),
                )
                .unwrap();
        }

        canvas.present();

        if audio_queue.is_none() {
            let timestamp = std::time::Instant::now();
            if timestamp > next_frame {
                next_frame = timestamp + ONE_FRAME;
            } else {
                std::thread::sleep(next_frame - timestamp);
                next_frame += ONE_FRAME;
            }
        }
    }
}
