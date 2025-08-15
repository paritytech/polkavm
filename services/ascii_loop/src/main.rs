#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::vec;
use alloc::string::String;

const SIZE0 : usize = 0x100000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x8000000; // 128MB
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::{FIRST_READABLE_ADDRESS, NONE};
use utils::functions::{call_log};
use utils::host_functions::{fetch};
fn encode_image(width: u16, height: u16, pixels: &[u8]) -> vec::Vec<u8> {
    let mut buf = vec::Vec::with_capacity(4 + pixels.len());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(pixels);
    buf
}

fn decode_image(data: &[u8]) -> Option<(u16, u16, &[u8])> {
    if data.len() < 4 {
        return None;
    }
    let width = u16::from_be_bytes([data[0], data[1]]);
    let height = u16::from_be_bytes([data[2], data[3]]);
    let expected = width as usize * height as usize * 3;
    if data.len() != 4 + expected {
        return None;
    }
    Some((width, height, &data[4..]))
}

fn bytes_to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}
static mut pic: [u8; 691200] = [0u8; 691200]; // 640*360*3
static mut gray_buffer: [u8; 230400] = [0u8; 230400]; // 640*360
static mut resized_buffer: [u8; 57600] = [0u8; 57600]; // 320*180  
static mut ascii_buffer: [u8; 57780] = [0u8; 57780]; // 320*180 + 180 newlines

// 配置常量
mod config {
    pub const SRC_WIDTH: u32 = 640;
    pub const SRC_HEIGHT: u32 = 360;
    pub const DST_WIDTH: u32 = SRC_WIDTH/4;
    pub const DST_HEIGHT: u32 = SRC_HEIGHT/5;
    pub const IMAGE_SIZE: usize = (SRC_WIDTH * SRC_HEIGHT * 3) as usize;
}

// 图像处理模块
mod image_processing {
    use super::config::*;
    use super::{gray_buffer, resized_buffer, ascii_buffer};

    /// 将RGB图像转换为灰度
    unsafe fn rgb_to_grayscale(image_bytes: &[u8]) -> usize {
        let mut gray_idx = 0;
        let max_pixels = (SRC_WIDTH * SRC_HEIGHT) as usize;
        let available_pixels = core::cmp::min(max_pixels, gray_buffer.len());
        
        for chunk in image_bytes.chunks_exact(3) {
            if gray_idx >= available_pixels {
                break;
            }
            let lum = ((chunk[0] as u32 * 30 + chunk[1] as u32 * 59 + chunk[2] as u32 * 11) / 100) as u8;
            gray_buffer[gray_idx] = lum;
            gray_idx += 1;
        }
        gray_idx
    }

    /// 最近邻插值缩放
    unsafe fn resize_image(gray_len: usize) -> usize {
        let mut resized_idx = 0;
        let max_resized = (DST_WIDTH * DST_HEIGHT) as usize;
        let available_resized = core::cmp::min(max_resized, resized_buffer.len());
        
        for y in 0..DST_HEIGHT {
            let src_y = y * SRC_HEIGHT / DST_HEIGHT;
            for x in 0..DST_WIDTH {
                if resized_idx >= available_resized {
                    break;
                }
                let src_x = x * SRC_WIDTH / DST_WIDTH;
                let i = (src_y * SRC_WIDTH + src_x) as usize;
                if i < gray_len && i < gray_buffer.len() {
                    resized_buffer[resized_idx] = gray_buffer[i];
                }
                resized_idx += 1;
            }
            if resized_idx >= available_resized {
                break;
            }
        }
        resized_idx
    }

    /// 将灰度图像转换为ASCII艺术
    unsafe fn grayscale_to_ascii() -> usize {
        let gradient = b"@%#*+=- . ";
        let mut ascii_idx = 0;
        
        for y in 0..DST_HEIGHT {
            for x in 0..DST_WIDTH {
                if ascii_idx >= ascii_buffer.len() {
                    break;
                }
                let resized_i = (y * DST_WIDTH + x) as usize;
                if resized_i < resized_buffer.len() {
                    let lum = resized_buffer[resized_i];
                    let idx = (lum as usize * (gradient.len() - 1)) / 255;
                    ascii_buffer[ascii_idx] = gradient[idx];
                    ascii_idx += 1;
                }
            }
            if ascii_idx < ascii_buffer.len() {
                ascii_buffer[ascii_idx] = b'\n';
                ascii_idx += 1;
            } else {
                break;
            }
        }
        ascii_idx
    }

    /// 完整的图像到ASCII转换流程
    pub fn convert_image_to_ascii(image_bytes: &[u8]) -> usize {
        unsafe {
            let gray_len = rgb_to_grayscale(image_bytes);
            let _resized_len = resize_image(gray_len);
            grayscale_to_ascii()
        }
    }
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    use config::*;
    use image_processing::convert_image_to_ascii;
    
    call_log(2, None, &format!("refine: start_address: {:x}, length: {}", start_address, length));
    
    // todo : use payload to change the algorithm behavior
    let ptr = unsafe { pic.as_ptr() as u64 };
    call_log(2, None, &format!("refine: ptr: {:x}", ptr));
    for _ in 0..1000000000{
        let fetch_result = unsafe { fetch(ptr, 0, IMAGE_SIZE as u64, 30, 0, 0) };
        call_log(2, None, &format!("fetch_result: {:x}", fetch_result));
        
        if fetch_result == NONE {
            call_log(2, None, "refine: fetch failed");
            return (FIRST_READABLE_ADDRESS as u64, 0);
        }
        
        call_log(2, None, &format!("Processing {}x{} image to {}x{} ASCII", 
            SRC_WIDTH, SRC_HEIGHT, DST_WIDTH, DST_HEIGHT));

        call_log(2, None, "Creating image slice...");
        // Safety: fetch_result is a pointer to the image buffer
        let image_bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, IMAGE_SIZE) };
        call_log(2, None, &format!("Image slice created, first bytes: {:?}", &image_bytes[..8]));
        
        call_log(2, None, "Starting ASCII conversion...");
        let ascii_len = convert_image_to_ascii(image_bytes);
        call_log(2, None, &format!("ASCII art generated, length: {}", ascii_len));
        
        // 使用log 10输出ASCII艺术
        let ascii_string = unsafe {
            core::str::from_utf8_unchecked(&ascii_buffer[..ascii_len])
        };
        call_log(10, None, ascii_string);
    }
    
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}