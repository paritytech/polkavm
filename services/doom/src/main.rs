#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE0 : usize = (1 << 24) - (1 << 12);

// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0); // 2^24 - 1 - 4096, should not greater than 2^24 - 1 (16777215)


const SIZE1 : usize = (1 << 15) * (1 << 12);
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new(); // 2^24 - 1 - 4096, should not greater than 2^24 - 1 (16777215)

// use picoalloc::{Allocator, Size};
// static ALLOCATOR: Allocator = Allocator::new(Size::from_bytes_usize(SIZE1).unwrap());

use crate::vm::Vm;
use polkavm::ProgramBlob;

mod vm;

#[polkavm_derive::polkavm_import]
extern "C" {
    // general
    #[polkavm_import(index = 100)]
    pub fn log(level: u64, target: u64, target_len: u64, message: u64, message_len: u64) -> u64;

}

pub fn call_log(level: u64, target: Option<&str>, msg: &str) {
    let (target_address, target_length) = if let Some(target_str) = target {
        let target_bytes = target_str.as_bytes();
        (target_bytes.as_ptr() as u64, target_bytes.len() as u64)
    } else {
        (0, 0)
    };
    let msg_bytes = msg.as_bytes();
    let msg_address = msg_bytes.as_ptr() as u64;
    let msg_length = msg_bytes.len() as u64;
    unsafe { log(level, target_address, target_length, msg_address, msg_length) };
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    call_log(2, None, &format!("Refining..."));
    const DOOM_PROGRAM: &[u8] = include_bytes!("../roms/doom.polkavm");
    const DOOM_ROM: &[u8] = include_bytes!("../roms/doom1.wad");
    
    let blob = ProgramBlob::parse(DOOM_PROGRAM.into()).unwrap();

    call_log(2, None, &format!("Blob parsed"));

    let mut vm = Vm::from_blob(blob).unwrap();
    
    call_log(2, None, &format!("Vm created"));

    vm.initialize(DOOM_ROM).unwrap();

    call_log(2, None, &format!("Vm initialized"));

    // let mut keys: [isize; 256] = [0; 256];

    call_log(2, None, &format!("Starting Doom..."));

    loop {
        let Ok((width, height, frame)) = vm.run_for_a_frame() else {
            break;
        };

        call_log(2, None, &format!("frame: {:?}x{:?}, len={:?}", width, height, frame.len()));
    }
    (0 , 0)
}


#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(_start_address: u64, _length: u64) -> (u64, u64) {
    (0 , 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    (0 , 0)
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}