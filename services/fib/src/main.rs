#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use simplealloc::SimpleAlloc;

#[global_allocator]
static ALLOCATOR: SimpleAlloc<4096> = SimpleAlloc::new();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 3)]
    pub fn write(ko: u32, kz: u32, bo: u32, bz: u32) -> u32;
    #[polkavm_import(index = 18)]
    pub fn import(import_index: u32, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 19)]
    pub fn export(out: *const u8, out_len: u32) -> u32;
}

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorized() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u32 {
    let mut buffer = [0u8; 12];
    let result = unsafe { import(0, buffer.as_mut_ptr(), buffer.len() as u32) };

    if result == 0 {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
    
        let new_fib_n = fib_n + fib_n_minus_1;
    
        buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
        buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());
    
    } else {
        buffer[0..4].copy_from_slice(&1_i32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_i32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_i32.to_le_bytes());
    }

    unsafe {
        export(buffer.as_mut_ptr(), buffer.len() as u32);
    }
    let buffer_addr = buffer.as_ptr() as u32;
    let buffer_len = buffer.len() as u32;
    unsafe {
        core::arch::asm!(
            "mv a3, {0}",
            "mv a4, {1}",
            in(reg) buffer_addr,
            in(reg) buffer_len,
        );
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    let key = [0u8; 1];
    let omega_9: u32 = 0xFEFF0000;
    let omega_10: u32 = 0xC;
    unsafe {
        write(key.as_ptr() as u32, key.len() as u32, omega_9, omega_10 as u32);
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}
