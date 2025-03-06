#![no_std]
#![no_main]

use utils::{PAGE_SIZE};

#[polkavm_derive::polkavm_export]
extern "C" fn main() -> u64 {
    let omega_7: u64;
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            out(reg) omega_7,
        );
    }

    let mut buffer = [0u8; PAGE_SIZE as usize];
    let buffer_addr = buffer.as_mut_ptr() as u64;
    let buffer_len = buffer.len() as u64;

    let mut sum: u32 = 0;

    for i in 0..omega_7.min(3) {
        let source = unsafe {
            core::slice::from_raw_parts((i * PAGE_SIZE) as *const u8, PAGE_SIZE as usize)
        };

        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&source[..4]);

        sum = sum.wrapping_add(u32::from_le_bytes(bytes));
    }

    buffer[..4].copy_from_slice(&sum.to_le_bytes());

    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) buffer_len,
        );
    }
    buffer_addr
}
