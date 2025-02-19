#![no_std]
#![no_main]

extern crate alloc;

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
    pub fn write(ko: u64, kz: u64, bo: u64, bz: u64) -> u64;
    #[polkavm_import(index = 18)]
    pub fn fetch(start_address: u64, offset: u64, maxlen: u64, omega_10: u64, omega_11: u64, omega_12: u64) -> u64;
    #[polkavm_import(index = 19)]
    pub fn export(out: u64, out_len: u64) -> u64;
}

pub const NONE: u64 = u64::MAX;

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    let mut buffer = [0u8; 12];
    let offset: u64 = 0;
    let maxlen: u64 = buffer.len() as u64;
    let result = unsafe { 
        fetch(
            buffer.as_mut_ptr() as u64, 
            offset,
            maxlen,
            5,
            0,
            0,
        )
    };

    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());

        let new_fib_n = fib_n + fib_n_minus_1;

        buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
        buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());

    } else {
        buffer[0..4].copy_from_slice(&1_u32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_u32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_u32.to_le_bytes());
    }

    unsafe {
        export(buffer.as_ptr() as u64, buffer.len() as u64);
    }
    let buffer_addr = buffer.as_ptr() as u64;
    let buffer_len = buffer.len() as u64;
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) buffer_len,
        );
    }
    buffer_addr
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    let omega_7: u64; // accumulate input start address
    let omega_8: u64; // accumulate input length
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            "mv {1}, a1",
            out(reg) omega_7,
            out(reg) omega_8,
        );
    }

    let key = [0u8; 1];
    let n: u64 = unsafe { ( *(omega_7 as *const u32)).into() }; 
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, omega_7, omega_8);
    }
    // Option<hash> test
    let hash_bytes: [u8; 32] = [1; 32];
    let mut buffer_addr: u64 = hash_bytes.as_ptr() as u64;
    let buffer_len: u64 = hash_bytes.len() as u64;

    if n % 3 == 0{
        // trigger PANIC 
        let buffer_addr = 0;
        unreachable!();
    } else if n % 2 == 0 {
        // Write to invalid memory address to obtain an empty hash
        let buffer_addr = 0;
    }

    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) buffer_len,
        );
    }
    buffer_addr
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    0
}
