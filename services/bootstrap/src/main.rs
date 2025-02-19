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

    #[polkavm_import(index = 9)]
    pub fn new(o: u64, l: u64, g: u64, m: u64) -> u64;

    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, g: u64, out: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    let omega_7: u64; // refine input start address
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            out(reg) omega_7,
        );
    }

    let output_len: u64 = 32 + 4; // 32 bytes hash + 4 bytes code length
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) output_len,
        );
    }
    omega_7 + 4 // eliminate the first 4 bytes (workitem service index)
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

    let code_length_address: u64 = omega_7 + omega_8 - 4;
    let code_length: u64 = unsafe { ( *(code_length_address as *const u32)).into() }; 
    
    let omega_9: u64 = 100;  // g -  the minimum gas required in order to execute the Accumulate entry-point of the service's code
    let omega_10: u64 = 100; // m -  the minimum required for the On Transfer entry-point
    // new: create service host call
    let result = unsafe { new(omega_7, code_length, omega_9, omega_10) };
    let result_bytes: [u8; 8] = result.to_le_bytes();
    let storage_key: [u8; 4] = [0; 4];

    let omega_7: u64 = storage_key.as_ptr() as u64; 
    let omega_8: u64 = storage_key.len() as u64;
    let omega_9: u64 = result_bytes.as_ptr() as u64; // new service index bytes address
    let omega_10: u64 = 4; // service index is u32
    // write: create service host call
    unsafe { write(omega_7, omega_8, omega_9, omega_10) };

    // transfer some token to the new service

    let memo = [0u8; 128];
    let omega_7 = result;
    let omega_8: u64 = 9999999;
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    0
}
