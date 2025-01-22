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
extern "C" fn is_authorized() -> u64 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    unsafe {
        core::arch::asm!(
            "li a3, 0xFEFF0004",
            "li a4, 0x24", 
        );
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    let mut omega_7: u64 = 0xFEFF0000; // CHECK
    let omega_8: u64 = unsafe { ( *(0xFEFF0020 as *const u64)).into() }; // get code length
    let omega_9: u64 = 100;  // g -  the minimum gas required in order to execute the Accumulate entry-point of the service's code
    let omega_10: u64 = 100; // m -  the minimum required for the On Transfer entry-point
    // new: create service host call
    let result = unsafe { new(omega_7, omega_8, omega_9, omega_10) };
    unsafe {
        let ptr1 = 0xFEFDE000 as *mut u64; // 2^32 − 2*ZZ − ZI − P (s) (Writable address)
        *ptr1 = 0;

        let ptr2 = 0xFEFDE004 as *mut u64; // 2^32 − 2*ZZ − ZI − P (s) + 4 (Writable address)
        *ptr2 = result;
    }
    let mut omega_7: u64 = 0xFEFDE000; // 2^32 − 2*ZZ − ZI − P (s) (Writable address), storage key {0,0,0,0}
    let omega_8: u64 = 4;
    let omega_9: u64 = 0xFEFDE004; // 2^32 − 2*ZZ − ZI − P (s) + 4 (Writable address), new service index u32
    let omega_10: u64 = 4;
    // write: create service host call
    unsafe { write(omega_7, omega_8, omega_9, omega_10) };

    // transfer some token to the new service

    let mut omega_7 = result;
    let omega_8: u64 = 9999999;
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = 0xFEFDE000; // memo start address
    unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    0
}
