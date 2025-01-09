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

    #[polkavm_import(index = 1)]
    pub fn lookup(service: u32, hash_ptr: *const u8, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 2)]
    pub fn read(service: u32, key_ptr: *const u8, key_len: u32, out: *mut u8, out_len: u32) -> u32;

    #[polkavm_import(index = 3)]
    pub fn write(ko: u64, kz: u64, bo: u64, bz: u64) -> u64;

    #[polkavm_import(index = 4)]
    pub fn info(service: u64, out: u64) -> u64;
    
    #[polkavm_import(index = 9)]
    pub fn new(service: u32, hash_ptr: *const u8, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 10)]
    pub fn upgrade(out: *const u8, g: u64, m: u64) -> u32;
    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, g: u64, out: u64) -> u64;
    #[polkavm_import(index = 12)]
    pub fn quit(d: u32, a: u64, g: u64, out: *mut u8) -> u32;
    #[polkavm_import(index = 13)]
    pub fn solicit(hash_ptr: *const u8, z: u32) -> u32;
    #[polkavm_import(index = 14)]
    pub fn forget(hash_ptr: *const u8, z: u32) -> u32;
    #[polkavm_import(index = 15)]
    pub fn historical_lookup(service: u32, hash_ptr: *const u8, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 16)]
    pub fn import(import_index: u32, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 17)]
    pub fn export(out: *const u8, out_len: u32) -> u32;
    #[polkavm_import(index = 18)]
    pub fn machine(out: *const u8, out_len: u32) -> u32;
    #[polkavm_import(index = 19)]
    pub fn peek(out: *const u8, out_len: u32, i: u32) -> u32;
    #[polkavm_import(index = 20)]
    pub fn poke(n: u32, a: u32, b: u32, l: u32) -> u32;
    #[polkavm_import(index = 21)]
    pub fn invoke(n: u32, out: *mut u8) -> u32;
    #[polkavm_import(index = 22)]
    pub fn expunge(n: u32) -> u32;
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
            "li a4, 0x8", 
        );
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    let omega_7: u64 = unsafe { *(0xFEFF0000 as *const u32) as u64 }; // reciver
    let omega_8: u64 = unsafe { *(0xFEFF0004 as *const u32) as u64 }; // amount to transfer
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = 0xFEFDE000; // memo start address
    
    let result = unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };
    result
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    // let mut omega_7: u64 = u64::MAX; // 2^64 - 1
    // let omega_8: u64 = 0xFEFDE000; 

    // let result = unsafe { info(omega_7, omega_8) };

    // unsafe {
    //     let ptr1 = 0xFEFDE029 as *mut u32; // 2^32 − 2*ZZ − ZI − P (s) (Writable address)
    //     *ptr1 = 0; // set storage key = {0,0,0,0}
    // }

    // let ko: u64 = 0xFEFDE029;
    // let kz: u64 = 4; // storage key = {0,0,0,0}
    // let vo: u64 = 0xFEFDE021;
    // let vz: u64 = 8; // service's ba;ance
    // // balance to storage
    // let result = unsafe { write(ko, kz, vo, vz) };
    // result
    0
}
