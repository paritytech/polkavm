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
    #[polkavm_import(index = 6)]
    pub fn assign(omega_7 : u64, omega_8 : u64)-> u64;
    #[polkavm_import(index = 18)]
    pub fn fetch(start_address: u64, offset: u64, maxlen: u64, omega_10: u64, omega_11: u64, omega_12: u64) -> u64;
    #[polkavm_import(index = 19)]
    pub fn export(out: u64, out_len: u64) -> u64;

}

pub const NONE: u64 = u64::MAX;

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {

    let mut authorization_hash = [0u8;32];
    //use BE to read the authorization hash "0x8c30f2c101674af1da31769e96ce72e81a4a44c89526d7d3ff0a1a511d5f3c9f"
    authorization_hash.copy_from_slice(&[
        0x8c, 0x30, 0xf2, 0xc1, 0x01, 0x67, 0x4a, 0xf1, 0xda, 0x31, 0x76, 0x9e, 0x96, 0xce, 0x72, 0xe8,
        0x1a, 0x4a, 0x44, 0xc8, 0x95, 0x26, 0xd7, 0xd3, 0xff, 0x0a, 0x1a, 0x51, 0x1d, 0x5f, 0x3c, 0x9f
    ]);
    let mut authorization_hashes = [0u8;32*80];
    for i in 0..80 {
        let start = i * 32;
        authorization_hashes[start..start + 32].copy_from_slice(&authorization_hash);
    }
    let hashes_address=(&authorization_hashes).as_ptr() as u64;
    unsafe{
        assign(0, hashes_address);
        assign(1, hashes_address);
    }
    let buffer_addr = authorization_hash.as_ptr() as u64;
    let buffer_len = authorization_hash.len() as u64;
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
