#![no_std]
#![no_main]

extern crate alloc;

use core::convert::TryInto;
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
    #[polkavm_import(index = 2)]
    pub fn read(service: u64, key_ptr: u64, key_len: u64, out: u64, out_len: u64) -> u64;
    #[polkavm_import(index = 3)]
    pub fn write(key_ptr: u64, key_len: u64, value: u64, value_len: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    unsafe {
        core::arch::asm!(
            "li a1, 0x8", 
        );
    }
    0xFEFF0004
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    let mut buffer0 = [0u8; 12];
    let mut buffer1 = [0u8; 12];
    let key = [0u8; 1];
    let mut buffer = [0u8; 12];

    unsafe {
        let service0 = ( *(0xFEFF0000 as *const u32)).into();
        let service1 = ( *(0xFEFF0004 as *const u32)).into();

        read(service0, key.as_ptr() as u64, key.len() as u64, buffer0.as_ptr() as u64, buffer0.len() as u64);
        read(service1, key.as_ptr() as u64, key.len() as u64, buffer1.as_ptr() as u64, buffer1.len() as u64);

        let s0_n = u32::from_le_bytes(buffer0[0..4].try_into().unwrap());
        let s0_vn = u32::from_le_bytes(buffer0[4..8].try_into().unwrap());
        let s0_vnminus1 = u32::from_le_bytes(buffer0[8..12].try_into().unwrap());

        let s1_vn = u32::from_le_bytes(buffer1[4..8].try_into().unwrap());
        let s1_vnminus1 = u32::from_le_bytes(buffer1[8..12].try_into().unwrap());

        let m_n = s0_n;
        let m_vn = s0_vn + s1_vn;
        let m_vnminus1 = s0_vnminus1 + s1_vnminus1;

        buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
        buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
        buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());

        write(key.as_ptr() as u64, key.len() as u64, buffer.as_ptr() as u64, buffer.len() as u64);
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    0
}
