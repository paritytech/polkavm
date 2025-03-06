#![no_std]
#![no_main]

use utils::{NONE};

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 99)]
    pub fn delay(seconds: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u32 {
    let seconds: u64 = unsafe { *(0xFEFF0004 as *const u32) as u64 };
    unsafe {
        delay(seconds);
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}
