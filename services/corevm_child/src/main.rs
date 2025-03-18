#![no_std]
#![no_main]
#![feature(asm_const)]

use polkavm_derive::min_stack_size;
min_stack_size!(40960);

use utils::constants::{PAGE_SIZE, FIRST_READABLE_ADDRESS};

#[polkavm_derive::polkavm_export]
extern "C" fn main(num_pages: u64) -> (u64, u64) {
    let mut sum: u32 = 0;
    unsafe {
        for i in 0..num_pages {
            let addr = (FIRST_READABLE_ADDRESS as u64) + i * PAGE_SIZE;
            let value = core::ptr::read_volatile(addr as *const u32);
            sum = sum.wrapping_add(value);
        }
    }

    let out_ptr = FIRST_READABLE_ADDRESS as *mut u32;
    unsafe { 
        core::ptr::write_volatile(out_ptr, sum);
    }
    return (FIRST_READABLE_ADDRESS as u64, PAGE_SIZE)
}
