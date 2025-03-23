#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use utils::functions::{call_log};
use alloc::format;

use polkavm_derive::min_stack_size;
min_stack_size!(40960);

use utils::constants::{FIRST_READABLE_ADDRESS, PAGE_SIZE};

#[polkavm_derive::polkavm_export]
extern "C" fn main(n: u64) -> (u64, u64) {
    let result: u32;
    if n < 2 {
        result = 1;     // fib(0) = 1, fib(1) = 1
        call_log(2, None, &format!("CHILD BASE n={:?} result={:?}", n, result));
    } else {
        // fib(n) = fib(n-2) + fib(n-1)
        unsafe {
            let addr2 = (FIRST_READABLE_ADDRESS as u64) + (n-2) * PAGE_SIZE;
            let fib_n_2 = core::ptr::read_volatile(addr2 as *const u32);
            let addr1 = (FIRST_READABLE_ADDRESS as u64) + (n-1) * PAGE_SIZE;
            let fib_n_1 = core::ptr::read_volatile(addr1 as *const u32);
            result = fib_n_2 + fib_n_1;
            call_log(2, None, &format!("CHILD NORM n={:?} fib_n_2={:?} (addr2={:?}) fib_n_1={:?} (addr1={:?}) result={:?}", n, fib_n_2, addr2, fib_n_1, addr1, result));
        }
    }
    
    let out_address = FIRST_READABLE_ADDRESS + (n * PAGE_SIZE) as u32;
    call_log(2, None, &format!("CHILD WRITE out_address={:?} result={:?}", out_address, result));
    unsafe {
        core::ptr::write_volatile(out_address  as *mut u32, result);
    }
    return (FIRST_READABLE_ADDRESS as u64, PAGE_SIZE);
}
