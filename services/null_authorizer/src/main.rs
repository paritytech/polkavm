#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;

const SIZE0 : usize = 0x10000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x10000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{call_log};
use utils::host_functions::{
    gas
};

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorize(_start_address: u64, _length: u64) -> (u64, u64) {
    for i in 0..3 {
        let gas_result = unsafe { gas() };
        call_log(2, None, &format!("null_authorizer gas call {:?} gas_result: {:?}", i, gas_result));
    }
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
