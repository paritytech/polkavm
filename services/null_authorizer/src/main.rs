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
use utils::functions::{parse_accumulate_args, parse_refine_args};
use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{call_log};
use utils::host_functions::{
    gas
};
#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let (_wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
        if let Some(args) = parse_refine_args(start_address, length) {
            (
                args.wi_service_index,
                args.wi_payload_start_address,
                args.wi_payload_length,
                args.wphash,
            )
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };

    return (wi_payload_start_address, wi_payload_length);
}
#[polkavm_derive::polkavm_export]
extern "C" fn is_authorize(_start_address: u64, _length: u64) -> (u64, u64) {
    for i in 0..3 {
        let gas_result = unsafe { gas() };
        call_log(2, None, &format!("null_authorizer gas call {:?} gas_result: {:?}", i, gas_result));
    }
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}


#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}