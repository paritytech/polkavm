#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::vec;
use alloc::boxed::Box;
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
use utils::functions::{parse_accumulate_args, parse_refine_args, call_log};
use utils::host_functions::{new, transfer, write};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
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
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
        if let Some(args) = parse_accumulate_args(start_address, length, 0) {
            (args.t, args.s, args.work_result_ptr, args.work_result_len)
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };

    // call_log(2, None, &format!("BOOTSTRAP INIT t={} s={}", _timeslot, _service_index));

    let code_length_address: u64 = work_result_address + work_result_length - 4;
    let code_length: u64 = unsafe { (*(code_length_address as *const u32)).into() };

    let omega_9: u64 = 100;
    let omega_10: u64 = 100;
    let result = unsafe { new(work_result_address, code_length, omega_9, omega_10) };
    let result_bytes = &result.to_le_bytes()[..4];

    // write result to storage
    let storage_key: [u8; 4] = [0; 4];
    unsafe {
        write(
            storage_key.as_ptr() as u64,
            storage_key.len() as u64,
            result_bytes.as_ptr() as u64,
            result_bytes.len() as u64,
        );
    }

    // do transfer
    let memo = [0u8; 128];
    unsafe {
        transfer(result, 500000, 100, memo.as_ptr() as u64);
    }

    // allocate return buffer
    let mut buffer = Box::new([0u8; 32]);
    buffer[..4].copy_from_slice(result_bytes);
    let ptr = Box::into_raw(buffer) as u64; // leak the box to get a raw pointer

    /*call_log(
        2,
        None,
        &format!("RETURN acc output_bytes_address {} len {}", ptr, 32),
    ); */

    (ptr, 32)
}


#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}


