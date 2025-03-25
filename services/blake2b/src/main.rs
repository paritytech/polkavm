#![no_std]
#![no_main]
#![feature(asm_const)]

// use core::convert::TryInto;
use polkavm_derive::min_stack_size;
min_stack_size!(8192); // depends on how many pages you need

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{parse_accumulate_args, parse_refine_args};
use utils::host_functions::write;
use utils::hash_functions::blake2b_hash;

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
    let (_wi_service_index, wi_payload_start_address, _wi_payload_length, _wphash) =
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

    let input_length_address: u64 = wi_payload_start_address;
    let input_length: u64 = unsafe { (*(input_length_address as *const u32)).into() }; // skip service index

    let input_address: u64 = wi_payload_start_address + 4;
    let input = unsafe { core::slice::from_raw_parts(input_address as *const u8, input_length as usize) };
    let output = blake2b_hash(input);

    let output_address = output.as_ptr() as u64;
    let output_length = output.len() as u64;
    (output_address, output_length)
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
        if let Some(args) = parse_accumulate_args(start_address, length, 0) {
            (args.t, args.s, args.work_result_ptr, args.work_result_len)
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };

    // write FIB result to storage
    let key = [0u8; 1];
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    (work_result_address, work_result_length)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
