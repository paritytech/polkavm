#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
const SIZE0: usize = 0x10000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1: usize = 0x10000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::call_log;
use utils::functions::parse_accumulate_args;
use utils::functions::parse_refine_args;
use utils::hash_functions::blake2b_hash;
use utils::host_functions::{assign, fetch};

fn build_combined_input(input: &[u8], wi_payload_address: u64, wi_payload_length: usize) -> Vec<u8> {
    let wi_payload = unsafe { core::slice::from_raw_parts(wi_payload_address as *const u8, wi_payload_length) };

    let mut buffer = Vec::with_capacity(input.len() + wi_payload.len());
    buffer.extend_from_slice(input);
    buffer.extend_from_slice(wi_payload);
    buffer
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let auth_output_address = start_address + length - 32;
    let input = unsafe { core::slice::from_raw_parts(auth_output_address as *const u8, 32) };

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

    if _wi_payload_length > 0 {
        let combined_input = build_combined_input(input, wi_payload_start_address, _wi_payload_length as usize);
        let input: &[u8] = &combined_input;
        call_log(2, None, &format!("auth_copy input={:x?}", input));
    } else {
        call_log(2, None, &format!("auth_copy input={:x?}", input));
    }

    // call_log(2, None, &format!("auth_copy start_address={:x?} load={:x?}", auth_output_address, input));
    let output = blake2b_hash(input);
    // call_log(2, None, &format!("auth_copy output={:x?}", output));
    let output_address = output.as_ptr() as u64;
    let output_length = output.len() as u64;
    // call_log(2, None, &format!("auth_copy output_address={:x?} output_length={}", output_address, output_length));
    return (output_address, output_length);
}

#[no_mangle]
static mut authorization_hashes: [u8; 32 * N_Q] = [0u8; 32 * N_Q];
const N_Q: usize = 80;

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    let (_timeslot, _service_index, _num) = if let Some(args) = parse_accumulate_args(start_address, length, 0) {
        (args.t, args.s, args.number_of_operands)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    let ptr = unsafe { output_bytes_32.as_ptr() as u64 };
    let result0 = unsafe { fetch(0, ptr, 8, 14, 0, 0) };
    unsafe {
        call_log(2, None, &format!("auth_copy {:?} result={}", output_bytes_32, result0));
    }

    unsafe {
        for i in 0..N_Q {
            let start = i * 32;
            authorization_hashes[start..start + 32].copy_from_slice(&output_bytes_32);
        }

        let authorization_hashes_address = authorization_hashes.as_ptr() as u64;
        assign(0, authorization_hashes_address);
        assign(1, authorization_hashes_address);

        (ptr, 32)
    }
}

static mut output_bytes_32: [u8; 32] = [0; 32];

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
