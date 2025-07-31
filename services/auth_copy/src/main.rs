#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
const SIZE0: usize = 0x10000;
use alloc::boxed::Box;
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
use utils::functions::{parse_accumulate_args, parse_accumulate_operand_args, parse_refine_args};
use utils::hash_functions::blake2b_hash;
use utils::host_functions::{assign, fetch}; // Added 'export' here

fn build_combined_input(input: &[u8], wi_payload_address: u64, wi_payload_length: usize) -> Vec<u8> {
    let wi_payload = unsafe { core::slice::from_raw_parts(wi_payload_address as *const u8, wi_payload_length) };

    let mut buffer = Vec::with_capacity(input.len() + wi_payload.len());
    buffer.extend_from_slice(input);
    buffer.extend_from_slice(wi_payload);
    buffer
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let operand_ptr = unsafe { operand.as_ptr() as u64 };
    // Datatype 8: Should be fetching 32 bytes of p_u
    let operand_len = unsafe { fetch(operand_ptr, 0, 4104, 8, 0, 0) };
    // TODO: our fetch resp doesn't have discriminator len yet but should, then requiring that we get it to match 14.10
    let input_slice = unsafe { core::slice::from_raw_parts(operand_ptr as *const u8, operand_len.try_into().unwrap()) };

    // use fetch to get 4 byte payload of work item 0
    let payload_4: [u8; 4] = [0u8; 4]; // 'a' value
    let payload_4_ptr = payload_4.as_ptr() as u64;
    // Datatype 13: Should be fetching 4 bytes for 'a'
    let payload_result0 = unsafe { fetch(payload_4_ptr, 0, 4, 13, 0, 0) };

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

    let output = blake2b_hash(input_slice);
    unsafe {
        output_bytes_36[0..32].copy_from_slice(&output[0..32]);
        output_bytes_36[32..36].copy_from_slice(&payload_4);
    }

    let output_address = unsafe { output_bytes_36.as_ptr() as u64 };
    let output_length = unsafe { output_bytes_36.len() as u64 };

    unsafe {
        call_log(
            2,
            None,
            &format!(
                "auth_copy ref input_slice={:x?} output_bytes_36={:x?} output_length={}",
                input_slice, output_bytes_36, output_length
            ),
        );
    }
    return (output_address, output_length);
}

const N_Q: usize = 80;
#[no_mangle]
static mut operand: [u8; 4104] = [0u8; 4104];
static mut output_bytes_36: [u8; 36] = [0u8; 36];
static mut authorization_hashes: [u8; 32 * N_Q] = [0u8; 32 * N_Q];

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    let output_bytes_36_ptr = unsafe { output_bytes_36.as_ptr() as u64 };
    let (_timeslot, _service_index, num_of_operands) = match parse_accumulate_args(start_address, length) {
        Some(args) => (args.t, args.s, args.number_of_operands),
        None => return (FIRST_READABLE_ADDRESS as u64, 0),
    };
    // fetch 36 byte output which will be (32 byte p_u_hash + 4 byte "a" from payload y)
    let operand_ptr = unsafe { operand.as_ptr() as u64 };
    let operand_len = unsafe { fetch(operand_ptr, 0, 4104, 15, 0, 0) };

    let (output_ptr, output_len) = match parse_accumulate_operand_args(operand_ptr, operand_len) {
        Some(args) => (args.output_ptr, args.output_len),
        None => return (FIRST_READABLE_ADDRESS as u64, 0),
    };
    // copy the last 36 bytes of the operand to output_bytes_36
    if output_len < 36 {
        unsafe {
            call_log(
                2,
                None,
                &format!("output_len ACC output_len={} is less than 36 bytes, returning error", output_len),
            );
        }
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    unsafe {
        // Reconstruct a slice from the raw output_ptr and output_len
        let output_slice = core::slice::from_raw_parts(output_ptr as *const u8, output_len as usize);

        // Copy the first 36 bytes into output_bytes_36
        output_bytes_36.copy_from_slice(&output_slice[..36]);

        let a = u32::from_le_bytes(output_bytes_36[32..36].try_into().unwrap());

        // Fill authorization_hashes with the first 32 bytes of output_bytes_36
        for i in 0..N_Q {
            authorization_hashes[i * 32..(i + 1) * 32].copy_from_slice(&output_bytes_36[..32]);
        }

        let authorization_hashes_address = authorization_hashes.as_ptr() as u64;
        assign(0, authorization_hashes_address, a as u64);
        assign(1, authorization_hashes_address, a as u64);
        call_log(2, None, &format!("assigned core 0+1 service {}", a));
    }
    (output_bytes_36_ptr, 32)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
