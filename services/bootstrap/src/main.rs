#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;

use alloc::format;

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
use utils::functions::{call_log, parse_accumulate_args, parse_accumulate_operand_args, parse_refine_args};
use utils::host_functions::{fetch, new, transfer, write};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
    let (_wi_index, _wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
        if let Some(args) = parse_refine_args(start_address, length) {
            (
                args.wi_index,
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

#[no_mangle]
static mut output_bytes_36: [u8; 36] = [0; 36];
static mut operand: [u8; 4104] = [0; 4104];

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse args
    let (_timeslot, _service_index, number_of_operands) = if let Some(args) = parse_accumulate_args(start_address, length) {
        (args.t, args.s, args.number_of_operands)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    let operand_ptr = unsafe { operand.as_ptr() as u64 };
    let ptr = unsafe { output_bytes_36.as_ptr() as u64 };

    for i in 0..number_of_operands {
        let operand_len = unsafe { fetch(operand_ptr, 0, 4104, 15, i.into(), 0) };

        let (output_ptr, output_len) = match parse_accumulate_operand_args(operand_ptr, operand_len) {
            Some(args) => (args.output_ptr, args.output_len),
            None => return (FIRST_READABLE_ADDRESS as u64, 0),
        };
        if output_len < 36 {
                call_log(
                    2,
                    None,
                    &format!("output_len ACC output_len={} is less than 36 bytes, returning error", output_len),
                );
            return (FIRST_READABLE_ADDRESS as u64, 0);
        }

        // Reconstruct a slice from the raw output_ptr and output_len
        let output_slice = unsafe { core::slice::from_raw_parts(output_ptr as *const u8, output_len as usize) };

        // Copy the first 36 bytes into output_bytes_36
        unsafe {
            output_bytes_36.copy_from_slice(&output_slice[..36]);
        }

        // len is the 32..36 of the output_bytes_36 which is output by refine result  
        let len: u32 = unsafe {
            u32::from_le_bytes(
                    output_bytes_36[32..36]      
                        .try_into()
                    .expect("slice length is exactly 4"),
            )
        };
       
        unsafe {
            call_log(2, None, &format!("createService output_bytes_address {} {:x?} len={}", ptr, output_bytes_36, len));
        } 
        let omega_9: u64 = 100;  // g
        let omega_10: u64 = 100;  // m
        let omega_11: u64 = 1024; // gratis f

        let result = unsafe { new(output_ptr, len as u64, omega_9, omega_10, omega_11) };
        let result_bytes = &result.to_le_bytes()[..4];
        // write result to storage
        let storage_key: [u8; 4] = (i as u32).to_le_bytes();
        unsafe {
            write(
                storage_key.as_ptr() as u64,
                storage_key.len() as u64,
                result_bytes.as_ptr() as u64,
                result_bytes.len() as u64,
            );
        }
        call_log(2, None, &format!("SERVICEID={} storage_key={:x?}", result, storage_key));
        let memo = [0u8; 128];
        unsafe {
            transfer(result, 500000, 100, memo.as_ptr() as u64);
        }
        let start = (i * 4) as usize;
        let end = ((i + 1) * 4) as usize;
        unsafe {
            output_bytes_36[start..end].copy_from_slice(result_bytes);
        }
    }
    (ptr, 32)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
