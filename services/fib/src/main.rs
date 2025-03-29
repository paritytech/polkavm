#![no_std]
#![no_main]

use utils::constants::{FIRST_READABLE_ADDRESS, NONE};
use utils::functions::{parse_accumulate_args, call_log};
use utils::host_functions::{export, fetch, write, gas};
extern crate alloc;
use alloc::format;

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    let mut buffer = [0u8; 12];
    let offset: u64 = 0;
    let maxlen: u64 = buffer.len() as u64;
    let result = unsafe { fetch(buffer.as_mut_ptr() as u64, offset, maxlen, 5, 0, 0) }; // fetch segment 0 from work item 0

    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());

        let new_fib_n = fib_n + fib_n_minus_1;

        buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
        buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());
    } else {
        buffer[0..4].copy_from_slice(&1_u32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_u32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_u32.to_le_bytes());
    }

    unsafe {
        export(buffer.as_ptr() as u64, buffer.len() as u64);
    }

    // set the output address to register a0 and output length to register a1
    let buffer_addr = buffer.as_ptr() as u64;
    let buffer_len = buffer.len() as u64;
    return (buffer_addr, buffer_len);
}

#[no_mangle]
static mut output_bytes_32: [u8; 32] = [0; 32];

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
    let _n: u64 = unsafe { (*(work_result_address as *const u32)).into() };
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let output_address: u64;
    let output_length: u64;
    unsafe {
        output_bytes_32[..work_result_length as usize]
            .copy_from_slice(&core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize));

        output_address = output_bytes_32.as_ptr() as u64;
        output_length = output_bytes_32.len() as u64;
    }

    return (output_address, output_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(start_address: u64, length: u64) -> (u64, u64) {
    call_log(2, None, &format!("FIB on_transfer: start_address={:?} length={:?}", start_address, length));
    // parse transfer args
    //let (timeslot, service_index, transfer_result_address, transfer_result_length) =
    //    if let Some(args) = parse_transfer_args(start_address, length) {
    //        (args.t, args.s, args.transfer_result_ptr, args.transfer_result_len)
    //    } else {
    //        return (FIRST_READABLE_ADDRESS as u64, 0);
    //    };

    //call_log(2, None, &format!("on_transfer: timeslot={:?} service_index={:?}", timeslot, service_index));
    // TODO: for each transfer, split into these fields but log the values of each tran
    //let sender_index : u32 = 0;
    //let receiver_index : u32 = 0;
    //let amount : u64 = 0;
    // memo_ptr  128 bytes
    //let gas_limit : u64 = 0;
    //call_log(2, None, &format!("on_transfer: timeslot={:?} service_index={:?}, receiver_index={:?} amount={:?} gas_limit={:?} transfer_result_length={:?} ", timeslot, service_index, receiver_index, amount, gas_limit, transfer_result_length));

    // TODO: write (key=senderindex, value=memo_ptr) 
    // unsafe {        write(sender_index_ptr, 4, memo_ptr, 128);     }


    return (FIRST_READABLE_ADDRESS as u64, 0);
}
