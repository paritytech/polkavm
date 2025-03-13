#![no_std]
#![no_main]

use utils::constants::{FIRST_READABLE_ADDRESS};
use utils::functions::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::host_functions::{transfer};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
    let (_wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
    if let Some(args) = parse_refine_args(start_address, length)
    {
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
    // fetch all_accumulation_o
    let all_accumulation_o_start_address = start_address + 4 + 4; // 4 bytes time slot + 4 bytes service index
    let remaining_length = length - 4 - 4; // 4 bytes time slot + 4 bytes service index
    
    let (work_result_address, work_result_length) =
    if let Some(tuple) = parse_wrangled_operand_tuple(all_accumulation_o_start_address, remaining_length, 0)
    {
        (tuple.work_result_ptr, tuple.work_result_len)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    // get amount address
    let amount_address = work_result_address + work_result_length - 4;
    let memo = [0u8; 128];

    let reciver: u64 = unsafe { *(work_result_address as *const u32) as u64 }; // reciver
    let amount: u64 = unsafe { *(amount_address as *const u32) as u64 }; // amount to transfer
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    
    let result = unsafe { transfer(reciver, amount, omega_9, omega_10) };
    let result_bytes = result.to_le_bytes();
    let result_address = result_bytes.as_ptr() as u64;
    let result_length = result_bytes.len() as u64;

    return (result_address, result_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
