#![no_std]
#![no_main]

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{parse_accumulate_args, parse_refine_args};
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

#[no_mangle]
static mut OUTPUT_BUFFER: [u8; 32] = [0; 32];

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
        if let Some(args) = parse_accumulate_args(start_address, length, 0) {
            (args.t, args.s, args.work_result_ptr, args.work_result_len)
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };

    // Work result here should contain 32 bytes hash and 4 bytes code length
    let code_length_address: u64 = work_result_address + work_result_length - 4;
    let code_length: u64 = unsafe { (*(code_length_address as *const u32)).into() };

    let omega_9: u64 = 100; // g -  the minimum gas required in order to execute the Accumulate entry-point of the service's code
    let omega_10: u64 = 100; // m -  the minimum required for the On Transfer entry-point
    // create new service with host New
    let result = unsafe { new(work_result_address, code_length, omega_9, omega_10) };
    let result_bytes = &result.to_le_bytes()[..4];

    // write the new service index to the storage
    let storage_key: [u8; 4] = [0; 4];
    let omega_7: u64 = storage_key.as_ptr() as u64;
    let omega_8: u64 = storage_key.len() as u64;
    let omega_9: u64 = result_bytes.as_ptr() as u64; // new service index bytes address
    let omega_10: u64 = result_bytes.len() as u64; // new service index bytes length
    unsafe { write(omega_7, omega_8, omega_9, omega_10) };

    // transfer some token to the new service
    let memo = [0u8; 128];
    let omega_7 = result; // receiver
    let omega_8: u64 = 500000; // amount
    let omega_9: u64 = 100; // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };

    // Option<hash> test
    // pad result to 32 bytes
    unsafe {
        OUTPUT_BUFFER[..result_bytes.len()].copy_from_slice(&result_bytes);
        let output_bytes_address: u64 = OUTPUT_BUFFER.as_ptr() as u64;
        let output_bytes_length: u64 = OUTPUT_BUFFER.len() as u64;
        return (output_bytes_address, output_bytes_length);
    }
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
