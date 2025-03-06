#![no_std]
#![no_main]

use utils::{NONE};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{write, new, transfer, log};
use utils::{call_info};

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    // read the input start address and length from register a0 and a1
    let omega_7: u64; // refine input start address
    let omega_8: u64; // refine input length

    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            "mv {1}, a1",
            out(reg) omega_7,
            out(reg) omega_8,
        );
    }

    // parse refine args
    let (wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
    if let Some(args) = parse_refine_args(omega_7, omega_8)
    {
        (
            args.wi_service_index,
            args.wi_payload_start_address,
            args.wi_payload_length,
            args.wphash,
        )
    } else {
        call_info("parse refine args failed");
        return NONE;
    };

    call_info("parse refine args success");
    
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) wi_payload_length,
        );
    }
    wi_payload_start_address
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    // read the input start address and length from register a0 and a1
    let omega_7: u64; // accumulate input start address
    let omega_8: u64; // accumulate input length

    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            "mv {1}, a1",
            out(reg) omega_7,
            out(reg) omega_8,
        );
    }
    // fetch service index
    let service_index_address = omega_7 + 4; // skip 4 bytes time slot
    let SERVICE_INDEX: u64   = unsafe { ( *(service_index_address as *const u32)).into() }; // 4 bytes service index

    // fetch all_accumulation_o
    let mut start_address = omega_7 + 4 + 4; // 4 bytes time slot + 4 bytes service index
    let mut remaining_length = omega_8 - 4 - 4; // 4 bytes time slot + 4 bytes service index
    
    let (work_result_address, work_result_length) =
    if let Some(tuple) = parse_wrangled_operand_tuple(start_address, remaining_length, 0)
    {
        (tuple.work_result_ptr, tuple.work_result_len)
    } else {
        return NONE;
    };

    // Work result here should contain 32 bytes hash and 4 bytes code length
    let code_length_address: u64 = work_result_address + work_result_length - 4;
    let code_length: u64 = unsafe { ( *(code_length_address as *const u32)).into() }; 
    
    let omega_9: u64 = 100;  // g -  the minimum gas required in order to execute the Accumulate entry-point of the service's code
    let omega_10: u64 = 100; // m -  the minimum required for the On Transfer entry-point
    // create new service with host New
    let result = unsafe { new(work_result_address, code_length, omega_9, omega_10) };
    let result_bytes: [u8; 8] = result.to_le_bytes();

    // write the new service index to the storage
    let storage_key: [u8; 4] = [0; 4];
    let omega_7: u64 = storage_key.as_ptr() as u64; 
    let omega_8: u64 = storage_key.len() as u64;
    let omega_9: u64 = result_bytes.as_ptr() as u64; // new service index bytes address
    let omega_10: u64 = 4; // service index is u32
    unsafe { write(omega_7, omega_8, omega_9, omega_10) };

    // transfer some token to the new service
    let memo = [0u8; 128];
    let omega_7 = result; // receiver
    let omega_8: u64 = 500000; // amount
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..result_bytes.len()].copy_from_slice(&result_bytes);
    let omega_7: u64 = output_bytes_32.as_ptr() as u64;
    let omega_8: u64 = output_bytes_32.len() as u64;

    // set the result address to register a0 and set the result length to register a1
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) omega_8,
        );
    }
    // this equals to a0 = omega_7
    omega_7
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    0
}
