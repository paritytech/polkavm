#![no_std]
#![no_main]

use utils::{NONE, OK, PAGE_SIZE, SEGMENT_SIZE};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{call_info, setup_page, get_page, serialize_gas_and_registers, deserialize_gas_and_registers};

use utils::{transfer};

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

    // set the output address to register a0 and output length to register a1
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

    // get amount address
    let amount_address = work_result_address + work_result_length - 4;
    let memo = [0u8; 128];

    let reciver: u64 = unsafe { *(work_result_address as *const u32) as u64 }; // reciver
    let amount: u64 = unsafe { *(amount_address as *const u32) as u64 }; // amount to transfer
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    
    let result = unsafe { transfer(reciver, amount, omega_9, omega_10) };
    result
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    // let mut omega_7: u64 = u64::MAX; // 2^64 - 1
    // let omega_8: u64 = 0xFEFDE000; 

    // let result = unsafe { info(omega_7, omega_8) };

    // unsafe {
    //     let ptr1 = 0xFEFDE029 as *mut u32; // 2^32 − 2*ZZ − ZI − P (s) (Writable address)
    //     *ptr1 = 0; // set storage key = {0,0,0,0}
    // }

    // let ko: u64 = 0xFEFDE029;
    // let kz: u64 = 4; // storage key = {0,0,0,0}
    // let vo: u64 = 0xFEFDE021;
    // let vz: u64 = 8; // service's ba;ance
    // // balance to storage
    // let result = unsafe { write(ko, kz, vo, vz) };
    // result
    0
}
