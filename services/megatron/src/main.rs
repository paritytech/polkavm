#![no_std]
#![no_main]

use utils::{NONE, OK, PAGE_SIZE, SEGMENT_SIZE};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{read, write, oyield, log};
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

    // get the two service index from the input
    let service0_bytes_start_addr: u64 = work_result_address; // 4 bytes service index
    let service1_bytes_start_addr: u64 = work_result_address + work_result_length - 4; // 4 bytes service index

    let buffer0 = [0u8; 12];
    let buffer1 = [0u8; 12];
    let key = [0u8; 1];
    let mut buffer = [0u8; 12];

    let service0: u64 = unsafe { ( *(service0_bytes_start_addr as *const u32)).into() }; 
    let service1: u64 = unsafe { ( *(service1_bytes_start_addr as *const u32)).into() }; 

    // read the two services' storage
    unsafe {
        read(service0, key.as_ptr() as u64, key.len() as u64, buffer0.as_ptr() as u64, 0 as u64, buffer0.len() as u64);
        read(service1, key.as_ptr() as u64, key.len() as u64, buffer1.as_ptr() as u64, 0 as u64, buffer1.len() as u64);
    }
    let s0_n = u32::from_le_bytes(buffer0[0..4].try_into().unwrap());
    let s0_vn = u32::from_le_bytes(buffer0[4..8].try_into().unwrap());
    let s0_vnminus1 = u32::from_le_bytes(buffer0[8..12].try_into().unwrap());

    let s1_vn = u32::from_le_bytes(buffer1[4..8].try_into().unwrap());
    let s1_vnminus1 = u32::from_le_bytes(buffer1[8..12].try_into().unwrap());

    // calculate the new values and write to storage
    let m_n = s0_n;
    let m_vn = s0_vn + s1_vn;
    let m_vnminus1 = s0_vnminus1 + s1_vnminus1;

    buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
    buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
    buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, buffer.as_ptr() as u64, buffer.len() as u64);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..buffer.len()].copy_from_slice(&buffer);
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;

    unsafe { oyield(omega_7); }

    // set the result length to register a1
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

