#![no_std]
#![no_main]

use utils::constants::{FIRST_READABLE_ADDRESS};
use utils::functions::{parse_refine_args, parse_accumulate_args};
use utils::host_functions::{read, write, oyield};

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
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
    if let Some(args) = parse_accumulate_args(start_address, length, 0)
    {
        (args.t, args.s, args.work_result_ptr, args.work_result_len)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    // get the two service index from the input
    let service0_bytes_start_addr: u64 = work_result_address; // 4 bytes service index
    let service1_bytes_start_addr: u64 = work_result_address + work_result_length - 4; // 4 bytes service index

    let buffer0 = [0u8; 12];
    let buffer1 = [0u8; 12];
    let key = [0u8; 1];
    let key_address = key.as_ptr() as u64;
    let key_length = key.len() as u64;
    let mut buffer = [0u8; 12];
    
    let service0: u64 = unsafe { ( *(service0_bytes_start_addr as *const u32)).into() }; 
    let service1: u64 = unsafe { ( *(service1_bytes_start_addr as *const u32)).into() }; 

    // read the two services' storage
    unsafe {
        read(service0, key_address, key_length, buffer0.as_ptr() as u64, 0 as u64, buffer0.len() as u64);
        read(service1, key_address, key_length, buffer1.as_ptr() as u64, 0 as u64, buffer1.len() as u64);
    }; 

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
    let output_bytes_address = output_bytes_32.as_ptr() as u64;
    let output_bytes_length = output_bytes_32.len() as u64;

    unsafe { oyield(output_bytes_address); }

    return (output_bytes_address, output_bytes_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
