#![no_std]
#![no_main]

use utils::constants::{NONE, FIRST_READABLE_ADDRESS};
use utils::functions::{parse_accumulate_args};
use utils::host_functions::{write, fetch, export};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    let mut buffer = [0u8; 16];
    let offset: u64 = 0;
    let maxlen: u64 = buffer.len() as u64;
    let result = unsafe { 
        fetch(
            buffer.as_mut_ptr() as u64, 
            offset,
            maxlen,
            5,
            0,
            0,
        )
    };

    // let mut prev_n : u32 = 0;
    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let t_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let t_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
        let t_n_minus_2 = u32::from_le_bytes(buffer[12..16].try_into().unwrap());
        // prev_n = n;

        let new_t_n = t_n + t_n_minus_1 + t_n_minus_2;

        let n_new = n + 1;

        buffer[0..4].copy_from_slice(&n_new.to_le_bytes());
        buffer[4..8].copy_from_slice(&new_t_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&t_n.to_le_bytes());
        buffer[12..16].copy_from_slice(&t_n_minus_1.to_le_bytes());
    } else {
        buffer[0..4].copy_from_slice(&1_i32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_i32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_i32.to_le_bytes());
        buffer[12..16].copy_from_slice(&0_i32.to_le_bytes());
    }

    unsafe {
        export(buffer.as_ptr() as u64, buffer.len() as u64);
    }
    
    // set the output address to register a0 and output length to register a1
    let buffer_addr = buffer.as_ptr() as u64;
    let buffer_len = buffer.len() as u64;

    // unsafe {
    //     export(buffer_addr, buffer_len);
    // }

    // Put N additional exports which are identical FOR NOW
    // for i in 0..prev_n {
    //     buffer[16..20].copy_from_slice(&(i + 1).to_le_bytes());
    //     unsafe {
    //         export(buffer_addr, buffer_len);
    //     }
    // }

    return (buffer_addr, buffer_len);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64)  {
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
    if let Some(args) = parse_accumulate_args(start_address, length, 0)
    {
        (args.t, args.s, args.work_result_ptr, args.work_result_len)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    // write TRIB result to storage
    let key = [0u8; 1];
    let _n: u64 = unsafe { ( *(work_result_address as *const u32)).into() }; 
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize].copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let output_bytes_address = output_bytes_32.as_ptr() as u64;
    let output_bytes_length = output_bytes_32.len() as u64;
    
    return (output_bytes_address, output_bytes_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
