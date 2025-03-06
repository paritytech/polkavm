#![no_std]
#![no_main]

use utils::{NONE};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{checkpoint, write, fetch, export};
use utils::{call_info};

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    let mut buffer = [0u8; 20];
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

    let mut prev_n : u32 = 0;
    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let t_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let t_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
        let t_n_minus_2 = u32::from_le_bytes(buffer[12..16].try_into().unwrap());
        prev_n = n;

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

    unsafe {
        export(buffer_addr, buffer_len);
    }

    // Put N additional exports which are identical FOR NOW
    for i in 0..prev_n {
        buffer[16..20].copy_from_slice(&(i + 1).to_le_bytes());
        unsafe {
            export(buffer_addr, buffer_len);
        }
    }



    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) buffer_len,
        );
    }
    // this equals to a0 = buffer_addr
    buffer_addr
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

    // write TRIB result to storage
    let key = [0u8; 1];
    let n: u64 = unsafe { ( *(work_result_address as *const u32)).into() }; 
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize].copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;

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
