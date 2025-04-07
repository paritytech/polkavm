#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::string::String;
use alloc::format;

use polkavm_derive::min_stack_size;
min_stack_size!(409600);

use utils::constants::{FIRST_READABLE_ADDRESS, PAGE_SIZE, SEGMENT_SIZE};
use utils::constants::{NONE, HOST, LOG};

use utils::functions::{call_log};
use utils::functions::{initialize_pvm_registers, serialize_gas_and_registers, deserialize_gas_and_registers};
use utils::functions::{parse_standard_program_initialization_args, standard_program_initialization_for_child};
use utils::functions::{parse_accumulate_args, parse_refine_args};

use utils::host_functions::{solicit};
use utils::host_functions::{export, expunge, fetch, historical_lookup, invoke, machine, poke, peek, zero, log};


#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let (wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
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

    // fetch extrinsic
    let mut extrinsic = [0u8; 36];
    let extrinsic_address = extrinsic.as_mut_ptr() as u64;
    let code_hash_address = extrinsic_address;
    let extrinsic_length = extrinsic.len() as u64;
    unsafe {
        let _ = fetch(extrinsic_address, 0, extrinsic_length, 3, 0, 0);
    }

    if wi_payload_length < 8 {
        call_log(2, None, &format!("Parent: First time setup: wi_payload_length={:?}", wi_payload_length));
        return (extrinsic_address, extrinsic_length);
    }

    let step_n: u32 = unsafe { (*(wi_payload_start_address as *const u32)).into() };

    let num_of_gloders_address: u64 = wi_payload_start_address + 4;
    let num_of_gloders: u32 = unsafe { (*(num_of_gloders_address as *const u32)).into() };

    let total_execution_steps_address: u64 = wi_payload_start_address + 8;
    let total_execution_steps: u32 = unsafe { (*(total_execution_steps_address as *const u32)).into() };

    // fetch child VM blob
    let mut raw_blob = [0u8; 81920];
    let raw_blob_address = raw_blob.as_mut_ptr() as u64;
    let mut raw_blob_length = raw_blob.len() as u64;
    raw_blob_length = unsafe {
        historical_lookup(
            wi_service_index as u64,
            code_hash_address,
            raw_blob_address,
            0,
            raw_blob_length,
        )
    };
    if raw_blob_length == NONE {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    // parse raw blob
    let (z, s, o_bytes_address, o_bytes_length, w_bytes_address, w_bytes_length, c_bytes_address, c_bytes_length) =
        if let Some(blob) = parse_standard_program_initialization_args(raw_blob_address, raw_blob_length) {
            (
                blob.z,
                blob.s,
                blob.o_bytes_address,
                blob.o_bytes_length,
                blob.w_bytes_address,
                blob.w_bytes_length,
                blob.c_bytes_address,
                blob.c_bytes_length,
            )
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };
    call_log(2, None, &format!("Parent: z={:?} s={:?} o_bytes_address={:?} o_bytes_length={:?} w_bytes_address={:?} w_bytes_length={:?} c_bytes_address={:?} c_bytes_length={:?}",
        z, s, o_bytes_address, o_bytes_length, w_bytes_address, w_bytes_length, c_bytes_address, c_bytes_length));

    // new child VM
    let new_machine_idx = unsafe { machine(c_bytes_address, c_bytes_length, 0) };
    call_log(2, None, &format!("Parent: machine new index={:?}", new_machine_idx));

    // StandardProgramInitializationForChild
    standard_program_initialization_for_child(
        z,
        s,
        o_bytes_address,
        o_bytes_length,
        w_bytes_address,
        w_bytes_length,
        new_machine_idx as u32,
    );

    // fetch segments, poke child VM
    let mut segment_buf = [0u8; SEGMENT_SIZE as usize];
    let mut segment_index = 0u64;
    let segment_buf_segment_address = segment_buf.as_mut_ptr() as u64;
    let segment_buf_page_address = segment_buf_segment_address + 8;
    let mut m: u64;
    let mut page_id: u64;
    let mut first_page_address: u64 = 0;
    loop {
        let fetch_result = unsafe { fetch(segment_buf_segment_address as u64, 0, SEGMENT_SIZE as u64, 6, segment_index, 0) };
        if fetch_result == NONE {
            break;
        }
        call_log(2, None, &format!("Parent: fetch segment_index={:?} fetch_result={:?}", segment_index, fetch_result));
        (m, page_id) = (
            u32::from_le_bytes(segment_buf[0..4].try_into().unwrap()) as u64,
            u32::from_le_bytes(segment_buf[4..8].try_into().unwrap()) as u64,
        );

        if segment_index == 0 {
            first_page_address = page_id * PAGE_SIZE as u64;
        }

        let zero_result = unsafe { zero(m, page_id, 1) };
        call_log(2, None, &format!("Parent: zero m={:?}, page_id={:?} zero_result={:?}",  m, page_id, zero_result));

        // poke(machine n, source s, dest o, # bytes z)
        let s = segment_buf.as_mut_ptr() as u64;
        let page_address = page_id * PAGE_SIZE as u64;
        let poke_result = unsafe { poke(m, segment_buf_page_address, page_address, PAGE_SIZE) };
        call_log(2, None, &format!("Parent: poke m={:?} s={:?} o={:?} z={:?} poke_result={:?}", m, s, page_address, PAGE_SIZE, poke_result));

        segment_index += 1;
    }

    // invoke child VM
    let init_gas: u64 = 0x10000;
    let mut child_vm_registers = initialize_pvm_registers();
    child_vm_registers[7] = first_page_address;
    child_vm_registers[8] = segment_index * PAGE_SIZE as u64;
    child_vm_registers[9] = step_n as u64;
    child_vm_registers[10] = num_of_gloders as u64;
    child_vm_registers[11] = total_execution_steps as u64;
    call_log(2, None, &format!("Parent: child_vm_registers={:?}", child_vm_registers));

    let g_w = serialize_gas_and_registers(init_gas, &child_vm_registers);
    let g_w_address = g_w.as_ptr() as u64;

    let mut invoke_result: u64;
    let mut omega_8: u64;

    loop {
        call_log(2, None, &format!("Parent: invoke child VM, segment_index={:?}", segment_index));
        (invoke_result, omega_8) = unsafe { invoke(new_machine_idx as u64, g_w_address) };
        call_log(2, None, &format!("Parent: invoke {:?} invoke_result={:?} omega_8={:?}", new_machine_idx, invoke_result, omega_8));

        // only process hostlog
        if invoke_result == HOST && omega_8 == LOG {
            (_, child_vm_registers) = deserialize_gas_and_registers(&g_w);
            let level = child_vm_registers[7];
            let target_address = child_vm_registers[8];
            let target_length = child_vm_registers[9];
            let message_address = child_vm_registers[10];
            let message_length = child_vm_registers[11];

            let target_buffer = [0u8; 64];
            let target_buffer_address = target_buffer.as_ptr() as u64;

            let message_buffer = [0u8; 256];
            let message_buffer_address = message_buffer.as_ptr() as u64;

            unsafe { peek(new_machine_idx, target_buffer_address, target_address, target_length) };
            unsafe { peek(new_machine_idx, message_buffer_address, message_address, message_length) };

            unsafe {
                log(level, target_buffer_address, target_length, message_buffer_address, message_length);
            }
        } else {
            (_, child_vm_registers) = deserialize_gas_and_registers(&g_w);
            break;
        }
    }

    // peek child VM output and export it
    let output_start_address = child_vm_registers[7];

    for i in 0..9 {
        segment_buf.fill(0);
        segment_buf[0..4].copy_from_slice(&(new_machine_idx as u32).to_le_bytes());
        let page_address = output_start_address + i * PAGE_SIZE as u64;
        let page_id = page_address / PAGE_SIZE as u64;
        segment_buf[4..8].copy_from_slice(&(page_id as u32).to_le_bytes());
        unsafe { peek(new_machine_idx, segment_buf_page_address, page_address, PAGE_SIZE) };
        unsafe { export(segment_buf_segment_address, SEGMENT_SIZE) };
    }

    unsafe { expunge(new_machine_idx as u64) };

    call_log(2, None, &format!("Parent: done with child VM, segment_index={:?}", segment_index));
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
    if let Some(args) = parse_accumulate_args(start_address, length, 0) {
        (args.t, args.s, args.work_result_ptr, args.work_result_len)
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    // first time setup: do nothing but solicit code for child VM
    if work_result_length == 36 {
        let code_hash_address = work_result_address;
        let code_length_address = work_result_address + 32;
        let code_length: u64 = unsafe { (*(code_length_address as *const u32)).into() };
        unsafe { solicit(code_hash_address, code_length) };
        call_log(2, None, &format!("Parent: solicit code_hash_address={:?} code_length={:?}", code_hash_address, code_length));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }
    return (FIRST_READABLE_ADDRESS as u64, 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
