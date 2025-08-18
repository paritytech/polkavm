#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE0 : usize = 0x10000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x8000000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::{FIRST_READABLE_ADDRESS, OOG, LOG,HOST};
use utils::functions::{
    parse_accumulate_args, parse_refine_args, parse_standard_program_initialization_args,
    call_log, standard_program_initialization_for_child, initialize_pvm_registers,
    serialize_gas_and_registers,extract_memory_from_machine,write_memory_to_machine, deserialize_gas_and_registers
};
use utils::host_functions::{write, machine, invoke, expunge,peek, log};
use utils::hash_functions::blake2b_hash;

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
   const CHILDCODE: &[u8] = include_bytes!("../../blake2b_child/blake2b_child.pvm");
    let raw_blob_address = CHILDCODE.as_ptr() as u64;
    let raw_blob_length = CHILDCODE.len() as u64;

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
            call_log(1, None, &format!("Parent: parse_standard_program_initialization_args failed for raw_blob_address: {:?} raw_blob_length: {:?}", raw_blob_address, raw_blob_length));
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
    call_log(2, None, &format!("Parent: standard_program_initialization_for_child done"));

    // init 100 gas
    // invoke child VM
    let mut init_gas: u64 = 100;
    let mut child_vm_registers = initialize_pvm_registers();
    let g_w = serialize_gas_and_registers(init_gas, &child_vm_registers);
    let g_w_address = g_w.as_ptr() as u64;

    let mut invoke_result: u64;
    let mut omega_8: u64;
        loop {
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
            if invoke_result == OOG {
                call_log(1, None, &format!("Parent: invoke OOG, omega_8={:?}", omega_8));
            }
            call_log(2, None, &format!("Parent: exiting invoke loop invoke_result={:?} omega_8={:?}", invoke_result, omega_8));
            let mut gas;
            (gas, child_vm_registers) = deserialize_gas_and_registers(&g_w);
            // print the gas usage 
            gas = init_gas - gas;
            call_log(2, None, &format!("Parent: gas used={:?} child_vm_registers={:?}", gas, child_vm_registers));
            break;
        }
    }
    let extracted_memory = match utils::functions::extract_memory_from_machine(
        z,
        s,
        o_bytes_address,
        o_bytes_length,
        w_bytes_address,
        w_bytes_length,
        new_machine_idx as u32,
    ) {
        Ok(mem) => mem,
        Err(e) => {
            call_log(1, None, &format!("Parent: extract_memory_from_machine failed: {:?}", e));
            return (FIRST_READABLE_ADDRESS as u64, 0);
        }
    };
    (_, child_vm_registers) = deserialize_gas_and_registers(&g_w);
    let pc_counter = unsafe { expunge(new_machine_idx as u64) }; 
    let new_machine_idx_2 = unsafe { machine(c_bytes_address, c_bytes_length, pc_counter) };
    call_log(2, None, &format!("Parent: machine new index after expunge={:?}", new_machine_idx_2));
    
    if let Err(e) = write_memory_to_machine(&extracted_memory, new_machine_idx_2 as u32) {
        call_log(1, None, &format!("Parent: write_memory_to_machine failed: {:?}", e));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }
    call_log(2, None, &format!("Parent: write_memory_to_machine done"));
    init_gas = 0x0FFF_FFFF_FFFF_FFFF;
    let new_gw = serialize_gas_and_registers(init_gas, &child_vm_registers);
    let g_w_address = new_gw.as_ptr() as u64;
    // invoke child VM again
    loop {
        (invoke_result, omega_8) = unsafe { invoke(new_machine_idx_2 as u64, g_w_address) };
        call_log(2, None, &format!("Parent: invoke {:?} invoke_result={:?} omega_8={:?}", new_machine_idx_2, invoke_result, omega_8));
        // only process hostlog
        if invoke_result == HOST && omega_8 == LOG {
            (_, child_vm_registers) = deserialize_gas_and_registers(&new_gw);
            let level = child_vm_registers[7];
            let target_address = child_vm_registers[8];
            let target_length = child_vm_registers[9];
            let message_address = child_vm_registers[10];
            let message_length = child_vm_registers[11];

            let target_buffer = [0u8; 64];
            let target_buffer_address = target_buffer.as_ptr() as u64;

            let message_buffer = [0u8; 256];
            let message_buffer_address = message_buffer.as_ptr() as u64;

            unsafe { peek(new_machine_idx_2, target_buffer_address, target_address, target_length) };
            unsafe { peek(new_machine_idx_2, message_buffer_address, message_address, message_length) };

            unsafe {
                log(level, target_buffer_address, target_length, message_buffer_address, message_length);
            }
        } else {
            call_log(2, None, &format!("Parent: exiting invoke loop invoke_result={:?} omega_8={:?}", invoke_result, omega_8));
            let mut gas;
            (gas, child_vm_registers) = deserialize_gas_and_registers(&new_gw);
            // print the gas usage 
            gas = init_gas - gas;
            call_log(2, None, &format!("Parent: gas used={:?} child_vm_registers={:?}", gas, child_vm_registers));
            break;
        }
    }
    let (gas, child_vm_registers) = deserialize_gas_and_registers(&new_gw);
    let hash_output_address = child_vm_registers[7];
    let hash_length = child_vm_registers[8];
    call_log(2, None, &format!("Parent: hash output address={:?} length={:?}", hash_output_address, hash_length));
    // read the hash result from out 
    let hash_result = [0u8; 32];
    let hash_result_address = hash_result.as_ptr() as u64;
    let peek_result = unsafe{peek(new_machine_idx_2 as u64, hash_result_address ,hash_output_address, 32)};
    if peek_result != 0 {
        call_log(1, None, &format!("Parent: peek failed with result: {:?}", peek_result));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }
   return (hash_result_address, 32); // return the address and length of the hash result

}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
