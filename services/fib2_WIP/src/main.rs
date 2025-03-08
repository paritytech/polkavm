#![no_std]
#![no_main]

use utils::{NONE, OK, PAGE_SIZE, SEGMENT_SIZE, PARENT_MACHINE_INDEX};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{call_info, setup_page, get_page, serialize_gas_and_registers, deserialize_gas_and_registers};
use utils::{write_result};

use utils::{historical_lookup, fetch, export, machine, peek, poke, zero, void, invoke, expunge};
use utils::{gas, lookup, read, write, info, bless, assign, checkpoint, new, upgrade, eject, query, solicit, forget, oyield, log};

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    // read the input start address and length from register a0 and a1
    let omega_7: u64;
    let omega_8: u64;
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
        return NONE;
    };

    // fetch extrinsic, this will fetch code hash and code length for child VM
    let mut extrinsic = [0u8; 36];
    let extrinsic_address: u64 = extrinsic.as_ptr() as u64;
    let code_hash_address = extrinsic_address;
    let extrinsic_length: u64 = extrinsic.len() as u64;

    let fetch_result = unsafe { fetch(extrinsic_address, 0, extrinsic_length, 3, 0, 0) };

    // first time setup: do nothing but put the code hash to output for solicit
    if wi_payload_length == 0 {
        unsafe {
            core::arch::asm!(
                "mv a1, {0}",
                in(reg) extrinsic_length,
            );
        }
        return extrinsic_address;
    }

    let mut buffer_0 = [0u8; SEGMENT_SIZE as usize];
    let mut buffer_1 = [0u8; SEGMENT_SIZE as usize];

    // fetch import segment
    let mut segment_index = 0;
    loop {
        let result = unsafe { fetch(buffer_0.as_ptr() as u64, 0, SEGMENT_SIZE, 6, segment_index as u64, 0) };
        if result == NONE {
            break;
        }
        segment_index += 1;
        setup_page(&buffer_0);
    }

    // fetch child vm blob
    let child_vm_blob = [0u8; 512]; // MAX_LEN = 512
    let child_vm_blob_address: u64 = child_vm_blob.as_ptr() as u64;
    let mut child_vm_blob_length: u64 = child_vm_blob.len() as u64;
    child_vm_blob_length = unsafe {
        historical_lookup(wi_service_index as u64, code_hash_address, child_vm_blob_address, 0, child_vm_blob_length)
    };
    if child_vm_blob_length == NONE {
        return NONE;
    }

    // New child VM
    let num_payload = wi_payload_length / 8;
    let mut child_vm_ids = [0u32; 16]; // 16 at most
    let num_child_vm = num_payload - 1;
    let payload =  unsafe { core::slice::from_raw_parts(wi_payload_start_address as *const u8, wi_payload_length as usize) };
    for i in 0..num_child_vm {
        let new_idx = unsafe { machine(child_vm_blob_address, child_vm_blob_length, 0) };
        child_vm_ids[i as usize] = new_idx as u32;
    }

    // Invoke child VMs and Export
    let mut output = [0u8; 12];

    let mut child_vm_registers = [0u64; 13];
    let init_gas: u64 = 0x10000;
    for i in 0..num_child_vm {
        let child_vm_id = child_vm_ids[i as usize];

        let child_payload = &payload[(i * 8) as usize..(i * 8 + 8) as usize];
        let _child_n = u32::from_le_bytes(child_payload[0..4].try_into().unwrap());
        let child_f = u32::from_le_bytes(child_payload[4..8].try_into().unwrap());
                
        if child_f == 1 {
            child_vm_registers[7] = 2; // set the number of arguments
            let g_w = serialize_gas_and_registers(init_gas, &child_vm_registers);
            let g_w_address = g_w.as_ptr() as u64;

            if child_vm_id == 0 {
                buffer_0 = [0u8; SEGMENT_SIZE as usize];
                buffer_1 = [0u8; SEGMENT_SIZE as usize];
            } else if child_vm_id == 1 {
                buffer_0 = get_page(0, 0);
                buffer_1 = [0u8; SEGMENT_SIZE as usize];
                buffer_1[8..12].copy_from_slice(&1_u32.to_le_bytes());
            } else {
                buffer_0 = get_page(child_vm_id - 2, 0);
                buffer_1 = get_page(child_vm_id - 1, 0);
            }

            buffer_0[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            buffer_0[4..8].copy_from_slice(&0_u32.to_le_bytes());
            buffer_1[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            buffer_1[4..8].copy_from_slice(&1_u32.to_le_bytes());

            // store 12 bytes result
            output[8..12].copy_from_slice(&buffer_0[8..12]);
            output[4..8].copy_from_slice(&buffer_1[8..12]);

            setup_page(&buffer_0);
            setup_page(&buffer_1);
            
            unsafe {
                invoke(child_vm_id as u64, g_w_address);
            };

            let buffer_0_data_address: u64 = buffer_0.as_ptr() as u64 + 8;

            let (_, new_child_vm_registers) = deserialize_gas_and_registers(&g_w);
            let output_address:u64 = new_child_vm_registers[7];
            
            unsafe {
                peek(child_vm_id as u64, buffer_0_data_address, output_address, PAGE_SIZE);
            }

            buffer_0[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            buffer_0[4..8].copy_from_slice(&0_u32.to_le_bytes());

            // store 12 bytes result
            output[0..4].copy_from_slice(&buffer_0[8..12]);

            setup_page(&buffer_0);
            call_info("FIB2 WIP invoke done");
        }
    }

    let parent_payload_index = num_payload - 1;
    let parent_payload = &payload[parent_payload_index as usize * 8..(parent_payload_index + 1) as usize * 8];
    let parent_n = u32::from_le_bytes(parent_payload[0..4].try_into().unwrap());
    let parent_f = u32::from_le_bytes(parent_payload[4..8].try_into().unwrap());

    let mut sum: u32 = 0;
    for i in 0..num_child_vm {
        let child_vm_id = child_vm_ids[i as usize];
        if parent_f == 16 {
            buffer_0 = get_page(child_vm_id, 0);

            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&buffer_0[8..12]);

            sum = sum.wrapping_add(u32::from_le_bytes(bytes));
        } else if parent_f == 17 {
            buffer_0 = get_page(child_vm_id, 0);

            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&buffer_0[8..12]);

            sum = sum.wrapping_mul(u32::from_le_bytes(bytes));
        }
    }

    let output_address = output.as_ptr() as u64;
    let output_length = output.len() as u64;

    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) output_length,
        );
    }
    output_address
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    // read the input start address and length from register a0 and a1
    let omega_7: u64;
    let omega_8: u64;
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
    let service_index: u64   = unsafe { ( *(service_index_address as *const u32)).into() };

    // fetch all_accumulation_o
    let start_address = omega_7 + 4 + 4; // 4 bytes time slot + 4 bytes service index
    let remaining_length = omega_8 - 4 - 4; // 4 bytes time slot + 4 bytes service index
    
    let (work_result_address, work_result_length) =
    if let Some(tuple) = parse_wrangled_operand_tuple(start_address, remaining_length, 0)
    {
        (tuple.work_result_ptr, tuple.work_result_len)
    } else {
        return NONE;
    };

    // first time setup: do nothing but solicit code for child VM
    if work_result_length == 36 {
        let code_hash_address = work_result_address;
        let code_length_address = work_result_address + 32;
        let code_length: u64 =  unsafe { ( *(code_length_address as *const u32)).into() };
        unsafe { solicit(code_hash_address, code_length) };
        return OK;
    }

    // write FIB result to storage
    let key = [0u8; 1];
    let n: u64 = unsafe { ( *(work_result_address as *const u32)).into() };
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Prepare some keys and hashes.
    // let jam_key: [u8; 3] = [b'j', b'a', b'm'];
    // let dot_val: [u8; 3] = [b'D', b'O', b'T'];
    
    // // blake2b("jam") = 0x6a0d4a19d199505713fc65f531038e73f1d885645632c8ae503c4f0c4d5e19a7
    // let jam_key_hash: [u8; 32] = [
    //     0x6a, 0x0d, 0x4a, 0x19, 0xd1, 0x99, 0x50, 0x57,
    //     0x13, 0xfc, 0x65, 0xf5, 0x31, 0x03, 0x8e, 0x73,
    //     0xf1, 0xd8, 0x85, 0x64, 0x56, 0x32, 0xc8, 0xae,
    //     0x50, 0x3c, 0x4f, 0x0c, 0x4d, 0x5e, 0x19, 0xa7
    // ];

    // // blake2b("dot") = 0xbfa9bb0fa4968747e63d3cf1e74a49ddc4a6eca89a6a6f339da3337fd2eb5507
    // let dot_val_hash: [u8; 32] = [
    //     0xbf, 0xa9, 0xbb, 0x0f, 0xa4, 0x96, 0x87, 0x47,
    //     0xe6, 0x3d, 0x3c, 0xf1, 0xe7, 0x4a, 0x49, 0xdd,
    //     0xc4, 0xa6, 0xec, 0xa8, 0x9a, 0x6a, 0x6f, 0x33,
    //     0x9d, 0xa3, 0x33, 0x7f, 0xd2, 0xeb, 0x55, 0x07
    // ];

    // let jam_key_address: u64 = jam_key.as_ptr() as u64;
    // let jam_key_length: u64 = jam_key.len() as u64;
    // let dot_val_address: u64 = dot_val.as_ptr() as u64;
    // let dot_val_length: u64 = dot_val.len() as u64;
    // let jam_key_hash_address: u64 = jam_key_hash.as_ptr() as u64;
    // let _jam_key_hash_length: u64 = jam_key_hash.len() as u64;
    // let dot_val_hash_address: u64 = dot_val_hash.as_ptr() as u64;
    // let _dot_val_hash_length: u64 = dot_val_hash.len() as u64;
    
    // let info_bytes = [0u8; 100];
    // let _info_address: u64 = info_bytes.as_ptr() as u64;
    // let _info_length: u64 = info_bytes.len() as u64;
    
    // let buffer = [0u8; 256];
    // let buffer_address = buffer.as_ptr() as u64;
    // let buffer_length = buffer.len() as u64;
    
    // // Depending on what "n" is, test different host functions
    // if n == 1 {
    //     // do nothing
    // } else if n == 2 {
    //     let read_none_result = unsafe { read(service_index, jam_key_address, jam_key_length, buffer_address, 0, buffer_length) };
    //     write_result(read_none_result, 1);
    
    //     let write_result1 = unsafe { write(jam_key_address, jam_key_length, dot_val_address, dot_val_length) };
    //     write_result(write_result1, 2);
    
    //     let read_ok_result = unsafe { read(service_index, jam_key_address, jam_key_length, buffer_address, 0, buffer_length) };
    //     write_result(read_ok_result, 5);
    
    //     let forget_result = unsafe { forget(jam_key_address, 0) };
    //     write_result(forget_result, 6);
    // } else if n == 3 {
    //     let solicit_result = unsafe { solicit(jam_key_hash_address, jam_key_length) };
    //     write_result(solicit_result, 1);
    
    //     let query_jamhash_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(query_jamhash_result, 2);
    
    //     let query_none_result = unsafe { query(dot_val_hash_address, dot_val_length) };
    //     write_result(query_none_result, 5);
    // } else if n == 4 {
    //     let forget_result = unsafe { forget(jam_key_hash_address, jam_key_length) };
    //     write_result(forget_result, 1);
    
    //     let query_jamhash_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(query_jamhash_result, 2);
    
    //     let lookup_none_result = unsafe { lookup(service_index, dot_val_hash_address, buffer_address, 0, dot_val_length) };
    //     write_result(lookup_none_result, 5);
    
    //     let assign_result = unsafe { assign(1000, jam_key_address) };
    //     write_result(assign_result, 6);
    // } else if n == 5 {
    //     let lookup_result = unsafe { lookup(service_index, jam_key_hash_address, buffer_address, 0, jam_key_length) };
    //     write_result(lookup_result, 1);
    
    //     let read_ok_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(read_ok_result, 2);
    
    //     let eject_who_result = unsafe { eject(service_index, jam_key_hash_address) };
    //     write_result(eject_who_result, 5);
    
    //     let overflow_s = 0xFFFFFFFFFFFFu64;
    //     let bless_who_result = unsafe { bless(overflow_s, 0, 0, jam_key_hash_address, 0) };
    //     write_result(bless_who_result, 6);
    // } else if n == 6 {
    //     let solicit_result = unsafe { solicit(jam_key_hash_address, jam_key_length) };
    //     write_result(solicit_result, 1);
    
    //     let query_jamhash_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(query_jamhash_result, 2);
    
    //     let core_index = 0;
    
    //     let mut auth_hashes = [0; 2560];
    //     let mut i = 0;
    //     while i < 80 {
    //         let offset = i * 32;
    //         auth_hashes[offset..offset + 32].copy_from_slice(&jam_key_hash);
    //         i += 1;
    //     }
    
    //     let assign_ok_result = unsafe { assign(core_index, auth_hashes.as_ptr() as u64) };
    //     write_result(assign_ok_result, 5);
    // } else if n == 7 {
    //     let forget_result = unsafe { forget(jam_key_hash_address, jam_key_length) };
    //     write_result(forget_result, 1);
    
    //     let query_jamhash_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(query_jamhash_result, 2);
    // } else if n == 8 {
    //     let lookup_result = unsafe { lookup(service_index, jam_key_hash_address, buffer_address, 0, jam_key_length) };
    //     write_result(lookup_result, 1);
    
    //     let query_jamhash_result = unsafe { query(jam_key_hash_address, jam_key_length) };
    //     write_result(query_jamhash_result, 2);
    // } else if n == 9 {
    //     let read_result = unsafe { read(service_index, jam_key_address, jam_key_length, buffer_address, 0, buffer_length) };
    //     write_result(read_result, 1);
    
    //     let write_result1 = unsafe { write(jam_key_address, jam_key_length, 0, 0) };
    //     write_result(write_result1, 2);
    
    //     let read_ok_result = unsafe { read(service_index, jam_key_address, jam_key_length, buffer_address, 0, buffer_length) };
    //     write_result(read_ok_result, 5);
    
    //     let solicit_result = unsafe { solicit(jam_key_hash_address, jam_key_length) };
    //     write_result(solicit_result, 6);
    // } else if n == 1024 {
    //     let g: u64 = 911911;
    //     let m: u64 = 911911;
    //     let new_result = unsafe { new(jam_key_hash_address, jam_key_length, g, m) };
    //     write_result(new_result, 1);
        
    //     let upgrade_result = unsafe { upgrade(jam_key_hash_address, g, m) };
    //     write_result(upgrade_result, 2);
    
    //     let s: u32 = 911;
    //     let s_bytes = s.to_le_bytes();
    //     let gas_bytes = g.to_le_bytes();
    //     let mut bless_input = [0u8; 12];
    //     bless_input[..4].copy_from_slice(&s_bytes);
    //     bless_input[4..12].copy_from_slice(&gas_bytes);
    //     let bless_input_address = bless_input.as_ptr() as u64;
    
    //     let bless_ok_result = unsafe { bless(0, 1, 1, bless_input_address, 1) };
    //     write_result(bless_ok_result, 5);
    // }
    
    
    // let info_result = unsafe { info(service_index, buffer_address) };
    // write_result(info_result, 8);
    
    // let gas_result = unsafe { gas() };
    // write_result(gas_result, 9);
    
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize].copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;
    
    // write yield
    // if n % 3 == 0 {
    //     if n != 9 { // n=3,6 should go through even though there is a panic, 9 does not.
    //         let gas_result = unsafe { checkpoint() };
    //     }
    //     let result42 = n + 42;
    //     write_result(result42, 7); // this should not be stored if n = 3, 6, 9 because its after the checkpoint
    //     unsafe {
    //         core::arch::asm!(
    //             "li a0, 0",
    //             "li a1, 1",
    //             "jalr x0, a0, 0", // djump(0+0) causes panic
    //         );
    //     }
    // } else {
    // }
    // unsafe { oyield(omega_7); }
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
