#![no_std]
#![no_main]
#![feature(asm_const)]

use polkavm_derive::min_stack_size;
min_stack_size!(40960); // depends on how many pages you need

use utils::constants::{FIRST_READABLE_ADDRESS, FIRST_READABLE_PAGE, NONE, CORE, OK, WHO, HUH, PAGE_SIZE, SEGMENT_SIZE};
use utils::functions::{write_result, call_log};
use utils::functions::{deserialize_gas_and_registers, get_page, initialize_pvm_registers, serialize_gas_and_registers, setup_page};
use utils::functions::{parse_accumulate_args, parse_refine_args};
extern crate alloc;
use alloc::format;

use utils::host_functions::{
    assign, bless, checkpoint, eject, forget, gas, info, lookup, new, oyield, query, read, solicit, upgrade, write,
};
use utils::host_functions::{export, expunge, fetch, historical_lookup, invoke, machine, poke, peek};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
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
        return (extrinsic_address, extrinsic_length);
    }

    // fetch child VM blob
    let mut child_vm_blob = [0u8; 512 as usize];
    let child_vm_blob_address = child_vm_blob.as_mut_ptr() as u64;
    let mut child_vm_blob_length = child_vm_blob.len() as u64;
    child_vm_blob_length = unsafe {
        historical_lookup(
            wi_service_index as u64,
            code_hash_address,
            child_vm_blob_address,
            0,
            child_vm_blob_length,
        )
    };
    if child_vm_blob_length == NONE {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    // new child VM
    let num_payload = wi_payload_length / 8;
    let mut child_vm_ids = [0u32; 16];
    let num_child_vm = num_payload - 1;
    let mut gas_result = unsafe { gas() };

    call_log(2, None, &format!("num_child_vm={:?} gas_result={:?}", num_child_vm, gas_result));

    let payload = unsafe { core::slice::from_raw_parts(wi_payload_start_address as *const u8, wi_payload_length as usize) };
    for i in 0..num_child_vm {
        let new_idx = unsafe { machine(child_vm_blob_address, child_vm_blob_length, 0) };
        child_vm_ids[i as usize] = new_idx as u32;
        call_log(2, None, &format!("machine new index={:?}", new_idx));
    }

    // fetch segments for all child VMs
    let mut segment_buf = [0u8; SEGMENT_SIZE as usize];
    let mut segment_index = 0u64;
    loop {
        let result = unsafe { fetch(segment_buf.as_mut_ptr() as u64, 0, SEGMENT_SIZE as u64, 6, segment_index, 0) };
        if result == NONE {
            break;
        }
        call_log(2, None, &format!("fetch segment_index={:?} fetch_result={:?}", segment_index, result));
        setup_page(&segment_buf);
        segment_index += 1;
    }

    // invoke all child VMs
    let init_gas: u64 = 0x10000;

    let mut buffer_0 = [0; SEGMENT_SIZE as usize];
    let mut buffer_1 = [0; SEGMENT_SIZE as usize];
    let mut result_buffer = [0; SEGMENT_SIZE as usize];
    let mut child_vm_registers = initialize_pvm_registers();

    let mut output_12_bytes: [u8; 16] = [0; 16];
    let output_12_bytes_address = output_12_bytes.as_ptr() as u64;
    let output_12_bytes_length = output_12_bytes.len() as u64;

    call_log(2, None, &format!("num_child_vm={:?}", num_child_vm));

    for i in 0..num_child_vm {
        buffer_0.fill(0);
        buffer_1.fill(0);
        result_buffer.fill(0);
        output_12_bytes.fill(0);

        let child_vm_id = child_vm_ids[i as usize];
        let child_payload = &payload[(i * 8) as usize..(i * 8 + 8) as usize];
        let child_n = u32::from_le_bytes(child_payload[0..4].try_into().unwrap());
        let child_f = u32::from_le_bytes(child_payload[4..8].try_into().unwrap());

        if child_f == 1 {
            child_vm_registers[7] = 2;
            let g_w = serialize_gas_and_registers(init_gas, &child_vm_registers);
            let g_w_address = g_w.as_ptr() as u64;

            if child_vm_id == 1 {
                buffer_0.copy_from_slice(&get_page(child_vm_id - 1, FIRST_READABLE_PAGE));
                buffer_1[8..12].copy_from_slice(&1_u32.to_le_bytes());
            } else if child_vm_id > 1 {
                buffer_0.copy_from_slice(&get_page(child_vm_id - 2, FIRST_READABLE_PAGE));
                buffer_1.copy_from_slice(&get_page(child_vm_id - 1, FIRST_READABLE_PAGE));
            }

            buffer_0[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            buffer_0[4..8].copy_from_slice(&FIRST_READABLE_PAGE.to_le_bytes());
            buffer_1[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            buffer_1[4..8].copy_from_slice(&(FIRST_READABLE_PAGE + 1).to_le_bytes());

            output_12_bytes[12..16].copy_from_slice(&buffer_0[8..12]);
            output_12_bytes[8..12].copy_from_slice(&buffer_1[8..12]);
            output_12_bytes[0..4].copy_from_slice(&child_n.to_le_bytes());

            setup_page(&buffer_0);
            setup_page(&buffer_1);

            let invoke_result = unsafe {
                invoke(child_vm_id as u64, g_w_address);
            };
            gas_result = unsafe { gas() };
            call_log(2, None, &format!("invoke i={:?} invoke_result={:?} gas={:?}", i, invoke_result, gas_result));

            let result_buffer_data_address = unsafe { result_buffer.as_mut_ptr().add(8) };
            let (_, new_child_vm_registers) = deserialize_gas_and_registers(&g_w);
            let output_address = new_child_vm_registers[7];

            let peek_result = unsafe {
                peek(child_vm_id as u64, result_buffer_data_address as u64, output_address, PAGE_SIZE);
            };
            gas_result = unsafe { gas() };
            call_log(2, None, &format!("peek i={:?}, expect OK {:?}, got {:?} gas={:?}", i, OK, peek_result, gas_result));

            result_buffer[0..4].copy_from_slice(&child_vm_id.to_le_bytes());
            result_buffer[4..8].copy_from_slice(&FIRST_READABLE_PAGE.to_le_bytes());

            output_12_bytes[4..8].copy_from_slice(&result_buffer[8..12]);

            setup_page(&result_buffer);
        }
        let export_result = unsafe {
            export(result_buffer.as_ptr() as u64, SEGMENT_SIZE);
        };

        gas_result = unsafe { gas() };
        call_log(2, None, &format!("export i={:?}: expect OK {:?}, got {:?} gas={:?}", i, OK, export_result, gas_result));
    }

    for i in 0..num_child_vm {
        let child_vm_id = child_vm_ids[i as usize];
        let expunge_result = unsafe {
            expunge(child_vm_id as u64);
        };
        gas_result = unsafe { gas() };
        call_log(2, None, &format!("expunge i={:?}: expect OK {:?}, got {:?} gas={:?}", i, OK, expunge_result, gas_result));
    }

    (output_12_bytes_address, output_12_bytes_length)
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse accumulate args
    let (_timeslot, service_index, work_result_address, work_result_length) =
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
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    // write FIB result to storage
    let key = [0u8; 1];
    let n: u64 = unsafe { (*(work_result_address as *const u32)).into() };
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Prepare some keys and hashes.
    let jam: [u8; 3] = [b'j', b'a', b'm'];
    let dot: [u8; 3] = [b'D', b'O', b'T'];

    // blake2b("jam") = 0x6a0d4a19d199505713fc65f531038e73f1d885645632c8ae503c4f0c4d5e19a7
    let jam_hash: [u8; 32] = [
        0x6a, 0x0d, 0x4a, 0x19, 0xd1, 0x99, 0x50, 0x57, 0x13, 0xfc, 0x65, 0xf5, 0x31, 0x03, 0x8e, 0x73, 0xf1, 0xd8, 0x85, 0x64, 0x56, 0x32,
        0xc8, 0xae, 0x50, 0x3c, 0x4f, 0x0c, 0x4d, 0x5e, 0x19, 0xa7,
    ];

    // blake2b("dot") = 0xbfa9bb0fa4968747e63d3cf1e74a49ddc4a6eca89a6a6f339da3337fd2eb5507
    let dot_hash: [u8; 32] = [
        0xbf, 0xa9, 0xbb, 0x0f, 0xa4, 0x96, 0x87, 0x47, 0xe6, 0x3d, 0x3c, 0xf1, 0xe7, 0x4a, 0x49, 0xdd, 0xc4, 0xa6, 0xec, 0xa8, 0x9a, 0x6a,
        0x6f, 0x33, 0x9d, 0xa3, 0x33, 0x7f, 0xd2, 0xeb, 0x55, 0x07,
    ];

    let jam_address: u64 = jam.as_ptr() as u64;
    let jam_length: u64 = jam.len() as u64;
    let dot_address: u64 = dot.as_ptr() as u64;
    let dot_length: u64 = dot.len() as u64;
    let jam_hash_address: u64 = jam_hash.as_ptr() as u64;
    let _jam_hash_length: u64 = jam_hash.len() as u64;
    let dot_hash_address: u64 = dot_hash.as_ptr() as u64;
    let _dot_hash_length: u64 = dot_hash.len() as u64;

    let info_bytes = [0u8; 100];
    let _info_address: u64 = info_bytes.as_ptr() as u64;
    let _info_length: u64 = info_bytes.len() as u64;

    let buffer = [0u8; 256];
    let buffer_address = buffer.as_ptr() as u64;
    let buffer_length = buffer.len() as u64;

    // Depending on what "n" is, test different host functions
    if n == 1 {
        let read_none_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(2, None, &format!("read from jam: expect NONE {:?}, got {:?} (recorded at key 1)", NONE,read_none_result));
        write_result(read_none_result, 1);

        let write_result1 = unsafe { write(jam_address, jam_length, dot_address, dot_length) };
        call_log(2, None, &format!("write to jam: expect NONE {:?}, got {:?} (recorded at key 2)", NONE, write_result1));
        write_result(write_result1, 2);

        let read_ok_result = unsafe {read( service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(2, None, &format!("read from jam: expect OK {:?}, got {:?} (recorded at key 5)", OK, read_ok_result));
        write_result(read_ok_result, 5);

        let forget_result = unsafe { forget(jam_address, 0) };
        call_log(2, None, &format!("forget jam: expect HUH {:?}, got {:?} (recorded at key 6)", HUH, forget_result));
        write_result(forget_result, 6);
    } else if n == 2 {
        let read_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(2, None, &format!("read jam: expect OK {:?}, got {:?} (recorded at key 1)", OK, read_result));
        write_result(read_result, 1);

        let write_result1 = unsafe { write(jam_address, jam_length, 0, 0) };
        call_log(2, None, &format!("write deleted jam: expect OK {:?}, got {:?} (recorded at key 2)", OK, write_result1));
        write_result(write_result1, 2);

        let read_ok_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(2, None, &format!("read jam: expect NONE {:?}, got {:?} (recorded at key 5)", NONE, read_ok_result));
        write_result(read_ok_result, 5);
    } else if n == 3 {
        let solicit_result = unsafe { solicit(jam_hash_address, jam_length) };
        call_log(2, None, &format!("solicit hash(jam): expect OK {:?}, got {:?} (recorded at key 1)", OK, solicit_result));
        write_result(solicit_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?}, got {:?} (recorded at key 2)", OK, query_jamhash_result));
        write_result(query_jamhash_result, 2);

        let query_none_result = unsafe { query(dot_hash_address, dot_length) };
        call_log(2, None, &format!("query hash(dot): expect NONE {:?}, got {:?} (recorded at key 5)", NONE, query_none_result));
        write_result(query_none_result, 5);
    } else if n == 4 {
        let forget_result = unsafe { forget(jam_hash_address, jam_length) };
        call_log(2, None, &format!("forget hash(jam): expect OK {:?}, got {:?} (recorded at key 1)", OK, forget_result));
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?} 2+2^32*x, got {:?} (recorded at key 2)", OK, forget_result));
        write_result(query_jamhash_result, 2);

        let lookup_none_result = unsafe { lookup(service_index as u64, dot_hash_address, buffer_address, 0, dot_length) };
        call_log(2, None, &format!("lookup hash(dot): expect NONE {:?}, got {:?} (recorded at key 5)", NONE, lookup_none_result));
        write_result(lookup_none_result, 5);

        let assign_result = unsafe { assign(1000, jam_address) };
        call_log(2, None, &format!("assign jam: expect CORE {:?}, got {:?} (recorded at key 6)", CORE, assign_result));
        write_result(assign_result, 6);
    } else if n == 5 {
        let lookup_result = unsafe { lookup(service_index as u64, jam_hash_address, buffer_address, 0, jam_length) };
        call_log(2, None, &format!("lookup hash(jam): expect OK {:?}, got {:?} (recorded at key 1)", OK, lookup_result));
        write_result(lookup_result, 1);

        let query_ok_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?}, got {:?} (recorded at key 2)", OK, query_ok_result));
        write_result(query_ok_result, 2);

        let eject_who_result = unsafe { eject(service_index as u64, jam_hash_address) };
        call_log(2, None, &format!("eject: expect WHO {:?}, got {:?} (recorded at key 5)", WHO, eject_who_result));
        write_result(eject_who_result, 5);

        let overflow_s = 0xFFFFFFFFFFFFu64;
        let bless_who_result = unsafe { bless(overflow_s, 0, 0, jam_hash_address, 0) };
        call_log(2, None, &format!("bless: expect WHO {:?}, got {:?} (recorded at key 6)", WHO, bless_who_result));
        write_result(bless_who_result, 6);
    } else if n == 6 {
        let solicit_result = unsafe { solicit(jam_hash_address, jam_length) };
        write_result(solicit_result, 1);
        call_log(2, None, &format!("solicit hash(jam), expect OK {:?}: insert t, got {:?} (recorded at key 1)", OK, solicit_result));

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?}, got {:?} (recorded at key 2)", OK, query_jamhash_result));
        write_result(query_jamhash_result, 2);

        let core_index = 0;

        let mut auth_hashes = [0; 2560];
        let mut i = 0;
        while i < 80 {
            let offset = i * 32;
            auth_hashes[offset..offset + 32].copy_from_slice(&jam_hash);
            i += 1;
        }

        let assign_ok_result = unsafe { assign(core_index, auth_hashes.as_ptr() as u64) };
        call_log(2, None, &format!("assign: expect OK {:?}, got {:?} (recorded at key 5)", OK, assign_ok_result));
        write_result(assign_ok_result, 5);
    } else if n == 7 {
        let forget_result = unsafe { forget(jam_hash_address, jam_length) };
        call_log(2, None, &format!("forget hash(jam): expect OK {:?}, got {:?} (recorded at key 1)", OK, forget_result));
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?} (2+2^32*x), got {:?} (recorded at key 1)", OK, query_jamhash_result));
        write_result(query_jamhash_result, 2);
    } else if n == 8 {
        let lookup_result = unsafe { lookup(service_index as u64, jam_hash_address, buffer_address, 0, jam_length) };
        call_log(2, None, &format!("lookup hash(jam), expect OK {:?}: 3 (2+2^32*x), got {:?} (recorded at key 1)", OK, lookup_result));
        write_result(lookup_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(2, None, &format!("query hash(jam): expect OK {:?} (2+2^32*x), got {:?} (recorded at key 2)", OK, query_jamhash_result));
        write_result(query_jamhash_result, 2);
    } else if n == 9 {
        let g: u64 = 911;
        let m: u64 = 911;
        let new_result = unsafe { new(jam_hash_address, jam_length, g, m) };
        call_log(2, None, &format!("new: expect OK {:?} (service_index), got {:?} (recorded at key 1)", OK, new_result));
        write_result(new_result, 1);

        let upgrade_result = unsafe { upgrade(jam_hash_address, g, m) };
        call_log(2, None, &format!("upgrade: expect OK {:?}, got {:?} (recorded at key 2)", OK, upgrade_result));
        write_result(upgrade_result, 2);

        let s: u32 = 911;
        let s_bytes = s.to_le_bytes();
        let gas_bytes = g.to_le_bytes();
        let mut bless_input = [0u8; 12];
        bless_input[..4].copy_from_slice(&s_bytes);
        bless_input[4..12].copy_from_slice(&gas_bytes);
        let bless_input_address = bless_input.as_ptr() as u64;

        let bless_ok_result = unsafe { bless(0, 1, 1, bless_input_address, 1) };
        call_log(2, None, &format!("bless: expect OK {:?}, got {:?} (recorded at key 5)", OK, bless_ok_result));
        write_result(bless_ok_result, 5);
    } else if n == 10 {
        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(2, None, &format!("write deleted DOT: expect NONE {:?}, got {:?} (recorded at key 1)", NONE, delete_result));
        write_result(delete_result, 1);

        let write_result1 = unsafe { write(dot_address, dot_length, jam_address, jam_length) };
        call_log(2, None, &format!("write to DOT: expect NONE {:?}, got {:?} (recorded at key 1)", NONE, write_result1));
        write_result(write_result1, 2);

        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(2, None, &format!("write deleted DOT: expect OK {:?}: 3, got {:?} (recorded at key 5)", OK, delete_result));
        write_result(delete_result, 5);

        let read_result = unsafe { read(service_index as u64, dot_address, dot_length, buffer_address, 0, buffer_length) };
        call_log(2, None, &format!("read from DOT: expect NONE {:?}, got {:?} (recorded at key 6)", NONE, read_result));
        write_result(read_result, 6);

        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(2, None, &format!("write deleted DOT: expect NONE {:?}, got {:?} (recorded at key 7)", NONE, delete_result));
        write_result(delete_result, 7);
    }

    let info_result = unsafe { info(service_index as u64, buffer_address) };
    write_result(info_result, 8);
    call_log(2, None, &format!("info: expect OK {:?}, got {:?} (recorded at key 8)", OK, info_result));

    let gas_result = unsafe { gas() };
    write_result(gas_result, 9);
    call_log(2, None, &format!("gas: expect OK {:?}, got {:?} (recorded at key 9)", OK, gas_result));

    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize]
        .copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let output_address = output_bytes_32.as_ptr() as u64;
    let output_length = output_bytes_32.len() as u64;

    // write yield
    if n % 3 == 0 {
        if n != 9 {
            // n=3,6 should go through even though there is a panic, 9 does not.
            unsafe {
                checkpoint();
            }
            call_log(2, None, "corevm checkpoint");
        }
        let result42 = n + 42;
        write_result(result42, 7); // this should not be stored if n = 3, 6, 9 because its after the checkpoint
        unsafe {
            core::arch::asm!(
                "jalr x0, a0, 0", // djump(0+0) causes panic
            );
        }
        call_log(1, None, "corevm PANIC");
    } else {
    }
    unsafe {
        oyield(output_address);
    }
    call_log(2, None, "yield, expect OK");
    return (output_address, output_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
