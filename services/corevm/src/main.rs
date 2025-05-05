#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE0 : usize = 0x100000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x100000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::{FIRST_READABLE_ADDRESS, INPUT_ARGS_PAGE, PAGE_SIZE, SEGMENT_SIZE};
use utils::constants::{NONE, CORE, OK, WHO, HUH, HALT, HOST, LOG};

use utils::functions::{write_result, call_log};
use utils::functions::{initialize_pvm_registers, serialize_gas_and_registers, deserialize_gas_and_registers};
use utils::functions::{parse_standard_program_initialization_args, standard_program_initialization_for_child};
use utils::functions::{parse_accumulate_args, parse_refine_args};

use utils::host_functions::{assign, bless, checkpoint, eject, forget, gas, info, lookup, new, oyield, query, read, solicit, upgrade, write, provide};
use utils::host_functions::{export, expunge, fetch, historical_lookup, invoke, machine, poke, peek, zero, log};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    // parse refine args
    let (wi_service_index, _wi_payload_start_address, wi_payload_length, _wphash) =
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
    let mut raw_blob = [0u8; 8192 as usize];
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
    call_log(3, None, &format!("z={:?} s={:?} o_bytes_address={:?} o_bytes_length={:?} w_bytes_address={:?} w_bytes_length={:?} c_bytes_address={:?} c_bytes_length={:?}",
        z, s, o_bytes_address, o_bytes_length, w_bytes_address, w_bytes_length, c_bytes_address, c_bytes_length));

    let mut gas_result = unsafe { gas() };
    call_log(4, None, &format!("gas_result={:?}", gas_result));

    // new child VM
    let new_machine_idx = unsafe { machine(c_bytes_address, c_bytes_length, 0) };
    call_log(3, None, &format!("machine new index={:?}", new_machine_idx));

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
    let mut m = 0u64;
    let mut page_id = 0u64;
    loop {
        let fetch_result = unsafe { fetch(segment_buf_segment_address as u64, 0, SEGMENT_SIZE as u64, 6, segment_index, 0) };
        if fetch_result == NONE {
            break;
        }
        call_log(4, None, &format!("fetch segment_index={:?} fetch_result={:?}", segment_index, fetch_result));
        (m, page_id) = (
            u32::from_le_bytes(segment_buf[0..4].try_into().unwrap()) as u64,
            u32::from_le_bytes(segment_buf[4..8].try_into().unwrap()) as u64,
        );

        let zero_result = unsafe { zero(m, page_id, 1) };
        call_log(4, None, &format!("zero m={:?}, page_id={:?} zero_result={:?}",  m, page_id, zero_result));

        // poke(machine n, source s, dest o, # bytes z)
        let s = segment_buf.as_mut_ptr() as u64;
        let page_address = page_id * PAGE_SIZE as u64;
        let poke_result = unsafe { poke(m, segment_buf_page_address, page_address, PAGE_SIZE) };
        call_log(4, None, &format!("poke m={:?} s={:?} o={:?} z={:?} poke_result={:?}", m, s, page_address, PAGE_SIZE, poke_result));

        segment_index += 1;
    }

    // zero one more page for result
    page_id = if page_id == 0 { INPUT_ARGS_PAGE as u64 } else { page_id + 1 };
    let zero_result = unsafe { zero(m, page_id, 1) };
    call_log(4, None, &format!("zero for result m={:?}, page_id={:?} zero_result={:?}",  m, page_id, zero_result));

    // invoke child VM
    let init_gas: u64 = 0x10000;
    let mut child_vm_registers = initialize_pvm_registers();
    child_vm_registers[7] = segment_index;

    call_log(4, None, &format!("before child_vm_registers={:?}", child_vm_registers));
    let g_w = serialize_gas_and_registers(init_gas, &child_vm_registers);
    let g_w_address = g_w.as_ptr() as u64;

    let mut invoke_result: u64;
    let mut omega_8: u64;

    loop {
        (invoke_result, omega_8) = unsafe { invoke(new_machine_idx as u64, g_w_address) };
        call_log(4, None, &format!("invoke {:?} invoke_result={:?} gas={:?}", new_machine_idx, invoke_result, gas_result));

        if invoke_result == HALT {
            break;
        }

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
        }
    }

    gas_result = unsafe { gas() };
    call_log(4, None, &format!("gas_result={:?}", gas_result));

    // peek child VM, process output and export segments
    let mut output_bytes: [u8; 12] = [0;12];
    let output_bytes_address = output_bytes.as_ptr() as u64;
    let output_bytes_length = output_bytes.len() as u64;
    let child_id = new_machine_idx as u32;
    for i in 0..segment_index+1 {
        let i_32 = i as u32;
        let child_page_address = ((INPUT_ARGS_PAGE + i_32) as u64) * PAGE_SIZE as u64;
        let peek_result = unsafe { peek(new_machine_idx, segment_buf_page_address, child_page_address, PAGE_SIZE) };

        call_log(3, None, &format!("peek child_vm_id={:?}, segment_buf_page_address={:?}, child_page_address={:?} PAGE_SIZE={:?} peek_result={:?}",
        new_machine_idx, segment_buf_page_address, child_page_address, PAGE_SIZE, peek_result));

        let seg_index = INPUT_ARGS_PAGE + i as u32;

        // page metadata (child_vm_id, page index)
        segment_buf[0..4].copy_from_slice(&child_id.to_le_bytes());
        segment_buf[4..8].copy_from_slice(&seg_index.to_le_bytes());
        call_log(4, None, &format!(" segment_buf[0..8]={:?}, child_id={:?}, seg_index={:?}", &segment_buf[0..8], child_id, seg_index));

        if i == segment_index {
            // output_bytes: n, f(n), f(n-1)
            output_bytes[0..4].copy_from_slice(&i_32.to_le_bytes());
            output_bytes[4..8].copy_from_slice(&segment_buf[8..12]);
        } else if i == segment_index - 1 {
            output_bytes[8..12].copy_from_slice(&segment_buf[8..12]);
        }

        let export_result = unsafe { export(segment_buf.as_ptr() as u64, SEGMENT_SIZE) };
        gas_result = unsafe { gas() };
        call_log(4, None, &format!("export i={:?}: expect ς+|e|={:?}, got {:?} gas={:?} output={:?}", i, i, export_result, gas_result, output_bytes[0]));
    }
    call_log(3, None, &format!("output_bytes={:?}|{:?}|{:?}", output_bytes[0], output_bytes[4], output_bytes[8]));

    // expunge child VM
    let expunge_result = unsafe { expunge(new_machine_idx as u64) };
    gas_result = unsafe { gas() };
    call_log(4, None, &format!("expunge {:?}: child VM instruction counter={:?} gas={:?}", new_machine_idx, expunge_result, gas_result));

    (output_bytes_address, output_bytes_length)
}

#[no_mangle]
static mut output_bytes_32: [u8; 32] = [0; 32];

fn log_level(cond: bool) -> u64 {
    if cond {
        return 2;
    }
    return 0;
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

    // Depending on what "n" is, test different host functions and use log level 0 vs 2 for intermediate values
    if n == 1 {
        let read_none_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(log_level(read_none_result == NONE), None, &format!("read from jam @n={:?}: expect NONE, got {:?} (recorded at key 1)", n, read_none_result));
        write_result(read_none_result, 1);

        let write_result1 = unsafe { write(jam_address, jam_length, dot_address, dot_length) };
        call_log(log_level(write_result1 == NONE), None, &format!("write to jam @n={:?}: expect NONE, got {:?} (recorded at key 2)", n, write_result1));
        write_result(write_result1, 2);

        let read_ok_result = unsafe {read( service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(log_level(read_ok_result == 3), None, &format!("read from jam @n={:?}: expect 3, got {:?} (recorded at key 5)", n, read_ok_result));
        write_result(read_ok_result, 5);

        let forget_result = unsafe { forget(jam_address, 0) };
        call_log(log_level(forget_result == HUH), None, &format!("forget jam @n={:?}: expect HUH, got {:?} (recorded at key 6)", n, forget_result));
        write_result(forget_result, 6);
    } else if n == 2 {
        let read_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(log_level(read_result == 3), None, &format!("read jam@n={:?}: expect 3, got {:?} (recorded at key 1)", n, read_result));
        write_result(read_result, 1);

        let write_result1 = unsafe { write(jam_address, jam_length, 0, 0) };
        call_log(log_level(write_result1 == 3), None, &format!("write deleted jam@n={:?}: expect prev len 3, got {:?} (recorded at key 2)", n, write_result1));
        write_result(write_result1, 2);

        let read_ok_result = unsafe {read(service_index as u64, jam_address, jam_length, buffer_address, 0, buffer_length)};
        call_log(log_level(read_ok_result == NONE), None, &format!("read jam@n={:?}: expect NONE, got {:?} (recorded at key 5)", n, read_ok_result));
        write_result(read_ok_result, 5);
    } else if n == 3 {
        let solicit_result = unsafe { solicit(jam_hash_address, jam_length) };
        call_log(log_level(solicit_result == OK), None, &format!("solicit hash(jam)@n={:?}: expect OK, got {:?} (recorded at key 1)", n, solicit_result));
        write_result(solicit_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_jamhash_result == 0), None, &format!("query hash(jam)@n={:?}: expect zero result, got {:?} (recorded at key 2)", n, query_jamhash_result));
        write_result(query_jamhash_result, 2);

        let query_none_result = unsafe { query(dot_hash_address, dot_length) };
        call_log(log_level(query_none_result == NONE), None, &format!("query hash(dot)@n={:?}: expect NONE, got {:?} (recorded at key 5)", n, query_none_result));
        write_result(query_none_result, 5);
    } else if n == 4 {
        let forget_result = unsafe { forget(jam_hash_address, jam_length) };
        call_log(log_level(forget_result == OK), None, &format!("forget hash(jam)@n={:?}: expect OK, got {:?} (recorded at key 1)", n, forget_result));
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_jamhash_result > 0), None, &format!("query hash(jam)@n={:?}: expect non-zero, got {:?} (recorded at key 2)", n, query_jamhash_result));
        write_result(query_jamhash_result, 2);

        let lookup_none_result = unsafe { lookup(service_index as u64, dot_hash_address, buffer_address, 0, dot_length) };
        call_log(log_level(lookup_none_result == NONE), None, &format!("lookup hash(dot)@n={:?}: expect NONE, got {:?} (recorded at key 5)", n, lookup_none_result));
        write_result(lookup_none_result, 5);

        let assign_result = unsafe { assign(1000, jam_address) };
        call_log(log_level(assign_result == CORE), None, &format!("assign jam@n={:?}: expect CORE, got {:?} (recorded at key 6)", n, assign_result));
        write_result(assign_result, 6);

        let provide_jamhash_result = unsafe { provide(666, jam_hash_address, jam_length) };
        call_log(log_level(provide_jamhash_result == WHO), None, &format!("provide hash(jam): expect WHO, got {:?} (recorded at key 7)", query_jamhash_result));
        write_result(provide_jamhash_result, 7);

        let provide_jamhash_result2 = unsafe { provide(service_index as u64, jam_hash_address, jam_length) };
        call_log(log_level(provide_jamhash_result2 == OK), None, &format!("provide hash(jam): expect OK, got {:?} (recorded at key 8)", provide_jamhash_result2));
        write_result(provide_jamhash_result, 8);        
    } else if n == 5 {
        let lookup_result = unsafe { lookup(service_index as u64, jam_hash_address, buffer_address, 0, jam_length) };
        call_log(log_level(lookup_result == 3), None, &format!("lookup hash(jam)@n={:?}: expect 3, got {:?} (recorded at key 1)", n, lookup_result));
        write_result(lookup_result, 1);

        let query_ok_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_ok_result > 0), None, &format!("query hash(jam)@n={:?}: expect non-zero, got {:?} (recorded at key 2)", n, query_ok_result));
        write_result(query_ok_result, 2);

        let eject_who_result = unsafe { eject(service_index as u64, jam_hash_address) };
        call_log(log_level(eject_who_result == WHO), None, &format!("eject@n={:?}: expect WHO, got {:?} (recorded at key 5)", n, eject_who_result));
        write_result(eject_who_result, 5);

        let overflow_s = 0xFFFFFFFFFFFFu64;
        let bless_who_result = unsafe { bless(overflow_s, 0, 0, jam_hash_address, 0) };
        call_log(log_level(bless_who_result == WHO), None, &format!("bless@n={:?}: expect WHO, got {:?} (recorded at key 6)", n, bless_who_result));
        write_result(bless_who_result, 6);

        let provide_jamhash_result = unsafe { provide(service_index as u64, jam_hash_address, jam_length) };
        call_log(log_level(provide_jamhash_result == OK), None, &format!("provide hash(jam): expect OK, got {:?} (recorded at key 7)", provide_jamhash_result));
        write_result(provide_jamhash_result, 7);   

        let provide_jamhash_result2 = unsafe { provide(service_index as u64, jam_hash_address, jam_length) };
        call_log(log_level(provide_jamhash_result2 == HUH), None, &format!("provide hash(jam): expect HUH, got {:?} (recorded at key 8)", provide_jamhash_result2));
        write_result(provide_jamhash_result2, 8);     
    } else if n == 6 {
        let solicit_result = unsafe { solicit(jam_hash_address, jam_length) };
        write_result(solicit_result, 1);
        call_log(log_level(solicit_result == OK), None, &format!("solicit hash(jam)@n={:?}: expect OK, got {:?} (recorded at key 1)", n, solicit_result));

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_jamhash_result > 0), None, &format!("query hash(jam)@n={:?}: expect 2+2^32*x, got {:?} (recorded at key 2)", n, query_jamhash_result));
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
        call_log(log_level(assign_ok_result == OK), None, &format!("assign@n={:?}: expect OK, got {:?} (recorded at key 5)", n, assign_ok_result));
        write_result(assign_ok_result, 5);
    } else if n == 7 {
        let lookup_result = unsafe { lookup(service_index as u64, jam_hash_address, buffer_address, 0, jam_length) };
        call_log(log_level(lookup_result == 3), None, &format!("lookup hash(jam)@n={:?}:: expect 3, got {:?} (recorded at key 1)", n, lookup_result));
        write_result(lookup_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_jamhash_result > 0), None, &format!("query hash(jam)@n={:?}: expect non-zero, got {:?} (recorded at key 2)", n, query_jamhash_result));
        write_result(query_jamhash_result, 2);
    } else if n == 8 {
        let forget_result = unsafe { forget(jam_hash_address, jam_length) };
        call_log(log_level(forget_result == OK), None, &format!("forget hash(jam)@n={:?}: expect OK, got {:?} (recorded at key 1)", n, forget_result));
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(jam_hash_address, jam_length) };
        call_log(log_level(query_jamhash_result > 0), None, &format!("query hash(jam)@n={:?}: expect non-zero, got {:?} (recorded at key 1)", n, query_jamhash_result));
        write_result(query_jamhash_result, 2);
    } else if n == 9 {
        let g: u64 = 911;
        let m: u64 = 911;
        let new_result = unsafe { new(jam_hash_address, jam_length, g, m) };
        call_log(log_level(new_result > 0), None, &format!("new @n={:?}: expect service_index, got {:?} (recorded at key 1)", n, new_result));
        write_result(new_result, 1);

        let upgrade_result = unsafe { upgrade(jam_hash_address, g, m) };
        call_log(log_level(upgrade_result == OK), None, &format!("upgrade @n={:?}: expect OK, got {:?} (recorded at key 2)", n, upgrade_result));
        write_result(upgrade_result, 2);

        let s: u32 = 911;
        let s_bytes = s.to_le_bytes();
        let gas_bytes = g.to_le_bytes();
        let mut bless_input = [0u8; 12];
        bless_input[..4].copy_from_slice(&s_bytes);
        bless_input[4..12].copy_from_slice(&gas_bytes);
        let bless_input_address = bless_input.as_ptr() as u64;

        let bless_ok_result = unsafe { bless(0, 1, 1, bless_input_address, 1) };
        call_log(log_level(bless_ok_result == OK), None, &format!("bless @n={:?}: expect OK, got {:?} (recorded at key 5)", n, bless_ok_result));
        write_result(bless_ok_result, 5);
    } else if n == 10 {
        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(log_level(delete_result == NONE), None, &format!("write deleted DOT @n={:?}: expect NONE, got {:?} (recorded at key 1)", n, delete_result));
        write_result(delete_result, 1);

        let write_result1 = unsafe { write(dot_address, dot_length, jam_address, jam_length) };
        call_log(log_level(write_result1 == NONE), None, &format!("write to DOT @n={:?}: expect NONE, got {:?} (recorded at key 1)", n, write_result1));
        write_result(write_result1, 2);

        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(log_level(delete_result == 3), None, &format!("write deleted DOT @n={:?}: expect prev len 3, got {:?} (recorded at key 5)", n, delete_result));
        write_result(delete_result, 5);

        let read_result = unsafe { read(service_index as u64, dot_address, dot_length, buffer_address, 0, buffer_length) };
        call_log(log_level(read_result == NONE), None, &format!("read from DOT @n={:?}: expect NONE, got {:?} (recorded at key 6)", n, read_result));
        write_result(read_result, 6);

        let delete_result = unsafe { write(dot_address, dot_length, 0, 0) };
        call_log(log_level(delete_result == NONE), None, &format!("write deleted DOT @n={:?}: expect NONE, got {:?} (recorded at key 7)", n, delete_result));
        write_result(delete_result, 7);
    }

    let info_result = unsafe { info(service_index as u64, buffer_address) };
    write_result(info_result, 8);
    call_log(log_level(info_result == OK), None, &format!("info@n={:?}: expect OK, got {:?} (recorded at key 8)", n, info_result));

    let gas_result = unsafe { gas() };
    write_result(gas_result, 9);
    call_log(4, None, &format!("gas: got {:?} (recorded at key 9)", gas_result));

    let output_address: u64;
    let output_length: u64;
    unsafe {
        output_bytes_32[..work_result_length as usize]
            .copy_from_slice(&core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize));

        output_address = output_bytes_32.as_ptr() as u64;
        output_length = output_bytes_32.len() as u64;
    }

    // write yield
    if n % 3 == 0 {
        if n != 9 {
            // n=3,6 should go through even though there is a panic, 9 does not.
            unsafe {
                checkpoint();
            }
            call_log(3, None, "corevm checkpoint");
        }
        let result42 = n + 42;
        write_result(result42, 7); // this should not be stored if n = 3, 6, 9 because its after the checkpoint
        unsafe {
            core::arch::asm!(
                "jalr x0, a0, 0", // djump(0+0) causes panic
            );
        }
        call_log(3, None, "corevm PANIC");
    }

    let yield_result = unsafe { oyield(output_address) };
    call_log(log_level(yield_result == OK), None, &format!("yield@n={:?}: expect OK, got {:?}", n, yield_result));
    return (output_address, output_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {

    let gas_result = unsafe { gas() };
    write_result(gas_result, 4);
    call_log(4, None, &format!("on_transfer gas: got {:?} (recorded at key 4)", gas_result));
    
    return (FIRST_READABLE_ADDRESS as u64, 0);
}