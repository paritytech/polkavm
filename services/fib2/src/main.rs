#![no_std]
#![no_main]

use utils::{NONE, OK, PAGE_SIZE, SEGMENT_SIZE};
use utils::{parse_refine_args, parse_wrangled_operand_tuple};
use utils::{call_info, setup_page, get_page, serialize_gas_and_registers, deserialize_gas_and_registers};
use utils::{write_result};

use utils::{historical_lookup, fetch, export, machine, peek, poke, zero, void, invoke, expunge, log};
use utils::{gas, lookup, read, write, info, bless, assign, checkpoint, new, upgrade, eject, query, solicit, forget, oyield};

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
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

    let mut prev_n : u32 = 0;
    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());

        let new_fib_n = fib_n + fib_n_minus_1;
        prev_n = n;
        buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
        buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());

    } else {
        buffer[0..4].copy_from_slice(&1_u32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_u32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_u32.to_le_bytes());
    }

    let buffer_addr = buffer.as_ptr() as u64;
    let buffer_len = buffer.len() as u64;

    unsafe {
        export(buffer_addr, buffer_len);
    }

    // Put N additional exports which are identical FOR NOW
    for i in 0..prev_n {
        buffer[12..16].copy_from_slice(&(i + 1).to_le_bytes());
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

    // write FIB result to storage
    let key = [0u8; 1];
    let n: u64 = unsafe { ( *(work_result_address as *const u32)).into() };
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Prepare some keys and hashes.
    let jam_key: [u8; 3] = [b'j', b'a', b'm'];
    let dot_val: [u8; 3] = [b'D', b'O', b'T'];
    
    // blake2b("jam") = 0x6a0d4a19d199505713fc65f531038e73f1d885645632c8ae503c4f0c4d5e19a7
    let jam_key_hash: [u8; 32] = [
        0x6a, 0x0d, 0x4a, 0x19, 0xd1, 0x99, 0x50, 0x57,
        0x13, 0xfc, 0x65, 0xf5, 0x31, 0x03, 0x8e, 0x73,
        0xf1, 0xd8, 0x85, 0x64, 0x56, 0x32, 0xc8, 0xae,
        0x50, 0x3c, 0x4f, 0x0c, 0x4d, 0x5e, 0x19, 0xa7
    ];

    // blake2b("dot") = 0xbfa9bb0fa4968747e63d3cf1e74a49ddc4a6eca89a6a6f339da3337fd2eb5507
    let dot_val_hash: [u8; 32] = [
        0xbf, 0xa9, 0xbb, 0x0f, 0xa4, 0x96, 0x87, 0x47,
        0xe6, 0x3d, 0x3c, 0xf1, 0xe7, 0x4a, 0x49, 0xdd,
        0xc4, 0xa6, 0xec, 0xa8, 0x9a, 0x6a, 0x6f, 0x33,
        0x9d, 0xa3, 0x33, 0x7f, 0xd2, 0xeb, 0x55, 0x07
    ];

    let JAM_KEY_ADDRESS: u64 = jam_key.as_ptr() as u64;
    let JAM_KEY_LENGTH: u64 = jam_key.len() as u64;

    let DOT_VAL_ADDRESS: u64 = dot_val.as_ptr() as u64;
    let DOT_VAL_LENGTH: u64 = dot_val.len() as u64;

    
    let JAM_KEY_HASH_ADDRESS: u64 = jam_key_hash.as_ptr() as u64;
    let JAM_KEY_HASH_LENGTH: u64 = jam_key_hash.len() as u64;

    let DOT_VAL_HASH_ADDRESS: u64 = dot_val_hash.as_ptr() as u64;
    let DOT_VAL_HASH_LENGTH: u64 = dot_val_hash.len() as u64;
    
    let info_bytes = [0u8; 100];
    let INFO_ADDRESS: u64 = info_bytes.as_ptr() as u64;
    let INFO_LENGTH: u64 = info_bytes.len() as u64;

    let mut buffer = [0u8; 256];
    let buffer_address = buffer.as_ptr() as u64;
    let buffer_length = buffer.len() as u64;

    // Depending on what "n" is, test different host functions
    if n == 1 {
        let read_none_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // NONE
        write_result(read_none_result, 1);

        let write_result1 = unsafe { write(JAM_KEY_ADDRESS, JAM_KEY_LENGTH, DOT_VAL_ADDRESS, DOT_VAL_LENGTH) }; // OK
        write_result(write_result1, 2);

        let read_ok_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // OK
        write_result(read_ok_result, 5);

        let forget_result = unsafe { forget(JAM_KEY_ADDRESS, 0) }; // HUH: not any lookup meet the condition
        write_result(forget_result, 6);
    } else if n == 2 {
        let read_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // OK: 3
        write_result(read_result, 1);

        let write_result1 = unsafe { write(JAM_KEY_ADDRESS, JAM_KEY_LENGTH, 0, 0) }; // delete OK
        write_result(write_result1, 2);

        let read_ok_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // NONE: deleted
        write_result(read_ok_result, 5);

        let solicit_result = unsafe { solicit(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: insert one timeslot
        write_result(solicit_result, 6);
    } else if n == 3 {
        let solicit_result = unsafe { solicit(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: initialize empty timeslot
        write_result(solicit_result, 1);

        let query_jamhash_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 0
        write_result(query_jamhash_result, 2);

        let query_none_result = unsafe { query(DOT_VAL_HASH_ADDRESS, DOT_VAL_LENGTH) }; // NONE: no such key
        write_result(query_none_result, 5);
    } else if n == 4 {
        let forget_result = unsafe { forget(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: insert one timeslot
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 2
        write_result(query_jamhash_result, 2);

        let lookup_none_result = unsafe { lookup(SERVICE_INDEX, DOT_VAL_HASH_ADDRESS, buffer_address, 0, DOT_VAL_LENGTH) }; // NONE: never written
        write_result(lookup_none_result, 5);

        let assign_result = unsafe { assign(1000, JAM_KEY_ADDRESS) }; // CORE: invalid core number
        write_result(assign_result, 6);
    } else if n == 5 {
        let lookup_result = unsafe { lookup(SERVICE_INDEX, JAM_KEY_HASH_ADDRESS, buffer_address, 0, JAM_KEY_LENGTH) }; // OK: |v| = 3
        write_result(lookup_result, 1);

        let read_ok_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 2
        write_result(read_ok_result, 2);

        let eject_who_result = unsafe { eject(SERVICE_INDEX, JAM_KEY_HASH_ADDRESS) }; // WHO: invalid service index
        write_result(eject_who_result, 5);

        let overflow_s = 0xFFFFFFFFFFFFu64;
        let bless_who_result = unsafe { bless(overflow_s, 0, 0, JAM_KEY_HASH_ADDRESS, 0) }; // WHO: invalid service index
        write_result(bless_who_result, 6);
    } else if n == 6 {
        let solicit_result = unsafe { solicit(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: insert one timeslot
        write_result(solicit_result, 1);

        let query_jamhash_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 3
        write_result(query_jamhash_result, 2);

        let core_index = 0;

        let mut auth_hashes = [0; 2560];
        let mut i = 0;
        while i < 80 {
            let offset = i * 32;
            auth_hashes[offset..offset + 32].copy_from_slice(&jam_key_hash);
            i += 1;
        }

        let assign_ok_result = unsafe { assign(core_index, auth_hashes.as_ptr() as u64) }; // OK
        write_result(assign_ok_result, 5);
    } else if n == 7 {
        let forget_result = unsafe { forget(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: adjust 3 timeslots to 2 timeslots
        write_result(forget_result, 1);

        let query_jamhash_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 2
        write_result(query_jamhash_result, 2);
    } else if n == 8 {
        let lookup_result = unsafe { lookup(SERVICE_INDEX, JAM_KEY_HASH_ADDRESS, buffer_address, 0, JAM_KEY_LENGTH) }; // OK: |v| = 3
        write_result(lookup_result, 1);

        let query_jamhash_result = unsafe { query(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: 2
        write_result(query_jamhash_result, 2);
    } else if n == 9 {
        let read_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // OK |v| = 3
        write_result(read_result, 1);

        let write_result1 = unsafe { write(JAM_KEY_ADDRESS, JAM_KEY_LENGTH, 0, 0) }; // delete OK
        write_result(write_result1, 2);

        let read_ok_result = unsafe { read(SERVICE_INDEX, JAM_KEY_ADDRESS, JAM_KEY_LENGTH, buffer_address, 0, buffer_length) }; // NONE: deleted
        write_result(read_ok_result, 5);

        let solicit_result = unsafe { solicit(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH) }; // OK: insert one timeslot
        write_result(solicit_result, 6);
    } else if n == 1024 {
        let g: u64 = 911911; // this will trigger error: ServiceItemTooLow: Accumulated gas is below the service minimum.
        let m: u64 = 911911;
        let new_result = unsafe { new(JAM_KEY_HASH_ADDRESS, JAM_KEY_LENGTH, g, m) }; // OK
        write_result(new_result, 1);
        
        // this will trigger error: BadCodeHash: Work result code hash doesn't match the one expected for the service.
        let upgrade_result = unsafe { upgrade(JAM_KEY_HASH_ADDRESS, g, m) };
        write_result(upgrade_result, 2);

        let s:u32 = 911;
        let s_bytes = s.to_le_bytes();
        let gas_bytes = g.to_le_bytes();
        let mut bless_input = [0u8; 12];
        bless_input[..4].copy_from_slice(&s_bytes);
        bless_input[4..12].copy_from_slice(&gas_bytes);
        let bless_input_address = bless_input.as_ptr() as u64;
    
        let bless_ok_result = unsafe { bless(0, 1, 1, bless_input_address, 1) };
        write_result(bless_ok_result, 5);
    }


    // write info to 8
    let info_result = unsafe { info(SERVICE_INDEX, buffer_address) };
    write_result(info_result, 8);

    // write gas to 9
    let gas_result = unsafe { gas() };
    write_result(gas_result, 9);

    // Prepare an output buffer (pad result to 32 bytes).
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize].copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;

    // write yield
    if n % 3 == 0 {
        if n != 9 { // n=3,6 should go through even though there is a panic, 9 does not.
            let gas_result = unsafe { checkpoint() };
        }
        let result42 = n + 42;
        write_result(result42, 7); // this should not be stored if n = 3, 6, 9 because its after the checkpoint
        unsafe {
            core::arch::asm!(
                "li a0, 0",
                "li a1, 1",
                "jalr x0, a0, 0", // djump(0+0) causes panic
            );
        }
    } else {
    }
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
