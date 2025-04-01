#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{parse_accumulate_args, parse_refine_args, parse_transfer_args, call_log, write_result};
use utils::host_functions::{oyield, read, write, transfer, gas};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let (_wi_service_index, wi_payload_start_address, wi_payload_length, _wphash) =
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

    return (wi_payload_start_address, wi_payload_length);
}

#[no_mangle]
static mut output_bytes_32: [u8; 32] = [0; 32];

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    // parse accumulate args
    let (_timeslot, _service_index, work_result_address, work_result_length) =
        if let Some(args) = parse_accumulate_args(start_address, length, 0) {
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

    let service0: u64 = unsafe { (*(service0_bytes_start_addr as *const u32)).into() };
    let service1: u64 = unsafe { (*(service1_bytes_start_addr as *const u32)).into() };

    // read the two services' storage
    unsafe {
        read(
            service0,
            key_address,
            key_length,
            buffer0.as_ptr() as u64,
            0 as u64,
            buffer0.len() as u64,
        );
        read(
            service1,
            key_address,
            key_length,
            buffer1.as_ptr() as u64,
            0 as u64,
            buffer1.len() as u64,
        );
    };

    let s0_n = u32::from_le_bytes(buffer0[0..4].try_into().unwrap());
    let s0_vn = u32::from_le_bytes(buffer0[4..8].try_into().unwrap());
    let s0_vnminus1 = u32::from_le_bytes(buffer0[8..12].try_into().unwrap());

    let s1_vn = u32::from_le_bytes(buffer1[4..8].try_into().unwrap());
    let s1_vnminus1 = u32::from_le_bytes(buffer1[8..12].try_into().unwrap());

    let m_n = s0_n;

    // calculate the new values and write to storage
    call_log(2, None, &format!("meg {:?} read from service {:?} fib(n={:?})={:?}", m_n, service0, m_n, s0_vn));
    call_log(2, None, &format!("meg {:?} read from service {:?} trib(n={:?})={:?}", m_n, service1, m_n, s1_vn));
    let m_vn = s0_vn + s1_vn;
    let m_vnminus1 = s0_vnminus1 + s1_vnminus1;
    let memo = [0u8; 128];
    let send0 = (m_n % 3 == 0) || (m_n % 3 == 2);
    let send1 = (m_n % 3 == 1) || (m_n % 3 == 2);
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    let amount0: u64 = m_n as u64;
    let amount1: u64 = (m_n * 2 + 1) as u64;
    let gas_avail: u64 = 100;
    let mut num_transfers = 0;

    for i in 0..m_n {
        if send0 {
            let result0 = unsafe { transfer(service0, amount0, gas_avail, omega_10) };
            num_transfers += 1;
            call_log(2, None, &format!("{:?} transfer(dest:{:?}, amount={:?}, gas_avail={:?}) Result: {:?}", i, service0, amount0, gas_avail, result0));
        }
        if send1 {
            let result1 = unsafe { transfer(service1, amount1, gas_avail, omega_10) };
            num_transfers += 1;
            call_log(2, None, &format!("{:?} transfer(dest:{:?}, amount={:?}, gas_avail={:?}) Result: {:?}", i, service1, amount1, gas_avail, result1));
        }
    }

    buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
    buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
    buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());

    call_log(2, None, &format!("meg({:?})={:?}", m_n, m_vn));
    unsafe { write(key.as_ptr() as u64, key.len() as u64, buffer.as_ptr() as u64, buffer.len() as u64); }
    call_log(2, None, &format!("meg {:?} write(n={:?}) num_transfers={:?}", m_n, m_vn, num_transfers));

    // Option<hash> test
    // pad result to 32 bytes
    let output_address: u64;
    let output_length: u64;
    unsafe {
        output_bytes_32[..buffer.len() as usize].copy_from_slice(&buffer);
        output_address = output_bytes_32.as_ptr() as u64;
        output_length = output_bytes_32.len() as u64;
    }

    unsafe { oyield(output_address); }

    return (output_address, output_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(start_address: u64, length: u64) -> (u64, u64) {
    // Note: This part executes only if there are deferred transfers AND this service is the receiver.
    let gas_result = unsafe { gas() };
    write_result(gas_result, 4);
    call_log(2, None, &format!("Megatron on_transfer gas: got {:?} (recorded at key 4)", gas_result));

    return (FIRST_READABLE_ADDRESS as u64, 0);
}
