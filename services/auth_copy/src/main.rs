#![no_std]
#![no_main]

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::parse_accumulate_args;
use utils::host_functions::assign;

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let auth_output_address = start_address + length - 32;
    let output_len = 32;
    return (auth_output_address, output_len);
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

    let hash = unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) };
    assert!(32 == hash.len());
    const N_Q: usize = 80;
    let mut authorization_hashes = [0u8; 32 * N_Q];
    for i in 0..N_Q {
        let start = i * 32;
        authorization_hashes[start..start + 32].copy_from_slice(hash);
    }

    let authorization_hashes_address = authorization_hashes.as_ptr() as u64;
    unsafe {
        assign(0, authorization_hashes_address);
        assign(1, authorization_hashes_address);
    }
    return (work_result_address, work_result_length);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
