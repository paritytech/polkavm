#![no_std]
#![no_main]

use utils::constants::{FIRST_READABLE_ADDRESS};
use utils::functions::{parse_refine_args};

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 99)]
    pub fn delay(seconds: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let (_wi_service_index, wi_payload_start_address, _wi_payload_length, _wphash) =
    if let Some(args) = parse_refine_args(start_address, length)
    {
        (
            args.wi_service_index,
            args.wi_payload_start_address,
            args.wi_payload_length,
            args.wphash,
        )
    } else {
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    let seconds: u64 = unsafe { *(wi_payload_start_address as *const u32) as u64 };
    unsafe {
        delay(seconds);
    }
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
