#![no_std]
#![no_main]

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::{call_log};
extern crate alloc;
use alloc::format;

use utils::host_functions::{
    gas
};

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorize(_start_address: u64, _length: u64) -> (u64, u64) {
    for i in 0..3 {
        let gas_result = unsafe { gas() };
        call_log(2, None, &format!("null_authorizer gas call {:?} gas_result: {:?}", i, gas_result));
    }
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
