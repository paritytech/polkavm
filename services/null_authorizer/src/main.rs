#![no_std]
#![no_main]

use utils::constants::{FIRST_READABLE_ADDRESS};

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorize(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
