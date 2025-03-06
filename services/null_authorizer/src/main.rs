#![no_std]
#![no_main]

use utils::{NONE};

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorize() -> u64 {
    0
}


