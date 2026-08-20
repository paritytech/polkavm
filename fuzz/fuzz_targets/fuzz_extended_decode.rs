#![no_main]

use libfuzzer_sys::fuzz_target;
use polkavm_common::program::{extended_assert_decode_is_deterministic, ISA_Latest64};

fuzz_target!(|code: &[u8]| {
    extended_assert_decode_is_deterministic(ISA_Latest64, code);
});
