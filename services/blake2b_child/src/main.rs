#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::string::String;

const SIZE0 : usize = 0x100000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x100000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::functions::{call_log};
use utils::hash_functions::blake2b_hash;

const PAGE_SIZE: usize = 4096;
const PAGE_DIM: usize = 64;
const NUM_PAGES: usize = 9;
const PAGES_PER_ROW: usize = 3;
const TOTAL_ROWS: usize = PAGE_DIM * 3;
const TOTAL_COLS: usize = PAGE_DIM * 3;

const TOTAL_BYTES: usize = PAGE_SIZE * NUM_PAGES;

const ROWS_WITH_GHOST: usize = TOTAL_ROWS + 2;
const COLS_WITH_GHOST: usize = TOTAL_COLS + 2;


fn bytes_to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

#[polkavm_derive::polkavm_export]
extern "C" fn main() -> (u64, u64) {
    // Use predefined test data instead of reading from arbitrary memory address
    let input_data: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
    let input_slice = &input_data[..];
    
    // log the input data
    let input_hex = bytes_to_hex(input_slice);
    call_log(2, None, &format!("Input data: {}", input_hex));
    
    let hash_times = 100;
    let mut hash_result = [0u8; 32]; // Blake2b always returns 32 bytes
    
    for i in 0..hash_times {
        // hash the input data (for first iteration) or previous hash result
        if i == 0 {
            hash_result = blake2b_hash(input_slice);
        } else {
            hash_result = blake2b_hash(&hash_result);
        }
        let hash_hex = bytes_to_hex(&hash_result);
        call_log(2, None, &format!("Hash {}: {}", i + 1, hash_hex));
    }
    
    // log the final hash result
    let final_hash_hex = bytes_to_hex(&hash_result);
    call_log(2, None, &format!("Final hash result: {}", final_hash_hex));
    
    // return the address and length of the final hash result
    let result_address = hash_result.as_ptr() as u64;
    let result_length = hash_result.len() as u64;
    return (result_address, result_length);
}
