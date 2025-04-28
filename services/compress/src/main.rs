#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE : usize = (1 << 24) - (1 << 12);

// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE);

// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE> = SimpleAlloc::new();


use utils::functions::{call_log};
use utils::constants::{FIRST_READABLE_ADDRESS, PAGE_SIZE, SEGMENT_SIZE};


use lzss::{Lzss, SliceReader, SliceWriter};
use rand::{RngCore, SeedableRng};
use rand::rngs::SmallRng;


#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    type MyLzss = Lzss<10, 4, 0x20, { 1 << 10 }, { 2 << 10 }>;

    const SIZE: usize = SEGMENT_SIZE as usize * 1;
    // let mut input = [0u8; SIZE as usize];
    // // put some data in the input buffer
    // for i in 0..SIZE as usize {
    //     input[i] = (i % 256) as u8;
    // }

    let mut input = [0u8; SIZE];
    let mut rng = SmallRng::seed_from_u64(0x12345678);
    rng.fill_bytes(&mut input);


    // let input = b"Example Data";
    let mut output = [0; SIZE as usize];
    let compress_result = MyLzss::compress(
        SliceReader::new(&input),
        SliceWriter::new(&mut output),
    );
    call_log(2, None, &format!("Result: {:?}", compress_result));
    call_log(2, None, &format!("Compress Output: {:?}", output));

    // Decompress the data
    let decompress_result = MyLzss::decompress(
        SliceReader::new(&output),
        SliceWriter::new(&mut input),
    );

    call_log(2, None, &format!("Decompress Result: {:?}", decompress_result));
    call_log(2, None, &format!("Decompress Output: {:?}", input));







    (FIRST_READABLE_ADDRESS as u64, 0)
}



#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
