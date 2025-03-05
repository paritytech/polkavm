#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use simplealloc::SimpleAlloc;

#[global_allocator]
static ALLOCATOR: SimpleAlloc<4096> = SimpleAlloc::new();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, g: u64, out: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    // get the input start address form register a0
    let omega_7: u64; // refine input start address
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            out(reg) omega_7,
        );
    }

    let output_len: u64 = 8; // 4 bytes service index + 4 bytes amount

    // set the output address to register a0 and output length to register a1
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) output_len,
        );
    }
    // this equals to a0 = omega_7 + 4
    omega_7 + 4 // eliminate the first 4 bytes (workitem service index)
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

    // fetch all_accumulation_o 
    let mut start_address = omega_7 + 4 + 4; // 4 bytes time slot + 4 bytes service index
    let mut remaining_length = omega_8 - 4 - 4; // 4 bytes time slot + 4 bytes service index
    let all_accumulation_o = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };

    // fetch the number of accumulation_o
    let all_accumulation_o_discriminator_length = extract_discriminator(all_accumulation_o);
    let num_of_accumulation_o = decode_e(&all_accumulation_o[..all_accumulation_o_discriminator_length as usize]);

    // update the address pointer and remaining length
    start_address += all_accumulation_o_discriminator_length as u64;
    remaining_length -= all_accumulation_o_discriminator_length as u64;

    // set variables for storing work result address and length
    let mut work_result_address: u64 = 0;
    let mut work_result_length: u64 = 0;

    // set variables for storing auth output address and length
    let mut auth_output_address: u64 = 0;
    let mut auth_output_length: u64 = 0;
    
    for n in 0.. num_of_accumulation_o {
        // we only use the 0th accumulation_o
        if n > 0 {
            break;
        }
        // fetch work result prefix
        let accumulation_o = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };
        let work_result_prefix = &accumulation_o[..1];

        start_address += 1;
        remaining_length -= 1;
        
        // fetch work result
        if work_result_prefix[0] == 0 {  
            let accumulation_o = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };
            let work_result_discriminator_length = extract_discriminator(accumulation_o);
            work_result_length = if work_result_discriminator_length > 0 {
                decode_e(&accumulation_o[..work_result_discriminator_length as usize])
            } else {
                0
            };
            
            start_address += work_result_discriminator_length as u64;
            remaining_length -= work_result_discriminator_length as u64;
            
            // store the work result address
            work_result_address = start_address;
            
            // update the address pointer and remaining length
            start_address += work_result_length as u64;
            remaining_length -= work_result_length as u64;
        }

        // skip l, k which are two 32 bytes hashes
        start_address += 32 + 32;
        remaining_length -= 32 + 32;

        // fetch auth output prefix
        let accumulation_o = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };
        let auth_output_discriminator_length = extract_discriminator(accumulation_o);
        auth_output_length = if auth_output_discriminator_length > 0 {
            decode_e(&accumulation_o[..auth_output_discriminator_length as usize])
        } else {
            0
        };

        start_address += auth_output_discriminator_length as u64;
        remaining_length -= auth_output_discriminator_length as u64;

        // store the auth output address
        auth_output_address = start_address;

        // update the address pointer and remaining length
        start_address += auth_output_length as u64;
        remaining_length -= auth_output_length as u64;
    }

    // get amount address
    let amount_address = work_result_address + work_result_length - 4;
    let memo = [0u8; 128];

    let reciver: u64 = unsafe { *(work_result_address as *const u32) as u64 }; // reciver
    let amount: u64 = unsafe { *(amount_address as *const u32) as u64 }; // amount to transfer
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    
    let result = unsafe { transfer(reciver, amount, omega_9, omega_10) };
    result
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
    // let mut omega_7: u64 = u64::MAX; // 2^64 - 1
    // let omega_8: u64 = 0xFEFDE000; 

    // let result = unsafe { info(omega_7, omega_8) };

    // unsafe {
    //     let ptr1 = 0xFEFDE029 as *mut u32; // 2^32 − 2*ZZ − ZI − P (s) (Writable address)
    //     *ptr1 = 0; // set storage key = {0,0,0,0}
    // }

    // let ko: u64 = 0xFEFDE029;
    // let kz: u64 = 4; // storage key = {0,0,0,0}
    // let vo: u64 = 0xFEFDE021;
    // let vz: u64 = 8; // service's ba;ance
    // // balance to storage
    // let result = unsafe { write(ko, kz, vo, vz) };
    // result
    0
}

// some helpful functions
fn extract_discriminator(input: &[u8]) -> u8 {
    if input.is_empty() {
        return 0;
    }

    let first_byte = input[0];
    match first_byte {
        1..=127 => 1,
        128..=191 => 2,
        192..=223 => 3,
        224..=239 => 4,
        240..=247 => 5,
        248..=251 => 6,
        252..=253 => 7,
        254..=u8::MAX => 8,
        _ => 0,
    }
}

fn power_of_two(exp: u32) -> u64 {
    1 << exp
}

fn decode_e_l(encoded: &[u8]) -> u64 {
    let mut x: u64 = 0;
    for &byte in encoded.iter().rev() {
        x = x.wrapping_mul(256).wrapping_add(byte as u64);
    }
    x
}

fn decode_e(encoded: &[u8]) -> u64 {
    let first_byte = encoded[0];
    if first_byte == 0 {
        return 0;
    }
    if first_byte == 255 {
        return decode_e_l(&encoded[1..9]);
    }
    for l in 0..8 {
        let left_bound  = 256 - power_of_two(8 - l);
        let right_bound = 256 - power_of_two(8 - (l + 1));

        if (first_byte as u64) >= left_bound && (first_byte as u64) < right_bound {
            let x1 = (first_byte as u64) - left_bound;
            let x2 = decode_e_l(&encoded[1..(1 + l as usize)]);
            let x = x1 * power_of_two(8 * l) + x2;
            return x;
        }
    }
    0
}