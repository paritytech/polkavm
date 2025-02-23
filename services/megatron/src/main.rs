#![no_std]
#![no_main]

extern crate alloc;

use core::convert::TryInto;
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
    #[polkavm_import(index = 2)]
    pub fn read(service: u64, key_ptr: u64, key_len: u64, out: u64, out_offet: u64, out_len: u64) -> u64;
    #[polkavm_import(index = 3)]
    pub fn write(key_ptr: u64, key_len: u64, value: u64, value_len: u64) -> u64;
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    // get the input start address form register a0
    let omega_7: u64; // refine input start address 0xFEFF0000
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            out(reg) omega_7,
        );
    }

    let output_len: u64 = 8; // 4 bytes service index + 4 bytes service index

    // set the output address to register a0 and output length to register a1
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) output_len,
        );
    }
    // this equals to a0 = omega_7 + 4
    omega_7 + 4 // eliminate the first 4 bytes (workitem service index) 0xFEFF0000 + 4 = 0xFEFF0004
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u64 {
    // read the input start address from register a0 and input length from register a1
    let omega_7: u64;
    let omega_8: u64;
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

    // get the two service index from the input
    let service0_bytes_start_addr: u64 = work_result_address; // 4 bytes service index
    let service1_bytes_start_addr: u64 = work_result_address + work_result_length - 4; // 4 bytes service index

    let buffer0 = [0u8; 12];
    let buffer1 = [0u8; 12];
    let key = [0u8; 1];
    let mut buffer = [0u8; 12];

    let service0: u64 = unsafe { ( *(service0_bytes_start_addr as *const u32)).into() }; 
    let service1: u64 = unsafe { ( *(service1_bytes_start_addr as *const u32)).into() }; 

    // read the two services' storage
    unsafe {
        read(service0, key.as_ptr() as u64, key.len() as u64, buffer0.as_ptr() as u64, 0 as u64, buffer0.len() as u64);
        read(service1, key.as_ptr() as u64, key.len() as u64, buffer1.as_ptr() as u64, 0 as u64, buffer1.len() as u64);
    }
    let s0_n = u32::from_le_bytes(buffer0[0..4].try_into().unwrap());
    let s0_vn = u32::from_le_bytes(buffer0[4..8].try_into().unwrap());
    let s0_vnminus1 = u32::from_le_bytes(buffer0[8..12].try_into().unwrap());

    let s1_vn = u32::from_le_bytes(buffer1[4..8].try_into().unwrap());
    let s1_vnminus1 = u32::from_le_bytes(buffer1[8..12].try_into().unwrap());

    // calculate the new values and write to storage
    let m_n = s0_n;
    let m_vn = s0_vn + s1_vn;
    let m_vnminus1 = s0_vnminus1 + s1_vnminus1;

    buffer[0..4].copy_from_slice(&m_n.to_le_bytes());
    buffer[4..8].copy_from_slice(&m_vn.to_le_bytes());
    buffer[8..12].copy_from_slice(&m_vnminus1.to_le_bytes());
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, buffer.as_ptr() as u64, buffer.len() as u64);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..buffer.len()].copy_from_slice(&buffer);
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;

    if m_n % 3 == 0 {
        // trigger PANIC
        unsafe {
            core::arch::asm!(
                "li a0, 0",
                "li a1, 1",
                "jalr x0, a0, 0", // djump(0+0) causes panic
            );
        }
    } else if m_n % 2 == 0 {
        // Write to invalid memory address to obtain an empty hash
        unsafe {
            core::arch::asm!(
                "li a1, 1"
            );
        }   
        return 1;
    }

    // set the result length to register a1
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) omega_8,
        );
    }
    // this equals to a0 = omega_7
    omega_7
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u64 {
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

