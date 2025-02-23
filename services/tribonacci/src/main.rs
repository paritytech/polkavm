#![no_std]
#![no_main]

extern crate alloc;

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
    #[polkavm_import(index = 3)]
    pub fn write(ko: u64, kz: u64, bo: u64, bz: u64) -> u64;
    #[polkavm_import(index = 18)]
    pub fn fetch(start_address: u64, offset: u64, maxlen: u64, omega_10: u64, omega_11: u64, omega_12: u64) -> u64;
    #[polkavm_import(index = 19)]
    pub fn export(out: u64, out_len: u64) -> u64;
}

pub const NONE: u64 = u64::MAX;

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    let mut buffer = [0u8; 16];
    let offset: u64 = 0;
    let maxlen: u64 = buffer.len() as u64;
    let result = unsafe { 
        fetch(
            buffer.as_mut_ptr() as u64, 
            offset,
            maxlen,
            5,
            0,
            0,
        )
    };

    if result != NONE {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let t_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let t_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
        let t_n_minus_2 = u32::from_le_bytes(buffer[12..16].try_into().unwrap());

        let new_t_n = t_n + t_n_minus_1 + t_n_minus_2;

        let n_new = n + 1;

        buffer[0..4].copy_from_slice(&n_new.to_le_bytes());
        buffer[4..8].copy_from_slice(&new_t_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&t_n.to_le_bytes());
        buffer[12..16].copy_from_slice(&t_n_minus_1.to_le_bytes());
    } else {
        buffer[0..4].copy_from_slice(&1_i32.to_le_bytes());
        buffer[4..8].copy_from_slice(&1_i32.to_le_bytes());
        buffer[8..12].copy_from_slice(&0_i32.to_le_bytes());
        buffer[12..16].copy_from_slice(&0_i32.to_le_bytes());
    }

    unsafe {
        export(buffer.as_ptr() as u64, buffer.len() as u64);
    }
    
    // set the output address to register a0 and output length to register a1
    let buffer_addr = buffer.as_ptr() as u64;
    let buffer_len = buffer.len() as u64;

    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) buffer_len,
        );
    }
    // this equals to a0 = buffer_addr
    buffer_addr
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

    // write TRIB result to storage
    let key = [0u8; 1];
    let n: u64 = unsafe { ( *(work_result_address as *const u32)).into() }; 
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..work_result_length as usize].copy_from_slice(&unsafe { core::slice::from_raw_parts(work_result_address as *const u8, work_result_length as usize) });
    let omega_7 = output_bytes_32.as_ptr() as u64;
    let omega_8 = output_bytes_32.len() as u64;

    if n % 3 == 0 {
        // trigger PANIC
        unsafe {
            core::arch::asm!(
                "li a0, 0",
                "li a1, 1",
                "jalr x0, a0, 0", // djump(0+0) causes panic
            );
        }
    } else if n % 2 == 0 {
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
