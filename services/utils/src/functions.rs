use crate::constants::*;
use crate::host_functions::*;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}

// Helpful functions and structs for refine
#[repr(C)]
#[derive(Debug, Clone)]
pub struct RefineArgs {
    pub wi_service_index: u32,
    pub wi_payload_start_address: u64,
    pub wi_payload_length: u64,
    pub wphash: [u8; 32],
}

impl Default for RefineArgs {
    fn default() -> Self {
        Self {
            wi_service_index: 0,
            wi_payload_start_address: 0,
            wi_payload_length: 0,
            wphash: [0u8; 32],
        }
    }
}

pub fn parse_refine_args(mut start_address: u64, mut remaining_length: u64) -> Option<RefineArgs> {
    if remaining_length < 4 {
        return None;
    }
    let wi_service_index = unsafe { ( *(start_address as *const u32)).into() }; 
    start_address += 4;
    remaining_length = remaining_length.saturating_sub(4);

    if remaining_length == 0 {
        return None;
    }
    let payload_slice = unsafe {
        core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
    };
    let discriminator_len = extract_discriminator(payload_slice);
    let payload_len = if discriminator_len > 0 {
        decode_e(&payload_slice[..discriminator_len as usize])
    } else {
        0
    };

    start_address += discriminator_len as u64;
    remaining_length = remaining_length.saturating_sub(discriminator_len as u64);

    if remaining_length < payload_len {
        return None;
    }
    let wi_payload_start_address = start_address;
    let wi_payload_length = payload_len;
    start_address += payload_len;
    remaining_length = remaining_length.saturating_sub(payload_len);

    if remaining_length < 32 {
        return None;
    }
    let mut wphash = [0u8; 32];
    let hash_slice = unsafe {
        core::slice::from_raw_parts(start_address as *const u8, 32)
    };
    wphash.copy_from_slice(hash_slice);

    Some(RefineArgs {
        wi_service_index,
        wi_payload_start_address,
        wi_payload_length,
        wphash,
    })
}

pub fn setup_page(segment: &[u8]) {
    if segment.len() < 8 {
        return call_info("setup_page: buffer too small");
    }

    let (m, page_id) = (
        u32::from_le_bytes(segment[0..4].try_into().unwrap()) as u64,
        u32::from_le_bytes(segment[4..8].try_into().unwrap()) as u64,
    );

    let page_address = page_id * PAGE_SIZE;
    let data = &segment[8..];

    if unsafe { zero(m, page_id, 1) } != OK {
        return call_info("setup_page: zero failed");
    }

    if unsafe { poke(m, data.as_ptr() as u64, page_address, PAGE_SIZE) } != OK {
        call_info("setup_page: poke failed");
    }
}

pub fn get_page(vm_id: u32, page_id: u32) -> [u8; SEGMENT_SIZE as usize] {
    let mut result = [0u8; SEGMENT_SIZE as usize];
    
    result[0..4].copy_from_slice(&vm_id.to_le_bytes());
    result[4..8].copy_from_slice(&page_id.to_le_bytes());

    let page_address = (page_id as u64) * PAGE_SIZE;
    let result_address = unsafe { result.as_mut_ptr().add(8) };
    let result_length = (SEGMENT_SIZE - 8) as u64;

    let peek_result = unsafe{ peek(vm_id as u64, result_address as u64, page_address as u64, result_length) };
    if peek_result != OK {
        call_info("get_page: peek failed");
    }
    result
}

pub fn serialize_gas_and_registers(gas: u64, child_vm_registers: &[u64; 13]) -> [u8; 112] {
    let mut result = [0u8; 112];

    result[0..8].copy_from_slice(&gas.to_le_bytes());

    for (i, &reg) in child_vm_registers.iter().enumerate() {
        let start = 8 + i * 8;
        result[start..start + 8].copy_from_slice(&reg.to_le_bytes());
    }
    result
}

pub fn deserialize_gas_and_registers(data: &[u8; 112]) -> (u64, [u64; 13]) {
    let gas = u64::from_le_bytes(data[0..8].try_into().unwrap());

    let mut child_vm_registers = [0u64; 13];
    for i in 0..13 {
        let start = 8 + i * 8;
        child_vm_registers[i] = u64::from_le_bytes(data[start..start + 8].try_into().unwrap());
    }

    (gas, child_vm_registers)
}

pub fn initialize_pvm_registers() -> [u64; 13] {
    let mut registers = [0u64; 13];
    registers[0] = INIT_RA;
    registers[1] = (1u64 << 32) - 2 * Z_Z - Z_I;
    registers[7] = (1u64 << 32) - Z_Z - Z_I;
    registers[8] = 0;
    return registers
}

// Helpful functions and structs for accumulate
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AccumulateArgs {
    pub t: u32,
    pub s: u32,
    pub h: [u8; 32],
    pub e: [u8; 32],
    pub a: [u8; 32],
    pub o_ptr: u64,
    pub o_len: u64,
    pub y: [u8; 32],
    pub work_result_ptr: u64,
    pub work_result_len: u64,
}

impl Default for AccumulateArgs {
    fn default() -> Self {
        Self {
            t: 0,
            s: 0,
            h: [0u8; 32],
            e: [0u8; 32],
            a: [0u8; 32],
            o_ptr: 0,
            o_len: 0,
            y: [0u8; 32],
            work_result_ptr: 0,
            work_result_len: 0,
        }
    }
}

pub fn parse_accumulate_args(start_address: u64, length: u64, m: u64) -> Option<AccumulateArgs> {
    if length == 0 {
        return None;
    }
    let mut current_address = start_address;
    let mut remaining_length = length;

    let mut args = AccumulateArgs::default();

    if remaining_length < 8 {
        return None;
    }
    let t_slice = unsafe {
        core::slice::from_raw_parts(current_address as *const u8, 4)
    };
    let s_slice = unsafe {
        core::slice::from_raw_parts((current_address + 4) as *const u8, 4)
    };
    let global_t = u32::from_le_bytes(t_slice[0..4].try_into().unwrap());
    let global_s = u32::from_le_bytes(s_slice[0..4].try_into().unwrap());

    args.t = global_t;
    args.s = global_s;

    current_address += 8;
    remaining_length = remaining_length.saturating_sub(8);

    let full_slice = unsafe {
        core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize)
    };
    let discriminator_len = extract_discriminator(full_slice);
    if discriminator_len as usize > full_slice.len() {
        return None;
    }
    let num_of_operands = decode_e(&full_slice[..discriminator_len as usize]);

    current_address += discriminator_len as u64;
    remaining_length = remaining_length.saturating_sub(discriminator_len as u64);

    if m >= num_of_operands {
        return None;
    }

    for i in 0..num_of_operands {
        if remaining_length < 96 {
            return None;
        }
        let hash_slice = unsafe {
            core::slice::from_raw_parts(current_address as *const u8, 96)
        };
        args.h.copy_from_slice(&hash_slice[0..32]);
        args.e.copy_from_slice(&hash_slice[32..64]);
        args.a.copy_from_slice(&hash_slice[64..96]);
        current_address += 96;
        remaining_length = remaining_length.saturating_sub(96);

        {
            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize)
            };
            let auth_output_discriminator_len = extract_discriminator(accumulation_slice);
            let auth_output_len = if auth_output_discriminator_len > 0 {
                decode_e(&accumulation_slice[..auth_output_discriminator_len as usize])
            } else {
                0
            };

            current_address += auth_output_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(auth_output_discriminator_len as u64);

            args.o_ptr = current_address;
            args.o_len = remaining_length.min(auth_output_len);

            current_address += auth_output_len;
            remaining_length = remaining_length.saturating_sub(auth_output_len);
        }

        if remaining_length < 32 {
            return None;
        }
        let y_slice = unsafe {
            core::slice::from_raw_parts(current_address as *const u8, 32)
        };
        args.y.copy_from_slice(y_slice);
        current_address += 32;
        remaining_length = remaining_length.saturating_sub(32);

        let accumulation_slice = unsafe {
            core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize)
        };
        if accumulation_slice.is_empty() {
            return None;
        }
        let work_result_prefix = accumulation_slice[0];
        current_address += 1;
        remaining_length = remaining_length.saturating_sub(1);

        if work_result_prefix == 0 {
            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize)
            };
            let wr_discriminator_len = extract_discriminator(accumulation_slice);
            let wr_len = if wr_discriminator_len > 0 {
                decode_e(&accumulation_slice[..wr_discriminator_len as usize])
            } else {
                0
            };

            current_address += wr_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(wr_discriminator_len as u64);

            args.work_result_ptr = current_address;
            args.work_result_len = remaining_length.min(wr_len);

            current_address += wr_len;
            remaining_length = remaining_length.saturating_sub(wr_len);
        }

        if i == m {
            return Some(args);
        }
    }
    None
}

pub fn write_result(result: u64, key: u8) {
    let key_bytes = key.to_le_bytes();
    let result_bytes = result.to_le_bytes();
    unsafe {
        write(key_bytes.as_ptr() as u64, key_bytes.len() as u64, result_bytes.as_ptr() as u64, result_bytes.len() as u64);
    }
}

pub fn call_info(msg: &str) {
    let ascii_bytes = msg.as_bytes();
    let ascii_address = ascii_bytes.as_ptr() as u64;
    let ascii_length = ascii_bytes.len() as u64;
    unsafe { log(2, 0, 0, ascii_address, ascii_length) };
}

// Helpful functions for both refine and accumulate
pub fn extract_discriminator(input: &[u8]) -> u8 {
    if input.is_empty() {
        return 0;
    }

    let first_byte = input[0];
    match first_byte {
        0..=127 => 1,
        128..=191 => 2,
        192..=223 => 3,
        224..=239 => 4,
        240..=247 => 5,
        248..=251 => 6,
        252..=253 => 7,
        254..=u8::MAX => 8,
    }
}

pub fn power_of_two(exp: u32) -> u64 {
    1 << exp
}

pub fn decode_e_l(encoded: &[u8]) -> u64 {
    let mut x: u64 = 0;
    for &byte in encoded.iter().rev() {
        x = x.wrapping_mul(256).wrapping_add(byte as u64);
    }
    x
}

pub fn decode_e(encoded: &[u8]) -> u64 {
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
