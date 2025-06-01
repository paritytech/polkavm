/*
 * Table of Contents:
 * 1. Parsing Functions
 *    - parse_refine_args and RefineArgs struct
 *    - parse_accumulate_args and AccumulateArgs struct
 *    - parse_transfer_args and TransferArgs struct
 *    - parse_standard_program_initialization_args and StandardProgramInitializationArgs struct
 *
 * 2. Child VM Related Functions
 *    - standard_program_initialization_for_child
 *    - setup_page
 *    - get_page
 *    - serialize_gas_and_registers
 *    - deserialize_gas_and_registers
 *    - initialize_pvm_registers
 *
 * 3. Logging Functions
 *    - write_result
 *    - call_log
 *
 * 4. Helper Functions
 *    - extract_discriminator
 *    - power_of_two
 *    - decode_e_l
 *    - decode_e
 *    - ceiling_divide
 *    - p_func
 *    - z_func
 */

extern crate alloc;
use alloc::format;

use crate::constants::*;
use crate::host_functions::*;

// Parse refine args
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
    let mut args = RefineArgs::default();
    if remaining_length < 4 {
        return None;
    }
    let t_full_slice = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };
    let t_len = extract_discriminator(t_full_slice) as u64;
    if t_len == 0 || remaining_length < t_len {
        return None;
    }

    // Decode t and update pointers
    let t_slice = &t_full_slice[..t_len as usize];
    args.wi_service_index = decode_e(t_slice) as u32;
    start_address += t_len;
    remaining_length -= t_len;

    let payload_slice = unsafe { core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize) };
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
    args.wi_payload_start_address = start_address;
    args.wi_payload_length = payload_len;
    start_address += payload_len;
    remaining_length = remaining_length.saturating_sub(payload_len);

    if remaining_length < 32 {
        return None;
    }
    let hash_slice = unsafe { core::slice::from_raw_parts(start_address as *const u8, 32) };
    args.wphash.copy_from_slice(hash_slice);

    return Some(args);
}

// Parse accumulate args
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
    pub g: u64,
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
            g: 0,
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

    //    call_log(2, None, &format!("parse_accumulate_args start_address={} length={}", start_address, length));

    // Create a slice of the available data to parse t
    let t_full_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
    let t_len = extract_discriminator(t_full_slice) as u64;
    if t_len == 0 || remaining_length < t_len {
        return None;
    }

    // Decode t and update pointers
    let t_slice = &t_full_slice[..t_len as usize];
    args.t = decode_e(t_slice) as u32;
    current_address += t_len;
    remaining_length -= t_len;

    // Create a new slice for the remaining data to parse s
    let s_full_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
    let s_len = extract_discriminator(s_full_slice) as u64;
    if s_len == 0 || remaining_length < s_len {
        return None;
    }

    // Decode s and update pointers
    let s_slice = &s_full_slice[..s_len as usize];
    args.s = decode_e(s_slice) as u32;
    current_address += s_len;
    remaining_length -= s_len;

    let full_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
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
        let hash_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 96) };
        args.h.copy_from_slice(&hash_slice[0..32]);
        args.e.copy_from_slice(&hash_slice[32..64]);
        args.a.copy_from_slice(&hash_slice[64..96]);
        current_address += 96;
        remaining_length = remaining_length.saturating_sub(96);

        {
            let accumulation_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
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
        let y_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 32) };
        args.y.copy_from_slice(y_slice);
        current_address += 32;
        remaining_length = remaining_length.saturating_sub(32);

        // // 0.6.5 -- add gas limit
        // let g_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 8) };
        // let global_g= u64::from_le_bytes(g_slice[0..8].try_into().unwrap());
        // args.g = global_g;
        // current_address += 8;
        // remaining_length = remaining_length.saturating_sub(8);


        // 0.6.5 -- special case for g (should be removed in the future)
        let g_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
        let g_len = extract_discriminator(g_slice);
        let g = if g_len > 0 {
            decode_e(&g_slice[..g_len as usize])
        } else {
            0
        };
        args.g = g;

        current_address += g_len as u64;
        remaining_length = remaining_length.saturating_sub(g_len as u64);
        // 0.6.5 -- special case for g (should be removed in the future)


        let accumulation_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
        if accumulation_slice.is_empty() {
            return None;
        }
        let work_result_prefix = accumulation_slice[0];
        current_address += 1;
        remaining_length = remaining_length.saturating_sub(1);

        if work_result_prefix == 0 {
            let accumulation_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
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

// Parse transfer args
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TransferArgs {
    pub t: u32,
    pub s: u32,
    pub ts: u32,
    pub td: u32,
    pub ta: u64,
    pub tm: [u8; 128],
    pub tg: u64,
}

impl Default for TransferArgs {
    fn default() -> Self {
        Self {
            t: 0,
            s: 0,
            ts: 0,
            td: 0,
            ta: 0,
            tm: [0u8; 128],
            tg: 0,
        }
    }
}

pub fn parse_transfer_args(start_address: u64, length: u64, m: u64) -> Option<TransferArgs> {
    if length == 0 {
        return None;
    }
    let mut current_address = start_address;
    let mut remaining_length = length;

    let mut args = TransferArgs::default();

    if remaining_length < 8 {
        return None;
    }
    ///---
    
    let t_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 4) };
    let s_slice = unsafe { core::slice::from_raw_parts((current_address + 4) as *const u8, 4) };
    // TODO: use decode_e here to strip out global_t and global_s from current_address
    let global_t = u32::from_le_bytes(t_slice[0..4].try_into().unwrap());
    let global_s = u32::from_le_bytes(s_slice[0..4].try_into().unwrap());

    args.t = global_t;
    args.s = global_s;

    current_address += 8;
    remaining_length = remaining_length.saturating_sub(8);
    // ----- 
    let full_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, remaining_length as usize) };
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
        if remaining_length < 8 {
            return None;
        }
        let ts_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 4) };
        let td_slice = unsafe { core::slice::from_raw_parts((current_address + 4) as *const u8, 4) };
        let global_ts = u32::from_le_bytes(ts_slice[0..4].try_into().unwrap());
        let global_td = u32::from_le_bytes(td_slice[0..4].try_into().unwrap());

        args.ts = global_ts;
        args.td = global_td;

        current_address += 8;
        remaining_length = remaining_length.saturating_sub(8);

        if remaining_length < 8 {
            return None;
        }
        let ta_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 8) };
        args.ta = decode_e_l(ta_slice);
        current_address += 8;
        remaining_length = remaining_length.saturating_sub(8);

        if remaining_length < 128 {
            return None;
        }
        let tm_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 128) };
        args.tm.copy_from_slice(tm_slice);
        current_address += 128;
        remaining_length = remaining_length.saturating_sub(128);

        if remaining_length < 8 {
            return None;
        }

        let tg_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 8) };
        args.tg = decode_e_l(tg_slice);
        current_address += 8;
        remaining_length = remaining_length.saturating_sub(8);

        if i == m {
            return Some(args);
        } 
    }
    None
}

// Parse standard program initialization args
#[repr(C)]
#[derive(Debug, Clone)]
pub struct StandardProgramInitializationArgs {
    pub o_len_bytes : [u8; 3],
    pub w_len_bytes : [u8; 3],
    pub z_bytes : [u8; 2],
    pub s_bytes : [u8; 3],
    pub z : u64,
    pub s : u64,
    pub o_bytes_address : u64,
    pub o_bytes_length : u64,
    pub w_bytes_address : u64,
    pub w_bytes_length : u64,
    pub c_len_bytes : [u8; 4],
    pub c_bytes_address : u64,
    pub c_bytes_length : u64,
}

impl Default for StandardProgramInitializationArgs {
    fn default() -> Self {
        Self {
            o_len_bytes: [0u8; 3],
            w_len_bytes: [0u8; 3],
            z_bytes: [0u8; 2],
            s_bytes: [0u8; 3],
            z: 0,
            s: 0,
            o_bytes_address: 0,
            o_bytes_length: 0,
            w_bytes_address: 0,
            w_bytes_length: 0,
            c_len_bytes: [0u8; 4],
            c_bytes_address: 0,
            c_bytes_length: 0,
        }
    }
}

pub fn parse_standard_program_initialization_args(start_address: u64, length: u64,) -> Option<StandardProgramInitializationArgs> {
    if length == 0 {
        return None;
    }

    let mut args = StandardProgramInitializationArgs::default();

    let mut current_address = start_address;
    let mut remaining_length = length;

    if remaining_length < 3 {
        return None;
    }
    let o_len_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 3) };
    args.o_len_bytes.copy_from_slice(o_len_slice);
    let o_len = decode_e_l(&args.o_len_bytes);
    current_address += 3;
    remaining_length -= 3;

    if remaining_length < 3 {
        return None;
    }
    let w_len_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 3) };
    args.w_len_bytes.copy_from_slice(w_len_slice);
    let w_len = decode_e_l(&args.w_len_bytes);
    current_address += 3;
    remaining_length -= 3;

    if remaining_length < 2 {
        return None;
    }
    let z_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 2) };
    args.z = decode_e_l(z_slice);
    args.z_bytes.copy_from_slice(z_slice);
    current_address += 2;
    remaining_length -= 2;

    if remaining_length < 3 {
        return None;
    }
    let s_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 3) };
    args.s = decode_e_l(s_slice);
    args.s_bytes.copy_from_slice(s_slice);
    current_address += 3;
    remaining_length -= 3;

    if remaining_length < o_len {
        return None;
    }
    args.o_bytes_address = current_address;
    args.o_bytes_length = o_len;
    current_address += o_len;
    remaining_length -= o_len;

    if remaining_length < w_len {
        return None;
    }
    args.w_bytes_address = current_address;
    args.w_bytes_length = w_len;
    current_address += w_len;
    remaining_length -= w_len;

    if remaining_length < 4 {
        return None;
    }
    let c_len_slice = unsafe { core::slice::from_raw_parts(current_address as *const u8, 4) };
    args.c_len_bytes.copy_from_slice(c_len_slice);
    let c_len = decode_e_l(&args.c_len_bytes);
    current_address += 4;
    remaining_length -= 4;

    if remaining_length < c_len {
        return None;
    }
    args.c_bytes_address = current_address;
    args.c_bytes_length = c_len;

    Some(args)
}

pub fn standard_program_initialization_for_child(z: u64, s: u64, o_bytes_address: u64, o_bytes_length: u64, w_bytes_address: u64, w_bytes_length: u64, machine_index: u32) {
    let o_bytes_page_len = ceiling_divide(o_bytes_length, PAGE_SIZE);
    let w_bytes_page_len = ceiling_divide(w_bytes_length, PAGE_SIZE) + z;
    let stack_page_len = ceiling_divide(s, PAGE_SIZE);

    let o_start_addreess = Z_Z;
    let o_start_page = Z_Z / PAGE_SIZE;
    let zero_result = unsafe { zero(machine_index as u64, o_start_page, o_bytes_page_len) };
    if zero_result != OK {
        return call_log(2, None, "StandardProgramInitializationForChild: zero failed for o_bytes");
    }

    let w_start_address = 2 * Z_Z + z_func(o_bytes_length);
    let w_start_page = w_start_address / PAGE_SIZE; 
    let zero_result = unsafe { zero(machine_index as u64, w_start_page, w_bytes_page_len) };
    if zero_result != OK {
        return call_log(2, None, "StandardProgramInitializationForChild: zero failed for w_bytes");
    }

    let s_start_address = (1u64 << 32) - 2 * Z_Z - Z_I - p_func(s);
    let s_start_page = s_start_address / PAGE_SIZE;
    let zero_result = unsafe { zero(machine_index as u64, s_start_page, stack_page_len) };
    if zero_result != OK {
        return call_log(2, None, "StandardProgramInitializationForChild: zero failed for stack");
    }

    let poke_result = unsafe { poke(machine_index as u64, o_bytes_address, o_start_addreess, o_bytes_length) };
    if poke_result != OK {
        return call_log(2, None, "StandardProgramInitializationForChild: poke failed for o_bytes");
    }

    let poke_result = unsafe { poke(machine_index as u64, w_bytes_address, w_start_address, w_bytes_length) };
    if poke_result != OK {
        return call_log(2, None, "StandardProgramInitializationForChild: poke failed for w_bytes");
    }
    call_log(2, None, "StandardProgramInitializationForChild: success");
}

// Child VM related functions
pub fn setup_page(segment: &[u8]) {
    if segment.len() < 8 {
        return call_log(0, None, "setup_page: buffer too small");
    }

    let (m, page_id) = (
        u32::from_le_bytes(segment[0..4].try_into().unwrap()) as u64,
        u32::from_le_bytes(segment[4..8].try_into().unwrap()) as u64,
    );

    let page_address = page_id * PAGE_SIZE;
    let data = &segment[8..];

    if unsafe { zero(m, page_id, 1) } != OK {
        return call_log(0, None, "setup_page: zero failed");
    }

    if unsafe { poke(m, data.as_ptr() as u64, page_address, PAGE_SIZE) } != OK {
        call_log(0, None, "setup_page: poke failed");
    }
}

pub fn get_page(vm_id: u32, page_id: u32) -> [u8; SEGMENT_SIZE as usize] {
    let mut result = [0u8; SEGMENT_SIZE as usize];

    result[0..4].copy_from_slice(&vm_id.to_le_bytes());
    result[4..8].copy_from_slice(&page_id.to_le_bytes());

    let page_address = (page_id as u64) * PAGE_SIZE;
    let result_address = unsafe { result.as_mut_ptr().add(8) };
    let result_length = (SEGMENT_SIZE - 8) as u64;

    let peek_result = unsafe { peek(vm_id as u64, result_address as u64, page_address as u64, result_length) };
    if peek_result != OK {
        call_log(0, None, "get_page: peek failed");
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
    return registers;
}

// Logging related functions
pub fn write_result(result: u64, key: u8) {
    let key_bytes = key.to_le_bytes();
    let result_bytes = result.to_le_bytes();
    unsafe {
        write(
            key_bytes.as_ptr() as u64,
            key_bytes.len() as u64,
            result_bytes.as_ptr() as u64,
            result_bytes.len() as u64,
        );
    }
    call_log(2, None, &format!("write_result key={:x?}, result={:x?}", key_bytes, result));
}

pub fn call_log(level: u64, target: Option<&str>, msg: &str) {
    let (target_address, target_length) = if let Some(target_str) = target {
        let target_bytes = target_str.as_bytes();
        (target_bytes.as_ptr() as u64, target_bytes.len() as u64)
    } else {
        (0, 0)
    };
    let msg_bytes = msg.as_bytes();
    let msg_address = msg_bytes.as_ptr() as u64;
    let msg_length = msg_bytes.len() as u64;
    unsafe { log(level, target_address, target_length, msg_address, msg_length) };
}

// Helper functions
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
        let left_bound = 256 - power_of_two(8 - l);
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

fn ceiling_divide(a: u64, b: u64) -> u64 {
    if b == 0 {
        0
    } else {
        (a + b - 1) / b
    }
}

fn p_func(x: u64) -> u64 {
    Z_P * ceiling_divide(x, Z_P)
}

pub fn z_func(x: u64) -> u64 {
    Z_Z * ceiling_divide(x, Z_Z)
}