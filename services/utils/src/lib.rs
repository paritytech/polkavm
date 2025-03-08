#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 0)]
    pub fn gas() -> u64;
    // accumulate
    #[polkavm_import(index = 1)]
    pub fn lookup(s: u64, h: u64, o: u64, f: u64, l: u64) -> u64;
    #[polkavm_import(index = 2)]
    pub fn read(s: u64, ko: u64, kz: u64, o: u64, f: u64, l: u64) -> u64;
    #[polkavm_import(index = 3)]
    pub fn write(ko: u64, kz: u64, bo: u64, bz: u64) -> u64;
    #[polkavm_import(index = 4)]
    pub fn info(s: u64, o: u64) -> u64;
    #[polkavm_import(index = 5)]
    pub fn bless(m: u64, a: u64, v: u64, o: u64, n: u64) -> u64;
    #[polkavm_import(index = 6)]
    pub fn assign(c: u64, o: u64) -> u64;

    #[polkavm_import(index = 8)]
    pub fn checkpoint() -> u64;
    #[polkavm_import(index = 9)]
    pub fn new(o: u64, l: u64, g: u64, m: u64) -> u64;
    #[polkavm_import(index = 10)]
    pub fn upgrade(o: u64, g: u64, m: u64) -> u64;
    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, l: u64, o: u64) -> u64;
    #[polkavm_import(index = 12)]
    pub fn eject(d: u64, o: u64) -> u64;
    #[polkavm_import(index = 13)]
    pub fn query(o: u64, z: u64) -> u64;
    #[polkavm_import(index = 14)]
    pub fn solicit(o: u64, z: u64) -> u64;
    #[polkavm_import(index = 15)]
    pub fn forget(o: u64, z: u64) -> u64;
    #[polkavm_import(index = 16)]
    pub fn oyield(o: u64) -> u64;

    // refine
    #[polkavm_import(index = 17)]
    pub fn historical_lookup(service: u64, h: u64, o: u64, f: u64, l: u64) -> u64;
    #[polkavm_import(index = 18)]
    pub fn fetch(start_address: u64, offset: u64, maxlen: u64, omega_10: u64, omega_11: u64, omega_12: u64) -> u64;
    #[polkavm_import(index = 19)]
    pub fn export(out: u64, out_len: u64) -> u64;
    #[polkavm_import(index = 20)]
    pub fn machine(po: u64, pz: u64, i: u64) -> u64;
    #[polkavm_import(index = 21)]
    pub fn peek(n: u64, o: u64, s: u64, z: u64) -> u64;
    #[polkavm_import(index = 22)]
    pub fn poke(n: u64, s: u64, o: u64, z: u64) -> u64;
    #[polkavm_import(index = 23)]
    pub fn zero(n: u64, p: u64, c: u64) -> u64;
    #[polkavm_import(index = 24)]
    pub fn void(n: u64, p: u64, c: u64) -> u64;
    #[polkavm_import(index = 25)]
    pub fn invoke(n: u64, o: u64) -> u64;
    #[polkavm_import(index = 26)]
    pub fn expunge(n: u64) -> u64;

    #[polkavm_import(index = 100)]
    pub fn log(level: u64, target: u64, target_len: u64, message: u64, message_len: u64) -> u64; //https://hackmd.io/@polkadot/jip1
}

pub const NONE: u64 = u64::MAX;
pub const OK: u64 = 0;
pub const PAGE_SIZE: u64 = 4096;
pub const SEGMENT_SIZE: u64 = 4104;
pub const PARENT_MACHINE_INDEX: u64 = (1u64 << 32) - 1;


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
    let result_address = result.as_ptr() as u64 + 8;
    let result_length = (SEGMENT_SIZE - 8) as u64;

    let peek_result = unsafe{ peek(vm_id as u64, result_address, page_address as u64, result_length) };
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

// Helpful functions and structs for accumulate
#[repr(C)]
#[derive(Debug, Clone)]
pub struct WrangledOperandTuple {
    pub h: [u8; 32],
    pub e: [u8; 32],
    pub a: [u8; 32],

    pub o_ptr: u64,
    pub o_len: u64,

    pub y: [u8; 32],

    pub work_result_ptr: u64,
    pub work_result_len: u64,
}

impl Default for WrangledOperandTuple {
    fn default() -> Self {
        Self {
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

pub fn parse_wrangled_operand_tuple(all_accumulation_o_ptr: u64, all_accumulation_o_len: u64, m: u64,) -> Option<WrangledOperandTuple> {
    if all_accumulation_o_len == 0 {
        return None;
    }
    let mut start_address = all_accumulation_o_ptr;
    let mut remaining_length = all_accumulation_o_len;

    let full_slice = unsafe {
        core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
    };
    let all_accumulation_o_discriminator_length = extract_discriminator(full_slice);
    if all_accumulation_o_discriminator_length as usize > full_slice.len() {
        return None;
    }
    let num_of_accumulation_o = decode_e(&full_slice[..all_accumulation_o_discriminator_length as usize]);

    start_address += all_accumulation_o_discriminator_length as u64;
    remaining_length = remaining_length.saturating_sub(all_accumulation_o_discriminator_length as u64);

    if m >= num_of_accumulation_o {
        return None;
    }

    for i in 0..num_of_accumulation_o {
        let mut wrangled = WrangledOperandTuple::default();

        if remaining_length < 96 {
            return None;
        }
        let hash_slice = unsafe {
            core::slice::from_raw_parts(start_address as *const u8, 96)
        };
        wrangled.h.copy_from_slice(&hash_slice[0..32]);
        wrangled.e.copy_from_slice(&hash_slice[32..64]);
        wrangled.a.copy_from_slice(&hash_slice[64..96]);
        start_address += 96;
        remaining_length = remaining_length.saturating_sub(96);

        {
            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
            };
            let auth_output_discriminator_len = extract_discriminator(accumulation_slice);
            let auth_output_len = if auth_output_discriminator_len > 0 {
                decode_e(&accumulation_slice[..auth_output_discriminator_len as usize])
            } else {
                0
            };

            start_address += auth_output_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(auth_output_discriminator_len as u64);

            wrangled.o_ptr = start_address;
            wrangled.o_len = remaining_length.min(auth_output_len);

            start_address += auth_output_len;
            remaining_length = remaining_length.saturating_sub(auth_output_len);
        }

        if remaining_length < 32 {
            return None;
        }
        let y_slice = unsafe {
            core::slice::from_raw_parts(start_address as *const u8, 32)
        };
        wrangled.y.copy_from_slice(y_slice);
        start_address += 32;
        remaining_length = remaining_length.saturating_sub(32);

        let accumulation_slice = unsafe {
            core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
        };
        if accumulation_slice.is_empty() {
            return None;
        }
        let work_result_prefix = accumulation_slice[0];
        start_address += 1;
        remaining_length = remaining_length.saturating_sub(1);

        if work_result_prefix == 0 {
            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
            };
            let wr_discriminator_len = extract_discriminator(accumulation_slice);
            let wr_len = if wr_discriminator_len > 0 {
                decode_e(&accumulation_slice[..wr_discriminator_len as usize])
            } else {
                0
            };

            start_address += wr_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(wr_discriminator_len as u64);

            wrangled.work_result_ptr = start_address;
            wrangled.work_result_len = remaining_length.min(wr_len);

            start_address += wr_len;
            remaining_length = remaining_length.saturating_sub(wr_len);
        }

        if i == m {
            return Some(wrangled);
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

pub fn call_info(msg: &str) {
    let ascii_bytes = msg.as_bytes();
    let ascii_address = ascii_bytes.as_ptr() as u64;
    let ascii_length = ascii_bytes.len() as u64;
    unsafe { log(2, 0, 0, ascii_address, ascii_length) };
}

