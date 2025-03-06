#![no_std]
#![no_main]

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
    #[polkavm_import(index = 9)]
    pub fn new(o: u64, l: u64, g: u64, m: u64) -> u64;
    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, g: u64, out: u64) -> u64;
    #[polkavm_import(index = 100)]
    pub fn log(level: u64, target: u64, target_len: u64, message: u64, message_len: u64) -> u64;
}

pub const NONE: u64 = u64::MAX;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct RefineArgs {
    pub WIServiceIndex: u32,
    pub WIPayloadStartAddress: u64,
    pub WIPayloadLength: u64,
    pub WPHash: [u8; 32],
}

impl Default for RefineArgs {
    fn default() -> Self {
        Self {
            WIServiceIndex: 0,
            WIPayloadStartAddress: 0,
            WIPayloadLength: 0,
            WPHash: [0u8; 32],
        }
    }
}

pub fn parse_refine_args(mut start_address: u64, mut remaining_length: u64) -> Option<RefineArgs> {
    if remaining_length < 4 {
        return None;
    }
    let wi_service_index = {
        let mut buf = [0u8; 4];
        unsafe {
            core::ptr::copy_nonoverlapping(start_address as *const u8, buf.as_mut_ptr(), 4);
        }
        u32::from_le_bytes(buf)
    };
    start_address += 4;
    remaining_length = remaining_length.saturating_sub(4);

    if remaining_length == 0 {
        return None;
    }
    let payload_slice = unsafe {
        core::slice::from_raw_parts(start_address as *const u8, remaining_length as usize)
    };
    let discriminator_len = extract_discriminator(payload_slice);
    if discriminator_len == 0 || discriminator_len as usize > payload_slice.len() {
        return None;
    }
    let payload_len = decode_e(&payload_slice[..discriminator_len as usize]);
    start_address += discriminator_len as u64;
    remaining_length = remaining_length.saturating_sub(discriminator_len as u64);

    if remaining_length < payload_len {
        return None;
    }
    let wipayload_start_address = start_address;
    let wipayload_length = payload_len;
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
        WIServiceIndex: wi_service_index,
        WIPayloadStartAddress: wipayload_start_address,
        WIPayloadLength: wipayload_length,
        WPHash: wphash,
    })
}

pub fn call_info(msg: &str) {
    let ascii_bytes = msg.as_bytes();
    let ascii_address = ascii_bytes.as_ptr() as u64;
    let ascii_length = ascii_bytes.len() as u64;
    unsafe { log(2, 0, 0, ascii_address, ascii_length) };
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct WrangledOperandTuple {
    pub WorkResultPtr: u64,
    pub WorkResultLen: u64,

    pub PauloadHash: [u8; 32],
    pub WorkpackageHash: [u8; 32],

    pub AuthOutputPtr: u64,
    pub AuthOutputLen: u64,
}

impl Default for WrangledOperandTuple {
    fn default() -> Self {
        Self {
            WorkResultPtr: 0,
            WorkResultLen: 0,
            PauloadHash: [0u8; 32],
            WorkpackageHash: [0u8; 32],
            AuthOutputPtr: 0,
            AuthOutputLen: 0,
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
        core::slice::from_raw_parts(
            start_address as *const u8,
            remaining_length as usize
        )
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
        let accumulation_slice = unsafe {
            core::slice::from_raw_parts(
                start_address as *const u8,
                remaining_length as usize
            )
        };

        if accumulation_slice.is_empty() {
            return None;
        }
        let work_result_prefix = accumulation_slice[0];

        start_address += 1;
        remaining_length = remaining_length.saturating_sub(1);

        let mut wrangled = WrangledOperandTuple::default();

        if work_result_prefix == 0 {

            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(
                    start_address as *const u8,
                    remaining_length as usize
                )
            };
            let wr_discriminator_len = extract_discriminator(accumulation_slice);
            let wr_len = if wr_discriminator_len > 0 {
                decode_e(&accumulation_slice[..wr_discriminator_len as usize])
            } else {
                0
            };

            start_address += wr_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(wr_discriminator_len as u64);

            wrangled.WorkResultPtr = start_address;
            wrangled.WorkResultLen = remaining_length.min(wr_len);

            start_address += wr_len;
            remaining_length = remaining_length.saturating_sub(wr_len);
        } else {}

        if remaining_length < 64 {
            return None;
        }
        let two_hashes_slice = unsafe {
            core::slice::from_raw_parts(
                start_address as *const u8,
                64
            )
        };
        wrangled.PauloadHash.copy_from_slice(&two_hashes_slice[0..32]);
        wrangled.WorkpackageHash.copy_from_slice(&two_hashes_slice[32..64]);

        start_address += 64;
        remaining_length -= 64;

        {
            let accumulation_slice = unsafe {
                core::slice::from_raw_parts(
                    start_address as *const u8,
                    remaining_length as usize
                )
            };
            let auth_output_discriminator_len = extract_discriminator(accumulation_slice);
            let auth_output_len = if auth_output_discriminator_len > 0 {
                decode_e(&accumulation_slice[..auth_output_discriminator_len as usize])
            } else {
                0
            };

            start_address += auth_output_discriminator_len as u64;
            remaining_length = remaining_length.saturating_sub(auth_output_discriminator_len as u64);

            wrangled.AuthOutputPtr = start_address;
            wrangled.AuthOutputLen = remaining_length.min(auth_output_len);

            start_address += auth_output_len;
            remaining_length = remaining_length.saturating_sub(auth_output_len);
        }

        if i == m {
            return Some(wrangled);
        }
    }
    None
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

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    // read the input start address and length from register a0 and a1
    let omega_7: u64; // refine input start address
    let omega_8: u64; // refine input length

    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            "mv {1}, a1",
            out(reg) omega_7,
            out(reg) omega_8,
        );
    }

    // parse refine args
    let (WIServiceIndex, WIPayloadStartAddress, WIPayloadLength, WPHash) =
    if let Some(args) = parse_refine_args(omega_7, omega_8)
    {
        (args.WIServiceIndex, args.WIPayloadStartAddress, args.WIPayloadLength, args.WPHash)
    } else {
        call_info("parse refine args failed");
        return NONE;
    };

    call_info("parse refine args success");
    
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) WIPayloadLength,
        );
    }
    // this equals to a0 = buffer_addr
    WIPayloadStartAddress
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
    // fetch service index
    let service_index_address = omega_7 + 4; // skip 4 bytes time slot
    let SERVICE_INDEX: u64   = unsafe { ( *(service_index_address as *const u32)).into() }; // 4 bytes service index

    // fetch all_accumulation_o
    let mut start_address = omega_7 + 4 + 4; // 4 bytes time slot + 4 bytes service index
    let mut remaining_length = omega_8 - 4 - 4; // 4 bytes time slot + 4 bytes service index
    
    let (work_result_address, work_result_length) =
    if let Some(tuple) = parse_wrangled_operand_tuple(start_address, remaining_length, 0)
    {
        (tuple.WorkResultPtr, tuple.WorkResultLen)
    } else {
        return NONE;
    };

    // Work result here should contain 32 bytes hash and 4 bytes code length
    let code_length_address: u64 = work_result_address + work_result_length - 4;
    let code_length: u64 = unsafe { ( *(code_length_address as *const u32)).into() }; 
    
    let omega_9: u64 = 100;  // g -  the minimum gas required in order to execute the Accumulate entry-point of the service's code
    let omega_10: u64 = 100; // m -  the minimum required for the On Transfer entry-point
    // create new service with host New
    let result = unsafe { new(work_result_address, code_length, omega_9, omega_10) };
    let result_bytes: [u8; 8] = result.to_le_bytes();

    // write the new service index to the storage
    let storage_key: [u8; 4] = [0; 4];
    let omega_7: u64 = storage_key.as_ptr() as u64; 
    let omega_8: u64 = storage_key.len() as u64;
    let omega_9: u64 = result_bytes.as_ptr() as u64; // new service index bytes address
    let omega_10: u64 = 4; // service index is u32
    unsafe { write(omega_7, omega_8, omega_9, omega_10) };

    // transfer some token to the new service
    let memo = [0u8; 128];
    let omega_7 = result; // receiver
    let omega_8: u64 = 500000; // amount
    let omega_9: u64 = 100;  // g -  the minimum gas
    let omega_10: u64 = memo.as_ptr() as u64; // memo
    unsafe { transfer(omega_7, omega_8, omega_9, omega_10) };

    // Option<hash> test
    // pad result to 32 bytes
    let mut output_bytes_32 = [0u8; 32];
    output_bytes_32[..result_bytes.len()].copy_from_slice(&result_bytes);
    let omega_7: u64 = output_bytes_32.as_ptr() as u64;
    let omega_8: u64 = output_bytes_32.len() as u64;

    // set the result address to register a0 and set the result length to register a1
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
