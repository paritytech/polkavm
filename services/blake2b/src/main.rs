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
}

pub const NONE: u64 = u64::MAX;

use core::convert::TryInto;

const IV: [u64; 8] = [
    0x6A09_E667_F3BC_C908,
    0xBB67_AE85_84CA_A73B,
    0x3C6E_F372_FE94_F82B,
    0xA54F_F53A_5F1D_36F1,
    0x510E_527F_ADE6_82D1,
    0x9B05_688C_2B3E_6C1F,
    0x1F83_D9AB_FB41_BD6B,
    0x5BE0_CD19_137E_2179,
];

const SIGMA: [[u8; 16]; 12] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],
    [ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8],
    [ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13],
    [ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9],
    [12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11],
    [13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10],
    [ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5],
    [10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0],
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
];

#[inline(always)]
fn rotr64(x: u64, n: u32) -> u64 {
    (x >> n) | (x << (64 - n))
}

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr64(v[b] ^ v[c], 63);
}

fn compress(h: &mut [u64; 8], block: &[u8], t0: u64, t1: u64, is_final: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        let start = i * 8;
        m[i] = u64::from_le_bytes(block[start..start + 8].try_into().unwrap());
    }

    let mut v = [0u64; 16];
    for i in 0..8 {
        v[i] = h[i];
    }
    v[8]  = IV[0];
    v[9]  = IV[1];
    v[10] = IV[2];
    v[11] = IV[3];
    v[12] = IV[4] ^ t0;
    v[13] = IV[5] ^ t1;
    v[14] = IV[6];
    v[15] = IV[7];
    if is_final {
        v[14] = !v[14];
    }

    for r in 0..12 {
        g(&mut v, 0, 4, 8, 12, m[SIGMA[r][0] as usize], m[SIGMA[r][1] as usize]);
        g(&mut v, 1, 5, 9, 13, m[SIGMA[r][2] as usize], m[SIGMA[r][3] as usize]);
        g(&mut v, 2, 6, 10, 14, m[SIGMA[r][4] as usize], m[SIGMA[r][5] as usize]);
        g(&mut v, 3, 7, 11, 15, m[SIGMA[r][6] as usize], m[SIGMA[r][7] as usize]);

        g(&mut v, 0, 5, 10, 15, m[SIGMA[r][8] as usize], m[SIGMA[r][9] as usize]);
        g(&mut v, 1, 6, 11, 12, m[SIGMA[r][10] as usize], m[SIGMA[r][11] as usize]);
        g(&mut v, 2, 7, 8, 13, m[SIGMA[r][12] as usize], m[SIGMA[r][13] as usize]);
        g(&mut v, 3, 4, 9, 14, m[SIGMA[r][14] as usize], m[SIGMA[r][15] as usize]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

struct Blake2b {
    h: [u64; 8],
    t0: u64,
    t1: u64,
    buf: [u8; 128],
    buflen: usize,
    finalized: bool,
}

impl Blake2b {
    pub const DIGEST_LENGTH: usize = 32;

    fn new() -> Self {
        let digest_length = Self::DIGEST_LENGTH;
        let param = (digest_length as u32) | (1 << 16) | (1 << 24);
        let mut h = [0u64; 8];
        h[0] = IV[0] ^ (param as u64);
        h[1] = IV[1];
        h[2] = IV[2];
        h[3] = IV[3];
        h[4] = IV[4];
        h[5] = IV[5];
        h[6] = IV[6];
        h[7] = IV[7];
        Blake2b {
            h,
            t0: 0,
            t1: 0,
            buf: [0u8; 128],
            buflen: 0,
            finalized: false,
        }
    }

    fn update(&mut self, data: &[u8]) {
        if self.finalized {
            return;
        }
        let mut offset = 0;
        while offset < data.len() {
            if self.buflen == 128 {
                self.t0 = self.t0.wrapping_add(128);
                if self.t0 < 128 {
                    self.t1 = self.t1.wrapping_add(1);
                }
                compress(&mut self.h, &self.buf, self.t0, self.t1, false);
                self.buflen = 0;
            }
            let n = core::cmp::min(128 - self.buflen, data.len() - offset);
            self.buf[self.buflen..self.buflen + n]
                .copy_from_slice(&data[offset..offset + n]);
            self.buflen += n;
            offset += n;
        }
    }

    fn finalize(&mut self) -> [u8; Self::DIGEST_LENGTH] {
        if self.finalized {
            return [0u8; Self::DIGEST_LENGTH];
        }
        self.t0 = self.t0.wrapping_add(self.buflen as u64);
        if self.t0 < self.buflen as u64 {
            self.t1 = self.t1.wrapping_add(1);
        }
        let mut block = [0u8; 128];
        block[..self.buflen].copy_from_slice(&self.buf[..self.buflen]);
        compress(&mut self.h, &block, self.t0, self.t1, true);
        self.finalized = true;
        let mut out = [0u8; Self::DIGEST_LENGTH];
        let num_words = Self::DIGEST_LENGTH / 8;
        for i in 0..num_words {
            out[i * 8..(i + 1) * 8].copy_from_slice(&self.h[i].to_le_bytes());
        }
        out
    }
}

fn blake2b_hash(data: &[u8]) -> [u8; Blake2b::DIGEST_LENGTH] {
    let mut hasher = Blake2b::new();
    hasher.update(data);
    hasher.finalize()
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u64 {
    let omega_7: u64;
    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            out(reg) omega_7,
        );
    }

    let input_address: u64 = omega_7 + 4 + 4; // skip time service index and input length
    let input_length_address: u64 = omega_7 + 4; // skip service index
    let input_length: u64  = unsafe { ( *(input_length_address  as *const u32)).into() }; // skip service index


    let input = unsafe { core::slice::from_raw_parts(input_address as *const u8, input_length as usize) };
    let output = blake2b_hash(input);
    
    let output_address = output.as_ptr() as u64;
    let output_length = output.len() as u64;
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) output_length,
        );
    }
    output_address
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

    // write FIB result to storage
    let key = [0u8; 1];
    unsafe {
        write(key.as_ptr() as u64, key.len() as u64, work_result_address, work_result_length);
    }

    // set the result address to register a0 and set the result length to register a1
    unsafe {
        core::arch::asm!(
            "mv a1, {0}",
            in(reg) work_result_length,
        );
    }
    // this equals to a0 = work_result_address
    work_result_address
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