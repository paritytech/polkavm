// const IV: [u64; 8] = [
//     0x6A09_E667_F3BC_C908,
//     0xBB67_AE85_84CA_A73B,
//     0x3C6E_F372_FE94_F82B,
//     0xA54F_F53A_5F1D_36F1,
//     0x510E_527F_ADE6_82D1,
//     0x9B05_688C_2B3E_6C1F,
//     0x1F83_D9AB_FB41_BD6B,
//     0x5BE0_CD19_137E_2179,
// ];

// const SIGMA: [[u8; 16]; 12] = [
//     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//     [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
//     [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
//     [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
//     [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
//     [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
//     [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
//     [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
//     [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
//     [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
//     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
//     [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
// ];

// #[inline(always)]
// fn rotr64(x: u64, n: u32) -> u64 {
//     (x >> n) | (x << (64 - n))
// }

// #[inline(always)]
// fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
//     v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
//     v[d] = rotr64(v[d] ^ v[a], 32);
//     v[c] = v[c].wrapping_add(v[d]);
//     v[b] = rotr64(v[b] ^ v[c], 24);
//     v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
//     v[d] = rotr64(v[d] ^ v[a], 16);
//     v[c] = v[c].wrapping_add(v[d]);
//     v[b] = rotr64(v[b] ^ v[c], 63);
// }

// fn compress(h: &mut [u64; 8], block: &[u8], t0: u64, t1: u64, is_final: bool) {
//     let mut m = [0u64; 16];
//     for i in 0..16 {
//         let start = i * 8;
//         m[i] = u64::from_le_bytes(block[start..start + 8].try_into().unwrap());
//     }

//     let mut v = [0u64; 16];
//     for i in 0..8 {
//         v[i] = h[i];
//     }
//     v[8] = IV[0];
//     v[9] = IV[1];
//     v[10] = IV[2];
//     v[11] = IV[3];
//     v[12] = IV[4] ^ t0;
//     v[13] = IV[5] ^ t1;
//     v[14] = IV[6];
//     v[15] = IV[7];
//     if is_final {
//         v[14] = !v[14];
//     }

//     for r in 0..12 {
//         g(&mut v, 0, 4, 8, 12, m[SIGMA[r][0] as usize], m[SIGMA[r][1] as usize]);
//         g(&mut v, 1, 5, 9, 13, m[SIGMA[r][2] as usize], m[SIGMA[r][3] as usize]);
//         g(&mut v, 2, 6, 10, 14, m[SIGMA[r][4] as usize], m[SIGMA[r][5] as usize]);
//         g(&mut v, 3, 7, 11, 15, m[SIGMA[r][6] as usize], m[SIGMA[r][7] as usize]);

//         g(&mut v, 0, 5, 10, 15, m[SIGMA[r][8] as usize], m[SIGMA[r][9] as usize]);
//         g(&mut v, 1, 6, 11, 12, m[SIGMA[r][10] as usize], m[SIGMA[r][11] as usize]);
//         g(&mut v, 2, 7, 8, 13, m[SIGMA[r][12] as usize], m[SIGMA[r][13] as usize]);
//         g(&mut v, 3, 4, 9, 14, m[SIGMA[r][14] as usize], m[SIGMA[r][15] as usize]);
//     }

//     for i in 0..8 {
//         h[i] ^= v[i] ^ v[i + 8];
//     }
// }

// struct Blake2b {
//     h: [u64; 8],
//     t0: u64,
//     t1: u64,
//     buf: [u8; 128],
//     buflen: usize,
//     finalized: bool,
// }

// impl Blake2b {
//     pub const DIGEST_LENGTH: usize = 32;

//     fn new() -> Self {
//         let digest_length = Self::DIGEST_LENGTH;
//         let param = (digest_length as u32) | (1 << 16) | (1 << 24);
//         let mut h = [0u64; 8];
//         h[0] = IV[0] ^ (param as u64);
//         h[1] = IV[1];
//         h[2] = IV[2];
//         h[3] = IV[3];
//         h[4] = IV[4];
//         h[5] = IV[5];
//         h[6] = IV[6];
//         h[7] = IV[7];
//         Blake2b {
//             h,
//             t0: 0,
//             t1: 0,
//             buf: [0u8; 128],
//             buflen: 0,
//             finalized: false,
//         }
//     }

//     fn update(&mut self, data: &[u8]) {
//         if self.finalized {
//             return;
//         }
//         let mut offset = 0;
//         while offset < data.len() {
//             if self.buflen == 128 {
//                 self.t0 = self.t0.wrapping_add(128);
//                 if self.t0 < 128 {
//                     self.t1 = self.t1.wrapping_add(1);
//                 }
//                 compress(&mut self.h, &self.buf, self.t0, self.t1, false);
//                 self.buflen = 0;
//             }
//             let n = core::cmp::min(128 - self.buflen, data.len() - offset);
//             self.buf[self.buflen..self.buflen + n].copy_from_slice(&data[offset..offset + n]);
//             self.buflen += n;
//             offset += n;
//         }
//     }

//     fn finalize(&mut self) -> [u8; Self::DIGEST_LENGTH] {
//         if self.finalized {
//             return [0u8; Self::DIGEST_LENGTH];
//         }
//         self.t0 = self.t0.wrapping_add(self.buflen as u64);
//         if self.t0 < self.buflen as u64 {
//             self.t1 = self.t1.wrapping_add(1);
//         }
//         let mut block = [0u8; 128];
//         block[..self.buflen].copy_from_slice(&self.buf[..self.buflen]);
//         compress(&mut self.h, &block, self.t0, self.t1, true);
//         self.finalized = true;
//         let mut out = [0u8; Self::DIGEST_LENGTH];
//         let num_words = Self::DIGEST_LENGTH / 8;
//         for i in 0..num_words {
//             out[i * 8..(i + 1) * 8].copy_from_slice(&self.h[i].to_le_bytes());
//         }
//         out
//     }
// }

// pub fn blake2b_hash(data: &[u8]) -> [u8; Blake2b::DIGEST_LENGTH] {
//     let mut hasher = Blake2b::new();
//     hasher.update(data);
//     hasher.finalize()
// }



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
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
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

#[inline(always)]
fn compress(h: &mut [u64; 8], block: &[u8], t0: u64, t1: u64, is_final: bool) {
    let m: [u64; 16] = unsafe { core::ptr::read_unaligned(block.as_ptr() as *const [u64; 16]) };

    let mut v = [
        h[0], h[1], h[2], h[3],
        h[4], h[5], h[6], h[7],
        IV[0], IV[1], IV[2], IV[3],
        IV[4] ^ t0, IV[5] ^ t1, IV[6], IV[7],
    ];
    if is_final {
        v[14] = !v[14];
    }

    for r in 0..12 {
        let s = &SIGMA[r];
        g(&mut v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
        g(&mut v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
        g(&mut v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
        g(&mut v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);

        g(&mut v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
        g(&mut v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
        g(&mut v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
        g(&mut v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
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

    #[inline(always)]
    fn new() -> Self {
        let digest_length = Self::DIGEST_LENGTH;
        let param = (digest_length as u32) | (1 << 16) | (1 << 24);
        let h = [
            IV[0] ^ (param as u64),
            IV[1],
            IV[2],
            IV[3],
            IV[4],
            IV[5],
            IV[6],
            IV[7],
        ];
        Blake2b {
            h,
            t0: 0,
            t1: 0,
            buf: [0u8; 128],
            buflen: 0,
            finalized: false,
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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
        for i in 0..(Self::DIGEST_LENGTH / 8) {
            out[i * 8..(i + 1) * 8].copy_from_slice(&self.h[i].to_le_bytes());
        }
        out
    }
}

pub fn blake2b_hash(data: &[u8]) -> [u8; Blake2b::DIGEST_LENGTH] {
    let mut hasher = Blake2b::new();
    hasher.update(data);
    hasher.finalize()
}
