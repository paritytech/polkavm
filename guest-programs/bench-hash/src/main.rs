#![no_std]
#![no_main]

include!("../../bench-common.rs");

const MAX_LEN: usize = 1024 * 1024;

struct State {
    memory: alloc::vec::Vec<u8>,
}

define_benchmark! {
    heap_size = MAX_LEN + 64 * 1024,
    state = State {
        memory: alloc::vec::Vec::new(),
    },
}

fn benchmark_initialize(state: &mut State) {
    state.memory.resize(MAX_LEN, 0);
    for (index, byte) in state.memory.iter_mut().enumerate() {
        *byte = index as u8;
    }
}

// Entry point for the generic harness; the real benchmarks are the
// parameterized `benchmark_*` exports below, driven by benchtool's
// `bench-hash` subcommand.
fn benchmark_run(state: &mut State) {
    core::hint::black_box(chained(&mut state.memory, 32, 1, blake2_256_once));
}

/// The output buffer fits the largest digest (keccak_512); each hash function
/// writes its own digest length and returns it.
type Output = [u8; 64];

fn blake2b_once(input: &[u8], out: &mut Output, hash_length: usize) -> usize {
    let hash = blake2b_simd::Params::new().hash_length(hash_length).hash(input);
    out[..hash_length].copy_from_slice(hash.as_bytes());
    hash_length
}

fn blake2_128_once(input: &[u8], out: &mut Output) -> usize {
    blake2b_once(input, out, 16)
}

fn blake2_256_once(input: &[u8], out: &mut Output) -> usize {
    blake2b_once(input, out, 32)
}

fn keccak_256_once(input: &[u8], out: &mut Output) -> usize {
    use sha3::Digest;
    let mut hasher = sha3::Keccak256::new();
    hasher.update(input);
    out[..32].copy_from_slice(&hasher.finalize());
    32
}

fn keccak_512_once(input: &[u8], out: &mut Output) -> usize {
    use sha3::Digest;
    let mut hasher = sha3::Keccak512::new();
    hasher.update(input);
    out[..64].copy_from_slice(&hasher.finalize());
    64
}

fn sha2_256_once(input: &[u8], out: &mut Output) -> usize {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(input);
    out[..32].copy_from_slice(&hasher.finalize());
    32
}

/// N seeded XxHash64 passes, like `sp_core::hashing::twox_*`.
fn twox_once<const WORDS: usize>(input: &[u8], out: &mut Output) -> usize {
    use core::hash::Hasher;
    for seed in 0..WORDS {
        let mut hasher = twox_hash::XxHash64::with_seed(seed as u64);
        hasher.write(input);
        out[seed * 8..(seed + 1) * 8].copy_from_slice(&hasher.finish().to_le_bytes());
    }
    WORDS * 8
}

/// Hashes `memory[..len]` `times` times, feeding each round's output back
/// into the start of the buffer so every iteration depends on the previous
/// one (no loop-invariant hoisting), and returns the first 8 bytes of the
/// final digest so the whole chain is observable (no dead-code elimination).
fn chained(
    memory: &mut [u8],
    len: u64,
    times: u64,
    hash_once: impl Fn(&[u8], &mut Output) -> usize,
) -> u64 {
    let len = (len as usize).min(memory.len()).max(1);
    let mut out = [0u8; 64];
    for _ in 0..times {
        let out_len = hash_once(&memory[..len], &mut out);
        let feedback = len.min(out_len);
        memory[..feedback].copy_from_slice(&out[..feedback]);
    }
    u64::from_le_bytes(out[..8].try_into().unwrap())
}

macro_rules! export_hash_benchmark {
    ($name:ident, $hash_once:expr) => {
        #[cfg_attr(target_env = "polkavm", polkavm_derive::polkavm_export)]
        #[no_mangle]
        pub extern "C" fn $name(len: u64, times: u64) -> u64 {
            let state = unsafe { &mut *core::ptr::addr_of_mut!(STATE) };
            chained(&mut state.memory, len, times, $hash_once)
        }
    };
}

// Named after the `sp_io::hashing` host functions they mirror.
export_hash_benchmark!(benchmark_blake2_128, blake2_128_once);
export_hash_benchmark!(benchmark_blake2_256, blake2_256_once);
export_hash_benchmark!(benchmark_keccak_256, keccak_256_once);
export_hash_benchmark!(benchmark_keccak_512, keccak_512_once);
export_hash_benchmark!(benchmark_sha2_256, sha2_256_once);
export_hash_benchmark!(benchmark_twox_64, twox_once::<1>);
export_hash_benchmark!(benchmark_twox_128, twox_once::<2>);
export_hash_benchmark!(benchmark_twox_256, twox_once::<4>);
