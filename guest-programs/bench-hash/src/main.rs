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
    core::hint::black_box(chained(&mut state.memory, 32, 1, blake2b_once));
}

fn blake2b_once(input: &[u8], out: &mut [u8; 32]) {
    let hash = blake2b_simd::Params::new().hash_length(32).hash(input);
    out.copy_from_slice(hash.as_bytes());
}

fn keccak256_once(input: &[u8], out: &mut [u8; 32]) {
    use sha3::Digest;
    let mut hasher = sha3::Keccak256::new();
    hasher.update(input);
    out.copy_from_slice(&hasher.finalize());
}

fn sha256_once(input: &[u8], out: &mut [u8; 32]) {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(input);
    out.copy_from_slice(&hasher.finalize());
}

/// Hashes `memory[..len]` `times` times, feeding each round's output back
/// into the start of the buffer so every iteration depends on the previous
/// one (no loop-invariant hoisting), and returns a fold of the final output
/// so the whole chain is observable (no dead-code elimination).
fn chained(
    memory: &mut [u8],
    len: u64,
    times: u64,
    hash_once: impl Fn(&[u8], &mut [u8; 32]),
) -> u64 {
    let len = (len as usize).min(memory.len()).max(1);
    let mut out = [0u8; 32];
    for _ in 0..times {
        hash_once(&memory[..len], &mut out);
        let feedback = len.min(32);
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

export_hash_benchmark!(benchmark_blake2b, blake2b_once);
export_hash_benchmark!(benchmark_keccak256, keccak256_once);
export_hash_benchmark!(benchmark_sha256, sha256_once);
