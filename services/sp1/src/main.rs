#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use simplealloc::SimpleAlloc;
use bincode;
use sp1_sdk::{ProverClient, proof::{SP1ProofWithPublicValues}, SP1VerifyingKey};

#[global_allocator]
static ALLOCATOR: SimpleAlloc<4096> = SimpleAlloc::new();

#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 2)]
    pub fn read(service: u32, key_ptr: *const u8, key_len: u32, out: *mut u8, out_len: u32) -> u32;
    #[polkavm_import(index = 3)]
    pub fn write(key_ptr: *const u8, key_len: u32, value: *const u8, value_len: u32) -> u32;
}

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorized() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(payload: &[u8]) -> u32 {
    let mut buffer = [0u8; 12];

    // Ensure payload length is sufficient
    if payload.len() < 256 {
        return 0; // Invalid payload size
    }

    // Initialize the prover client
    let client = ProverClient::new();

    // Read the verifier key from payload into vk_data
    let vk_data = payload[0..256].to_vec();
    let vk: SP1VerifyingKey = match bincode::deserialize(&vk_data) {
        Ok(key) => key,
        Err(_) => {
            return 0; // Deserialization error
        }
    };

    // Ensure buffer is filled with meaningful data
    if payload.len() < 268 {
        return 0; // Payload too short for block number and proof data
    }
    buffer.copy_from_slice(&payload[256..268]);

    // Extract the block number (first 8 bytes of extrinsic)
    let mut block_number_bytes = [0u8; 8];
    block_number_bytes.copy_from_slice(&buffer[..8]);
    let block_number = u64::from_le_bytes(block_number_bytes);

    // Extract the proof data (remaining bytes of extrinsic)
    let proof_data = &payload[268..];
    let proof: SP1ProofWithPublicValues = match bincode::deserialize(proof_data) {
        Ok(proof) => proof,
        Err(_) => {
            return 0; // Invalid proof data
        }
    };

    // Verify the SP1 proof -- if it succeeds, output it!
    match client.verify(&proof, &vk) {
        Ok(_) => 1, // Proof is valid
        Err(_) => 0, // Proof is invalid
    }
}


#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}




