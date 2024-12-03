use std::fs::{File, read_dir};
use std::io::Read;
use bincode;
use sp1_sdk::{ProverClient, proof::{SP1ProofWithPublicValues}, SP1VerifyingKey};
use regex::Regex;

const PROOF_DIR: &str = "proofs"; // Directory containing proof_{blocknumber}.bin files
const VERIFIER_KEY_FILENAME: &str = "proofs/verifier_key.bin";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Initialize the prover client
    let client = ProverClient::new();

    // Read the verifier key file ===> this will go into the payload
    let mut vk_file = File::open(VERIFIER_KEY_FILENAME).expect("Failed to open verifier key file");
    let mut vk_data = Vec::new();
    vk_file
        .read_to_end(&mut vk_data)
        .expect("Failed to read verifier key file");
    let vk: SP1VerifyingKey =
        bincode::deserialize(&vk_data).expect("Failed to deserialize verifier key");

    let mut success = 0;
    let mut fail = 0;
    
    // Iterate over proof_{blocknumber}.bin files in PROOF_DIR directory ==> this will go into the extrinsics
    let proof_regex = Regex::new(r"^proof_(\d+)\.bin$").unwrap();
    for entry in read_dir(PROOF_DIR).expect("Failed to read proof directory") {
        let entry = entry.expect("Failed to read directory entry");
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();

        // Check if the filename matches the proof pattern
        if let Some(captures) = proof_regex.captures(&filename_str) {
            // Extract the block number from the filename ==> this will go into the extrinsics as well 
            let block_number: u64 = captures[1].parse().expect("Failed to parse block number");

            // Read the proof file
            let mut proof_file = File::open(entry.path()).expect("Failed to open proof file");
            let mut proof_data = Vec::new();
            proof_file
                .read_to_end(&mut proof_data)
                .expect("Failed to read proof file");

            // Deserialize SP1 proof
            let proof: SP1ProofWithPublicValues =
                bincode::deserialize(&proof_data).expect("Failed to deserialize proof");

            // Verify SP1 proof ==> this goes into the refine result along with the block_number
            match client.verify(&proof, &vk) {
                Ok(_) => { success +=1; println!("Proof for block {} succeeded.", block_number) },
                Err(e) => { fail += 1; println!("Proof for block {} failed: {:?}", block_number, e)},
            }
        }
    }

    println!("Verified {} proofs, {} failed.", success, fail);
    Ok(())
}

