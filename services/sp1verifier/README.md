
# OP Succinct RSP Verifier JAM Service

Here we take a set of a small number of SP1 proofs (generated with GCP nodes with CUDA) and verify them.

Currently this is done in a `main.rs` file, but we can now turn it into an refine/accumulate service `sp1verifier`, and tested with a set of work packages in `TestSP1Verifier`.

## Milestone 1: Basic SP1 Verification

### Refine

Work package:
* payload: `verifier_key.bin` goes `vk`
* extrinsics: `proof_{blocknumber}.bin` goes into `proof`
* work item: `blocknumber` (4 bytes)
* key operation: `client.verify(&proof, &vk)` results in 5 bytes: 4 bytes for `blocknumber` and 1 byte for `result`
  - Ok: (blocknumber, 1)
  - Err: (blocknumber, 0)

### Accumulate:

* input: (blocknumber, result)
* key operation: `write`
  - key = blocknumber (uint32)
  - value = result (0 or 1)

## Milestone 2: Incorporate groth16 Aggregated SP1 Proof with solicited blockhashes

* use `blockhash` instead of block number as the index into JAM State, use 4+1+32 byte results `(blocknumber, result, blockhash)`
* blocks are solicited by blockhash 
* every 32 blocks a [groth16 proof](https://docs.succinct.xyz/generating-proofs/proof-types.html#compressed) will aggregate 32 SP1 proofs for tentative or finalized checkpoints

## Milestone 3: Stitch L1+L2 together with Ordered Accumulation

* L1 blocks are ETH blocks under chain_id 1
* L2 blocks are OP Succinct rollup under chain_id 10 or chain_id 8453





