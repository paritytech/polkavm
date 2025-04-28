#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE0 : usize = (1 << 24) - (1 << 12);

// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0); // 2^24 - 1 - 4096, should not greater than 2^24 - 1 (16777215)


const SIZE1 : usize = (1 << 15) * (1 << 12);
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new(); // 2^24 - 1 - 4096, should not greater than 2^24 - 1 (16777215)


// use picoalloc::{Allocator, Size};
// static ALLOCATOR: Allocator = Allocator::new(Size::from_bytes_usize(SIZE).unwrap());


use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::call_log;

use revm::{
    bytecode::Bytecode,
    context::{Context, Evm, TxEnv},
    context_interface::result::{ExecutionResult, Output},
    database::{BenchmarkDB, CacheDB, EmptyDB, BENCH_CALLER, BENCH_TARGET},
    handler::{EvmTr, EthPrecompiles},
    primitives::{hex, Bytes, TxKind, bytes, U256},
    ExecuteCommitEvm,
    ExecuteEvm,
    MainBuilder,
    MainContext,
};

use bytes::{Bytes as RawBytes, BytesMut};

const BYTES: &str = include_str!("../add_example.hex");

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    call_log(2, None, &format!("refine called"));
    
    
    // Bytecode decoding example
    
    let bytecode = Bytecode::new_raw(Bytes::from(hex::decode(BYTES).unwrap()));
    call_log(2, None, &format!("bytecode decoded: {:?}", bytecode));



    // Bytecode execution example

    /*
    let bytecode = Bytes::from(hex::decode(BYTES).unwrap());
    let ctx = Context::mainnet()
        .modify_tx_chained(|tx| {
            tx.kind = TxKind::Create;
            tx.data = bytecode.clone();
        })
        .with_db(CacheDB::<EmptyDB>::default());

    let mut evm = ctx.build_mainnet();

    call_log(2, None, &format!("EVM created"));

    let ref_tx = evm.replay_commit().unwrap();

    call_log(2, None, &format!("EVM replayed"));
    let ExecutionResult::Success {
        output: Output::Create(_, Some(address)),
        ..
    } = ref_tx
    else {
        call_log(2, None, &format!("Failed to create contract"));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    };

    call_log(2, None, &format!("Created contract at {:?}", address));
    // call contract with custom opcode
    evm.ctx().modify_tx(|tx| {
        tx.kind = TxKind::Call(address);
        tx.data = bytes!("1a43c338"); // first 4 bytes of Keccak-256("compute()")
        tx.nonce += 1;
    });


    let ras = match evm.replay() {
        Ok(r) => r,
        Err(e) => {
            call_log(2, None, &format!("EVM failed: {:?}", e));
            return (FIRST_READABLE_ADDRESS as u64, 0);  
        }
    };
    

    let output_slice = ras.result.output().unwrap();
    call_log(2, None, &format!("output {:?}", output_slice));
    */
    
    
    
    // transfer example

    /* 
    let mut evm = Context::mainnet()
    .with_db(BenchmarkDB::new_bytecode(Bytecode::new()))
    .modify_tx_chained(|tx| {
        // Execution globals block hash/gas_limit/coinbase/timestamp..
        tx.caller = BENCH_CALLER;
        tx.kind = TxKind::Call(BENCH_TARGET);
        tx.value = U256::from(911);
    })
    .build_mainnet();

    call_log(2, None, &format!("EVM created"));

    let _res_and_state = evm.replay();

    call_log(2, None, &format!("_res_and_state {:?}", _res_and_state));
    */
    

    (FIRST_READABLE_ADDRESS as u64, 0)
}


#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(_start_address: u64, _length: u64) -> (u64, u64) {
    (FIRST_READABLE_ADDRESS as u64, 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    (FIRST_READABLE_ADDRESS as u64, 0)
}
