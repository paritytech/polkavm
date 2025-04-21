#![no_std]
#![no_main]

extern crate alloc;
use alloc::format;
use alloc::vec;

use polkavm_derive::min_stack_size;
min_stack_size!(409600);

use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::call_log;

// use revm::database::{BenchmarkDB, BENCH_CALLER, BENCH_TARGET};
// use revm::{
//     bytecode::Bytecode,
//     primitives::{bytes, hex, Bytes, TxKind},
//     Context, ExecuteEvm, MainBuilder, MainContext,
// };

use revm::database::{BenchmarkDB, BENCH_CALLER, BENCH_TARGET};
use revm::{
    bytecode::Bytecode,
    primitives::{TxKind, U256},
    Context, ExecuteEvm, MainBuilder, MainContext,
};


use revm::{
    context::TxEnv,
    context::Evm,
    handler::instructions::EthInstructions,
    handler::EthPrecompiles,
};

use bytes::{Bytes as RawBytes, BytesMut};

const BYTES: &str = include_str!("../add_example.hex");

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    // call_log(2, None, &format!("refine called"));
    // let bytecode = Bytecode::new_raw(Bytes::from(hex::decode(BYTES).unwrap()));

    // call_log(2, None, &format!("bytecode decoded: {:?}", bytecode));
    
    // let mut evm = Context::mainnet()
    //     .with_db(BenchmarkDB::new_bytecode(bytecode.clone()))
    //     .modify_tx_chained(|tx| {
    //         tx.caller = BENCH_CALLER;
    //         tx.kind = TxKind::Call(BENCH_TARGET);
    //         tx.data = bytes!("30627b7c");
    //         tx.gas_limit = 1_000_000_000;
    //         tx.blob_hashes = vec![bytecode.hash_slow()];
    //         ..Default::default()

    //     })
    //     .build_mainnet();

    // let result = evm.replay().unwrap();
    
    // call_log(2, None, &format!("result: {:?}", result));
    // call_log(2, None, &format!("Done replaying"));


    // let mut evm = Context::mainnet()
    //     .with_db(BenchmarkDB::new_bytecode(Bytecode::new()))
    //     .modify_tx_chained(|tx| {
    //         // Execution globals block hash/gas_limit/coinbase/timestamp..
    //         tx.caller = BENCH_CALLER;
    //         tx.kind = TxKind::Call(BENCH_TARGET);
    //         tx.value = U256::from(911);
    //     })
    //     .build_mainnet();

    // call_log(2, None, &format!("starting replay"));
    // let _res_and_state = evm.replay();
    // call_log(2, None, &format!("_res_and_state: {:?}", _res_and_state));

    
    
    let precompiles = EthPrecompiles::default();

    call_log(2, None, &format!("inited precompiles"));

    
    // let ctx = Context::mainnet();

    // call_log(2, None, &format!("init ctx"));

    // let instruction = EthInstructions::new_mainnet();

    // call_log(2, None, &format!("init instruction"));

    

    // let mut evm = Evm::new(ctx, instruction, precompiles);

    // call_log(2, None, &format!("init evm"));

    // let _res_and_state = evm.transact(TxEnv::default());

    // call_log(2, None, &format!("_res_and_state: {:?}", _res_and_state));

    // call_log(2, None, &format!("Done replaying"));
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
