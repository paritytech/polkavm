#![no_std]
#![no_main]
#![feature(asm_const)] 

extern crate alloc;
use alloc::format;
use alloc::vec;

const SIZE0 : usize = 0x10000;
// allocate memory for stack
use polkavm_derive::min_stack_size;
min_stack_size!(SIZE0);

const SIZE1 : usize = 0x200000;
// allocate memory for heap
use simplealloc::SimpleAlloc;
#[global_allocator]
static ALLOCATOR: SimpleAlloc<SIZE1> = SimpleAlloc::new();

use utils::constants::{FIRST_READABLE_ADDRESS, NONE};
use utils::functions::{parse_accumulate_args, parse_transfer_args, call_log, write_result};
use utils::host_functions::{export, fetch, write, gas};

// NEW: bring in the wasm-snap types and core helpers
use wasm_snap::{WasmVm, Value};
use core::cmp::min;

static mut buffer: [u8; 16] = [0; 16];

// Static module bytes for the demo VM (no_std friendly)
static mut WASM_BYTES: [u8; 41] = [
    0x00, 0x61, 0x73, 0x6d, // WASM magic
    0x01, 0x00, 0x00, 0x00, // Version
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
];

// Static mutable buffers for snapshot bytes and lengths (avoid std/heap for storage)
static mut SNAPSHOT_BUF: [u8; 4096] = [0; 4096];
static mut SNAPSHOT_LEN: usize = 0;

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    // Initialize VM from static bytes
    let mut vm = WasmVm::new();
    if let Err(e) = vm.init(unsafe { &WASM_BYTES }) {
        call_log(1, None, &format!("refine: vm.init failed: {:?}", e));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    // Execute a couple of steps (no assert!, just check and log)
    match vm.step() {
        Ok(true) => {},
        Ok(false) => { call_log(1, None, "refine: first step returned finished unexpectedly"); }
        Err(e) => { call_log(1, None, &format!("refine: first step error: {:?}", e)); }
    }
    match vm.step() {
        Ok(true) => {},
        Ok(false) => { call_log(1, None, "refine: second step returned finished unexpectedly"); }
        Err(e) => { call_log(1, None, &format!("refine: second step error: {:?}", e)); }
    }

    // Take a snapshot into a static buffer (avoid keeping Vec around)
    let state_vec = vm.snapshot();
    let copy_len = min(state_vec.len(), unsafe { SNAPSHOT_BUF.len() });
    unsafe {
        SNAPSHOT_BUF[..copy_len].copy_from_slice(&state_vec[..copy_len]);
        SNAPSHOT_LEN = copy_len;
    }
    call_log(2, None, &format!("snapshot size: {} bytes", copy_len));

    // Restore into a fresh VM and continue
    let mut vm2 = WasmVm::new();
    if let Err(e) = vm2.init(unsafe { &WASM_BYTES }) {
        call_log(1, None, &format!("refine: vm2.init failed: {:?}", e));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }
    let snap_slice: &[u8] = unsafe { &SNAPSHOT_BUF[..SNAPSHOT_LEN] };
    if let Err(e) = vm2.restore(snap_slice) {
        call_log(1, None, &format!("refine: vm2.restore failed: {:?}", e));
        return (FIRST_READABLE_ADDRESS as u64, 0);
    }

    // Continue stepping until finished
    loop {
        match vm2.step() {
            Ok(true) => {},
            Ok(false) => break,
            Err(e) => { call_log(1, None, &format!("refine: vm2.step error: {:?}", e)); break; }
        }
    }

    // Verify a simple result from state if available
    let st = vm2.dump_state();
    if st.is_finished {
        call_log(2, None, "refine: vm2 finished execution");
        if let Some(val) = st.locals.get(0) {
            match val {
                Value::I32(v) => {
                    if *v == 52 { call_log(2, None, "refine: locals[0] == 52 OK"); }
                    else { call_log(1, None, &format!("refine: unexpected locals[0]={}", v)); }
                }
                other => { call_log(1, None, &format!("refine: unexpected locals[0] type: {:?}", other)); }
            }
        }
    } else {
        call_log(1, None, "refine: vm2 not finished at end of loop");
    }

    // Also try calling a function directly
    match vm2.call_function("add", &[Value::I32(7), Value::I32(8)]) {
        Ok(results) => {
            if results.as_slice() == [Value::I32(15)] { call_log(2, None, "refine: add(7,8)=15 OK"); }
            else { call_log(1, None, &format!("refine: unexpected add result: {:?}", results)); }
        }
        Err(e) => { call_log(1, None, &format!("refine: call_function error: {:?}", e)); }
    }

    call_log(2, None, "snapshot/restore demo OK");
    (FIRST_READABLE_ADDRESS as u64, 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}