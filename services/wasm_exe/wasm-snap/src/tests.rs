#![cfg(test)]

use alloc::vec;

use crate::{WasmVm, WasmError, Value};

// Simple WASM module for testing
const TEST_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, // WASM magic
    0x01, 0x00, 0x00, 0x00, // Version
    // Type section: (func (param i32 i32) (result i32))
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    // Function section
    0x03, 0x02, 0x01, 0x00,
    // Export section: export "add" function
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    // Code section: local.get 0, local.get 1, i32.add
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b
];

#[test]
fn test_vm_creation() {
    let vm = WasmVm::new();
    assert!(!vm.is_finished());
    assert_eq!(vm.instruction_count(), 0);
}

#[test]
fn test_vm_init() {
    let mut vm = WasmVm::new();
    let result = vm.init(TEST_WASM);
    assert!(result.is_ok());
}

#[test]
fn test_vm_init_invalid_wasm() {
    let mut vm = WasmVm::new();
    let invalid_wasm = &[0x00, 0x01, 0x02, 0x03]; // Invalid WASM
    let result = vm.init(invalid_wasm);
    assert!(matches!(result, Err(WasmError::InvalidModule)));
}

#[test]
fn test_state_dump() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();
    
    let state = vm.dump_state();
    assert_eq!(state.pc, 0);
    assert_eq!(state.call_depth, 0);
    assert_eq!(state.instruction_count, 0);
    assert!(!state.is_finished);
}

#[test]
fn test_step_execution() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();
    
    let initial_count = vm.instruction_count();
    let step_result = vm.step();
    
    assert!(step_result.is_ok());
    assert!(vm.instruction_count() > initial_count);
}

#[test]
fn test_function_call() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();
    
    let args = vec![Value::I32(5), Value::I32(3)];
    let result = vm.call_function("add", &args);
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    
    if let Value::I32(sum) = results[0] {
        assert_eq!(sum, 8);
    } else {
        panic!("Expected I32 result");
    }
}

#[test]
fn test_function_not_found() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();
    
    let args = vec![Value::I32(5)];
    let result = vm.call_function("nonexistent", &args);
    
    assert!(matches!(result, Err(WasmError::FunctionNotFound)));
}

#[test]
fn test_reset() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();
    
    // Execute some steps
    vm.step().unwrap();
    vm.step().unwrap();
    
    let count_before_reset = vm.instruction_count();
    assert!(count_before_reset > 0);
    
    // Reset and check
    vm.reset();
    assert_eq!(vm.instruction_count(), 0);
    assert!(!vm.is_finished());
}

#[test]
fn test_dump_and_resume_then_validate() {
    let mut vm = WasmVm::new();
    vm.init(TEST_WASM).unwrap();

    // Step 1: should push 42 onto the stack
    assert!(vm.step().unwrap());
    let st1 = vm.dump_state();
    assert_eq!(st1.pc, 1);
    assert_eq!(st1.instruction_count, 1);
    assert_eq!(st1.value_stack, vec![Value::I32(42)]);
    assert_eq!(&*st1.last_instruction, "i32.const");

    // Step 2: should push 10
    assert!(vm.step().unwrap());
    let st2 = vm.dump_state();
    assert_eq!(st2.pc, 2);
    assert_eq!(st2.instruction_count, 2);
    assert_eq!(st2.value_stack, vec![Value::I32(42), Value::I32(10)]);
    assert_eq!(&*st2.last_instruction, "i32.const");

    // Continue until finished
    while vm.step().unwrap() {}
    assert!(vm.is_finished());

    // After our simulated program: locals[0] should hold 52 and stack is empty
    let st_end = vm.dump_state();
    assert_eq!(st_end.value_stack.len(), 0);
    assert_eq!(st_end.locals.get(0), Some(&Value::I32(52)));

    // Now reset and verify a function result via simulated call
    vm.reset();
    let results = vm.call_function("add", &[Value::I32(10), Value::I32(20)]).unwrap();
    assert_eq!(results, vec![Value::I32(30)]);
}
