#![no_std]
#![no_main]

include!("../../bench-common.rs");

struct State {
    memory: alloc::vec::Vec<u8>,
}

define_benchmark! {
    heap_size = 1024 * 1024 * 4,
    state = State {
        memory: alloc::vec::Vec::new(),
    },
}

fn benchmark_initialize(state: &mut State) {
    state.memory.resize(1024 * 1024 * 4 - 32, 0);
}

fn benchmark_run(state: &mut State) {
    state.memory[1..].fill(1);
}

#[cfg(target_env = "polkavm")]
#[polkavm_derive::polkavm_export]
#[no_mangle]
extern "C" fn benchmark_custom_accelerated(offset: usize, length: usize, mut times: usize) {
    while core::hint::black_box(times) > 0 {
        unsafe {
            let dst = STATE.memory.get_unchecked_mut(offset..offset + length).as_mut_ptr();
            polkavm_derive::memset(dst, 1, length);
        }
        times -= 1;
    }
}

#[cfg(target_env = "polkavm")]
#[polkavm_derive::polkavm_export]
#[no_mangle]
extern "C" fn benchmark_custom_naive(offset: usize, length: usize, mut times: usize) {
    while core::hint::black_box(times) > 0 {
        unsafe {
            let dst = STATE.memory.get_unchecked_mut(offset..offset + length).as_mut_ptr();
            for offset in 0..length {
                dst.add(offset).write_volatile(1);
            }
        }
        times -= 1;
    }
}

#[cfg(target_env = "polkavm")]
#[polkavm_derive::polkavm_export]
#[no_mangle]
extern "C" fn benchmark_custom_compiler_builtins(offset: usize, length: usize, mut times: usize) {
    while core::hint::black_box(times) > 0 {
        unsafe {
            let dst = STATE.memory.get_unchecked_mut(offset..offset + length).as_mut_ptr();
            compiler_builtins_crate::mem::memset(dst, 1, length);
        }
        times -= 1;
    }
}
