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

// use utils::constants::FIRST_READABLE_ADDRESS;
use utils::functions::call_log;

use polkavm::{Config, Engine, InterruptKind, Linker, Module, ProgramBlob, Reg};

#[polkavm_derive::polkavm_export]
extern "C" fn refine(_start_address: u64, _length: u64) -> (u64, u64) {
    let raw_blob = include_bytes!("./example-hello-world.polkavm");
    let blob = ProgramBlob::parse(raw_blob[..].into()).unwrap();

    let config = Config::from_env().unwrap();
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_blob(&engine, &Default::default(), blob).unwrap();

    // High-level API.
    let mut linker: Linker = Linker::new();

    // Define a host function.
    linker.define_typed("get_third_number", || -> u32 { 100 }).unwrap();

    // Link the host functions with the module.
    let instance_pre = linker.instantiate_pre(&module).unwrap();

    // Instantiate the module.
    let mut instance = instance_pre.instantiate().unwrap();

    // Grab the function and call it.
    // println!("Calling into the guest program (high level):");
    call_log(2, None, &format!("Calling into the guest program (high level):"));

    let result = instance
        .call_typed_and_get_result::<u32, (u32, u32)>(&mut (), "add_numbers", (1, 10))
        .unwrap();
    // println!("  1 + 10 + 100 = {}", result);
    call_log(2, None, &format!("  1 + 10 + 100 = {:?}", result));

    // Low-level API.
    let entry_point = module.exports().find(|export| export == "add_numbers").unwrap().program_counter();
    let mut instance = module.instantiate().unwrap();
    instance.set_next_program_counter(entry_point);
    instance.set_reg(Reg::A0, 1);
    instance.set_reg(Reg::A1, 10);
    instance.set_reg(Reg::RA, polkavm::RETURN_TO_HOST);
    instance.set_reg(Reg::SP, module.default_sp());

    // println!("Calling into the guest program (low level):");
    call_log(2, None, &format!("Calling into the guest program (low level):"));
    loop {
        let interrupt_kind = instance.run().unwrap();
        match interrupt_kind {
            InterruptKind::Finished => break,
            InterruptKind::Ecalli(num) => {
                let Some(name) = module.imports().get(num) else {
                    panic!("unexpected external call: {num}");
                };

                if name == "get_third_number" {
                    instance.set_reg(Reg::A0, 100);
                } else {
                    panic!("unexpected external call: {name} ({num})")
                }
            }
            _ => panic!("unexpected interruption: {interrupt_kind:?}"),
        }
    }

    // println!("  1 + 10 + 100 = {}", instance.reg(Reg::A0));
    call_log(2, None, &format!("  1 + 10 + 100 = {:?}", instance.reg(Reg::A0)));
    (0 , 0)
}


#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(_start_address: u64, _length: u64) -> (u64, u64) {
    (0 , 0)
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u32) -> (u64, u64) {
    (0 , 0)
}