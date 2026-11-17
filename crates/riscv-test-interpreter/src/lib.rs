#![deny(unused_must_use)]

mod interpreter;
pub mod width;

pub use interpreter::{Exception, Interpreter, MemoryProtection, MemoryRegion, StopReason};
pub use polkavm_linker::riscv::Reg;
