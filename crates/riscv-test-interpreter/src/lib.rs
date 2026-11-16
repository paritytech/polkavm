#![deny(unused_must_use)]

mod interpreter;

pub use interpreter::{Exception, Interpreter, MemoryProtection, StopReason};
pub use polkavm_linker::riscv::Reg;
