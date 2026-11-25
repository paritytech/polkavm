#![deny(unused_must_use)]
#![deny(clippy::as_conversions)]

mod interpreter;

pub use interpreter::{Exception, Interpreter, MemoryProtection, MemoryRegion, StopReason};
pub use polkavm_linker::riscv::Reg;
