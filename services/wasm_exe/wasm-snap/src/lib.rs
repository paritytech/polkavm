#![no_std]

//! A no_std WebAssembly execution crate that provides step-by-step execution
//! and VM state inspection capabilities using wasmi_core.

extern crate alloc;

pub mod vm;
pub mod state;
pub mod error;
pub mod value;

pub use vm::WasmVm;
pub use state::VmState;
pub use error::{WasmError, Result};
pub use value::{Value, ValType};

#[cfg(test)]
mod tests;

#[cfg(feature = "std")]
extern crate std;
