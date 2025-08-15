use core::fmt;

/// Error types for WASM execution
#[derive(Debug, Clone, PartialEq)]
pub enum WasmError {
    /// Module validation failed
    InvalidModule,
    /// Module instantiation failed
    InstantiationFailed,
    /// Execution error
    ExecutionFailed,
    /// Function not found
    FunctionNotFound,
    /// Invalid function signature
    InvalidSignature,
    /// Stack overflow
    StackOverflow,
    /// Out of memory
    OutOfMemory,
    /// Trap occurred during execution
    Trap,
    /// Invalid instruction
    InvalidInstruction,
    /// Step execution not possible
    StepNotPossible,
    /// Invalid or corrupted VM state snapshot
    InvalidState,
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmError::InvalidModule => write!(f, "Invalid WASM module"),
            WasmError::InstantiationFailed => write!(f, "Module instantiation failed"),
            WasmError::ExecutionFailed => write!(f, "Execution failed"),
            WasmError::FunctionNotFound => write!(f, "Function not found"),
            WasmError::InvalidSignature => write!(f, "Invalid function signature"),
            WasmError::StackOverflow => write!(f, "Stack overflow"),
            WasmError::OutOfMemory => write!(f, "Out of memory"),
            WasmError::Trap => write!(f, "Trap occurred"),
            WasmError::InvalidInstruction => write!(f, "Invalid instruction"),
            WasmError::StepNotPossible => write!(f, "Step execution not possible"),
            WasmError::InvalidState => write!(f, "Invalid VM state snapshot"),
        }
    }
}

/// Result type for WASM operations
pub type Result<T> = core::result::Result<T, WasmError>;
