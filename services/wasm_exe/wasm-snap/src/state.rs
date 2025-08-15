use alloc::vec::Vec;
use core::fmt;
use heapless::String;
use crate::value::{Value, ValType};

/// Maximum length for debug strings
const MAX_DEBUG_STRING_LEN: usize = 256;

/// Represents the current state of the WASM VM
#[derive(Debug, Clone)]
pub struct VmState {
    /// Current program counter
    pub pc: usize,
    /// Call stack depth
    pub call_depth: usize,
    /// Current locals in the active frame
    pub locals: Vec<Value>,
    /// Value stack
    pub value_stack: Vec<Value>,
    /// Current function index being executed
    pub current_function: Option<u32>,
    /// Number of instructions executed
    pub instruction_count: u64,
    /// Whether execution is finished
    pub is_finished: bool,
    /// Last instruction name (for debugging)
    pub last_instruction: String<MAX_DEBUG_STRING_LEN>,
}

impl VmState {
    /// Create a new empty VM state
    pub fn new() -> Self {
        Self {
            pc: 0,
            call_depth: 0,
            locals: Vec::new(),
            value_stack: Vec::new(),
            current_function: None,
            instruction_count: 0,
            is_finished: false,
            last_instruction: String::new(),
        }
    }

    /// Update the program counter
    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    /// Increment instruction count
    pub fn increment_instruction_count(&mut self) {
        self.instruction_count = self.instruction_count.saturating_add(1);
    }

    /// Set the current function
    pub fn set_current_function(&mut self, func_idx: Option<u32>) {
        self.current_function = func_idx;
    }

    /// Mark execution as finished
    pub fn set_finished(&mut self, finished: bool) {
        self.is_finished = finished;
    }

    /// Update call depth
    pub fn set_call_depth(&mut self, depth: usize) {
        self.call_depth = depth;
    }

    /// Set locals for the current frame
    pub fn set_locals(&mut self, locals: Vec<Value>) {
        self.locals = locals;
    }

    /// Set the value stack
    pub fn set_value_stack(&mut self, stack: Vec<Value>) {
        self.value_stack = stack;
    }

    /// Set the last executed instruction name
    pub fn set_last_instruction(&mut self, instruction: &str) {
        self.last_instruction.clear();
        let _ = self.last_instruction.push_str(instruction);
    }
}

impl Default for VmState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== WASM VM State ===")?;
        writeln!(f, "PC: {}", self.pc)?;
        writeln!(f, "Call Depth: {}", self.call_depth)?;
        writeln!(f, "Instructions Executed: {}", self.instruction_count)?;
        writeln!(f, "Finished: {}", self.is_finished)?;
        
        if let Some(func_idx) = self.current_function {
            writeln!(f, "Current Function: {}", func_idx)?;
        } else {
            writeln!(f, "Current Function: None")?;
        }

        if !self.last_instruction.is_empty() {
            writeln!(f, "Last Instruction: {}", self.last_instruction)?;
        }

        writeln!(f, "Value Stack ({} items):", self.value_stack.len())?;
        for (i, value) in self.value_stack.iter().enumerate() {
            writeln!(f, "  [{}]: {:?}", i, value)?;
        }

        writeln!(f, "Locals ({} items):", self.locals.len())?;
        for (i, local) in self.locals.iter().enumerate() {
            writeln!(f, "  [{}]: {:?}", i, local)?;
        }

        Ok(())
    }
}

/// Helper function to format a Value for display
pub fn format_value(value: &Value) -> String<64> {
    let mut result = String::new();
    match value {
        Value::I32(v) => { 
            let _ = result.push_str("i32(");
            // Simple integer to string conversion for no_std
            let mut num = *v;
            if num == 0 {
                let _ = result.push('0');
            } else {
                let mut digits = heapless::Vec::<char, 16>::new();
                let negative = num < 0;
                if negative {
                    num = -num;
                }
                while num > 0 {
                    let digit = (num % 10) as u8 + b'0';
                    let _ = digits.push(digit as char);
                    num /= 10;
                }
                if negative {
                    let _ = result.push('-');
                }
                for &digit in digits.iter().rev() {
                    let _ = result.push(digit);
                }
            }
            let _ = result.push(')');
        },
        Value::I64(_v) => { 
            let _ = result.push_str("i64(");
            // Simplified - placeholder
            let _ = result.push_str("...");
            let _ = result.push(')');
        },
        Value::F32(_v) => { let _ = result.push_str("f32(...)"); },
        Value::F64(_v) => { let _ = result.push_str("f64(...)"); },
    }
    result
}
