use alloc::vec;
use alloc::vec::Vec;

use crate::{
    error::{WasmError, Result},
    state::VmState,
    value::{Value, ValType},
};

/// A placeholder host function type used to describe imported host functions
/// in this simplified VM. It records parameter and result value types.
pub struct HostFuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

impl HostFuncType {
    pub fn new(params: Vec<ValType>, results: Vec<ValType>) -> Self {
        Self { params, results }
    }
}

/// A WebAssembly virtual machine that supports step-by-step execution
/// This is a simplified implementation for demonstration purposes
pub struct WasmVm {
    wasm_bytes: Option<Vec<u8>>,
    state: VmState,
}

impl WasmVm {
    /// Create a new WASM VM instance
    pub fn new() -> Self {
        Self {
            wasm_bytes: None,
            state: VmState::new(),
        }
    }

    /// Initialize the VM with WASM bytecode
    pub fn init(&mut self, wasm_bytes: &[u8]) -> Result<()> {
        // Basic validation - check WASM magic number
        if wasm_bytes.len() < 8 {
            return Err(WasmError::InvalidModule);
        }
        
        let magic = &wasm_bytes[0..4];
        if magic != [0x00, 0x61, 0x73, 0x6d] { // "\0asm"
            return Err(WasmError::InvalidModule);
        }

        let version = &wasm_bytes[4..8];
        if version != [0x01, 0x00, 0x00, 0x00] { // version 1
            return Err(WasmError::InvalidModule);
        }

        // Store the wasm bytes for later use
        self.wasm_bytes = Some(wasm_bytes.to_vec());

        // Reset VM state
        self.state = VmState::new();

        Ok(())
    }

    /// Get the current VM state
    pub fn dump_state(&self) -> &VmState {
        &self.state
    }

    /// Get a mutable reference to the VM state (for advanced usage)
    pub fn dump_state_mut(&mut self) -> &mut VmState {
        &mut self.state
    }

    /// Execute a single step of the WASM bytecode
    /// This is a simplified step execution for demonstration
    pub fn step(&mut self) -> Result<bool> {
        if self.state.is_finished {
            return Ok(false);
        }

        if self.wasm_bytes.is_none() {
            return Err(WasmError::InvalidModule);
        }

        // Simulate instruction execution
        self.state.increment_instruction_count();
        self.state.set_pc(self.state.pc + 1);
        
        // Simulate different types of instructions
        match self.state.instruction_count % 5 {
            1 => {
                self.state.set_last_instruction("i32.const");
                self.state.value_stack.push(Value::I32(42));
            }
            2 => {
                self.state.set_last_instruction("i32.const");
                self.state.value_stack.push(Value::I32(10));
            }
            3 => {
                self.state.set_last_instruction("i32.add");
                if self.state.value_stack.len() >= 2 {
                    let b = self.state.value_stack.pop().unwrap();
                    let a = self.state.value_stack.pop().unwrap();
                    if let (Value::I32(x), Value::I32(y)) = (a, b) {
                        self.state.value_stack.push(Value::I32(x + y));
                    }
                }
            }
            4 => {
                self.state.set_last_instruction("local.set");
                if let Some(value) = self.state.value_stack.pop() {
                    if self.state.locals.is_empty() {
                        self.state.locals.push(value);
                    } else {
                        self.state.locals[0] = value;
                    }
                }
            }
            0 => {
                self.state.set_last_instruction("return");
                self.state.set_finished(true);
                return Ok(false);
            }
            _ => unreachable!(),
        }

        // Stop after reasonable number of steps for demo
        if self.state.instruction_count > 20 {
            self.state.set_finished(true);
            return Ok(false);
        }

        Ok(true) // Continue execution
    }

    /// Execute a function by name with the given arguments
    /// This is a simplified implementation that simulates function execution
    pub fn call_function(&mut self, func_name: &str, args: &[Value]) -> Result<Vec<Value>> {
        if self.wasm_bytes.is_none() {
            return Err(WasmError::InvalidModule);
        }

        // Update state before execution
        self.state.set_current_function(Some(0));
        self.state.set_call_depth(self.state.call_depth + 1);
        self.state.set_locals(args.to_vec());

        // Simulate function execution based on name
        let results = match func_name {
            "add" => {
                if args.len() >= 2 {
                    if let (Value::I32(a), Value::I32(b)) = (args[0], args[1]) {
                        vec![Value::I32(a + b)]
                    } else {
                        return Err(WasmError::InvalidSignature);
                    }
                } else {
                    return Err(WasmError::InvalidSignature);
                }
            }
            "multiply" => {
                if args.len() >= 2 {
                    if let (Value::I32(a), Value::I32(b)) = (args[0], args[1]) {
                        vec![Value::I32(a * b)]
                    } else {
                        return Err(WasmError::InvalidSignature);
                    }
                } else {
                    return Err(WasmError::InvalidSignature);
                }
            }
            "test" => {
                // Simulate a simple test function
                vec![]
            }
            _ => return Err(WasmError::FunctionNotFound),
        };

        // Update state after execution
        self.state.set_call_depth(self.state.call_depth.saturating_sub(1));
        self.state.set_finished(true);
        self.state.increment_instruction_count();

        Ok(results)
    }

    /// Execute the entire WASM module's start function (if any)
    pub fn run(&mut self) -> Result<()> {
        if self.wasm_bytes.is_none() {
            return Err(WasmError::InvalidModule);
        }

        // Simulate running start function
        while !self.state.is_finished && self.state.instruction_count < 50 {
            if !self.step()? {
                break;
            }
        }

        self.state.set_finished(true);
        Ok(())
    }

    /// Add a host function to the linker (simplified for demo)
    pub fn add_host_function<T>(&mut self, _module: &str, _name: &str, _func_type: HostFuncType, _func: T) -> Result<()>
    where
        T: Fn(&[Value], &mut [Value]) -> Result<()> + Send + Sync + 'static,
    {
        // In a real implementation, this would register the host function
        // For now, just return success
        Ok(())
    }

    /// Reset the VM state
    pub fn reset(&mut self) {
        self.state = VmState::new();
    }

    /// Check if execution is finished
    pub fn is_finished(&self) -> bool {
        self.state.is_finished
    }

    /// Get the current instruction count
    pub fn instruction_count(&self) -> u64 {
        self.state.instruction_count
    }

    /// Serialize the current VM state into a compact byte buffer.
    /// This is a minimal custom format suitable for demo purposes.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Header magic: b"SNAP"
        buf.extend_from_slice(b"SNAP");
        // pc (u64), instruction_count (u64), finished (u8), current_function (i32 or -1), call_depth (u64)
        fn put_u64(buf: &mut Vec<u8>, v: u64) { buf.extend_from_slice(&v.to_le_bytes()); }
        fn put_i32(buf: &mut Vec<u8>, v: i32) { buf.extend_from_slice(&v.to_le_bytes()); }
        fn put_u8(buf: &mut Vec<u8>, v: u8) { buf.push(v); }

        put_u64(&mut buf, self.state.pc as u64);
        put_u64(&mut buf, self.state.instruction_count as u64);
        put_u8(&mut buf, self.state.is_finished as u8);
        let cur_fn = self.state.current_function.map(|v| v as i32).unwrap_or(-1);
        put_i32(&mut buf, cur_fn);
        put_u64(&mut buf, self.state.call_depth as u64);

        // Encode values helper
        fn put_val(buf: &mut Vec<u8>, v: &crate::value::Value) {
            match *v {
                crate::value::Value::I32(x) => { buf.push(0); buf.extend_from_slice(&x.to_le_bytes()); },
                crate::value::Value::I64(x) => { buf.push(1); buf.extend_from_slice(&x.to_le_bytes()); },
                crate::value::Value::F32(x) => { buf.push(2); buf.extend_from_slice(&x.to_bits().to_le_bytes()); },
                crate::value::Value::F64(x) => { buf.push(3); buf.extend_from_slice(&x.to_bits().to_le_bytes()); },
            }
        }

        // Locals
        put_u64(&mut buf, self.state.locals.len() as u64);
        for v in &self.state.locals { put_val(&mut buf, v); }
        // Value stack
        put_u64(&mut buf, self.state.value_stack.len() as u64);
        for v in &self.state.value_stack { put_val(&mut buf, v); }

        // Last instruction (length + utf8 bytes, truncated to 255)
        let li = &self.state.last_instruction;
        let bytes = li.as_bytes();
        let len = core::cmp::min(bytes.len(), 255);
        put_u8(&mut buf, len as u8);
        buf.extend_from_slice(&bytes[..len]);

        buf
    }

    /// Restore VM state from a previously produced snapshot.
    pub fn restore(&mut self, snap: &[u8]) -> Result<()> {
        use crate::error::WasmError;
        let mut i = 0usize;
        fn get<const N: usize>(b: &[u8], i: &mut usize) -> core::result::Result<[u8; N], ()> {
            if *i + N > b.len() { return Err(()); }
            let mut out = [0u8; N];
            out.copy_from_slice(&b[*i..*i+N]);
            *i += N; Ok(out)
        }
        fn get_u64(b: &[u8], i: &mut usize) -> core::result::Result<u64, ()> { Ok(u64::from_le_bytes(get::<8>(b,i)?)) }
        fn get_i32(b: &[u8], i: &mut usize) -> core::result::Result<i32, ()> { Ok(i32::from_le_bytes(get::<4>(b,i)?)) }
        fn get_u8(b: &[u8], i: &mut usize) -> core::result::Result<u8, ()> { Ok(u8::from_le_bytes(get::<1>(b,i)?)) }

        // Check magic
        if snap.len() < 4 || &snap[..4] != b"SNAP" { return Err(WasmError::InvalidState); }
        i = 4;

        let pc = get_u64(snap, &mut i).map_err(|_| WasmError::InvalidState)? as usize;
        let ic = get_u64(snap, &mut i).map_err(|_| WasmError::InvalidState)? as u64;
        let finished = get_u8(snap, &mut i).map_err(|_| WasmError::InvalidState)? != 0;
        let cur_fn = get_i32(snap, &mut i).map_err(|_| WasmError::InvalidState)?;
        let call_depth = get_u64(snap, &mut i).map_err(|_| WasmError::InvalidState)? as usize;

        // Read values helper
        fn get_val(b: &[u8], i: &mut usize) -> core::result::Result<crate::value::Value, ()> {
            let tag = b.get(*i).ok_or(())?; *i += 1;
            match *tag {
                0 => { let x = i32::from_le_bytes(get::<4>(b,i)?); Ok(crate::value::Value::I32(x)) },
                1 => { let x = i64::from_le_bytes(get::<8>(b,i)?); Ok(crate::value::Value::I64(x)) },
                2 => { let bits = u32::from_le_bytes(get::<4>(b,i)?); Ok(crate::value::Value::F32(f32::from_bits(bits))) },
                3 => { let bits = u64::from_le_bytes(get::<8>(b,i)?); Ok(crate::value::Value::F64(f64::from_bits(bits))) },
                _ => Err(()),
            }
        }

        // Locals
        let locals_len = get_u64(snap, &mut i).map_err(|_| WasmError::InvalidState)? as usize;
        let mut locals = Vec::with_capacity(locals_len);
        for _ in 0..locals_len { locals.push(get_val(snap, &mut i).map_err(|_| WasmError::InvalidState)?); }
        // Value stack
        let stack_len = get_u64(snap, &mut i).map_err(|_| WasmError::InvalidState)? as usize;
        let mut value_stack = Vec::with_capacity(stack_len);
        for _ in 0..stack_len { value_stack.push(get_val(snap, &mut i).map_err(|_| WasmError::InvalidState)?); }

        // Last instruction
        let li_len = get_u8(snap, &mut i).map_err(|_| WasmError::InvalidState)? as usize;
        if i + li_len > snap.len() { return Err(WasmError::InvalidState); }
        let li_bytes = &snap[i..i+li_len];
        i += li_len;
        let li_str = core::str::from_utf8(li_bytes).map_err(|_| WasmError::InvalidState)?;

        // Apply
        self.state.set_pc(pc);
        self.state.instruction_count = ic;
        self.state.set_finished(finished);
        self.state.set_current_function(if cur_fn < 0 { None } else { Some(cur_fn as u32) });
        self.state.set_call_depth(call_depth);
        self.state.set_locals(locals);
        self.state.set_value_stack(value_stack);
        self.state.set_last_instruction(li_str);

        Ok(())
    }
}
