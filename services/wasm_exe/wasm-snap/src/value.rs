use core::fmt;

/// Custom Value type for our WASM VM implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
}

impl Value {
    /// Get the type of this value
    pub fn value_type(&self) -> ValType {
        match self {
            Value::I32(_) => ValType::I32,
            Value::I64(_) => ValType::I64,
            Value::F32(_) => ValType::F32,
            Value::F64(_) => ValType::F64,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::I32(v) => write!(f, "i32({})", v),
            Value::I64(v) => write!(f, "i64({})", v),
            Value::F32(v) => write!(f, "f32({})", v),
            Value::F64(v) => write!(f, "f64({})", v),
        }
    }
}

/// Value types supported by WebAssembly
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
}

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValType::I32 => write!(f, "i32"),
            ValType::I64 => write!(f, "i64"),
            ValType::F32 => write!(f, "f32"),
            ValType::F64 => write!(f, "f64"),
        }
    }
}
