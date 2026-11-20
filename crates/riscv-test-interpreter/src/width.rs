pub fn sign_extend_8(value: i8) -> u64 {
    cast_to_u64(i64::from(value))
}

pub fn sign_extend_16(value: i16) -> u64 {
    cast_to_u64(i64::from(value))
}

pub fn sign_extend_32(value: i32) -> u64 {
    cast_to_u64(i64::from(value))
}

pub fn zero_extend_8(value: u8) -> u64 {
    u64::from(value)
}

pub fn zero_extend_16(value: u16) -> u64 {
    u64::from(value)
}

pub fn zero_extend_32(value: u32) -> u64 {
    u64::from(value)
}

pub fn truncate_to_i32(value: u64) -> i32 {
    value as i32
}

pub fn truncate_to_u32(value: u64) -> u32 {
    value as u32
}

pub fn truncate_to_i64(value: i128) -> i64 {
    value as i64
}

pub fn truncate_to_u64(value: u128) -> u64 {
    value as u64
}

pub fn cast_to_i32(value: u32) -> i32 {
    value as i32
}

pub fn cast_to_u32(value: i32) -> u32 {
    value as u32
}

pub fn cast_to_i64(value: u64) -> i64 {
    value as i64
}

pub fn cast_to_u64(value: i64) -> u64 {
    value as u64
}
