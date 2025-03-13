pub const NONE: u64 = u64::MAX;
pub const OOB: u64 = u64::MAX - 2;
pub const OK: u64 = 0;

pub const SEGMENT_SIZE: u64 = 4104;
pub const PARENT_MACHINE_INDEX: u64 = (1u64 << 32) - 1;

// memory related
pub const Z_Z: u64 = 1u64 << 16;
pub const Z_I: u64 = 1u64 << 24;
pub const INIT_RA: u64 = (1u64 << 32) - (1u64 << 16);
pub const PAGE_SIZE: u64 = 4096;
pub const FIRST_READABLE_ADDRESS: u32 = 16 * 4096;
pub const FIRST_READABLE_PAGE: u32 = 16;
