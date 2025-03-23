pub const NONE: u64 = u64::MAX;
pub const OOB: u64 = u64::MAX - 2;
pub const OK: u64 = 0;
pub const WHAT: u64 = u64::MAX - 1;       // (1 << 64) - 2
pub const WHO: u64 = u64::MAX - 3;        // (1 << 64) - 4
pub const FULL: u64 = u64::MAX - 4;       // (1 << 64) - 5
pub const CORE: u64 = u64::MAX - 5;       // (1 << 64) - 6
pub const CASH: u64 = u64::MAX - 6;       // (1 << 64) - 7
pub const LOW: u64 = u64::MAX - 7;        // (1 << 64) - 8
pub const HUH: u64 = u64::MAX - 8;        // (1 << 64) - 9

pub const SEGMENT_SIZE: u64 = 4104;
pub const PARENT_MACHINE_INDEX: u64 = (1u64 << 32) - 1;

// memory related
pub const Z_Z: u64 = 1u64 << 16;
pub const Z_I: u64 = 1u64 << 24;
pub const INIT_RA: u64 = (1u64 << 32) - (1u64 << 16);
pub const PAGE_SIZE: u64 = 4096;
pub const FIRST_READABLE_ADDRESS: u32 = 16 * 4096;
pub const FIRST_READABLE_PAGE: u32 = 16;
