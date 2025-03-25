// Host-Call Result Constants
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

// PVM result codes
pub const HALT: u64 = 0;
pub const PANIC: u64 = 1;
pub const FAULT: u64 = 2;
pub const HOST: u64 = 3;
pub const OOG: u64 = 4;

pub const SEGMENT_SIZE: u64 = 4104;
pub const PARENT_MACHINE_INDEX: u64 = (1u64 << 32) - 1;

// memory related
pub const Z_Z: u64 = 1u64 << 16;
pub const Z_I: u64 = 1u64 << 24;
pub const Z_P: u64 = 1u64 << 12;
pub const INIT_RA: u64 = (1u64 << 32) - (1u64 << 16);
pub const PAGE_SIZE: u64 = 4096;
pub const FIRST_READABLE_ADDRESS: u32 = 16 * 4096;
pub const FIRST_READABLE_PAGE: u32 = 16;
pub const INPUT_ARGS_ADDRESS: u32 = ((1u64 << 32) - Z_Z - Z_I) as u32;
pub const INPUT_ARGS_PAGE: u32 = INPUT_ARGS_ADDRESS / (PAGE_SIZE as u32);

// host functions id
pub const GAS: u64 = 0;              // Get remaining gas
pub const LOOKUP: u64 = 1;           // Fetch preimage content
pub const READ: u64 = 2;             // Read service storage
pub const WRITE: u64 = 3;            // Write to service storage
pub const INFO: u64 = 4;             // Get service metadata
pub const BLESS: u64 = 5;            // Bless a service (grant privilege)
pub const ASSIGN: u64 = 6;           // Assign a core for execution
pub const CHECKPOINT: u64 = 7;       // Commit state snapshot
pub const DESIGNATE: u64 = 8;        // Designate validators for a task
pub const EXPORT: u64 = 9;           // Export data segment
pub const FORGET: u64 = 10;          // Forget a preimage
pub const EJECT: u64 = 11;           // Remove a service
pub const KICKOFF: u64 = 12;         // Start inner-PVM execution
pub const MAKE: u64 = 13;            // Instantiate an inner-PVM
pub const NEW: u64 = 14;             // Create a new service
pub const POKE: u64 = 15;            // Write to inner-PVM memory
pub const PEEK: u64 = 16;            // Read from inner-PVM memory
pub const QUERY: u64 = 17;           // Query preimage status
pub const SOLICIT: u64 = 18;         // Request a preimage from network
pub const TRANSFER: u64 = 19;        // Transfer balance or resources
pub const UPGRADE: u64 = 20;         // Upgrade a service’s code
pub const VOID: u64 = 21;            // Clear inner-PVM memory
pub const EXPUNGE: u64 = 22;         // Destroy inner-PVM instance
pub const FETCH: u64 = 23;           // Fetch off-chain data
pub const ZERO: u64 = 24;            // Zero out memory page
pub const YIELD: u64 = 25;           // Return trie accumulation output
pub const LOG: u64 = 100;            // https://hackmd.io/@polkadot/jip1
