pub const BENCH_PINKY_32: &[u8] = include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/release/bench-pinky");
pub const BENCH_PINKY_64: &[u8] = include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/release/bench-pinky");
pub const TEST_BLOB_32: &[u8] = include_bytes!("../../../guest-programs/target/riscv32emac-unknown-none-polkavm/no-lto/test-blob");
pub const TEST_BLOB_64: &[u8] = include_bytes!("../../../guest-programs/target/riscv64emac-unknown-none-polkavm/no-lto/test-blob");

pub fn get_test_blob(is_64_bit: bool) -> &'static [u8] {
    if is_64_bit {
        TEST_BLOB_64
    } else {
        TEST_BLOB_32
    }
}
