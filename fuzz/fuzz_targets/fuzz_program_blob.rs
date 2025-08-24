#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use polkavm_common::program::{EstimateInterpreterMemoryUsageArgs, ProgramBlob, ProgramParts};
use polkavm_common::utils::ArcBytes;

#[derive(Debug, Arbitrary)]
struct FuzzParams {
    ro_data_size: u32,
    rw_data_size: u32,
    code_size: u32,
    stack_size: u32,

    page_size: u32,
    instruction_count: u32,
    basic_block_count: u32,
    max_cache_size_bytes: u32,
    max_block_size: u32,
    use_bounded_cache: bool,
}

fuzz_target!(|input: FuzzParams| {
    // Limit sizes to reasonable values to avoid OOM during fuzzing
    let alloc_ro_data_size = core::cmp::min(input.ro_data_size, 1024 * 1024) as usize;
    let alloc_rw_data_size = core::cmp::min(input.rw_data_size, 1024 * 1024) as usize;
    let alloc_code_size = core::cmp::min(input.code_size, 1024 * 1024) as usize;

    let mut parts = ProgramParts::default();
    parts.is_64_bit = true;
    parts.ro_data_size = input.ro_data_size;
    parts.rw_data_size = input.rw_data_size;
    parts.stack_size = input.stack_size;
    parts.ro_data = ArcBytes::from(vec![0; alloc_ro_data_size]);
    parts.rw_data = ArcBytes::from(vec![0; alloc_rw_data_size]);
    parts.code_and_jump_table = ArcBytes::from(vec![0; alloc_code_size]);

    if let Ok(blob) = ProgramBlob::from_parts(parts) {
        let args = if input.use_bounded_cache {
            EstimateInterpreterMemoryUsageArgs::BoundedCache {
                page_size: input.page_size,
                instruction_count: input.instruction_count,
                basic_block_count: input.basic_block_count,
                max_cache_size_bytes: input.max_cache_size_bytes,
                max_block_size: input.max_block_size,
            }
        } else {
            EstimateInterpreterMemoryUsageArgs::UnboundedCache {
                page_size: input.page_size,
                instruction_count: input.instruction_count,
                basic_block_count: input.basic_block_count,
            }
        };

        let _ = blob.estimate_interpreter_memory_usage(args);
    }
});
