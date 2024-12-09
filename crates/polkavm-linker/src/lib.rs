#![doc = include_str!("../README.md")]
#![deny(unused_must_use)]

use std::path::PathBuf;

mod dwarf;
mod elf;
mod fast_range_map;
mod program_from_elf;
mod reader_wrapper;
mod riscv;
mod utils;

pub use crate::program_from_elf::{program_from_elf, Config, ProgramFromElfError};
pub use polkavm_common::program::{ProgramBlob, ProgramParseError, ProgramParts};

pub static TARGET_JSON_32_BIT: &str = include_str!("../riscv32emac-unknown-none-polkavm.json");
pub static TARGET_JSON_64_BIT: &str = include_str!("../riscv64emac-unknown-none-polkavm.json");

fn target_json_path_impl(json_name: &str, json_contents: &str) -> Result<PathBuf, String> {
    let version = env!("CARGO_PKG_VERSION");
    let cache_path = dirs::cache_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("./"))
        .join(".polkavm-linker")
        .join(version);
    if let Err(error) = std::fs::create_dir_all(&cache_path) {
        return Err(format!(
            "failed to fetch path to the PolkaVM target JSON: failed to create {cache_path:?}: {error}"
        ));
    }

    let json_path = cache_path.join(json_name);

    // If the file exists only write if the contents changed, to not unnecesarily bump the file's mtime.
    if std::fs::read_to_string(&json_path)
        .ok()
        .map_or(true, |existing_contents| existing_contents != json_contents)
    {
        std::fs::write(&json_path, json_contents)
            .map_err(|error| format!("failed to fetch path to the PolkaVM target JSON: failed to write to {json_path:?}: {error}"))?;
    }

    json_path
        .canonicalize()
        .map_err(|error| format!("failed to fetch path to the PolkaVM target JSON: failed to canonicalize path {json_path:?}: {error}"))
}

pub fn target_json_32_path() -> Result<PathBuf, String> {
    target_json_path_impl("riscv32emac-unknown-none-polkavm.json", TARGET_JSON_32_BIT)
}

pub fn target_json_64_path() -> Result<PathBuf, String> {
    target_json_path_impl("riscv64emac-unknown-none-polkavm.json", TARGET_JSON_64_BIT)
}
