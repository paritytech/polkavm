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

pub use crate::program_from_elf::{program_from_elf, Config, OptLevel, ProgramFromElfError};
pub use polkavm_common::assembler::assemble;
pub use polkavm_common::program::{ProgramBlob, ProgramParseError, ProgramParts};

pub static TARGET_JSON_32_BIT_OLD: &str = include_str!("../targets/legacy/riscv32emac-unknown-none-polkavm.json");
pub static TARGET_JSON_64_BIT_OLD: &str = include_str!("../targets/legacy/riscv64emac-unknown-none-polkavm.json");

pub static TARGET_JSON_32_BIT_NEW: &str = include_str!("../targets/1_91/riscv32emac-unknown-none-polkavm.json");
pub static TARGET_JSON_64_BIT_NEW: &str = include_str!("../targets/1_91/riscv64emac-unknown-none-polkavm.json");

struct VersionDetector {
    major: u32,
    minor: u32,
    date: u32,
    is_nightly: bool,
}

const YEAR: u32 = 10000;
const MONTH: u32 = 100;

impl VersionDetector {
    fn new() -> Result<Self, String> {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
        let version = std::process::Command::new(&rustc)
            .arg("--version")
            .output()
            .map_err(|err| format!("failed to detect rustc version: failed to run '{rustc} --version': {err}"))?
            .stdout;

        let full_version = String::from_utf8(version)
            .map_err(|_| format!("failed to detect rustc version: '{rustc} --version' returned non-valid UTF-8"))?;
        let full_version = full_version.trim();
        // Examples of version strings:
        //   'rustc 1.86.0 (05f9846f8 2025-03-31)'
        //   'rustc 1.90.0-nightly (e9182f195 2025-07-13)'

        (move || -> Option<Self> {
            let version = full_version.split(' ').nth(1)?;
            let is_nightly = full_version.contains("-nightly");
            let date = {
                let mut it = full_version[full_version.rfind(' ')? + 1..full_version.len() - 1].split('-');
                let year: u32 = it.next()?.parse().ok()?;
                let month: u32 = it.next()?.parse().ok()?;
                let day: u32 = it.next()?.parse().ok()?;
                year * YEAR + month * MONTH + day
            };
            let mut it = version.split('.');
            let major: u32 = it.next()?.parse().ok()?;
            let minor: u32 = it.next()?.parse().ok()?;

            Some(Self {
                major,
                minor,
                date,
                is_nightly,
            })
        })()
        .ok_or_else(|| format!("failed to detect rustc version: failed to parse output of '{rustc}' --version: {full_version:?}"))
    }

    fn check_feature(&self, req_minor: u32, year: u32, month: u32, day: u32) -> bool {
        self.major > 1
            || (self.major == 1 && self.minor >= req_minor && !self.is_nightly)
            || (self.major == 1 && self.minor >= (req_minor + 1))
            || (self.major == 1 && self.is_nightly && self.date >= year * YEAR + month * MONTH + day)
    }
}

fn target_json_path_impl(
    json_name: &str,
    json_contents_old: &str,
    json_contents_new: &str,
    rustc_version: RustcVersion,
) -> Result<PathBuf, String> {
    let rustc_version = match rustc_version {
        RustcVersion::Autodetect => VersionDetector::new()?,
        RustcVersion::Rustc_1_91 => VersionDetector {
            major: 1,
            minor: 91,
            date: 2025 * YEAR + 10 * MONTH + 30,
            is_nightly: false,
        },
        RustcVersion::Legacy => VersionDetector {
            major: 1,
            minor: 90,
            date: 2025 * YEAR + 9 * MONTH + 18,
            is_nightly: false,
        },
    };

    // https://github.com/rust-lang/rust/pull/144443
    let (json_contents, subdirectory) = if rustc_version.check_feature(91, 2025, 9, 1) {
        (json_contents_new, "1_91")
    } else {
        (json_contents_old, "legacy")
    };

    let version = env!("CARGO_PKG_VERSION");
    let cache_path = dirs::cache_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("./"))
        .join(".polkavm-linker")
        .join(version)
        .join(subdirectory);

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

#[allow(non_camel_case_types)]
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum RustcVersion {
    #[default]
    Autodetect,
    Legacy,
    Rustc_1_91,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct TargetJsonArgs {
    pub rustc_version: RustcVersion,
    pub is_64_bit: bool,
}

impl Default for TargetJsonArgs {
    fn default() -> Self {
        TargetJsonArgs {
            rustc_version: Default::default(),
            is_64_bit: true,
        }
    }
}

pub fn target_json_path(args: TargetJsonArgs) -> Result<PathBuf, String> {
    if args.is_64_bit {
        target_json_path_impl(
            "riscv64emac-unknown-none-polkavm.json",
            TARGET_JSON_64_BIT_OLD,
            TARGET_JSON_64_BIT_NEW,
            args.rustc_version,
        )
    } else {
        target_json_path_impl(
            "riscv32emac-unknown-none-polkavm.json",
            TARGET_JSON_32_BIT_OLD,
            TARGET_JSON_32_BIT_NEW,
            args.rustc_version,
        )
    }
}
