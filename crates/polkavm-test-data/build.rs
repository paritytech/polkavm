#![allow(clippy::exit)]

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=OUT_DIR");
    build("test-blob", "no-lto", false);
    build("test-blob", "no-lto", true);
    build("bench-pinky", "release", false);
    build("bench-pinky", "release", true);
}

fn build(project: &str, profile: &str, target_64bit: bool) {
    let path = PathBuf::new().join(std::env!("CARGO_MANIFEST_DIR")).join("../../guest-programs");
    let home = dirs::home_dir().unwrap();
    let rust_flags = std::format!(
        "--remap-path-prefix={}= --remap-path-prefix={}=~",
        home.to_str().unwrap(),
        path.to_str().unwrap()
    );

    let target = if target_64bit {
        polkavm_linker::target_json_64_path().unwrap()
    } else {
        polkavm_linker::target_json_32_path().unwrap()
    };

    let mut cmd = Command::new("cargo");
    cmd.env_clear()
        .current_dir(path.to_str().unwrap())
        .env("PATH", std::env!("PATH"))
        .env("RUSTFLAGS", rust_flags)
        .env("RUSTUP_HOME", std::env!("RUSTUP_HOME"))
        .arg("build")
        .arg("-q")
        .arg("--profile")
        .arg(profile)
        .arg("--bin")
        .arg(project)
        .arg("-p")
        .arg(project)
        .arg("--target")
        .arg(target)
        .arg("-Zbuild-std=core,alloc");

    let res = cmd.output().unwrap();
    if !res.status.success() {
        let err = String::from_utf8_lossy(&res.stderr).to_string();
        println!("cargo::error={err}");
        std::process::exit(1);
    }
}
