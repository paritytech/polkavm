#![allow(clippy::exit)]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

static EXCLUDED_ENVS: &[&str] = &["CARGO", "RUSTC", "RUSTUP"];

fn main() {
    println!("cargo:warning=Boostrapping polkavm-test-data");
    let path = PathBuf::new().join(std::env!("CARGO_MANIFEST_DIR")).join("../../guest-programs");
    let home = dirs::home_dir().unwrap();
    let rust_flags = std::format!(
        "--remap-path-prefix={}= --remap-path-prefix={}=~",
        home.to_str().unwrap(),
        path.to_str().unwrap()
    );

    let mut envs: HashMap<String, String> = std::env::vars()
        .filter(|(k, _)| !EXCLUDED_ENVS.iter().any(|e| k.contains(e)))
        .collect();
    envs.insert("RUSTFLAGS".into(), rust_flags);
    for (key, value) in &envs {
        println!("cargo:warning={}={}", key, value);
    }

    build("test-blob", "no-lto", &path, &envs, false);
    build("test-blob", "no-lto", &path, &envs, true);
    build("bench-pinky", "release", &path, &envs, false);
    build("bench-pinky", "release", &path, &envs, true);

    println!("cargo:rerun-if-changed=build.rs");
}

fn build(project: &str, profile: &str, path: &Path, envs: &HashMap<String, String>, target_64bit: bool) {
    println!("cargo:warning=project={project}, profile={profile}");

    let target = if target_64bit {
        polkavm_linker::target_json_64_path().unwrap()
    } else {
        polkavm_linker::target_json_32_path().unwrap()
    };

    println!("cargo:warning=target={}", target.to_str().unwrap());

    let mut cmd = Command::new("cargo");
    cmd.env_clear()
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
        .arg("-Zbuild-std=core,alloc")
        .current_dir(path.to_str().unwrap())
        .envs(envs);

    println!("cargo:warning=Starting cargo build command...");
    let res = cmd.output().unwrap();
    if !res.status.success() {
        let err = String::from_utf8_lossy(&res.stderr).to_string();
        println!("cargo:error={err}");
        std::process::exit(1);
    }
}
