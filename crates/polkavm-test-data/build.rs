use std::path::Path;
use std::process::Command;

fn build(project: &str, profile: &str, target: &str) -> Result<(), String> {
    let polkavm = Path::new(std::env!("CARGO_MANIFEST_DIR")).join("../..");
    let project_path = polkavm.join("guest-programs").join(project).into_os_string().into_string().unwrap();

    let remap = std::format!(
        "--remap-path-prefix={}= --remap-path-prefix={}=",
        polkavm.into_os_string().into_string().unwrap(),
        std::env!("HOME"),
    );

    let mut cmd = Command::new("cargo");

    cmd.current_dir(project_path)
        .env_clear()
        .env("PATH", std::env!("PATH"))
        .env("RUSTFLAGS", remap)
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
        .arg("-Zunstable-options")
        .arg("-Zbuild-std=core,alloc");

    let res = cmd.output().unwrap();

    if res.status.success() {
        return Ok(());
    }

    return Err(String::from_utf8_lossy(&res.stderr).to_string());
}

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-env-changed=OUT_DIR");

    let target_32 = polkavm_linker::target_json_32_path()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    let target_64 = polkavm_linker::target_json_64_path()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();

    build("test-blob", "no-lto", &target_32)?;
    build("test-blob", "no-lto", &target_64)?;
    build("bench-pinky", "release", &target_32)?;
    build("bench-pinky", "release", &target_64)?;

    Ok(())
}
