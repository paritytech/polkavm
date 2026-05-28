#[cfg(target_arch = "x86_64")]
mod system;

#[cfg(target_arch = "x86_64")]
mod gastool_x86;

#[cfg(target_arch = "x86_64")]
fn main() {
    gastool_x86::main();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("gastool is only supported on x86_64-linux");
    std::process::exit(1);
}
