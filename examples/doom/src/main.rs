#[cfg(target_arch = "x86_64")]
include!("doom_x86.rs");

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("doom-host is only supported on x86_64");
}
