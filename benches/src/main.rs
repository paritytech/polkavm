#[cfg(target_arch = "x86_64")]
include!("benches_x86.rs");

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("benches are only supported on x86_64");
}
