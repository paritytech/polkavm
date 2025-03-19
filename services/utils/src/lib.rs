#![no_std]

pub mod constants;
pub mod functions;
pub mod host_functions;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}
