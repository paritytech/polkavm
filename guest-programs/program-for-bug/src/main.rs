#![no_std]
#![no_main]

extern crate core;

use core::fmt::Write;
use polkavm_derive::polkavm_export;

pub enum Foo {
    Success,
    CalleeTrapped,
    Unknown,
}

impl ::core::fmt::Debug for Foo {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::write_str(
            f,
            match self {
                Foo::Success => "Success",
                Foo::CalleeTrapped => "CalleeTrapped",
                Foo::Unknown => "Unknown",
            },
        )
    }
}

struct Writer;
impl core::fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe {
            crate::debug_message(s.as_ptr(), s.len() as u32);
        }
        Ok(())
    }
}

#[polkavm_derive::polkavm_import]
extern "C" {
    pub fn debug_message(str_ptr: *const u8, str_len: u32);
}

#[polkavm_export(abi = polkavm_derive::default_abi)]
pub fn deploy() {
    let mut m = Writer {};
    let _ = write!(&mut m, "{:?}", Foo::Success);
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp");
        core::hint::unreachable_unchecked();
    }
}
