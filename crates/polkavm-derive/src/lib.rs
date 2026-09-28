#![no_std]
#![doc = include_str!("../README.md")]

pub use polkavm_derive_impl_macro::__PRIVATE_DO_NOT_USE_polkavm_define_abi as polkavm_define_abi;
pub use polkavm_derive_impl_macro::__PRIVATE_DO_NOT_USE_polkavm_export as polkavm_export;
pub use polkavm_derive_impl_macro::__PRIVATE_DO_NOT_USE_polkavm_import as polkavm_import;

pub mod default_abi {
    polkavm_derive_impl_macro::__PRIVATE_DO_NOT_USE_polkavm_impl_abi_support!();
}

/// A hardware accelerated memset.
#[inline]
#[allow(unused_assignments)]
#[allow(unused_mut)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
pub unsafe fn memset(mut dst: *mut u8, value: usize, mut count: usize) {
    #[cfg(all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "e"))]
    unsafe {
        core::arch::asm!(
            ".insn r 0xb, 2, 0, zero, zero, zero",
            inout("a0") dst,
            in("a1") value,
            inout("a2") count,
        );
    }

    #[cfg(not(all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "e")))]
    unsafe {
        core::ptr::write_bytes(dst, value as u8, count);
    }
}

#[inline]
pub fn heap_base() -> *mut core::ffi::c_void {
    #[cfg(all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "e"))]
    unsafe {
        let mut output;
        core::arch::asm!(
            ".insn r 0xb, 3, 0, {dst}, zero, zero",
            dst = out(reg) output,
        );
        output
    }

    #[cfg(not(all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "e")))]
    {
        core::ptr::null_mut()
    }
}

/// Sets the minimum stack size.
#[cfg(any(all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "e"), doc))]
#[macro_export]
macro_rules! min_stack_size {
    ($size:expr) => {
        ::core::arch::global_asm!(
            ".pushsection .polkavm_min_stack_size,\"R\",@note\n",
            ".4byte {size}",
            ".popsection\n",
            size = const $size,
        );
    }
}

#[cfg(target_pointer_width = "32")]
#[cfg(any(all(target_arch = "riscv32", target_feature = "e"), doc))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __atomic_fetch_add_8(address: *mut u64, value: u64) -> u64 {
    unsafe {
        let old_value = *address;
        *address += value;
        old_value
    }
}
