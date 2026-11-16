#[allow(dead_code)]
#[derive(Copy, Clone, Default)]
pub struct GuestInit<'a> {
    pub page_size: u32,
    pub ro_data: &'a [u8],
    pub rw_data: &'a [u8],
    pub ro_data_size: u32,
    pub rw_data_size: u32,
    pub stack_size: u32,
    pub aux_data_size: u32,
}

impl<'a> GuestInit<'a> {
    pub fn memory_map(&self) -> Result<polkavm_common::abi::MemoryMap, &'static str> {
        polkavm_common::abi::MemoryMapBuilder::new(self.page_size)
            .ro_data_size(self.ro_data_size)
            .rw_data_size(self.rw_data_size)
            .stack_size(self.stack_size)
            .aux_data_size(self.aux_data_size)
            .build()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct Segfault {
    /// The address of the page which was accessed.
    pub page_address: u32,

    /// The size of the page.
    pub page_size: u32,

    /// If `true` the access was a write to a read-only page which exists.
    pub is_write_protected: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum InterruptKind {
    /// The execution finished normally.
    ///
    /// This happens when the program jumps to the address `0xffff0000`.
    Finished,

    /// The execution finished abnormally with a trap.
    ///
    /// This can happen for a few reasons:
    ///   - if the `trap` instruction is executed,
    ///   - if an invalid instruction is executed,
    ///   - if a jump to an invalid address is made,
    ///   - if a segmentation fault is triggered (when dynamic paging is not enabled for this VM)
    Trap,

    /// The execution triggered an external call with an `ecalli` instruction.
    Ecalli(u32),

    /// The execution triggered a segmentation fault.
    ///
    /// This happens when a program accesses a memory page that is not mapped,
    /// or tries to write to a read-only page.
    ///
    /// Requires dynamic paging to be enabled with [`ModuleConfig::set_dynamic_paging`](crate::ModuleConfig::set_dynamic_paging), otherwise is never emitted.
    Segfault(Segfault),

    /// The execution ran out of gas.
    ///
    /// Requires gas metering to be enabled with [`ModuleConfig::set_gas_metering`](crate::ModuleConfig::set_gas_metering), otherwise is never emitted.
    NotEnoughGas,

    /// Executed a single instruction.
    ///
    /// Requires execution step-tracing to be enabled with [`ModuleConfig::set_step_tracing`](crate::ModuleConfig::set_step_tracing), otherwise is never emitted.
    Step,
}
