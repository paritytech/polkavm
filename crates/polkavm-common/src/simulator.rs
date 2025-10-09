#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(unsafe_code)]

use crate::cast::cast;
use crate::program::{Opcode, ParsingVisitor, RawReg};
use crate::utils::{Bitness, BitnessT, GasVisitorT, B64};
use alloc::string::String;
use alloc::vec;

#[cfg(feature = "simd")]
use picosimd::amd64::{
    avx2::i8x32,
    avx2_composite::{i16x32, i32x32},
    sse::i8x16,
};

#[cfg(not(feature = "simd"))]
use picosimd::fallback::{i16x32, i32x32, i8x16, i8x32};

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
macro_rules! unsafe_avx2 {
    ($($t:tt)*) => { $($t)* }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
macro_rules! unsafe_avx2 {
    ($($t:tt)*) => { unsafe { $($t)* } }
}

#[derive(Copy, Clone, Debug, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum CacheModel {
    L1Hit,
    L2Hit,
    L3Hit,
}

/// The maximum number of instructions slots available per cycle.
const MAX_DECODE_PER_CYCLE: u32 = 4;

/// The maximum number of instructions in-flight.
const REORDER_BUFFER_SIZE: usize = 32;

/// The maximum number of cycles refunded at the end of each basic block.
const GAS_COST_SLACK: i32 = 3;

#[derive(Copy, Clone, Debug)]
pub struct InstCost {
    pub latency: i8,
    pub decode_slots: u32,
    pub alu_slots: u32,
    pub mul_slots: u32,
    pub div_slots: u32,
    pub load_slots: u32,
    pub store_slots: u32,
}

const MAX_ALU_SLOTS: u32 = 4;
const MAX_LOAD_SLOTS: u32 = 4;
const MAX_STORE_SLOTS: u32 = 4;
const MAX_MUL_SLOTS: u32 = 1;
const MAX_DIV_SLOTS: u32 = 1;

const fn bits_needed(value: u32) -> u32 {
    (value + 1).next_power_of_two().ilog2()
}

const ALU_BITS: u32 = bits_needed(MAX_ALU_SLOTS);
const LOAD_BITS: u32 = bits_needed(MAX_LOAD_SLOTS);
const STORE_BITS: u32 = bits_needed(MAX_STORE_SLOTS);
const MUL_BITS: u32 = bits_needed(MAX_MUL_SLOTS);
const DIV_BITS: u32 = bits_needed(MAX_DIV_SLOTS);

#[allow(clippy::int_plus_one)]
const _: () = {
    assert!((1 << ALU_BITS) - 1 >= MAX_ALU_SLOTS);
    assert!((1 << LOAD_BITS) - 1 >= MAX_LOAD_SLOTS);
    assert!((1 << STORE_BITS) - 1 >= MAX_STORE_SLOTS);
    assert!((1 << MUL_BITS) - 1 >= MAX_MUL_SLOTS);
    assert!((1 << DIV_BITS) - 1 >= MAX_DIV_SLOTS);
};

const ALU_OFFSET: u32 = 0;
const LOAD_OFFSET: u32 = ALU_OFFSET + ALU_BITS + 1;
const STORE_OFFSET: u32 = LOAD_OFFSET + LOAD_BITS + 1;
const MUL_OFFSET: u32 = STORE_OFFSET + STORE_BITS + 1;
const DIV_OFFSET: u32 = MUL_OFFSET + MUL_BITS + 1;

const RESOURCES_UNDERFLOW_MASK: u32 = (1 << (ALU_BITS + ALU_OFFSET))
    | (1 << (LOAD_BITS + LOAD_OFFSET))
    | (1 << (STORE_BITS + STORE_OFFSET))
    | (1 << (MUL_BITS + MUL_OFFSET))
    | (1 << (DIV_BITS + DIV_OFFSET));

#[cfg(all(test, feature = "logging"))]
struct DebugResources(u32);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugResources {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.debug_struct("Resources")
            .field("alu", &((self.0 >> ALU_OFFSET) & ((1 << ALU_BITS) - 1)))
            .field("load", &((self.0 >> LOAD_OFFSET) & ((1 << LOAD_BITS) - 1)))
            .field("store", &((self.0 >> STORE_OFFSET) & ((1 << STORE_BITS) - 1)))
            .field("mul", &((self.0 >> MUL_OFFSET) & ((1 << MUL_BITS) - 1)))
            .field("div", &((self.0 >> DIV_OFFSET) & ((1 << DIV_BITS) - 1)))
            .finish()
    }
}

#[cfg(all(test, feature = "logging"))]
struct DebugDeps([i32; 32]);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugDeps {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("{")?;
        let iter = self.0.into_iter().enumerate().filter(|(_, deps)| *deps != 0);
        let mut remaining = iter.clone().count();
        for (nth, mut deps) in iter {
            write!(fmt, "{nth}={{")?;
            while deps != 0 {
                let slot = deps.trailing_zeros();
                deps &= !(1 << slot);
                write!(fmt, "{slot}")?;
                if deps != 0 {
                    fmt.write_str(",")?;
                }
            }
            fmt.write_str("}")?;
            remaining -= 1;
            if remaining > 0 {
                fmt.write_str(", ")?;
            }
        }
        fmt.write_str("}")?;

        Ok(())
    }
}

#[cfg(all(test, feature = "logging"))]
struct DebugMask([i8; 32]);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugMask {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("{")?;
        let iter = self.0.into_iter().enumerate().filter(|(_, mask)| *mask != 0);
        let mut remaining = iter.clone().count();
        for (nth, mask) in iter {
            if mask == 0 {
                continue;
            } else if mask == -1 {
                write!(fmt, "{nth}")?;
            } else {
                write!(fmt, "{nth}={{{mask}}}")?;
            }

            remaining -= 1;
            if remaining > 0 {
                fmt.write_str(", ")?;
            }
        }
        fmt.write_str("}")?;

        Ok(())
    }
}

#[cfg(all(test, feature = "logging"))]
struct DebugEntryByRegister([i8; 16]);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugEntryByRegister {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("{")?;
        let mut remaining = self.0.iter().filter(|&&entry| entry != -1).count();
        for (reg, entry) in crate::program::Reg::ALL.into_iter().zip(self.0.into_iter()) {
            if entry == -1 {
                continue;
            }

            write!(fmt, "{reg}={entry}")?;
            remaining -= 1;
            if remaining > 0 {
                fmt.write_str(", ")?;
            }
        }
        fmt.write_str("}")?;

        Ok(())
    }
}

#[cfg(all(test, feature = "logging"))]
struct DebugCyclesRemaining([i8; 32]);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugCyclesRemaining {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("{")?;
        let mut remaining = self.0.len();
        for (index, count) in self.0.into_iter().enumerate() {
            write!(fmt, "{index}={count}")?;
            remaining -= 1;
            if remaining > 0 {
                fmt.write_str(", ")?;
            }
        }
        fmt.write_str("}")?;

        Ok(())
    }
}

#[cfg(all(test, feature = "logging"))]
struct DebugState([i8; 32]);

#[cfg(all(test, feature = "logging"))]
impl core::fmt::Debug for DebugState {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.write_str("{")?;
        let iter = self.0.into_iter().enumerate().filter(|(_, state)| *state != 0);
        let mut remaining = iter.clone().count();
        for (nth, state) in iter {
            remaining -= 1;
            let state = match state {
                1 => 'D',
                2 => 'w',
                3 => 'e',
                4 => 'X',
                _ => {
                    write!(fmt, "{nth}={state}")?;
                    if remaining > 0 {
                        fmt.write_str(", ")?;
                    }
                    continue;
                }
            };

            write!(fmt, "{nth}={state}")?;
            if remaining > 0 {
                fmt.write_str(", ")?;
            }
        }
        fmt.write_str("}")?;

        Ok(())
    }
}

impl InstCost {
    #[inline(always)]
    const fn resources(&self) -> u32 {
        assert!(self.alu_slots <= MAX_ALU_SLOTS);
        assert!(self.mul_slots <= MAX_MUL_SLOTS);
        assert!(self.div_slots <= MAX_DIV_SLOTS);
        assert!(self.load_slots <= MAX_LOAD_SLOTS);
        assert!(self.store_slots <= MAX_STORE_SLOTS);

        (self.alu_slots << ALU_OFFSET)
            | (self.load_slots << LOAD_OFFSET)
            | (self.store_slots << STORE_OFFSET)
            | (self.mul_slots << MUL_OFFSET)
            | (self.div_slots << DIV_OFFSET)
    }
}

const EMPTY_COST: InstCost = InstCost {
    latency: 0,
    decode_slots: 1,
    alu_slots: 0,
    mul_slots: 0,
    div_slots: 0,
    load_slots: 0,
    store_slots: 0,
};

#[derive(Copy, Clone, Debug)]
pub enum EventKind {
    Decode,
    WaitingForDependencies,
    Executing,
    Executed,
    WaitingForRetirement,
    Retired,
}

impl From<EventKind> for char {
    fn from(kind: EventKind) -> char {
        match kind {
            EventKind::Decode => 'D',
            EventKind::WaitingForDependencies => '=',
            EventKind::Executing => 'e',
            EventKind::Executed => 'E',
            EventKind::WaitingForRetirement => '-',
            EventKind::Retired => 'R',
        }
    }
}

pub trait Tracer: Sized {
    // A flag to make it easier for the optimizer to get rid of dead code.
    const SHOULD_CALL_ON_EVENT: bool;

    fn should_enable_fast_forward(&self) -> bool {
        true
    }

    fn on_event(&mut self, _cycle: u32, _instruction: u32, _event: EventKind) {}
}

impl Tracer for () {
    const SHOULD_CALL_ON_EVENT: bool = false;
}

pub struct Simulator<'a, B, T: Tracer = ()> {
    // The bytecode of the whole program.
    code: &'a [u8],
    /// The current cycle on which we're on.
    cycles: u32,
    /// The current instruction on which we're at when feeding code into the simulator.
    instructions: u32,
    /// Have we finished the simulation?
    finished: bool,
    /// Number of decode slots still available during this cycle.
    decode_slots_remaining_this_cycle: u32,
    /// Number of currently available resource, packed into a single field.
    resources_available: u32,
    /// The number of instructions currently in the reorder buffer.
    instructions_in_flight: u32,
    /// The offset of the first instruction in the reorder buffer (which is a circular buffer).
    reorder_buffer_head: u32,
    /// The next slot in the reorder buffer (which is a circular buffer).
    reorder_buffer_tail: u32,
    /// Which exact instruction does the reorder buffer contain at a given possition?
    /// Used only when emitting events.
    rob_instruction: [u32; REORDER_BUFFER_SIZE],
    /// The state of each entry in the reorder buffer.
    rob_state: i8x32,
    /// The number of cycles remaining for each instruction in the reorder buffer.
    rob_cycles_remaining: i8x32,
    /// The resources required to start execution for each instruction in the reorder buffer.
    rob_required_resources: i16x32,
    /// A bitmask which contains each instruction's dependencies.
    rob_dependencies: i32x32,
    /// A bitmask which contains each instruction's reverse dependencies.
    rob_depended_by: i32x32,
    /// A bitmask of all of the registers which a given instruction in the reorder buffer has written into.
    registers_written_by_rob_entry: i16x32,
    /// The index of the reorder buffer entry which has last written into a given register.
    rob_entry_by_register: i8x16,
    /// The cache model used for memory accesses.
    cache_model: CacheModel,
    /// When set this overrides the branch costs to be always either cheap (== brach hit) or expensive (==branch miss).
    force_branch_is_cheap: Option<bool>,

    tracer: T,
    _phantom: core::marker::PhantomData<B>,
}

impl<'a, B, T> Simulator<'a, B, T>
where
    T: Tracer,
    B: BitnessT,
{
    pub fn new(code: &'a [u8], cache_model: CacheModel, tracer: T) -> Self {
        unsafe_avx2! {
            let mut simulator = Simulator {
                code,
                rob_instruction: [0; REORDER_BUFFER_SIZE],
                cycles: 0,
                instructions: 0,
                finished: false,
                decode_slots_remaining_this_cycle: 0,
                resources_available: 0,
                rob_state: i8x32::zero(),
                rob_cycles_remaining: i8x32::zero(),
                rob_required_resources: i16x32::zero(),
                rob_dependencies: i32x32::zero(),
                rob_depended_by: i32x32::zero(),
                registers_written_by_rob_entry: i16x32::zero(),
                rob_entry_by_register: i8x16::zero(),
                reorder_buffer_tail: 0,
                cache_model,
                tracer,
                force_branch_is_cheap: None,
                instructions_in_flight: 0,
                reorder_buffer_head: 0,
                _phantom: core::marker::PhantomData,
            };

            simulator.clear();
            simulator
        }
    }

    pub fn set_force_branch_is_cheap(&mut self, value: Option<bool>) {
        self.force_branch_is_cheap = value;
    }

    fn clear(&mut self) {
        self.cycles = 0;
        self.instructions = 0;
        self.finished = false;
        self.instructions_in_flight = 0;
        self.decode_slots_remaining_this_cycle = MAX_DECODE_PER_CYCLE;
        self.resources_available = InstCost {
            alu_slots: MAX_ALU_SLOTS,
            mul_slots: MAX_MUL_SLOTS,
            div_slots: MAX_DIV_SLOTS,
            load_slots: MAX_LOAD_SLOTS,
            store_slots: MAX_STORE_SLOTS,
            ..EMPTY_COST
        }
        .resources()
            | RESOURCES_UNDERFLOW_MASK;

        self.reorder_buffer_tail = 0;
        self.reorder_buffer_head = 0;

        unsafe_avx2! {
            self.rob_entry_by_register = i8x16::negative_one();
            self.rob_state = i8x32::zero();
            self.rob_cycles_remaining = i8x32::zero();
            self.rob_required_resources = i16x32::zero();
            self.rob_dependencies = i32x32::zero();
            self.rob_depended_by = i32x32::zero();
            self.registers_written_by_rob_entry = i16x32::zero();
        }

        if T::SHOULD_CALL_ON_EVENT {
            self.rob_instruction.fill(0);
        }
    }

    fn emit_event(&mut self, slot: u32, kind: EventKind) {
        if T::SHOULD_CALL_ON_EVENT {
            self.tracer.on_event(self.cycles, self.rob_instruction[cast(slot).to_usize()], kind);
        }
    }

    fn tick_cycle<const FAST_FORWARD: bool>(&mut self) {
        unsafe_avx2! {
            self.tick_cycle_avx2::<FAST_FORWARD>();
        }
    }

    #[cfg_attr(all(feature = "simd", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    #[inline(never)]
    fn emit_events_avx2(&mut self, mask: i8x32, event_kind: EventKind) {
        if !T::SHOULD_CALL_ON_EVENT {
            return;
        }

        let mut bits = mask.most_significant_bits();
        while bits != 0 {
            let slot = bits.trailing_zeros();
            self.emit_event(slot, event_kind);
            bits &= !(1 << slot);
        }
    }

    fn instructions_in_flight(&self) -> u32 {
        self.instructions_in_flight
    }

    #[cfg_attr(all(feature = "simd", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    fn tick_cycle_avx2<const FAST_FORWARD: bool>(&mut self) {
        let state_decoding = i8x32::splat(1);
        let state_waiting = i8x32::splat(2);
        let state_executing = i8x32::splat(3);
        let state_executed = i8x32::splat(4);

        #[cfg(test)]
        let original_state = self.rob_state;
        #[cfg(test)]
        let original_cycles_remaining = self.rob_cycles_remaining;
        #[cfg(test)]
        let original_dependencies = self.rob_dependencies;
        #[cfg(test)]
        let original_depended_by = self.rob_depended_by;
        #[cfg(test)]
        let original_entry_by_register = self.rob_entry_by_register;
        #[cfg(test)]
        let original_required_resources = self.rob_required_resources;
        #[cfg(test)]
        let original_decode_slots = self.decode_slots_remaining_this_cycle;
        #[cfg(test)]
        let original_reorder_buffer_head = self.reorder_buffer_head;
        #[cfg(test)]
        let original_resources_available = self.resources_available;
        #[cfg(test)]
        let original_instructions_in_flight = self.instructions_in_flight;

        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: state={:?}",
            self.cycles,
            DebugState(self.rob_state.to_array())
        );
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: cycles={:?}",
            self.cycles,
            DebugCyclesRemaining(self.rob_cycles_remaining.to_array())
        );
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: dependencies={:?}",
            self.cycles,
            DebugDeps(self.rob_dependencies.to_array())
        );
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: depended_by={:?}",
            self.cycles,
            DebugDeps(self.rob_depended_by.to_array())
        );
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: entry_by_register={:?}",
            self.cycles,
            DebugEntryByRegister(self.rob_entry_by_register.to_array())
        );
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "tick_cycle_avx2[{}]: resources_available={:?}",
            self.cycles,
            DebugResources(self.resources_available)
        );

        debug_assert_eq!(
            self.rob_state.simd_eq(i8x32::zero()).most_significant_bits().count_zeros(),
            self.instructions_in_flight
        );

        // Retire unneeded instructions.
        {
            let is_waiting_for_retirement: i8x32 = self.rob_state.simd_eq(state_executed);
            let leading_count_to_retire = is_waiting_for_retirement
                .most_significant_bits()
                .rotate_right(self.reorder_buffer_head)
                .trailing_ones() as i32;

            let is_retired_this_cycle = i8x32::from_i1x32_sext(
                (cast(1_u64 << leading_count_to_retire).truncate_to_u32().wrapping_sub(1)).rotate_left(self.reorder_buffer_head) as i32,
            );

            // Mark every instruction which depended on instructions which just retired as not depending on them anymore.
            self.rob_dependencies = self
                .rob_dependencies
                .and_not(i32x32::splat(is_retired_this_cycle.most_significant_bits()));

            // Mark retired instructions as not depended by any other instruction.
            self.rob_depended_by = self.rob_depended_by.and_not(i32x32::from_i8x32_sext(is_retired_this_cycle));

            // Reset the state of retired instructions.
            self.rob_state = self.rob_state.and_not(is_retired_this_cycle);

            let retired_count = is_retired_this_cycle.most_significant_bits().count_ones();
            #[cfg(all(test, feature = "logging"))]
            if retired_count > 0 {
                log::debug!(
                    "tick_cycle_avx2[{}]: instructions_in_flight: {} -> {}",
                    self.cycles,
                    self.instructions_in_flight,
                    self.instructions_in_flight - retired_count
                );
            }

            self.instructions_in_flight -= retired_count;
            self.reorder_buffer_head = (self.reorder_buffer_head + retired_count) % (REORDER_BUFFER_SIZE as u32);

            self.emit_events_avx2(is_retired_this_cycle, EventKind::Retired);
            self.emit_events_avx2(
                is_waiting_for_retirement.and_not(is_retired_this_cycle),
                EventKind::WaitingForRetirement,
            );

            debug_assert_eq!(
                self.rob_state.simd_eq(i8x32::zero()).most_significant_bits().count_zeros(),
                self.instructions_in_flight
            );
        }

        {
            const RESOURCES_UNDERFLOW_MASK_I16: i16 = RESOURCES_UNDERFLOW_MASK as u16 as i16;
            let is_executed: i8x32 = self.rob_cycles_remaining.simd_lt(i8x32::splat(1));
            let is_executed_mask: i32 = is_executed.most_significant_bits();
            let has_no_dependencies: i8x32 = (self.rob_dependencies.and_not(i32x32::splat(is_executed_mask)))
                .simd_eq(i32x32::zero())
                .clamp_to_i8_range();

            let mut is_waiting_to_start: i8x32 = self.rob_state.simd_eq(state_waiting) & has_no_dependencies;

            for _ in 0..5 {
                #[cfg(all(test, feature = "logging"))]
                if is_waiting_to_start.most_significant_bits() != 0 {
                    log::debug!(
                        "tick_cycle_avx2[{}]: is_waiting_to_start={:?}",
                        self.cycles,
                        DebugMask(is_waiting_to_start.to_array())
                    );
                }
                debug_assert_eq!(self.resources_available & RESOURCES_UNDERFLOW_MASK, RESOURCES_UNDERFLOW_MASK);

                let new_resources: i16x32 = i16x32::splat(self.resources_available as i16) - self.rob_required_resources;
                let have_enough_resources: i8x32 = (new_resources.and(i16x32::splat(RESOURCES_UNDERFLOW_MASK_I16)))
                    .simd_eq(i16x32::splat(RESOURCES_UNDERFLOW_MASK_I16))
                    .clamp_to_i8_range();
                let have_enough_resources = have_enough_resources.and(is_waiting_to_start);
                let mask = have_enough_resources.most_significant_bits().rotate_left(self.reorder_buffer_head);
                let position = mask.trailing_zeros();

                if position != 32 {
                    let position = position.wrapping_sub(self.reorder_buffer_head) % (REORDER_BUFFER_SIZE as u32);
                    let resources_consumed = self.rob_required_resources.as_slice()[cast(position).to_usize()];
                    self.resources_available -= resources_consumed as u32;
                    self.rob_state.as_slice_mut()[cast(position).to_usize()] += 1;
                    is_waiting_to_start.as_slice_mut()[cast(position).to_usize()] = 0;
                }
            }
            self.emit_events_avx2(self.rob_state.simd_eq(state_waiting), EventKind::WaitingForDependencies);
        }

        // Progress execution. (executing -> executing, executing -> executed)
        let mut cycle_count = 1;
        {
            let is_executing: i8x32 = self.rob_state.simd_eq(state_executing);
            if FAST_FORWARD {
                let max_cycles =
                    ((self.rob_cycles_remaining & is_executing) | (is_executing ^ i8x32::negative_one())).horizontal_min_unsigned();
                let max_cycles = cast(max_cycles).to_signed();

                #[cfg(all(test, feature = "logging"))]
                log::debug!("tick_cycle_avx2[{}]: max_cycles={}", self.cycles, max_cycles);
                if max_cycles > 0 && self.decode_slots_remaining_this_cycle == MAX_DECODE_PER_CYCLE {
                    cycle_count = max_cycles;
                }
            }

            self.rob_cycles_remaining = self.rob_cycles_remaining.saturating_sub(i8x32::splat(cycle_count) & is_executing);

            // Check which instructions just finished execution.
            let is_execution_finished: i8x32 = self.rob_cycles_remaining.simd_eq(i8x32::zero()) & is_executing;
            let is_execution_finished = is_execution_finished.to_i16x32_sext();

            #[cfg(all(test, feature = "logging"))]
            log::debug!(
                "tick_cycle_avx2[{}]: is_execution_finished={:?}",
                self.cycles,
                is_execution_finished
            );

            let retired_register_writes: i16 = (self.registers_written_by_rob_entry & is_execution_finished).bitwise_reduce();
            self.registers_written_by_rob_entry = self.registers_written_by_rob_entry.and_not(is_execution_finished);
            self.rob_entry_by_register = self.rob_entry_by_register.or(i8x16::from_i1x16_sext(retired_register_writes));

            // Release any resources used.
            let resources_released = cast((self.rob_required_resources & is_execution_finished).wrapping_reduce()).to_unsigned();
            self.resources_available += u32::from(resources_released);
            self.rob_required_resources = self.rob_required_resources.and_not(is_execution_finished);

            let is_last_cycle = self.rob_cycles_remaining.simd_eq(i8x32::negative_one());
            let has_cycles_remaining = self.rob_cycles_remaining.simd_gt(i8x32::negative_one());
            self.rob_state += i8x32::splat(1) & is_executing.and(is_last_cycle);
            self.emit_events_avx2(is_executing.and(is_last_cycle), EventKind::Executed);
            self.emit_events_avx2(is_executing.and(has_cycles_remaining), EventKind::Executing);
        }

        // Progress: decoding -> waiting
        {
            let is_decoding = self.rob_state.simd_eq(state_decoding);
            self.rob_state += i8x32::splat(1) & is_decoding;
        }

        self.decode_slots_remaining_this_cycle = MAX_DECODE_PER_CYCLE;
        self.cycles += cast(i32::from(cycle_count)).to_unsigned();

        #[cfg(all(test, feature = "logging"))]
        {
            if self.rob_state != original_state {
                log::debug!("tick_cycle_avx2[{}]: state changed!", self.cycles);
            } else {
                log::debug!("tick_cycle_avx2[{}]: state did NOT change!", self.cycles);
            }
        }

        #[cfg(test)]
        {
            assert!(
                self.instructions_in_flight != original_instructions_in_flight
                    || self.reorder_buffer_head != original_reorder_buffer_head
                    || self.decode_slots_remaining_this_cycle != original_decode_slots
                    || self.resources_available != original_resources_available
                    || self.rob_state != original_state
                    || self.rob_cycles_remaining.max_signed(i8x32::negative_one())
                        != original_cycles_remaining.max_signed(i8x32::negative_one())
                    || self.rob_dependencies != original_dependencies
                    || self.rob_depended_by != original_depended_by
                    || self.rob_entry_by_register != original_entry_by_register
                    || self.rob_required_resources != original_required_resources,
                "made no progress"
            );
        }
    }

    #[inline(always)]
    fn tick_cycle_if_cannot_decode(&mut self, decode_slots: u32) {
        let mut should_tick =
            self.decode_slots_remaining_this_cycle < decode_slots || self.instructions_in_flight() == (REORDER_BUFFER_SIZE as u32);
        while should_tick {
            self.tick_cycle::<false>();
            should_tick = self.instructions_in_flight() == (REORDER_BUFFER_SIZE as u32);
        }
    }

    #[inline(always)]
    fn wait_until_empty(&mut self) {
        #[cfg(all(test, feature = "logging"))]
        if self.instructions_in_flight() > 0 {
            log::debug!("wait_until_empty[{}]: starting fast forward!", self.cycles);
        }

        while self.instructions_in_flight() > 0 {
            if self.tracer.should_enable_fast_forward() {
                self.tick_cycle::<true>();
            } else {
                self.tick_cycle::<false>();
            }
        }
    }

    fn dispatch_generic(&mut self, dst: Option<RawReg>, src1: Option<RawReg>, src2: Option<RawReg>, cost: InstCost) {
        #[cfg(all(test, feature = "logging"))]
        log::debug!(
            "dispatch[{}]: instruction={:?}, dst={:?}, src=[{:?}, {:?}], slots={}, latency={}, alu={}, load={}, store={}, mul={}, div={}",
            self.cycles,
            self.instructions,
            dst.map(|reg| reg.get()),
            src1.map(|reg| reg.get()),
            src2.map(|reg| reg.get()),
            cost.decode_slots,
            cost.latency,
            cost.alu_slots,
            cost.load_slots,
            cost.store_slots,
            cost.mul_slots,
            cost.div_slots,
        );

        debug_assert!(cost.latency >= 0);
        unsafe_avx2! { self.dispatch_generic_avx2(dst, src1, src2, cost) }
    }

    #[cfg_attr(all(feature = "simd", target_arch = "x86_64"), target_feature(enable = "avx2"))]
    fn dispatch_generic_avx2(&mut self, dst: Option<RawReg>, src1: Option<RawReg>, src2: Option<RawReg>, cost: InstCost) {
        let dst = dst.map(|dst| dst.get());
        let src1 = src1.map(|src1| src1.get());
        let src2 = src2.map(|src2| src2.get());

        self.tick_cycle_if_cannot_decode(cost.decode_slots);
        if T::SHOULD_CALL_ON_EVENT {
            self.tracer.on_event(self.cycles, self.instructions, EventKind::Decode);
        }

        let slot = self.reorder_buffer_tail;
        self.reorder_buffer_tail = (self.reorder_buffer_tail + 1) % (REORDER_BUFFER_SIZE as u32);
        let slot_mask = i8x32::zero().set_dynamic(cast(slot).truncate_to_u8(), cast(0xff_u8).to_signed());

        self.rob_cycles_remaining = self.rob_cycles_remaining.set_dynamic(slot as u8, cost.latency);
        self.rob_required_resources.as_slice_mut()[slot as usize] = cost.resources() as u16 as i16;

        let dependency_1: Option<u32> = src1
            .map(|src1| self.rob_entry_by_register.as_slice()[src1.to_usize()])
            .map(i32::from)
            .map(|x| cast(x).to_unsigned());
        let dependency_2: Option<u32> = src2
            .map(|src2| self.rob_entry_by_register.as_slice()[src2.to_usize()])
            .map(i32::from)
            .map(|x| cast(x).to_unsigned());
        match (dependency_1, dependency_2) {
            (Some(dependency_1), Some(dependency_2)) => {
                let base_1 = (dependency_1 >> 31) ^ 1;
                let base_2 = (dependency_2 >> 31) ^ 1;
                let dependencies_mask = cast(base_1.wrapping_shl(dependency_1) | base_2.wrapping_shl(dependency_2)).to_signed();
                self.rob_dependencies.as_slice_mut()[slot as usize] = dependencies_mask;
                self.rob_depended_by.as_slice_mut()[(dependency_1 * base_1) as usize] |= cast(base_1 << slot).to_signed();
                self.rob_depended_by.as_slice_mut()[(dependency_2 * base_2) as usize] |= cast(base_2 << slot).to_signed();
            }
            (Some(dependency), None) | (None, Some(dependency)) => {
                let base = (dependency >> 31) ^ 1;
                self.rob_dependencies.as_slice_mut()[slot as usize] = cast(base.wrapping_shl(dependency)).to_signed();
                self.rob_depended_by.as_slice_mut()[(dependency * base) as usize] |= cast(base.wrapping_shl(slot)).to_signed();
            }
            (None, None) => {}
        }

        if let Some(dst) = dst {
            let dst_mask: i16x32 = i16x32::splat(cast(cast(1_u32 << dst.to_u32()).truncate_to_u16()).to_signed());
            self.registers_written_by_rob_entry =
                self.registers_written_by_rob_entry.and_not(dst_mask) | (slot_mask.to_i16x32_sext() & dst_mask);
            self.rob_entry_by_register.as_slice_mut()[dst.to_usize()] = cast(cast(slot).truncate_to_u8()).to_signed();
        }

        self.rob_state = self.rob_state.set_dynamic(slot as u8, 1);
        if T::SHOULD_CALL_ON_EVENT {
            self.rob_instruction[cast(slot).to_usize()] = self.instructions;
        }

        self.instructions_in_flight += 1;
        self.decode_slots_remaining_this_cycle -= cost.decode_slots;
        self.instructions += 1;

        debug_assert_eq!(
            self.rob_state.simd_eq(i8x32::zero()).most_significant_bits().count_zeros(),
            self.instructions_in_flight
        );
    }

    fn dispatch_move_reg_avx2(&mut self, dst: RawReg, src: RawReg) {
        let dst = dst.get();
        let src = src.get();

        self.tick_cycle_if_cannot_decode(1);
        if T::SHOULD_CALL_ON_EVENT {
            self.tracer.on_event(self.cycles, self.instructions, EventKind::Decode);
        }

        let entry_by_register = self.rob_entry_by_register.as_slice_mut();
        let registers_written_by_rob_entry = self.registers_written_by_rob_entry.as_slice_mut();
        let old_slot = entry_by_register[dst.to_usize()];
        if old_slot != -1 {
            registers_written_by_rob_entry[old_slot as usize] &= !(1_i16 << dst.to_usize());
        }

        let new_slot = entry_by_register[src.to_usize()];
        if new_slot != -1 {
            registers_written_by_rob_entry[new_slot as usize] |= 1 << dst.to_usize();
        }

        entry_by_register[dst.to_usize()] = new_slot;
        self.decode_slots_remaining_this_cycle -= 1;
        self.instructions += 1;
    }

    fn dispatch_3op(&mut self, dst: RawReg, src1: RawReg, src2: RawReg, cost: InstCost) {
        self.dispatch_generic(Some(dst), Some(src1), Some(src2), cost);
    }

    fn dispatch_2op(&mut self, dst: RawReg, src: RawReg, cost: InstCost) {
        self.dispatch_generic(Some(dst), Some(src), None, cost);
    }

    fn dispatch_1op_dst(&mut self, dst: RawReg, cost: InstCost) {
        self.dispatch_generic(Some(dst), None, None, cost);
    }

    fn dispatch_finish(&mut self, latency: i8) {
        self.dispatch_generic(
            None,
            None,
            None,
            InstCost {
                latency,
                decode_slots: 1,
                ..EMPTY_COST
            },
        );

        self.wait_until_empty();
        self.finished = true;
    }

    fn load_cost(&self) -> InstCost {
        const L1_HIT: i8 = 4;
        const L2_HIT: i8 = 25;
        const L3_HIT: i8 = 37;

        let latency = match self.cache_model {
            CacheModel::L1Hit => L1_HIT,
            CacheModel::L2Hit => L2_HIT,
            CacheModel::L3Hit => L3_HIT,
        };

        InstCost {
            latency,
            decode_slots: 1,
            alu_slots: 1,
            load_slots: 1,
            ..EMPTY_COST
        }
    }

    fn dispatch_indirect_load(&mut self, dst: RawReg, base: RawReg, _offset: u32, _size: u32) {
        self.dispatch_2op(dst, base, self.load_cost());
    }

    fn dispatch_load(&mut self, dst: RawReg, _offset: u32, _size: u32) {
        self.dispatch_1op_dst(dst, self.load_cost());
    }

    #[allow(clippy::unused_self)]
    fn store_cost(&self) -> InstCost {
        InstCost {
            latency: 25,
            decode_slots: 1,
            alu_slots: 1,
            store_slots: 1,
            ..EMPTY_COST
        }
    }

    fn dispatch_store(&mut self, src: RawReg, _offset: u32, _size: u32) {
        self.dispatch_generic(None, Some(src), None, self.store_cost());
    }

    fn dispatch_store_imm(&mut self, _offset: u32, _size: u32) {
        self.dispatch_generic(None, None, None, self.store_cost());
    }

    fn dispatch_store_indirect(&mut self, src: RawReg, base: RawReg, _offset: u32, _size: u32) {
        self.dispatch_generic(None, Some(src), Some(base), self.store_cost());
    }

    fn dispatch_store_imm_indirect(&mut self, base: RawReg, _offset: u32, _size: u32) {
        self.dispatch_generic(None, Some(base), None, self.store_cost());
    }

    fn get_branch_cost(&self, offset: u32, args_length: u32, jump_offset: u32) -> i8 {
        const BRANCH_PREDICTION_HIT_COST: i8 = 1;
        const BRANCH_PREDICTION_MISS_COST: i8 = 20;

        if let Some(is_hit) = self.force_branch_is_cheap {
            return if is_hit {
                BRANCH_PREDICTION_HIT_COST
            } else {
                BRANCH_PREDICTION_MISS_COST
            };
        }

        if self
            .code
            .get(cast(offset).to_usize() + cast(args_length).to_usize())
            .map(|&opcode| opcode == Opcode::unlikely as u8 || opcode == Opcode::trap as u8)
            .unwrap_or(true)
        {
            return BRANCH_PREDICTION_HIT_COST;
        }

        if self
            .code
            .get(cast(jump_offset).to_usize())
            .map(|&opcode| opcode == Opcode::unlikely as u8 || opcode == Opcode::trap as u8)
            .unwrap_or(true)
        {
            return BRANCH_PREDICTION_HIT_COST;
        }

        BRANCH_PREDICTION_MISS_COST
    }

    fn dispatch_branch(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, jump_offset: u32) {
        self.dispatch_generic(
            None,
            Some(s1),
            Some(s2),
            InstCost {
                latency: self.get_branch_cost(offset, args_length, jump_offset),
                decode_slots: 1,
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
        self.wait_until_empty();
        self.finished = true;
    }

    fn dispatch_branch_imm(&mut self, offset: u32, args_length: u32, s: RawReg, jump_offset: u32) {
        self.dispatch_generic(
            None,
            Some(s),
            None,
            InstCost {
                latency: self.get_branch_cost(offset, args_length, jump_offset),
                decode_slots: 1,
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
        self.wait_until_empty();
        self.finished = true;
    }

    fn dispatch_trivial_2op_1c(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 1,
                decode_slots: 1,
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_trivial_2op_2c(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 2,
                decode_slots: 1,
                alu_slots: 2,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_simple_alu_2op(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 1,
                decode_slots: 1 + u32::from(d.get() != s.get()),
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_simple_alu_2op_32bit(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 1 + i8::from(B::BITNESS == Bitness::B64),
                decode_slots: 1 + u32::from(d.get() != s.get()) + u32::from(B::BITNESS == Bitness::B64),
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_simple_alu_3op(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 1,
                decode_slots: 1 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())),
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_simple_alu_3op_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 1 + i8::from(B::BITNESS == Bitness::B64),
                decode_slots: 1 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())) + u32::from(B::BITNESS == Bitness::B64),
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    fn dispatch_shift(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 1,
                decode_slots: 2 + u32::from(d.get() != s1.get()),
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_shift_32(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 1 + i8::from(B::BITNESS == Bitness::B64),
                decode_slots: 2 + u32::from(d.get() != s1.get()) + u32::from(B::BITNESS == Bitness::B64),
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_shift_imm_alt(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 1,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_shift_imm_alt_32(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 2,
                decode_slots: 4,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_compare(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 3,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_compare_imm(&mut self, d: RawReg, s: RawReg) {
        self.dispatch_2op(
            d,
            s,
            InstCost {
                latency: 3,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_cmov(&mut self, d: RawReg, s: RawReg, c: RawReg) {
        self.dispatch_3op(
            d,
            s,
            c,
            InstCost {
                latency: 2,
                decode_slots: 2,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_cmov_imm(&mut self, d: RawReg, c: RawReg) {
        self.dispatch_2op(
            d,
            c,
            InstCost {
                latency: 2,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_min_max(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 3,
                decode_slots: 2 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())),
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    fn dispatch_division(&mut self, d: RawReg, s1: RawReg, s2: RawReg) {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 60,
                decode_slots: 4,
                alu_slots: 1,
                div_slots: 1,
                ..EMPTY_COST
            },
        )
    }
}

impl<'a, B, T> GasVisitorT for Simulator<'a, B, T>
where
    B: BitnessT,
    T: Tracer,
{
    #[inline]
    fn take_block_cost(&mut self) -> Option<u32> {
        if (self.instructions_in_flight() == 0) & self.finished {
            let cycles = self.cycles;
            self.clear();

            let cycles = cast((cast(cycles).to_signed() - GAS_COST_SLACK).max(1)).to_unsigned();
            Some(cycles)
        } else {
            None
        }
    }

    fn is_at_start_of_basic_block(&self) -> bool {
        self.instructions == 0
    }
}

impl<'a, B, T> ParsingVisitor for Simulator<'a, B, T>
where
    B: BitnessT,
    T: Tracer,
{
    type ReturnTy = ();

    // Simple ALU instructions (3 op)

    #[inline(always)]
    fn xor(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op(d, s1, s2)
    }

    #[inline(always)]
    fn and(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op(d, s1, s2)
    }

    #[inline(always)]
    fn or(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op(d, s1, s2)
    }

    #[inline(always)]
    fn add_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op(d, s1, s2)
    }

    #[inline(always)]
    fn sub_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op(d, s1, s2)
    }

    // Simple ALU instructions (3 op), 32-bit

    #[inline(always)]
    fn add_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op_32(d, s1, s2)
    }

    #[inline(always)]
    fn sub_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_3op_32(d, s1, s2)
    }

    // Simple ALU instructions (2 op)

    #[inline(always)]
    fn xor_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s)
    }

    #[inline(always)]
    fn and_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s)
    }

    #[inline(always)]
    fn or_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _imm: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s)
    }

    #[inline(always)]
    fn add_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _imm: u32) -> Self::ReturnTy {
        // TODO: in 'd != s' case we use a single `lea`, see if modeling that makes sense
        self.dispatch_simple_alu_2op(d, s)
    }

    #[inline(always)]
    fn shift_logical_right_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s1)
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s1)
    }

    #[inline(always)]
    fn shift_logical_left_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s1)
    }

    #[inline(always)]
    fn rotate_right_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _c: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s1)
    }

    // Simple ALU instructions (2 op), 32-bit

    #[inline(always)]
    fn add_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _imm: u32) -> Self::ReturnTy {
        // TODO: in 'd != s' case we use a single `lea`, see if modeling that makes sense
        self.dispatch_simple_alu_2op_32bit(d, s)
    }

    #[inline(always)]
    fn shift_logical_right_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op_32bit(d, s1)
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op_32bit(d, s1)
    }

    #[inline(always)]
    fn shift_logical_left_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op_32bit(d, s1)
    }

    #[inline(always)]
    fn rotate_right_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _c: u32) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op_32bit(d, s1)
    }

    #[inline(always)]
    fn reverse_byte(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_simple_alu_2op(d, s)
    }

    // Trivial (2 op, 1 cycle)

    #[inline(always)]
    fn count_leading_zero_bits_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn count_leading_zero_bits_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn count_set_bits_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn count_set_bits_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn sign_extend_8(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn sign_extend_16(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    #[inline(always)]
    fn zero_extend_16(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_1c(d, s)
    }

    // Trivial (3 op, 2 cycles)

    #[inline(always)]
    fn count_trailing_zero_bits_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_2c(d, s)
    }

    #[inline(always)]
    fn count_trailing_zero_bits_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg) -> Self::ReturnTy {
        self.dispatch_trivial_2op_2c(d, s)
    }

    // Shifts and rotates, 64-bit

    #[inline(always)]
    fn shift_logical_right_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift(d, s1, s2)
    }

    #[inline(always)]
    fn shift_arithmetic_right_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift(d, s1, s2)
    }

    #[inline(always)]
    fn shift_logical_left_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift(d, s1, s2)
    }

    #[inline(always)]
    fn rotate_left_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift(d, s1, s2)
    }

    #[inline(always)]
    fn rotate_right_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift(d, s1, s2)
    }

    // Shifts and rotates, 32-bit

    #[inline(always)]
    fn shift_logical_right_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift_32(d, s1, s2)
    }

    #[inline(always)]
    fn shift_arithmetic_right_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift_32(d, s1, s2)
    }

    #[inline(always)]
    fn shift_logical_left_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift_32(d, s1, s2)
    }

    #[inline(always)]
    fn rotate_left_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift_32(d, s1, s2)
    }

    #[inline(always)]
    fn rotate_right_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_shift_32(d, s1, s2)
    }

    // Shifts and rotates, alt

    #[inline(always)]
    fn shift_logical_right_imm_alt_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt(d, s2)
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_alt_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt(d, s2)
    }

    #[inline(always)]
    fn shift_logical_left_imm_alt_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt(d, s2)
    }

    #[inline(always)]
    fn rotate_right_imm_alt_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _c: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt(d, s)
    }

    // Shifts and rotates, alt (32-bit)

    #[inline(always)]
    fn shift_logical_right_imm_alt_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt_32(d, s2)
    }

    #[inline(always)]
    fn shift_arithmetic_right_imm_alt_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt_32(d, s2)
    }

    #[inline(always)]
    fn shift_logical_left_imm_alt_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s2: RawReg, _s1: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt_32(d, s2)
    }

    #[inline(always)]
    fn rotate_right_imm_alt_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, _c: u32) -> Self::ReturnTy {
        self.dispatch_shift_imm_alt_32(d, s)
    }

    // Register comparisons

    #[inline(always)]
    fn set_less_than_unsigned(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_compare(d, s1, s2)
    }

    #[inline(always)]
    fn set_less_than_signed(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_compare(d, s1, s2)
    }

    // Register comparisons (immediate)

    #[inline(always)]
    fn set_less_than_unsigned_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_compare_imm(d, s1)
    }

    #[inline(always)]
    fn set_less_than_signed_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_compare_imm(d, s1)
    }

    #[inline(always)]
    fn set_greater_than_unsigned_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_compare_imm(d, s1)
    }

    #[inline(always)]
    fn set_greater_than_signed_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_compare_imm(d, s1)
    }

    // Conditional moves

    #[inline(always)]
    fn cmov_if_zero(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, c: RawReg) -> Self::ReturnTy {
        self.dispatch_cmov(d, s, c)
    }

    #[inline(always)]
    fn cmov_if_not_zero(&mut self, _offset: u32, _args_length: u32, d: RawReg, s: RawReg, c: RawReg) -> Self::ReturnTy {
        self.dispatch_cmov(d, s, c)
    }

    #[inline(always)]
    fn cmov_if_zero_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, c: RawReg, _s: u32) -> Self::ReturnTy {
        self.dispatch_cmov_imm(d, c)
    }

    #[inline(always)]
    fn cmov_if_not_zero_imm(&mut self, _offset: u32, _args_length: u32, d: RawReg, c: RawReg, _s: u32) -> Self::ReturnTy {
        self.dispatch_cmov_imm(d, c)
    }

    // Minimum/maximum

    #[inline(always)]
    fn maximum(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_min_max(d, s1, s2)
    }

    #[inline(always)]
    fn maximum_unsigned(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_min_max(d, s1, s2)
    }

    #[inline(always)]
    fn minimum(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_min_max(d, s1, s2)
    }

    #[inline(always)]
    fn minimum_unsigned(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_min_max(d, s1, s2)
    }

    // Indirect loads

    #[inline(always)]
    fn load_indirect_u8(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 1)
    }

    #[inline(always)]
    fn load_indirect_i8(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 1)
    }

    #[inline(always)]
    fn load_indirect_u16(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 2)
    }

    #[inline(always)]
    fn load_indirect_i16(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 2)
    }

    #[inline(always)]
    fn load_indirect_u32(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 4)
    }

    #[inline(always)]
    fn load_indirect_i32(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 4)
    }

    #[inline(always)]
    fn load_indirect_u64(&mut self, _offset: u32, _args_length: u32, dst: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_indirect_load(dst, base, offset, 8)
    }

    // Direct loads

    #[inline(always)]
    fn load_u8(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 1)
    }

    #[inline(always)]
    fn load_i8(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 1)
    }

    #[inline(always)]
    fn load_u16(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 2)
    }

    #[inline(always)]
    fn load_i16(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 2)
    }

    #[inline(always)]
    fn load_u32(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 4)
    }

    #[inline(always)]
    fn load_i32(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 4)
    }

    #[inline(always)]
    fn load_u64(&mut self, _offset: u32, _args_length: u32, dst: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_load(dst, offset, 8)
    }

    // Indirect stores (imm)

    #[inline(always)]
    fn store_imm_indirect_u8(&mut self, _offset: u32, _args_length: u32, base: RawReg, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm_indirect(base, offset, 1)
    }

    #[inline(always)]
    fn store_imm_indirect_u16(&mut self, _offset: u32, _args_length: u32, base: RawReg, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm_indirect(base, offset, 2)
    }

    #[inline(always)]
    fn store_imm_indirect_u32(&mut self, _offset: u32, _args_length: u32, base: RawReg, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm_indirect(base, offset, 4)
    }

    #[inline(always)]
    fn store_imm_indirect_u64(&mut self, _offset: u32, _args_length: u32, base: RawReg, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm_indirect(base, offset, 8)
    }

    // Indirect stores

    #[inline(always)]
    fn store_indirect_u8(&mut self, _offset: u32, _args_length: u32, src: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store_indirect(src, base, offset, 1)
    }

    #[inline(always)]
    fn store_indirect_u16(&mut self, _offset: u32, _args_length: u32, src: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store_indirect(src, base, offset, 2)
    }

    #[inline(always)]
    fn store_indirect_u32(&mut self, _offset: u32, _args_length: u32, src: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store_indirect(src, base, offset, 4)
    }

    #[inline(always)]
    fn store_indirect_u64(&mut self, _offset: u32, _args_length: u32, src: RawReg, base: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store_indirect(src, base, offset, 8)
    }

    // Stores (imm)

    #[inline(always)]
    fn store_imm_u8(&mut self, _offset: u32, _args_length: u32, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm(offset, 1)
    }

    #[inline(always)]
    fn store_imm_u16(&mut self, _offset: u32, _args_length: u32, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm(offset, 2)
    }

    #[inline(always)]
    fn store_imm_u32(&mut self, _offset: u32, _args_length: u32, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm(offset, 4)
    }

    #[inline(always)]
    fn store_imm_u64(&mut self, _offset: u32, _args_length: u32, offset: u32, _value: u32) -> Self::ReturnTy {
        self.dispatch_store_imm(offset, 8)
    }

    // Stores

    #[inline(always)]
    fn store_u8(&mut self, _offset: u32, _args_length: u32, src: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store(src, offset, 1)
    }

    #[inline(always)]
    fn store_u16(&mut self, _offset: u32, _args_length: u32, src: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store(src, offset, 2)
    }

    #[inline(always)]
    fn store_u32(&mut self, _offset: u32, _args_length: u32, src: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store(src, offset, 4)
    }

    #[inline(always)]
    fn store_u64(&mut self, _offset: u32, _args_length: u32, src: RawReg, offset: u32) -> Self::ReturnTy {
        self.dispatch_store(src, offset, 8)
    }

    // Branches

    #[inline(always)]
    fn branch_less_unsigned(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    #[inline(always)]
    fn branch_less_signed(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    #[inline(always)]
    fn branch_greater_or_equal_unsigned(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    #[inline(always)]
    fn branch_greater_or_equal_signed(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    #[inline(always)]
    fn branch_eq(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    #[inline(always)]
    fn branch_not_eq(&mut self, offset: u32, args_length: u32, s1: RawReg, s2: RawReg, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch(offset, args_length, s1, s2, imm)
    }

    // Branches (with immediate)

    #[inline(always)]
    fn branch_eq_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_not_eq_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_less_unsigned_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_less_signed_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_greater_or_equal_unsigned_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_greater_or_equal_signed_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_less_or_equal_unsigned_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_less_or_equal_signed_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_greater_unsigned_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    #[inline(always)]
    fn branch_greater_signed_imm(&mut self, offset: u32, args_length: u32, s1: RawReg, _s2: u32, imm: u32) -> Self::ReturnTy {
        self.dispatch_branch_imm(offset, args_length, s1, imm);
    }

    // Division

    #[inline(always)]
    fn div_unsigned_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn div_signed_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn rem_unsigned_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn rem_signed_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn div_unsigned_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn div_signed_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn rem_unsigned_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    #[inline(always)]
    fn rem_signed_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_division(d, s1, s2)
    }

    // Misc

    #[inline(always)]
    fn and_inverted(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        // TODO: inaccurate
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 2,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn or_inverted(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        // TODO: inaccurate
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 2,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn xnor(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 2,
                decode_slots: 2 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())),
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    #[inline(always)]
    fn negate_and_add_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_2op(
            d,
            s1,
            InstCost {
                latency: 2,
                decode_slots: 3,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn negate_and_add_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_2op(
            d,
            s1,
            InstCost {
                latency: 3,
                decode_slots: 4,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn move_reg(&mut self, _offset: u32, _args_length: u32, dst: RawReg, src: RawReg) -> Self::ReturnTy {
        self.dispatch_move_reg_avx2(dst, src);
    }

    #[inline(always)]
    fn load_imm(&mut self, _offset: u32, _args_length: u32, dst: RawReg, _value: u32) -> Self::ReturnTy {
        self.dispatch_1op_dst(
            dst,
            InstCost {
                latency: 1,
                decode_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn load_imm64(&mut self, _offset: u32, _args_length: u32, dst: RawReg, _value: u64) -> Self::ReturnTy {
        self.dispatch_1op_dst(
            dst,
            InstCost {
                latency: 1,
                decode_slots: 2,
                ..EMPTY_COST
            },
        );
    }

    #[inline(always)]
    fn mul_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 4,
                decode_slots: 2 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())),
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 3,
                decode_slots: 1 + u32::from((d.get() != s1.get()) & (d.get() != s2.get())),
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_imm_32(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_2op(
            d,
            s1,
            InstCost {
                latency: 4,
                decode_slots: 2 + u32::from(d.get() != s1.get()),
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_imm_64(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, _s2: u32) -> Self::ReturnTy {
        self.dispatch_2op(
            d,
            s1,
            InstCost {
                latency: 3,
                decode_slots: 1 + u32::from(d.get() != s1.get()),
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_upper_signed_signed(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 4,
                decode_slots: 4,
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_upper_unsigned_unsigned(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 4,
                decode_slots: 4,
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    #[inline(always)]
    fn mul_upper_signed_unsigned(&mut self, _offset: u32, _args_length: u32, d: RawReg, s1: RawReg, s2: RawReg) -> Self::ReturnTy {
        self.dispatch_3op(
            d,
            s1,
            s2,
            InstCost {
                latency: 6,
                decode_slots: 4,
                alu_slots: 1,
                mul_slots: 1,
                ..EMPTY_COST
            },
        )
    }

    // End of block instructions

    #[cold]
    fn invalid(&mut self, _offset: u32, _args_length: u32) -> Self::ReturnTy {
        self.dispatch_finish(2);
    }

    #[inline(always)]
    fn trap(&mut self, _offset: u32, _args_length: u32) -> Self::ReturnTy {
        self.dispatch_finish(2);
    }

    #[inline(always)]
    fn fallthrough(&mut self, _offset: u32, _args_length: u32) -> Self::ReturnTy {
        self.dispatch_finish(2);
    }

    #[inline(always)]
    fn unlikely(&mut self, _offset: u32, _args_length: u32) -> Self::ReturnTy {
        self.dispatch_generic(
            None,
            None,
            None,
            InstCost {
                latency: 40,
                decode_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    #[inline(always)]
    fn jump(&mut self, _offset: u32, _args_length: u32, _target: u32) -> Self::ReturnTy {
        self.dispatch_finish(15);
    }

    #[inline(always)]
    fn load_imm_and_jump(&mut self, _offset: u32, _args_length: u32, _ra: RawReg, _value: u32, _target: u32) -> Self::ReturnTy {
        self.dispatch_finish(15);
    }

    #[inline(always)]
    fn jump_indirect(&mut self, _offset: u32, _args_length: u32, base: RawReg, _base_offset: u32) -> Self::ReturnTy {
        self.dispatch_generic(
            None,
            Some(base),
            None,
            InstCost {
                latency: 22,
                decode_slots: 1,
                ..EMPTY_COST
            },
        );
        self.wait_until_empty();
        self.finished = true;
    }

    #[inline(always)]
    fn load_imm_and_jump_indirect(
        &mut self,
        _offset: u32,
        _args_length: u32,
        _ra: RawReg,
        base: RawReg,
        _value: u32,
        _base_offset: u32,
    ) -> Self::ReturnTy {
        self.dispatch_generic(
            None,
            Some(base),
            None,
            InstCost {
                latency: 22,
                decode_slots: 1,
                ..EMPTY_COST
            },
        );
        self.wait_until_empty();
        self.finished = true;
    }

    // Special instructions

    #[inline(always)]
    fn ecalli(&mut self, _offset: u32, _args_length: u32, _imm: u32) -> Self::ReturnTy {
        self.dispatch_generic(
            None,
            None,
            None,
            InstCost {
                latency: 100,
                decode_slots: 4,
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    #[inline(always)]
    fn sbrk(&mut self, _offset: u32, _args_length: u32, dst: RawReg, src: RawReg) -> Self::ReturnTy {
        // TODO: YOLO assigned
        self.dispatch_2op(
            dst,
            src,
            InstCost {
                latency: 100,
                decode_slots: 4,
                alu_slots: 1,
                ..EMPTY_COST
            },
        );
    }

    #[inline(always)]
    fn memset(&mut self, _offset: u32, _args_length: u32) -> Self::ReturnTy {
        // TODO: YOLO assigned
        self.dispatch_generic(
            None,
            None,
            None,
            InstCost {
                latency: 100,
                decode_slots: 4,
                alu_slots: 1,
                ..EMPTY_COST
            },
        )
    }
}

pub fn timeline_for_instructions(
    code: &[u8],
    cache_model: CacheModel,
    instructions: &[crate::program::ParsedInstruction],
) -> (String, u32) {
    timeline_for_instructions_impl(code, cache_model, instructions, false)
}

fn timeline_for_instructions_impl(
    code: &[u8],
    cache_model: CacheModel,
    instructions: &[crate::program::ParsedInstruction],
    should_enable_fast_forward: bool,
) -> (String, u32) {
    use crate::program::InstructionFormat;
    use alloc::collections::BTreeMap;

    struct TimelineTracer<'a> {
        should_enable_fast_forward: bool,
        timeline: &'a mut BTreeMap<(u32, u32), EventKind>,
    }

    impl<'a> Tracer for TimelineTracer<'a> {
        const SHOULD_CALL_ON_EVENT: bool = true;

        fn should_enable_fast_forward(&self) -> bool {
            self.should_enable_fast_forward
        }

        fn on_event(&mut self, cycle: u32, instruction: u32, event: EventKind) {
            match self.timeline.entry((cycle, instruction)) {
                alloc::collections::btree_map::Entry::Vacant(entry) => {
                    #[cfg(all(test, feature = "logging"))]
                    log::debug!(
                        "on_event[{cycle}]: instruction={instruction} '{}' (event={event:?})",
                        char::from(event)
                    );
                    entry.insert(event);
                }
                alloc::collections::btree_map::Entry::Occupied(entry) => {
                    panic!(
                        "duplicate timeline update: cycle={cycle} instruction={instruction} old_event={:?} new_event={event:?}",
                        entry.get()
                    );
                }
            }
        }
    }

    let count = instructions
        .iter()
        .take_while(|inst| !inst.kind.opcode().starts_new_basic_block())
        .count();

    let instructions = &instructions[..(count + 1).min(instructions.len())];

    let mut timeline_map = BTreeMap::new();
    let mut sim = Simulator::<B64, _>::new(
        code,
        cache_model,
        TimelineTracer {
            should_enable_fast_forward,
            timeline: &mut timeline_map,
        },
    );

    for &instruction in instructions {
        assert!(sim.take_block_cost().is_none());
        instruction.visit_parsing(&mut sim);
    }

    let total_cycles = cast(sim.cycles).to_usize();
    let block_cost = sim.take_block_cost().unwrap();
    #[cfg(all(test, feature = "logging"))]
    log::debug!("Total cycles: {total_cycles}");

    #[cfg(all(test, feature = "logging"))]
    log::debug!("Block cost: {block_cost}");

    let mut timeline = vec!['.'; total_cycles * instructions.len()];
    for ((cycle, instruction), event) in timeline_map {
        let index = instruction as usize * total_cycles + cycle as usize;
        timeline[index] = char::from(event);
    }

    let format = InstructionFormat {
        is_64_bit: true,
        ..InstructionFormat::default()
    };

    let mut timeline_s = String::new();
    for (nth_instruction, instruction) in instructions.iter().enumerate() {
        use core::fmt::Write;

        let line = &timeline[nth_instruction * total_cycles..(nth_instruction + 1) * total_cycles];
        timeline_s.extend(line.iter().copied());
        timeline_s.push_str("  ");
        writeln!(&mut timeline_s, "{}", instruction.display(&format)).unwrap();
    }

    if should_enable_fast_forward {
        let mut timeline_new = String::with_capacity(timeline_s.len());
        let mut is_in_cycles = true;
        let mut last = '.';
        for mut ch in timeline_s.chars() {
            if ch == ' ' {
                is_in_cycles = false;
            } else if ch == '\n' {
                is_in_cycles = true;
                last = '.';
            } else if ch == '.' {
                if last != 'R' && last != 'D' && is_in_cycles {
                    ch = last;
                }
            } else {
                last = ch;
            }
            timeline_new.push(ch);
        }
        timeline_s = timeline_new;
    }

    (timeline_s, block_cost)
}

pub fn trap_cost(cache_model: CacheModel) -> u32 {
    let mut sim = Simulator::<B64, _>::new(&[], cache_model, ());
    crate::program::ParsedInstruction {
        kind: crate::program::Instruction::trap,
        offset: crate::program::ProgramCounter(0),
        next_offset: crate::program::ProgramCounter(0),
    }
    .visit_parsing(&mut sim);
    sim.take_block_cost().unwrap()
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use super::{timeline_for_instructions, timeline_for_instructions_impl, CacheModel};
    use crate::assembler::assemble;
    use crate::program::{ProgramBlob, ISA64_V1};

    #[cfg(test)]
    fn test_config() -> CacheModel {
        CacheModel::L1Hit
    }

    #[cfg(test)]
    fn assert_timeline(config: CacheModel, program: &str, expected_timeline: &str) {
        use crate::cast::cast;

        let _ = env_logger::try_init();

        let program = assemble(program).unwrap();
        let blob = ProgramBlob::parse(program.into()).unwrap();
        let instructions: Vec<_> = blob.instructions(ISA64_V1).collect();

        let (timeline_s, cycles) = timeline_for_instructions(blob.code(), config, &instructions);
        let mut expected_timeline_s = String::new();
        let mut expected_cycles = 0;
        for line in expected_timeline.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            expected_timeline_s.push_str(line);
            expected_timeline_s.push('\n');

            expected_cycles = expected_cycles.max(line.split("  ").next().unwrap().len() as u32);
        }

        if timeline_s != expected_timeline_s {
            panic!("Timeline mismatch!\n\nExpected timeline:\n{expected_timeline_s}\nActual timeline:\n{timeline_s}");
        }

        let expected_cycles = cast(expected_cycles).to_signed() - 3;
        assert_eq!(cast(cycles).to_signed(), expected_cycles);

        #[cfg(feature = "logging")]
        log::debug!("Rerunning with fast-forward enabled...");

        let (timeline_ff_s, cycles_ff) = timeline_for_instructions_impl(blob.code(), config, &instructions, true);
        assert_eq!(cycles_ff, cycles);
        if timeline_ff_s != expected_timeline_s {
            panic!("Timeline mismatch for fast-forward!\n\nExpected timeline:\n{expected_timeline_s}\nActual timeline:\n{timeline_ff_s}");
        }
    }

    #[test]
    fn test_parallel_simple() {
        assert_timeline(
            test_config(),
            "
                a0 = a1 + a2
                a1 = a1 + a2
                trap
            ",
            "
                DeER.  a0 = a1 + a2
                DeER.  a1 = a1 + a2
                DeeER  trap
            ",
        );
    }

    #[test]
    fn test_sequential_simple() {
        assert_timeline(
            test_config(),
            "
                a0 = a1 + a2
                a1 = a0 + a2
                trap
            ",
            "
                DeER..  a0 = a1 + a2
                D=eER.  a1 = a0 + a2
                .DeeER  trap
            ",
        );
    }

    #[test]
    fn test_sequential_decode_limits() {
        assert_timeline(
            test_config(),
            "
                a0 = 0x12345678aabbccdd
                a1 = 0x12345678aabbccdd
                a2 = 0x12345678aabbccdd
                a3 = 0x12345678aabbccdd
                trap
            ",
            "
                DeER...  a0 = 0x12345678aabbccdd
                DeER...  a1 = 0x12345678aabbccdd
                .DeER..  a2 = 0x12345678aabbccdd
                .DeER..  a3 = 0x12345678aabbccdd
                ..DeeER  trap
            ",
        );
    }

    #[test]
    fn test_resource_limits_mul() {
        assert_timeline(
            test_config(),
            "
                a0 = a1 * a2
                a1 = a3 * a4
                trap
            ",
            "
                DeeeER...  a0 = a1 * a2
                D===eeeER  a1 = a3 * a4
                .DeeE---R  trap
            ",
        );
    }

    #[test]
    fn test_mul_with_dep() {
        assert_timeline(
            test_config(),
            "
                a0 = a1 + a2
                a4 = a0 * a3
                trap
            ",
            "
                DeER...  a0 = a1 + a2
                D=eeeER  a4 = a0 * a3
                .DeeE-R  trap
            ",
        );
    }

    #[test]
    fn test_register_move() {
        assert_timeline(
            test_config(),
            "
                s0 = 1
                a0 = s0
                a1 = a0 + 1
                trap
            ",
            "
                DeER..  s0 = 0x1
                D.....  a0 = s0
                D=eER.  a1 = a0 + 0x1
                .DeeER  trap
            ",
        )
    }

    #[test]
    fn test_memory_accesses() {
        assert_timeline(
            test_config(),
            "
                a0 = s1
                ra = u64 [sp + 0x30]
                s0 = u64 [sp + 0x28]
                s1 = u64 [sp + 0x20]
                sp = sp + 0x38
                ret
            ",
            "
                D............................  a0 = s1
                DeeeeER......................  ra = u64 [sp + 0x30]
                DeeeeER......................  s0 = u64 [sp + 0x28]
                DeeeeER......................  s1 = u64 [sp + 0x20]
                .DeE--R......................  sp = sp + 0x38
                .D===eeeeeeeeeeeeeeeeeeeeeeER  ret
            ",
        )
    }

    #[test]
    fn test_empty() {
        assert_timeline(
            test_config(),
            "
                fallthrough
            ",
            "
                DeeER  fallthrough
            ",
        );
    }

    #[test]
    fn test_overwrite_register() {
        assert_timeline(
            test_config(),
            "
                s0 = u64 [sp]
                s0 = a1 + a2
                s0 = u64 [s0]
                jump [s0]
            ",
            "
                DeeeeER.......................  s0 = u64 [sp]
                DeE---R.......................  s0 = a1 + a2
                D=eeeeER......................  s0 = u64 [s0]
                .D====eeeeeeeeeeeeeeeeeeeeeeER  jump [s0]
            ",
        );
    }

    #[test]
    fn test_load_and_jump() {
        assert_timeline(
            test_config(),
            "
                @0:
                a2 = u8 [a0 + 11]
                jump @0 if a2 == 0
            ",
            "
                DeeeeER.  a2 = u8 [a0 + 0xb]
                D====eER  jump 0 if a2 == 0
            ",
        );
    }

    #[test]
    fn test_complex() {
        assert_timeline(
            test_config(),
            "
                a2 = i16 [a0 + 0x6]
                a1 = a1 & 0x7
                a3 = 0x1
                a1 = a1 << 0x8
                a2 = a2 & 0xfffffffffffff8ff
                a1 = a1 | a2
                a2 = a1 + a3
                u8 [a0 + 0x2] = a3
                trap
            ",
            "
                DeeeeER.......................  a2 = i16 [a0 + 0x6]
                DeE---R.......................  a1 = a1 & 0x7
                DeE---R.......................  a3 = 0x1
                D=eE--R.......................  a1 = a1 << 0x8
                .D===eER......................  a2 = a2 & 0xfffffffffffff8ff
                .D====eER.....................  a1 = a1 | a2
                .D=====eER....................  a2 = a1 + a3
                ..DeeeeeeeeeeeeeeeeeeeeeeeeeER  u8 [a0 + 0x2] = a3
                ..DeeE-----------------------R  trap
            ",
        );
    }

    #[test]
    fn test_even_more_complex() {
        assert_timeline(
            test_config(),
            "
                @0:
                i32 a1 = clz a0
                i32 a0 = a0 << a1
                a1 = a1 << 0x17
                i32 a2 = a0 >> 0x8
                a3 = a0 >> 0x7
                a3 = a3 & ~a2
                i32 a2 = a2 - a1
                a0 = a0 << 0x18
                a3 = a3 & 0x1
                i32 a0 = a0 - a3
                i32 a0 = a0 >> 0x1f
                a1 = a2 + 0x4e800000
                i32 a0 = a0 + a1
                a1 = 0x46008c00
                ra = 0x24
                jump @0
            ",
            "
                DeER.....................  i32 a1 = clz a0
                D=eeER...................  i32 a0 = a0 << a1
                .DeE-R...................  a1 = a1 << 0x17
                .D==eeER.................  i32 a2 = a0 >> 0x8
                ..D=eE-R.................  a3 = a0 >> 0x7
                ...D==eeER...............  a3 = a3 & ~a2
                ....D=eeER...............  i32 a2 = a2 - a1
                ....DeE--R...............  a0 = a0 << 0x18
                ....D===eER..............  a3 = a3 & 0x1
                .....D===eeER............  i32 a0 = a0 - a3
                .....D=====eeER..........  i32 a0 = a0 >> 0x1f
                ......D=eE----R..........  a1 = a2 + 0x4e800000
                ......D======eeER........  i32 a0 = a0 + a1
                .......DeE------R........  a1 = 0x46008c00
                .......DeE------R........  ra = 0x24
                .......DeeeeeeeeeeeeeeeER  jump 0
            ",
        );
    }

    #[test]
    fn test_super_complex() {
        assert_timeline(
            test_config(),
            "
                @0:
                unlikely
                t1 = u8 [s0]
                a1 = u8 [s0 + 0x11]
                a2 = 0x172d0
                a3 = u8 [s0 + 0x16]
                t0 = sp + 0x58
                a1 = a1 << 0x3
                a1 = a1 + a2
                a2 = u8 [a1]
                a5 = u8 [a1 + 0x1]
                s1 = u8 [a1 + 0x2]
                a4 = u8 [a1 + 0x3]
                a3 = a3 + t0
                a5 = a5 << 0x8
                s1 = s1 << 0x10
                a4 = a4 << 0x18
                a2 = a2 | a5
                a5 = u8 [a1 + 0x4]
                a0 = u8 [a1 + 0x5]
                a4 = a4 | s1
                s1 = u8 [a1 + 0x6]
                a1 = u8 [a1 + 0x7]
                a0 = a0 << 0x8
                a0 = a0 | a5
                s1 = s1 << 0x10
                a1 = a1 << 0x18
                a1 = a1 | s1
                a2 = a2 | a4
                a0 = a0 | a1
                a1 = s0 - t1
                a0 = a0 << 0x20
                a0 = a0 | a2
                u64 [sp + 0x58] = a0
                a0 = u8 [a3]
                a1 = u8 [a1 + 0x4]
                a0 = a1 * a0
                a1 = u8 [s0 + 0x23]
                jump @0 if a1 != 0
            ",
            "
                DeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER............................  unlikely
                DeeeeE------------------------------------R............................  t1 = u8 [s0]
                DeeeeE------------------------------------R............................  a1 = u8 [s0 + 0x11]
                DeE---------------------------------------R............................  a2 = 0x172d0
                .DeeeeE-----------------------------------R............................  a3 = u8 [s0 + 0x16]
                .DeE--------------------------------------R............................  t0 = sp + 0x58
                .D===eE-----------------------------------R............................  a1 = a1 << 0x3
                ..D===eE----------------------------------R............................  a1 = a1 + a2
                ..D====eeeeE------------------------------R............................  a2 = u8 [a1]
                ..D====eeeeE------------------------------R............................  a5 = u8 [a1 + 0x1]
                ..D====eeeeE------------------------------R............................  s1 = u8 [a1 + 0x2]
                ...D===eeeeE------------------------------R............................  a4 = u8 [a1 + 0x3]
                ...D==eE----------------------------------R............................  a3 = a3 + t0
                ...D=======eE-----------------------------R............................  a5 = a5 << 0x8
                ...D=======eE-----------------------------R............................  s1 = s1 << 0x10
                ....D======eE-----------------------------R............................  a4 = a4 << 0x18
                ....D=======eE----------------------------R............................  a2 = a2 | a5
                ....D======eeeeE--------------------------R............................  a5 = u8 [a1 + 0x4]
                ....D=======eeeeE-------------------------R............................  a0 = u8 [a1 + 0x5]
                .....D======eE----------------------------R............................  a4 = a4 | s1
                .....D=======eeeeE------------------------R............................  s1 = u8 [a1 + 0x6]
                .....D=======eeeeE------------------------R............................  a1 = u8 [a1 + 0x7]
                .....D==========eE------------------------R............................  a0 = a0 << 0x8
                ......D==========eE-----------------------R............................  a0 = a0 | a5
                ......D==========eE-----------------------R............................  s1 = s1 << 0x10
                ......D==========eE-----------------------R............................  a1 = a1 << 0x18
                ......D===========eE----------------------R............................  a1 = a1 | s1
                .......D=======eE-------------------------R............................  a2 = a2 | a4
                .......D===========eE---------------------R............................  a0 = a0 | a1
                .......D========eE------------------------R............................  a1 = s0 - t1
                ........D===========eE--------------------R............................  a0 = a0 << 0x20
                ........D============eE-------------------R............................  a0 = a0 | a2
                ...........................................DeeeeeeeeeeeeeeeeeeeeeeeeeER  u64 [sp + 0x58] = a0
                ...........................................DeeeeE---------------------R  a0 = u8 [a3]
                ...........................................DeeeeE---------------------R  a1 = u8 [a1 + 0x4]
                ...........................................D====eeeE------------------R  a0 = a1 * a0
                ............................................DeeeeE--------------------R  a1 = u8 [s0 + 0x23]
                ............................................D====eE-------------------R  jump 0 if a1 != 0
            ",
        );
    }

    #[test]
    fn test_l3_loads() {
        assert_timeline(CacheModel::L3Hit,
            "
                a0 = u64 [a0]
                a0 = u64 [a0]
                a0 = u64 [a0]
                a0 = u64 [a0]
                ret
            ",
            "
                DeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER...............................................................................................................  a0 = u64 [a0]
                D=====================================eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER..........................................................................  a0 = u64 [a0]
                D==========================================================================eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER.....................................  a0 = u64 [a0]
                D===============================================================================================================eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER  a0 = u64 [a0]
                .DeeeeeeeeeeeeeeeeeeeeeeE-----------------------------------------------------------------------------------------------------------------------------R  ret
            ",
        )
    }

    #[test]
    fn test_ecalli() {
        assert_timeline(
            test_config(),
            "
                ecalli 27
                ret
            ",
            "
                DeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER  ecalli 27
                .DeeeeeeeeeeeeeeeeeeeeeeE-----------------------------------------------------------------------------R  ret
            ",
        );
    }
}
