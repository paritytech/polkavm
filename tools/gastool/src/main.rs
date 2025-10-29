#![allow(clippy::as_underscore)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::exit)]

use clap::Parser;
use core::sync::atomic::{AtomicBool, Ordering};
use memoffset::offset_of;
use polkavm::program::Instruction;
use polkavm::{
    CacheModel, Config, CostModel, CostModelKind, CustomCodegen, Engine, Error, InterruptKind, Module, ModuleConfig, ProgramBlob,
    ProgramCounter, ProgramParts, Reg,
};
use polkavm_assembler::amd64::addr::*;
use polkavm_assembler::amd64::inst::*;
use polkavm_assembler::amd64::Reg::rsp;
use polkavm_assembler::amd64::RegIndex::*;
use polkavm_assembler::amd64::{Condition, LoadKind, RegSize, Size};
use polkavm_assembler::{Assembler, Label};
use polkavm_common::program::InstructionFormat;
use polkavm_common::writer::ProgramBlobBuilder;
use polkavm_linux_raw as linux_raw;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

mod system;

use crate::system::{configure_perf_counter, restart_with_sudo_or_exit, SystemSetup};

const PCG_MULTIPLIER: u64 = 6364136223846793005;

fn pcg_init(out: &mut Vec<Instruction>, state_reg: Reg, seed: u128) -> u64 {
    let state = seed as u64;
    let increment = (((seed >> 64) as u64) << 1) | 1;
    let state = state.wrapping_add(increment);
    let state = state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(increment);
    out.push(Instruction::load_imm64(state_reg.into(), state));
    increment
}

fn pcg_rand_u32(out: &mut Vec<Instruction>, increment: u64, state_reg: Reg, tmp_reg: Reg, output_reg: Reg) {
    use Instruction::*;
    let state_reg = state_reg.into();
    let tmp_reg = tmp_reg.into();
    let output_reg = output_reg.into();

    // let xsh = (((state >> 18) ^ state) >> 27) as u32;
    out.push(shift_logical_right_imm_64(output_reg, state_reg, 18));
    out.push(xor(output_reg, output_reg, state_reg));
    out.push(shift_logical_right_imm_64(output_reg, output_reg, 27));
    out.push(shift_logical_left_imm_64(output_reg, output_reg, 32));
    out.push(shift_logical_right_imm_64(output_reg, output_reg, 32));

    // let rot = (state >> 59) as u32;
    out.push(shift_logical_right_imm_64(tmp_reg, state_reg, 59));

    // xsh.rotate_right(rot)
    out.push(rotate_right_32(output_reg, output_reg, tmp_reg));
    out.push(shift_logical_left_imm_64(output_reg, output_reg, 32));
    out.push(shift_logical_right_imm_64(output_reg, output_reg, 32));

    // Progress to next state.
    out.push(load_imm64(tmp_reg, PCG_MULTIPLIER));
    out.push(mul_64(state_reg, state_reg, tmp_reg));
    out.push(load_imm64(tmp_reg, increment));
    out.push(add_64(state_reg, state_reg, tmp_reg));
}

fn host_rng() -> oorandom::Rand32 {
    oorandom::Rand32::new_inc(RNG_SEED as u64, (RNG_SEED >> 64) as u64)
}

#[test]
fn test_pcg() {
    fn run_instruction(regs: &mut [u64; 13], instruction: Instruction) {
        match instruction {
            Instruction::load_imm64(reg, value) => regs[reg.get() as usize] = value,
            Instruction::mul_64(dst, src1, src2) => {
                regs[dst.get() as usize] = regs[src1.get() as usize].wrapping_mul(regs[src2.get() as usize])
            }
            Instruction::add_64(dst, src1, src2) => {
                regs[dst.get() as usize] = regs[src1.get() as usize].wrapping_add(regs[src2.get() as usize])
            }
            Instruction::xor(dst, src1, src2) => regs[dst.get() as usize] = regs[src1.get() as usize] ^ regs[src2.get() as usize],
            Instruction::shift_logical_right_imm_64(dst, src1, src2) => {
                regs[dst.get() as usize] = regs[src1.get() as usize].wrapping_shr(src2)
            }
            Instruction::shift_logical_left_imm_64(dst, src1, src2) => {
                regs[dst.get() as usize] = regs[src1.get() as usize].wrapping_shl(src2)
            }
            Instruction::rotate_right_32(dst, src1, src2) => {
                regs[dst.get() as usize] =
                    (i64::from((regs[src1.get() as usize] as u32).rotate_right(regs[src2.get() as usize] as u32) as i32)) as u64
            }
            _ => unimplemented!("unimplemented instruction: {instruction:?}"),
        }
    }

    use polkavm_common::program::Reg::*;

    let mut rng = host_rng();
    let mut regs = [0; 13];

    let mut instructions = Vec::new();
    let pcg_increment = pcg_init(&mut instructions, S0, RNG_SEED);
    for &instruction in &instructions {
        run_instruction(&mut regs, instruction);
    }
    instructions.clear();
    assert_eq!(pcg_increment, rng.state().1);
    assert_eq!(regs[S0 as usize], rng.state().0);

    pcg_rand_u32(&mut instructions, pcg_increment, S0, T0, A0);

    for n in 0..100 {
        let expected = rng.rand_u32();
        for &instruction in &instructions {
            run_instruction(&mut regs, instruction);
        }

        let actual = regs[A0 as usize] as u32;
        assert_eq!(actual, expected, "iteration failed: {n}");
    }
}

static RUNNING: AtomicBool = AtomicBool::new(true);

const MAX_CORE_PMC_COUNT: usize = 6;
const MAX_L3_PMC_COUNT: usize = 2;

#[derive(Copy, Clone)]
#[repr(C)]
struct Sample {
    cycles_start: u64,
    core_pmc_start: [u64; MAX_CORE_PMC_COUNT],
    l3_pmc_start: [u64; MAX_L3_PMC_COUNT],
    cycles_end: u64,
    core_pmc_end: [u64; MAX_CORE_PMC_COUNT],
    l3_pmc_end: [u64; MAX_L3_PMC_COUNT],
}

impl Sample {
    fn elapsed(&self) -> u64 {
        self.cycles_end.wrapping_sub(self.cycles_start)
    }

    fn get_pmc(&self, pmu: Pmu, index: usize) -> u64 {
        match pmu {
            Pmu::Core => self.core_pmc_end[index].wrapping_sub(self.core_pmc_start[index]),
            Pmu::L3 => self.l3_pmc_end[index].wrapping_sub(self.l3_pmc_start[index]),
        }
    }
}

const ECALLI_BENCHMARK_PROLOGUE: u32 = 0;
const ECALLI_BENCHMARK_EPILOGUE: u32 = 1;
const ECALLI_BENCHMARK_ON_FINISH: u32 = 2;

fn syscall1(asm: &mut Assembler, nr: impl Into<u64>, arg0: impl Into<u64>) {
    asm.push(push(rax));
    asm.push(push(rdi));
    asm.push(push(rcx));
    asm.push(push(r11));
    asm.push(mov_imm64(rax, nr));
    asm.push(mov_imm64(rdi, arg0));
    asm.push(syscall());
    let label_ok = asm.forward_declare_label();
    asm.push(test((RegSize::R64, rax, rax)));
    asm.push(jcc_label8(Condition::Equal, label_ok));
    asm.push(ud2());
    asm.define_label(label_ok);
    asm.push(pop(r11));
    asm.push(pop(rcx));
    asm.push(pop(rdi));
    asm.push(pop(rax));
}

fn syscall4(
    asm: &mut Assembler,
    nr: impl Into<u64>,
    arg0: impl Into<u64>,
    arg1: impl Into<u64>,
    arg2: impl Into<u64>,
    arg3: impl Into<u64>,
) {
    asm.push(push(rax));
    asm.push(push(rdi));
    asm.push(push(rsi));
    asm.push(push(rdx));
    asm.push(push(r10));
    asm.push(push(rcx));
    asm.push(push(r11));
    asm.push(mov_imm64(rax, nr));
    asm.push(mov_imm64(rdi, arg0));
    asm.push(mov_imm64(rsi, arg1));
    asm.push(mov_imm64(rdx, arg2));
    asm.push(mov_imm64(r10, arg3));
    asm.push(syscall());
    let label_ok = asm.forward_declare_label();
    asm.push(test((RegSize::R64, rax, rax)));
    asm.push(jcc_label8(Condition::Equal, label_ok));
    asm.push(ud2());
    asm.define_label(label_ok);
    asm.push(pop(r11));
    asm.push(pop(rcx));
    asm.push(pop(r10));
    asm.push(pop(rdx));
    asm.push(pop(rsi));
    asm.push(pop(rdi));
    asm.push(pop(rax));
}

fn syscall6(
    asm: &mut Assembler,
    nr: impl Into<u64>,
    arg0: impl Into<u64>,
    arg1: impl Into<u64>,
    arg2: impl Into<u64>,
    arg3: impl Into<u64>,
    arg4: impl Into<u64>,
    arg5: impl Into<u64>,
) {
    asm.push(push(rax));
    asm.push(push(rdi));
    asm.push(push(rsi));
    asm.push(push(rdx));
    asm.push(push(r10));
    asm.push(push(r8));
    asm.push(push(r9));
    asm.push(push(rcx));
    asm.push(push(r11));
    asm.push(mov_imm64(rax, nr));
    asm.push(mov_imm64(rdi, arg0));
    asm.push(mov_imm64(rsi, arg1));
    asm.push(mov_imm64(rdx, arg2));
    asm.push(mov_imm64(r10, arg3));
    asm.push(mov_imm64(r8, arg4));
    asm.push(mov_imm64(r9, arg5));
    asm.push(syscall());
    let label_ok = asm.forward_declare_label();
    asm.push(test((RegSize::R64, rax, rax)));
    asm.push(jcc_label8(Condition::NotSign, label_ok));
    asm.push(ud2());
    asm.define_label(label_ok);
    asm.push(pop(r11));
    asm.push(pop(rcx));
    asm.push(pop(r9));
    asm.push(pop(r8));
    asm.push(pop(r10));
    asm.push(pop(rdx));
    asm.push(pop(rsi));
    asm.push(pop(rdi));
    asm.push(pop(rax));
}

const STACK_ADDRESS_HI: u32 = 0xfffe0000;

struct CodegenState {
    label_start: Label,
    label_done: Label,
    label_inner: Label,
}

struct BenchCodegen {
    state: Mutex<Option<CodegenState>>,
    samples: usize,
    repeat_execution: u32,
    allocate_hugepages: bool,
    lock_memory: bool,
    custom_codegen: HashMap<u32, Box<dyn Fn(&mut Assembler) + Send + Sync>>,
}

#[repr(C)]
struct BenchState {
    rax: u64,
    rcx: u64,
    rdx: u64,
    rbx: u64,
    rsp: u64,
    rbp: u64,
    rsi: u64,
    rdi: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,

    restart_count: u64,
    warmup_finished: u64,

    rseq: linux_raw::rseq,
    rseq_cs: linux_raw::rseq_cs,
    pointer: *mut Sample,
}

const RSEQ_SIGNATURE: u32 = 0x12121212;

#[allow(clippy::unused_self)]
impl BenchCodegen {
    fn new(samples: usize, repeat_execution: u32, allocate_hugepages: bool, lock_memory: bool) -> Self {
        assert!(repeat_execution > 0);
        BenchCodegen {
            state: Mutex::new(None),
            samples,
            repeat_execution,
            allocate_hugepages,
            lock_memory,
            custom_codegen: HashMap::new(),
        }
    }

    fn stack_space_used(&self) -> u32 {
        (core::mem::size_of::<BenchState>() + self.samples * core::mem::size_of::<Sample>()) as u32
    }

    fn state_origin(&self) -> u32 {
        STACK_ADDRESS_HI - self.stack_space_used()
    }

    fn rseq_address(&self) -> u32 {
        self.state_origin() + offset_of!(BenchState, rseq) as u32
    }

    fn rseq_cs_address(&self) -> u32 {
        self.state_origin() + offset_of!(BenchState, rseq_cs) as u32
    }

    fn pointer_address(&self) -> u32 {
        self.state_origin() + offset_of!(BenchState, pointer) as u32
    }

    fn samples_address(&self) -> u32 {
        self.state_origin() + core::mem::size_of::<BenchState>() as u32
    }

    fn samples_address_end(&self) -> u32 {
        self.samples_address() + self.samples as u32 * core::mem::size_of::<Sample>() as u32
    }

    #[rustfmt::skip]
    fn emit_rseq_handler(&self, asm: &mut Assembler, label_start: Label) -> Label {
        // Restartable sequences handler. We will jump here if the kernel interrupts our benchmark.
        asm.push_raw(&RSEQ_SIGNATURE.to_le_bytes());
        let label_abort = asm.create_label();

        // Restore registers.
        asm.push(mov_imm64(rsi, self.state_origin()));
        asm.push(inc(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, restart_count) as i32)));
        asm.push(mov_imm(reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, warmup_finished) as i32), imm64(0)));
        asm.push(load(LoadKind::U64, rax, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rax) as i32)));
        asm.push(load(LoadKind::U64, rcx, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rcx) as i32)));
        asm.push(load(LoadKind::U64, rdx, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rdx) as i32)));
        asm.push(load(LoadKind::U64, rbx, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rbx) as i32)));
        asm.push(load(LoadKind::U64, rsp, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rsp) as i32)));
        asm.push(load(LoadKind::U64, rbp, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rbp) as i32)));
        asm.push(load(LoadKind::U64, r8, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r8) as i32)));
        asm.push(load(LoadKind::U64, r9, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r9) as i32)));
        asm.push(load(LoadKind::U64, r10, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r10) as i32)));
        asm.push(load(LoadKind::U64, r11, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r11) as i32)));
        asm.push(load(LoadKind::U64, r12, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r12) as i32)));
        asm.push(load(LoadKind::U64, r13, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r13) as i32)));
        asm.push(load(LoadKind::U64, r14, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r14) as i32)));
        asm.push(load(LoadKind::U64, r15, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r15) as i32)));

        asm.push(load(LoadKind::U64, rsi, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rsi) as i32)));
        asm.push(jmp_label32(label_start));

        label_abort
    }

    #[rustfmt::skip]
    fn emit_init(&self, asm: &mut Assembler, label_start: Label, label_done: Label, label_abort: Label) {
        // Save registers for a potential restartable sequences restart.
        asm.push(push(rsi));
        asm.push(mov_imm64(rsi, self.state_origin()));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rax) as i32), rax));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rcx) as i32), rcx));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rdx) as i32), rdx));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rbx) as i32), rbx));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rbp) as i32), rbp));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rdi) as i32), rdi));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r8) as i32), r8));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r9) as i32), r9));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r10) as i32), r10));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r11) as i32), r11));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r12) as i32), r12));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r13) as i32), r13));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r14) as i32), r14));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, r15) as i32), r15));

        asm.push(push(rax));
        asm.push(mov(RegSize::R64, rax, rsp));
        asm.push(sub((rax, imm64(16))));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rsp) as i32), rax));

        asm.push(load(LoadKind::U64, rax, reg_indirect(RegSize::R64, rsp + 8)));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(BenchState, rsi) as i32), rax));

        asm.push(pop(rax));
        asm.push(pop(rsi));

        if self.allocate_hugepages {
            syscall6(
                asm,
                linux_raw::SYS_mmap,
                0_u64,
                1024 * 1024 * 1024_u64,
                linux_raw::PROT_READ | linux_raw::PROT_WRITE,
                linux_raw::MAP_FIXED | linux_raw::MAP_ANONYMOUS | linux_raw::MAP_PRIVATE | linux_raw::MAP_HUGETLB | linux_raw::MAP_HUGE_1GB,
                u64::MAX,
                0_u64,
            );
        }

        // This will allocate and page in all of the memory.
        if self.lock_memory {
            syscall1(asm, linux_raw::SYS_mlockall, linux_raw::MCL_CURRENT | linux_raw::MCL_FUTURE);
        }

        // Set up restartable sequences.
        asm.push(push(rax));
        asm.push(push(rbx));
        asm.push(push(rsi));
        asm.push(mov_imm64(rsi, self.rseq_cs_address()));
        asm.push(lea_rip_label(rax, label_start));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(linux_raw::rseq_cs, start_ip) as i32), rax));
        asm.push(lea_rip_label(rbx, label_done));
        asm.push(sub((RegSize::R64, rbx, rax)));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(linux_raw::rseq_cs, post_commit_offset) as i32), rbx));
        asm.push(lea_rip_label(rax, label_abort));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset_of!(linux_raw::rseq_cs, abort_ip) as i32), rax));
        asm.push(mov_imm64(rax, self.rseq_address()));
        asm.push(store(Size::U64, reg_indirect(RegSize::R64, rax + offset_of!(linux_raw::rseq, rseq_cs) as i32), rsi));
        asm.push(pop(rsi));
        asm.push(pop(rbx));
        asm.push(pop(rax));

        assert_eq!(self.rseq_cs_address() % 32, 0);
        assert_eq!(self.rseq_address() % 32, 0);
        syscall4(asm, linux_raw::SYS_rseq, self.rseq_address(), core::mem::size_of::<linux_raw::rseq>() as u64, 0_u64, RSEQ_SIGNATURE);
    }
}

impl CustomCodegen for BenchCodegen {
    fn should_emit_ecalli(&self, number: u32, asm: &mut Assembler) -> bool {
        match number {
            ECALLI_BENCHMARK_PROLOGUE => {
                let label_init = asm.forward_declare_label();
                let label_start = asm.forward_declare_label();
                let label_done = asm.forward_declare_label();

                asm.push(jmp_label32(label_init));

                let label_abort = self.emit_rseq_handler(asm, label_start);

                asm.define_label(label_init);
                self.emit_init(asm, label_start, label_done, label_abort);
                asm.define_label(label_start);

                // Preserve clobbered registers.
                asm.push(push(rax));
                asm.push(push(rbx));
                asm.push(push(rcx));
                asm.push(push(rdx));
                asm.push(push(rsi));

                // Start the critical section.
                asm.push(mov_imm64(rsi, self.rseq_cs_address()));
                asm.push(mov_imm64(rax, self.rseq_address()));
                asm.push(store(
                    Size::U64,
                    reg_indirect(RegSize::R64, rax + offset_of!(linux_raw::rseq, rseq_cs) as i32),
                    rsi,
                ));

                // Read pointer to the current sample.
                asm.push(mov_imm64(rsi, self.pointer_address()));
                asm.push(load(LoadKind::U64, rsi, reg_indirect(RegSize::R64, rsi)));

                // Serialize.
                asm.push(xor((RegSize::R32, rax, rax)));
                asm.push(cpuid());

                // Fetch performance counters.
                for counter in 0..MAX_CORE_PMC_COUNT {
                    let offset = offset_of!(Sample, core_pmc_start) as i32 + counter as i32 * 8;
                    asm.push(mov_imm(rcx, imm32(counter as u32)));
                    asm.push(rdpmc());
                    asm.push(shl_imm(RegSize::R64, rdx, 32));
                    asm.push(or((RegSize::R64, rdx, rax)));
                    asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset), rdx));
                }

                for counter in 0..MAX_L3_PMC_COUNT {
                    let offset = offset_of!(Sample, l3_pmc_start) as i32 + counter as i32 * 8;
                    asm.push(mov_imm(rcx, imm32(10 + counter as u32)));
                    asm.push(rdpmc());
                    asm.push(shl_imm(RegSize::R64, rdx, 32));
                    asm.push(or((RegSize::R64, rdx, rax)));
                    asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi + offset), rdx));
                }

                // Fetch the current cycle counter.
                asm.push(rdtscp());
                asm.push(shl_imm(RegSize::R64, rdx, 32));
                asm.push(or((RegSize::R64, rdx, rax)));
                assert_eq!(offset_of!(Sample, cycles_start), 0);
                asm.push(store(Size::U64, reg_indirect(RegSize::R64, rsi), rdx));

                // Serialize again.
                asm.push(xor((RegSize::R32, rax, rax)));
                asm.push(cpuid());

                // Restore clobbered registers.
                asm.push(pop(rsi));
                asm.push(pop(rdx));
                asm.push(pop(rcx));
                asm.push(pop(rbx));
                asm.push(pop(rax));

                if self.repeat_execution > 1 {
                    asm.push(mov_imm(r12, imm32(self.repeat_execution)));
                }

                let label_inner = asm.create_label();

                assert!(self
                    .state
                    .lock()
                    .unwrap()
                    .replace(CodegenState {
                        label_start,
                        label_done,
                        label_inner,
                    })
                    .is_none());

                false
            }
            ECALLI_BENCHMARK_EPILOGUE => {
                let state = self.state.lock().unwrap().take().unwrap();
                if self.repeat_execution > 1 {
                    asm.push(dec(RegSize::R64, r12));
                    asm.push(test((RegSize::R64, r12, r12)));
                    asm.push(jcc_label32(Condition::NotEqual, state.label_inner));
                }

                // Preserve clobbered registers.
                asm.push(push(rax));
                asm.push(push(rbx));
                asm.push(push(rcx));
                asm.push(push(rdx));

                // Serialize.
                asm.push(xor((RegSize::R32, rax, rax)));
                asm.push(cpuid());

                // Fetch the current cycle counter.
                asm.push(rdtscp());
                asm.push(shl_imm(RegSize::R64, rdx, 32));
                asm.push(or((RegSize::R64, rdx, rax)));
                asm.push(push(rdx));

                // Fetch performance counters.
                for counter in 0..MAX_CORE_PMC_COUNT {
                    asm.push(mov_imm(rcx, imm32(counter as u32)));
                    asm.push(rdpmc());
                    asm.push(shl_imm(RegSize::R64, rdx, 32));
                    asm.push(or((RegSize::R64, rdx, rax)));
                    asm.push(push(rdx));
                }

                for counter in 0..MAX_L3_PMC_COUNT {
                    asm.push(mov_imm(rcx, imm32(10 + counter as u32)));
                    asm.push(rdpmc());
                    asm.push(shl_imm(RegSize::R64, rdx, 32));
                    asm.push(or((RegSize::R64, rdx, rax)));
                    asm.push(push(rdx));
                }

                // Save the cycle counter in memory.
                asm.push(mov_imm64(rax, self.pointer_address()));
                asm.push(load(LoadKind::U64, rcx, reg_indirect(RegSize::R64, rax)));

                for counter in (0..MAX_L3_PMC_COUNT).rev() {
                    let offset = offset_of!(Sample, l3_pmc_end) as i32 + counter as i32 * 8;
                    asm.push(pop(rdx));
                    asm.push(store(Size::U64, reg_indirect(RegSize::R64, rcx + offset), rdx));
                }

                for counter in (0..MAX_CORE_PMC_COUNT).rev() {
                    let offset = offset_of!(Sample, core_pmc_end) as i32 + counter as i32 * 8;
                    asm.push(pop(rdx));
                    asm.push(store(Size::U64, reg_indirect(RegSize::R64, rcx + offset), rdx));
                }

                asm.push(pop(rdx));
                asm.push(store(
                    Size::U64,
                    reg_indirect(RegSize::R64, rcx + offset_of!(Sample, cycles_end) as i32),
                    rdx,
                ));

                // See whether this was a warmup run.
                let label_restart = asm.forward_declare_label();
                asm.push(mov_imm64(rdx, self.state_origin() + offset_of!(BenchState, warmup_finished) as u32));
                asm.push(load(LoadKind::U64, rbx, reg_indirect(RegSize::R64, rdx)));
                asm.push(mov_imm(reg_indirect(RegSize::R64, rdx), imm64(1)));
                asm.push(test((RegSize::R64, rbx, rbx)));
                asm.push(jcc_label32(Condition::Equal, label_restart));

                // Increment the pointer and stash it back.
                asm.push(add((rcx, imm64(core::mem::size_of::<Sample>() as i32))));
                asm.push(store(Size::U64, reg_indirect(RegSize::R64, rax), rcx));

                // See if we're done.
                asm.push(mov_imm64(rax, self.samples_address_end()));
                asm.push(cmp((RegSize::R64, rcx, rax)));
                asm.push(jcc_label32(Condition::Equal, state.label_done));

                // We're not done.
                asm.define_label(label_restart);
                asm.push(pop(rdx));
                asm.push(pop(rcx));
                asm.push(pop(rbx));
                asm.push(pop(rax));
                asm.push(jmp_label32(state.label_start));

                // We are done.
                asm.define_label(state.label_done);
                asm.push(pop(rdx));
                asm.push(pop(rcx));
                asm.push(pop(rbx));
                asm.push(pop(rax));

                syscall4(
                    asm,
                    linux_raw::SYS_rseq,
                    self.rseq_address(),
                    core::mem::size_of::<linux_raw::rseq>() as u64,
                    linux_raw::RSEQ_FLAG_UNREGISTER,
                    RSEQ_SIGNATURE,
                );

                false
            }
            _ => {
                if let Some(handler) = self.custom_codegen.get(&number) {
                    handler(asm);
                    false
                } else {
                    true
                }
            }
        }
    }
}

struct Context {
    worker_pid: u32,
    cpu: usize,
    engine: Engine,
    setup: SystemSetup,
    frequency: u64,
    l1_cache: raw_cpuid::CacheParameter,
    l2_cache: raw_cpuid::CacheParameter,
    l3_cache: raw_cpuid::CacheParameter,
    default_repeat_outer: usize,
}

#[derive(Clone)]
struct Stats {
    all: Vec<u64>,
    min: u64,
    max: u64,
    avg: u64,
    med: u64,
    p10: u64,
    p90: u64,
}

impl core::fmt::Display for Stats {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            fmt,
            "med={:>7} avg={:>7} min={:>7} max={:>7} max-min={:>7} [{:>7}, {:>7}, {:>7}, ..., {:>7}]",
            self.med,
            self.avg,
            self.min,
            self.max,
            self.max - self.min,
            self.all[0],
            self.all[1],
            self.all[2],
            self.all[self.all.len() - 1]
        )
    }
}

fn stats(iter: impl IntoIterator<Item = u64>) -> Stats {
    let mut iter = iter.into_iter();
    let mut min = iter.next().unwrap();
    let mut max = min;
    let mut all = vec![min];
    let mut sum = min;

    for value in iter {
        min = min.min(value);
        max = max.max(value);
        all.push(value);
        sum += value;
    }

    let mut sorted = all.clone();
    sorted.sort_unstable();

    Stats {
        all,
        min,
        max,
        med: sorted[sorted.len() / 2],
        avg: sum / sorted.len() as u64,
        p10: sorted[(sorted.len() as f32 / 100.0 * 10.0) as usize],
        p90: sorted[(sorted.len() as f32 / 100.0 * 90.0) as usize],
    }
}

#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CounterKind {
    BranchesRetired,
    BranchesRetiredMisprediction,
    L1DataAccess,
    L1InstructionMiss,
    L2InstructionMiss,
    L1DataMiss,
    L2DataMissFromL1Miss,
    L2DataMissFromHWPF1,
    L2DataMissFromHWPF2,
    Interrupts,
    MicroOpsDispatched,
    MicroOpsRetired,
    InstructionCacheStall,
    InstructionCacheStallUpstreamNoFetchAddress,
    InstructionCacheStallDownstreamQueueFull,
    L2ITLBHit,
    L2ITLBMiss,
    ICacheLinesFromMemory,
    L3CacheHit,
    L3CacheMiss,
}

enum Pmu {
    Core,
    L3,
}

impl CounterKind {
    fn pmu(self) -> Pmu {
        match self {
            CounterKind::BranchesRetired
            | CounterKind::BranchesRetiredMisprediction
            | CounterKind::L1DataAccess
            | CounterKind::L1InstructionMiss
            | CounterKind::L2InstructionMiss
            | CounterKind::L1DataMiss
            | CounterKind::L2DataMissFromL1Miss
            | CounterKind::L2DataMissFromHWPF1
            | CounterKind::L2DataMissFromHWPF2
            | CounterKind::Interrupts
            | CounterKind::MicroOpsDispatched
            | CounterKind::MicroOpsRetired
            | CounterKind::InstructionCacheStall
            | CounterKind::InstructionCacheStallUpstreamNoFetchAddress
            | CounterKind::InstructionCacheStallDownstreamQueueFull
            | CounterKind::L2ITLBHit
            | CounterKind::L2ITLBMiss
            | CounterKind::ICacheLinesFromMemory => Pmu::Core,
            CounterKind::L3CacheHit | CounterKind::L3CacheMiss => Pmu::L3,
        }
    }

    fn get(self) -> u64 {
        // Sources:
        //   https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/programmer-references/58550-0.01.pdf
        //   https://www.amd.com/content/dam/amd/en/documents/processor-tech-docs/programmer-references/56255_OSRR.pdf
        // Also see 'tools/perf/pmu-events/arch/x86' in Linux kernel sources.
        match self {
            CounterKind::BranchesRetired => 0x4300c2,
            CounterKind::BranchesRetiredMisprediction => 0x4300c3,
            CounterKind::L1DataAccess => 0x430729,
            CounterKind::L1InstructionMiss => 0x431060,
            CounterKind::L2InstructionMiss => 0x430164,
            CounterKind::L1DataMiss => 0x43e060,
            CounterKind::L2DataMissFromL1Miss => 0x430864,
            CounterKind::L2DataMissFromHWPF1 => 0x431f71,
            CounterKind::L2DataMissFromHWPF2 => 0x431f72,
            CounterKind::Interrupts => 0x43002c,
            CounterKind::MicroOpsDispatched => 0x4303aa,
            CounterKind::MicroOpsRetired => 0x4300c1,
            CounterKind::InstructionCacheStall => 0x430487,
            CounterKind::InstructionCacheStallUpstreamNoFetchAddress => 0x430287,
            CounterKind::InstructionCacheStallDownstreamQueueFull => 0x430187,
            CounterKind::L2ITLBHit => 0x430084,
            CounterKind::L2ITLBMiss => 0x43ff85,
            CounterKind::ICacheLinesFromMemory => 0x430083,
            CounterKind::L3CacheHit => 0x0300c0000040ff04,
            CounterKind::L3CacheMiss => 0x0300c00000400104,
        }
    }

    fn name(self) -> &'static str {
        match self {
            CounterKind::BranchesRetired => "Branch",
            CounterKind::BranchesRetiredMisprediction => "Branch misp",
            CounterKind::L1DataAccess => "L1 data access",
            CounterKind::L1InstructionMiss => "L1 inst miss",
            CounterKind::L2InstructionMiss => "L2 inst miss",
            CounterKind::L1DataMiss => "L1 data miss",
            CounterKind::L2DataMissFromL1Miss => "L2 data miss (from L1 miss)",
            CounterKind::L2DataMissFromHWPF1 => "L2 data miss (from HWPF1)",
            CounterKind::L2DataMissFromHWPF2 => "L2 data miss (from HWPF2)",
            CounterKind::Interrupts => "Interrupts",
            CounterKind::MicroOpsDispatched => "Micro ops (disp)",
            CounterKind::MicroOpsRetired => "Micro ops (ret)",
            CounterKind::InstructionCacheStall => "Inst stall (all)",
            CounterKind::InstructionCacheStallUpstreamNoFetchAddress => "Inst stall (DQ)",
            CounterKind::InstructionCacheStallDownstreamQueueFull => "Inst stall (BP)",
            CounterKind::L2ITLBHit => "L1 ITLB miss, L2 hit",
            CounterKind::L2ITLBMiss => "L1 ITLB miss, L2 miss",
            CounterKind::ICacheLinesFromMemory => "icacheline from mem",
            CounterKind::L3CacheHit => "L3 cache hit",
            CounterKind::L3CacheMiss => "L3 cache miss",
        }
    }
}

struct InstructionBuffer<'a> {
    code: &'a mut Vec<Instruction>,
}

impl<'a> core::ops::Deref for InstructionBuffer<'a> {
    type Target = Vec<Instruction>;
    fn deref(&self) -> &Self::Target {
        &*self.code
    }
}

impl<'a> core::ops::DerefMut for InstructionBuffer<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.code
    }
}

fn get_frequency(cpu: usize) -> Result<u64, Error> {
    let frequency = crate::system::read_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_cur_freq"))?
        .trim()
        .parse::<u64>()
        .unwrap()
        * 1000;

    log::info!("CPU frequency: {:.02} GHz ({frequency})", frequency as f64 / 1_000_000_000_f64);
    Ok(frequency)
}

impl Context {
    fn new(isolation_args: IsolationArgs) -> Result<Self, Error> {
        let cpu = isolation_args.cpu;
        let reserved_cpus = isolation_args.reserve_cpus;
        let minimize_noise = isolation_args.minimize_noise;

        restart_with_sudo_or_exit();

        let mut config = Config::from_env()?;
        config.set_backend(Some(polkavm::BackendKind::Compiler));
        config.set_sandbox(Some(polkavm::SandboxKind::Linux));
        config.set_worker_count(1);
        config.set_cache_enabled(false);
        config.set_allow_experimental(true);
        config.set_sandboxing_enabled(false);
        config.set_allow_dynamic_paging(true);
        let engine = Engine::new(&config)?;
        let worker_pids = engine.idle_worker_pids();
        assert_eq!(worker_pids.len(), 1);
        let worker_pid = worker_pids[0];

        let mut setup = SystemSetup::initialize(&reserved_cpus, minimize_noise)?;
        setup.add_process(worker_pid as i32)?;
        log::info!("System setup done!");

        let cpuid = raw_cpuid::CpuId::new();
        log::info!("CPU: {}", cpuid.get_processor_brand_string().unwrap().as_str());

        let frequency = get_frequency(cpu)?;

        let mut l1_cache = None;
        let mut l2_cache = None;
        let mut l3_cache = None;

        let cache_info = cpuid.get_cache_parameters().unwrap();
        for info in cache_info {
            let size = info.sets() * (info.coherency_line_size() * info.associativity());
            log::info!(
                "CPU cache: L{}{} {}kB {}-way associative with {} sets and {}b cache line",
                info.level(),
                match info.cache_type() {
                    raw_cpuid::CacheType::Data => "d",
                    raw_cpuid::CacheType::Instruction => "i",
                    raw_cpuid::CacheType::Unified => "",
                    _ => panic!("encountered unknown cache type: {info:?}"),
                },
                size / 1024,
                info.associativity(),
                info.sets(),
                info.coherency_line_size()
            );

            if info.level() == 1 && info.cache_type() == raw_cpuid::CacheType::Data {
                assert!(l1_cache.is_none(), "duplicate L1 cache detected");
                l1_cache = Some(info);
            } else if info.level() == 2 && info.cache_type() == raw_cpuid::CacheType::Unified {
                assert!(l2_cache.is_none(), "duplicate L2 cache detected");
                l2_cache = Some(info);
            } else if info.level() == 3 && info.cache_type() == raw_cpuid::CacheType::Unified {
                assert!(l3_cache.is_none(), "duplicate L3 cache detected");
                l3_cache = Some(info);
            }
        }

        let l1_cache = l1_cache.expect("no L1 cache detected");
        let l2_cache = l2_cache.expect("no L2 cache detected");
        let l3_cache = l3_cache.expect("no L2 cache detected");

        Ok(Self {
            worker_pid,
            cpu,
            engine,
            setup,
            frequency,
            l1_cache,
            l2_cache,
            l3_cache,
            default_repeat_outer: 1,
        })
    }

    fn benchmark<C>(&self, name: impl Into<String>, codegen: C) -> BenchmarkBuilder<C>
    where
        C: FnMut(InstructionBuffer),
    {
        BenchmarkBuilder {
            name: name.into(),
            engine: &self.engine,
            samples: 128,
            rw_data_size: 0,
            repeat_code: 1000000,
            repeat_execution_inner: 1,
            repeat_execution_outer: self.default_repeat_outer,
            allocate_hugepages: false,
            lock_memory: true,
            cost_divisor: 1,
            frequency: self.frequency,
            codegen,
            performance_counters: vec![
                CounterKind::MicroOpsDispatched,
                CounterKind::MicroOpsRetired,
                CounterKind::BranchesRetired,
                CounterKind::BranchesRetiredMisprediction,
                CounterKind::L1DataMiss,
                CounterKind::L3CacheMiss,
            ],
            worker_pid: self.worker_pid,
            cpu: self.cpu,
            init_code: Vec::new(),
            amd_l3_type: self.setup.amd_l3_type,
            custom_codegen: HashMap::new(),
            hostcall_handlers: HashMap::new(),
            setup_instance: None,
            on_finished: None,
            dynamic_paging: false,
            gas_metering: false,
            jump_table: Vec::new(),
        }
    }
}

struct BenchmarkBuilder<'a, C> {
    name: String,
    engine: &'a Engine,
    rw_data_size: u32,
    repeat_code: usize,
    repeat_execution_inner: u32,
    repeat_execution_outer: usize,
    allocate_hugepages: bool,
    lock_memory: bool,
    cost_divisor: u64,
    frequency: u64,
    samples: usize,
    codegen: C,
    performance_counters: Vec<CounterKind>,
    worker_pid: u32,
    cpu: usize,
    init_code: Vec<Instruction>,
    amd_l3_type: u32,
    custom_codegen: HashMap<u32, Box<dyn Fn(&mut Assembler) + Send + Sync>>,
    hostcall_handlers: HashMap<u32, Box<dyn FnMut(&mut polkavm::RawInstance)>>,
    setup_instance: Option<Box<dyn FnMut(&mut polkavm::RawInstance)>>,
    on_finished: Option<Box<dyn FnMut(&mut polkavm::RawInstance)>>,
    dynamic_paging: bool,
    gas_metering: bool,
    jump_table: Vec<u32>,
}

struct BenchmarkResult {
    counters: HashMap<CounterKind, Stats>,
    cycles: Stats,
    cost_per_operation: u32,
    total_inner_repetitions: u64,
}

impl BenchmarkResult {
    fn counter(&self, kind: CounterKind) -> Option<&Stats> {
        self.counters.get(&kind)
    }
}

impl<'a, C> BenchmarkBuilder<'a, C> {
    fn counters(mut self, list: impl IntoIterator<Item = CounterKind>) -> Self {
        self.performance_counters = list.into_iter().collect();
        self
    }

    fn rw_data_size(mut self, value: u32) -> Self {
        self.rw_data_size = value;
        self
    }

    fn repeat_code(mut self, value: usize) -> Self {
        self.repeat_code = value;
        self
    }

    #[allow(dead_code)]
    fn repeat_outer(mut self, value: usize) -> Self {
        self.repeat_execution_outer = value;
        self
    }

    fn repeat_inner(mut self, value: u32) -> Self {
        self.repeat_execution_inner = value;
        self
    }

    fn cost_divisor(mut self, value: u64) -> Self {
        self.cost_divisor = value;
        self
    }

    fn gas_metering(mut self, value: bool) -> Self {
        self.gas_metering = value;
        self
    }

    #[allow(dead_code)]
    fn dynamic_paging(mut self, value: bool) -> Self {
        self.dynamic_paging = value;
        self
    }

    fn allocate_hugepages(mut self, value: bool) -> Self {
        self.allocate_hugepages = value;
        self
    }

    #[allow(dead_code)]
    fn lock_memory(mut self, value: bool) -> Self {
        self.lock_memory = value;
        self
    }

    #[allow(dead_code)]
    fn hostcall_codegen<F>(mut self, number: u32, generator: F) -> Self
    where
        F: Fn(&mut Assembler) + Send + Sync + 'static,
    {
        assert!(!self.custom_codegen.contains_key(&number));
        self.custom_codegen.insert(number, Box::new(generator));
        self
    }

    fn hostcall_handler<F>(mut self, number: u32, handler: F) -> Self
    where
        F: FnMut(&mut polkavm::RawInstance) + 'static,
    {
        assert!(!self.hostcall_handlers.contains_key(&number));
        self.hostcall_handlers.insert(number, Box::new(handler));
        self
    }

    fn setup_instance<F>(mut self, initialize: F) -> Self
    where
        F: FnMut(&mut polkavm::RawInstance) + 'static,
    {
        assert!(self.setup_instance.is_none());
        self.setup_instance = Some(Box::new(initialize));
        self
    }

    fn on_finished<F>(mut self, initialize: F) -> Self
    where
        F: FnMut(&mut polkavm::RawInstance) + 'static,
    {
        assert!(self.on_finished.is_none());
        self.on_finished = Some(Box::new(initialize));
        self
    }

    fn setup_code<F>(mut self, mut codegen: F) -> Self
    where
        F: FnMut(InstructionBuffer),
    {
        codegen(InstructionBuffer { code: &mut self.init_code });
        self
    }

    fn jump_table(mut self, jump_table: Vec<u32>) -> Self {
        self.jump_table = jump_table;
        self
    }

    fn run(mut self) -> BenchmarkResult
    where
        C: FnMut(InstructionBuffer),
    {
        use polkavm_common::program::asm::*;
        use polkavm_common::program::Reg::*;

        log::info!("Benchmarking '{}':", self.name);

        let mut cg = BenchCodegen::new(self.samples, self.repeat_execution_inner, self.allocate_hugepages, self.lock_memory);
        cg.custom_codegen = self.custom_codegen;

        let pointer_address = cg.pointer_address();
        let samples_address = cg.samples_address();
        let stack_space_used = cg.stack_space_used();
        let state_origin = cg.state_origin();

        let mut perf_counters = Vec::new();
        for &counter in &self.performance_counters {
            let (kind, pid) = match counter.pmu() {
                Pmu::Core => (crate::system::PERF_TYPE_RAW, self.worker_pid),
                Pmu::L3 => {
                    // Just a sanity check.
                    let cpumask = std::fs::read_to_string("/sys/bus/event_source/devices/amd_l3/cpumask").unwrap();
                    if !cpumask.trim().split(',').any(|cpu| {
                        let cpu: usize = cpu.trim().parse().unwrap();
                        self.cpu == cpu
                    }) {
                        panic!("selected CPU is incompatible with the L3 cache CPU mask");
                    }

                    // NOTE: These counters can't be set per-process, hence the 'u32::MAX'.
                    (self.amd_l3_type, u32::MAX)
                }
            };

            perf_counters.push(configure_perf_counter(pid, self.cpu, kind, counter.get()).unwrap());
        }

        let mut module_config = ModuleConfig::new();
        module_config.set_gas_metering(if self.gas_metering {
            Some(polkavm::GasMeteringKind::Sync)
        } else {
            None
        });
        module_config.set_custom_codegen(cg);
        module_config.set_dynamic_paging(self.dynamic_paging);

        let mut code = Vec::new();
        code.extend_from_slice(&self.init_code);
        code.push(ecalli(ECALLI_BENCHMARK_PROLOGUE));
        for _ in 0..self.repeat_code {
            (self.codegen)(InstructionBuffer { code: &mut code });
        }
        code.push(ecalli(ECALLI_BENCHMARK_EPILOGUE));
        code.push(ecalli(ECALLI_BENCHMARK_ON_FINISH));
        code.push(ret());

        let mut builder = ProgramBlobBuilder::new_64bit();
        builder.add_export_by_basic_block(0, b"main");
        builder.set_stack_size(4096 + stack_space_used);
        builder.set_rw_data_size(self.rw_data_size);
        builder.set_code(&code, &self.jump_table);
        let blob = ProgramBlob::parse(builder.into_vec().unwrap().into()).unwrap();
        let module = Module::from_blob(self.engine, &module_config, blob).unwrap();

        let offset_list = module.program_counter_to_machine_code_offset().unwrap();
        let benchmark_code_length = offset_list[offset_list.len() - 3].1 - offset_list[1].1;

        let mut samples_all = Vec::new();
        let mut restart_count = 0;
        for _ in 0..self.repeat_execution_outer {
            if !RUNNING.load(Ordering::Relaxed) {
                panic!("sigint triggered");
            }

            let mut instance = module.instantiate().unwrap();
            instance.set_reg(RA, polkavm::RETURN_TO_HOST);
            instance.set_reg(SP, u64::from(STACK_ADDRESS_HI - stack_space_used));
            instance.set_gas(i64::MAX);
            instance.set_next_program_counter(ProgramCounter(0));
            instance
                .zero_memory(samples_address, self.samples as u32 * core::mem::size_of::<Sample>() as u32)
                .unwrap();
            instance.write_u64(pointer_address, u64::from(samples_address)).unwrap();
            if let Some(ref mut setup_instance) = self.setup_instance {
                setup_instance(&mut instance);
            }

            loop {
                let mut result = instance.run().unwrap();
                if let InterruptKind::Ecalli(ECALLI_BENCHMARK_ON_FINISH) = result {
                    if let Some(ref mut on_finished) = self.on_finished {
                        on_finished(&mut instance);
                    }
                    result = instance.run().unwrap();
                }

                if let InterruptKind::Ecalli(hostcall) = result {
                    if let Some(handler) = self.hostcall_handlers.get_mut(&hostcall) {
                        handler(&mut instance);
                        continue;
                    }
                }

                assert!(matches!(result, InterruptKind::Finished), "unexpected result: {result:?}");
                break;
            }

            let restart_count_local = instance
                .read_u64(state_origin + offset_of!(BenchState, restart_count) as u32)
                .unwrap();

            restart_count += restart_count_local;

            let samples = instance
                .read_memory(samples_address, self.samples as u32 * core::mem::size_of::<Sample>() as u32)
                .unwrap();

            samples_all.extend(samples);
        }

        #[allow(clippy::cast_ptr_alignment)] // Technically this is UB, but eh, it's test code and it works, so whatever.
        let samples = unsafe {
            core::slice::from_raw_parts(
                samples_all.as_ptr().cast::<Sample>(),
                samples_all.len() / core::mem::size_of::<Sample>(),
            )
        };

        let cycles = stats(samples.iter().map(|sample| sample.elapsed()));
        let total_inner_repetitions = self.repeat_code as u64 * u64::from(self.repeat_execution_inner) * self.cost_divisor;
        let cost_in_cycles = cycles.med as f64 / total_inner_repetitions as f64;
        let cost_in_time = ((cost_in_cycles / self.frequency as f64) * 1_000_000_000_000_f64) as u64;
        log::info!("Benchmark results for '{}':", self.name);
        log::info!("  {:<28}: {restart_count}", "Restart count");
        log::info!("  {:<28}: {benchmark_code_length}", "Raw code length");
        log::info!("  {:<28}: {}", "Total ops", total_inner_repetitions);
        log::info!("  {:<28}: {}", "Cycles", cycles);
        log::info!("  {:<28}: {:.01}", "Cost (in cycles)", cost_in_cycles);
        log::info!("  {:<28}: {}", "Cost (in ps)", cost_in_time);

        let mut counters = HashMap::new();
        let mut core_pmc_index = 0;
        let mut l3_pmc_index = 0;
        for counter in self.performance_counters.iter().copied() {
            let pmc_index = match counter.pmu() {
                Pmu::Core => &mut core_pmc_index,
                Pmu::L3 => &mut l3_pmc_index,
            };

            let index = *pmc_index;
            *pmc_index += 1;

            let pmc = stats(samples.iter().map(|sample| sample.get_pmc(counter.pmu(), index)));
            let per_inst = if matches!(
                counter,
                CounterKind::L1DataAccess
                    | CounterKind::L1DataMiss
                    | CounterKind::L2DataMissFromL1Miss
                    | CounterKind::L2DataMissFromHWPF1
                    | CounterKind::L2DataMissFromHWPF2
                    | CounterKind::L3CacheHit
                    | CounterKind::L3CacheMiss
            ) {
                format!(" per instr={:.02}", pmc.med as f64 / total_inner_repetitions as f64)
            } else {
                String::new()
            };

            log::info!("  {:<28}: {pmc}{per_inst}", counter.name());
            counters.insert(counter, pmc);
        }

        core::mem::drop(perf_counters);
        BenchmarkResult {
            counters,
            cycles,
            cost_per_operation: cost_in_time.try_into().unwrap(),
            total_inner_repetitions,
        }
    }
}

macro_rules! weights_io {
    (generate_code_impl
        $output:ident $model:ident
        $($inst:ident,)+
    ) => {
        let CostModel { $($inst,)+ } = $model;
        $({
            writeln!(&mut $output, "    {}: {},", stringify!($inst), $inst).unwrap();
        })+
    };

    (generate_json_impl
        $output:ident $model:ident
        $($inst:ident,)+
    ) => {
        let CostModel { $($inst,)+ } = $model;
        $({
            writeln!(&mut $output, "    \"{}\": {},", stringify!($inst), $inst).unwrap();
        })+
    };

    (generate_json $output:ident $model:ident) => {
        weights_io! {
            call generate_json_impl $output $model
        }
    };

    (generate_code $output:ident $model:ident) => {
        weights_io! {
            call generate_code_impl $output $model
        }
    };

    (model_from_map_impl $map:ident $model:ident $($inst:ident,)+) => {
        $(
            $model.$inst = $map.remove(stringify!($inst)).ok_or_else(|| format!("missing cost for: '{}'", stringify!($inst)))?;
        )+
    };

    (model_from_map $map:ident $model:ident) => {
        weights_io! {
            call model_from_map_impl $map $model
        }
    };

    (call
        $($args:ident)+
    ) => {
        weights_io! {
            $($args)+

            add_32,
            add_64,
            add_imm_32,
            add_imm_64,
            and,
            and_imm,
            and_inverted,
            branch_eq,
            branch_eq_imm,
            branch_greater_or_equal_signed,
            branch_greater_or_equal_signed_imm,
            branch_greater_or_equal_unsigned,
            branch_greater_or_equal_unsigned_imm,
            branch_greater_signed_imm,
            branch_greater_unsigned_imm,
            branch_less_or_equal_signed_imm,
            branch_less_or_equal_unsigned_imm,
            branch_less_signed,
            branch_less_signed_imm,
            branch_less_unsigned,
            branch_less_unsigned_imm,
            branch_not_eq,
            branch_not_eq_imm,
            cmov_if_not_zero,
            cmov_if_not_zero_imm,
            cmov_if_zero,
            cmov_if_zero_imm,
            count_leading_zero_bits_32,
            count_leading_zero_bits_64,
            count_set_bits_32,
            count_set_bits_64,
            count_trailing_zero_bits_32,
            count_trailing_zero_bits_64,
            div_signed_32,
            div_signed_64,
            div_unsigned_32,
            div_unsigned_64,
            ecalli,
            fallthrough,
            invalid,
            jump,
            jump_indirect,
            load_i16,
            load_i32,
            load_i8,
            load_imm,
            load_imm64,
            load_imm_and_jump,
            load_imm_and_jump_indirect,
            load_indirect_i16,
            load_indirect_i32,
            load_indirect_i8,
            load_indirect_u16,
            load_indirect_u32,
            load_indirect_u64,
            load_indirect_u8,
            load_u16,
            load_u32,
            load_u64,
            load_u8,
            maximum,
            maximum_unsigned,
            memset,
            minimum,
            minimum_unsigned,
            move_reg,
            mul_32,
            mul_64,
            mul_imm_32,
            mul_imm_64,
            mul_upper_signed_signed,
            mul_upper_signed_unsigned,
            mul_upper_unsigned_unsigned,
            negate_and_add_imm_32,
            negate_and_add_imm_64,
            or,
            or_imm,
            or_inverted,
            rem_signed_32,
            rem_signed_64,
            rem_unsigned_32,
            rem_unsigned_64,
            reverse_byte,
            rotate_left_32,
            rotate_left_64,
            rotate_right_32,
            rotate_right_64,
            rotate_right_imm_32,
            rotate_right_imm_64,
            rotate_right_imm_alt_32,
            rotate_right_imm_alt_64,
            sbrk,
            set_greater_than_signed_imm,
            set_greater_than_unsigned_imm,
            set_less_than_signed,
            set_less_than_signed_imm,
            set_less_than_unsigned,
            set_less_than_unsigned_imm,
            shift_arithmetic_right_32,
            shift_arithmetic_right_64,
            shift_arithmetic_right_imm_32,
            shift_arithmetic_right_imm_64,
            shift_arithmetic_right_imm_alt_32,
            shift_arithmetic_right_imm_alt_64,
            shift_logical_left_32,
            shift_logical_left_64,
            shift_logical_left_imm_32,
            shift_logical_left_imm_64,
            shift_logical_left_imm_alt_32,
            shift_logical_left_imm_alt_64,
            shift_logical_right_32,
            shift_logical_right_64,
            shift_logical_right_imm_32,
            shift_logical_right_imm_64,
            shift_logical_right_imm_alt_32,
            shift_logical_right_imm_alt_64,
            sign_extend_16,
            sign_extend_8,
            store_imm_indirect_u16,
            store_imm_indirect_u32,
            store_imm_indirect_u64,
            store_imm_indirect_u8,
            store_imm_u16,
            store_imm_u32,
            store_imm_u64,
            store_imm_u8,
            store_indirect_u16,
            store_indirect_u32,
            store_indirect_u64,
            store_indirect_u8,
            store_u16,
            store_u32,
            store_u64,
            store_u8,
            sub_32,
            sub_64,
            trap,
            unlikely,
            xnor,
            xor,
            xor_imm,
            zero_extend_16,
        }
    };
}

fn serialize_cost_model_to_json(model: &CostModel) -> String {
    use core::fmt::Write;
    let mut output = String::new();
    output.push_str("{\n");
    weights_io! {
        generate_json output model
    }
    output.pop();
    output.pop();
    output.push('\n');
    output.push_str("}\n");
    output
}

fn serialize_cost_model_to_code(model: &CostModel) -> String {
    use core::fmt::Write;
    let mut output = String::new();
    output.push_str("CostModel {\n");
    weights_io! {
        generate_code output model
    }
    output.push_str("}\n");
    output
}

fn deserialize_cost_model_from_map(mut map: BTreeMap<String, u32>) -> Result<CostModel, String> {
    let mut cost_model = CostModel::naive();
    weights_io! {
        model_from_map map cost_model
    }

    if !map.is_empty() {
        let extra_keys: Vec<_> = map.into_keys().map(|key| format!("'{}'", key)).collect();
        let extra_keys = extra_keys.join(", ");
        return Err(format!("failed to deserialize cost model: extra keys: {extra_keys}"));
    }

    Ok(cost_model)
}

const RNG_SEED: u128 = 0xc7ad76cc9cc25b37634aefbede429109_u128;

struct ChartData {
    label: String,
    data: Vec<(u64, f64, f64, f64)>,
}

fn graph_charts(path: &Path, charts: Vec<ChartData>) {
    use plotters::prelude::*;
    let root = BitMapBackend::new(path, (1920, 1080 * charts.len() as u32)).into_drawing_area();
    root.fill(&BLACK).unwrap();
    let areas = root.split_evenly((charts.len(), 1));

    for (index, chart) in charts.into_iter().enumerate() {
        let key_points_x: Vec<_> = chart.data.iter().map(|(x, _, _, _)| *x).collect();
        let x_min = chart.data.iter().map(|(x, _, _, _)| *x).min().unwrap();
        let x_max = chart.data.iter().map(|(x, _, _, _)| *x).max().unwrap();
        let y_max = chart.data.iter().map(|(_, _, _, max)| *max).max_by(|a, b| a.total_cmp(b)).unwrap();
        let mut cb = ChartBuilder::on(&areas[index])
            .margin(64)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .caption(
                chart.label,
                ("Segoe UI", 50.0, FontStyle::Normal, &WHITE).into_text_style(&areas[index]),
            )
            .build_cartesian_2d((x_min..x_max).log_scale().with_key_points(key_points_x), (0.0..y_max).log_scale())
            .unwrap();

        cb.configure_mesh()
            .label_style(("Segoe UI", 20.0, &RGBColor(255, 255, 255)))
            .bold_line_style(RGBColor(100, 100, 100))
            .light_line_style(RGBColor(1, 1, 1))
            .draw()
            .unwrap();

        cb.draw_series(
            chart
                .data
                .iter()
                .map(|&(x, min, avg, max)| ErrorBar::new_vertical(x, min, avg, max, RED.filled().stroke_width(1), 15)),
        )
        .unwrap();
    }

    root.present().unwrap();
}

fn main_cache_miss_benchmark(isolation_args: IsolationArgs, output_chart: Option<PathBuf>) -> Result<(), Error> {
    use polkavm_common::program::asm::*;
    use polkavm_common::program::Reg::*;

    setup_signal_handler();
    let ctx = Context::new(isolation_args)?;

    let counters = [
        CounterKind::L1DataAccess,
        CounterKind::L1DataMiss,
        CounterKind::L2DataMissFromL1Miss,
        CounterKind::L2DataMissFromHWPF1,
        CounterKind::L2DataMissFromHWPF2,
        CounterKind::L3CacheHit,
        CounterKind::L3CacheMiss,
    ];

    let cache = ctx.l3_cache;
    let cache_line_bits = (cache.coherency_line_size() as u64).ilog2();
    let set_bits = (cache.sets() as u64).ilog2();
    let increment = 1 << (cache_line_bits + set_bits);
    log::info!("Access increment: 0x{:x} ({}kB)", increment, increment / 1024);

    let mut all_results = Vec::new();
    const MAX_MEMORY: u64 = 2048 * 1024 * 1024;

    let mut memory = 1024 * 64;
    while memory <= MAX_MEMORY {
        let mut local_results = Vec::new();
        for kind in [CacheKind::L1, CacheKind::L2, CacheKind::L3] {
            let name = match kind {
                CacheKind::L1 => "L1",
                CacheKind::L2 => "L2",
                CacheKind::L3 => "L3",
            };

            let (iterations, increment) = kind.get_config(&ctx);

            let mut addresses = Vec::new();
            let mut offset = 0x20000;
            for _ in 0..iterations {
                addresses.push(offset);
                offset += increment;
            }

            let results = ctx
                .benchmark(format!("cache_miss_test_{name}_{memory}_{increment}"), |mut code| {
                    let mut offset = 0x20000;
                    for _ in 0..iterations {
                        code.push(load_indirect_u64(A0, A0, offset));
                        offset += increment;
                    }
                })
                .cost_divisor(iterations)
                .rw_data_size(memory as u32 + 4096)
                .repeat_code(1)
                .repeat_inner(10000)
                .counters(counters)
                .setup_instance(move |instance| {
                    for &address in &addresses {
                        instance.zero_memory(address, 4096).unwrap();
                    }
                })
                .run();

            local_results.push(results);
        }

        let results = local_results.into_iter().max_by_key(|result| result.cost_per_operation).unwrap();

        all_results.push((memory, results));
        memory *= 2;
    }

    let mut charts = Vec::new();
    charts.push(ChartData {
        label: "Cycles".into(),
        data: all_results
            .iter()
            .map(|(increment, result)| {
                (
                    *increment,
                    result.cycles.p10 as f64 / result.total_inner_repetitions as f64,
                    result.cycles.med as f64 / result.total_inner_repetitions as f64,
                    result.cycles.p90 as f64 / result.total_inner_repetitions as f64,
                )
            })
            .collect(),
    });

    for counter in counters {
        charts.push(ChartData {
            label: counter.name().into(),
            data: all_results
                .iter()
                .map(|(increment, result)| {
                    let stats = result.counter(counter).unwrap();
                    (*increment, stats.min as f64, stats.avg as f64, stats.max as f64)
                })
                .collect(),
        })
    }

    if let Some(path) = output_chart {
        log::info!("Writing chart to {}...", path.display());
        graph_charts(&path, charts);
    }

    Ok(())
}

#[derive(Copy, Clone)]
enum CacheKind {
    L1,
    L2,
    L3,
}

impl CacheKind {
    fn get_cache(self, ctx: &Context) -> raw_cpuid::CacheParameter {
        match self {
            CacheKind::L1 => ctx.l1_cache,
            CacheKind::L2 => ctx.l2_cache,
            CacheKind::L3 => ctx.l3_cache,
        }
    }

    fn get_config(self, ctx: &Context) -> (u64, u32) {
        let cache = self.get_cache(ctx);
        let iterations = match self {
            CacheKind::L1 => ctx.l1_cache.associativity() as u64 + 1,
            // Not exactly sure why, but these actually need *more* accesses than their associativity would suggest to always trigger a cache miss.
            CacheKind::L2 => ctx.l2_cache.associativity() as u64 + 4,
            CacheKind::L3 => ctx.l3_cache.associativity() as u64 + 22,
        };

        let cache_line_bits = (cache.coherency_line_size() as u64).ilog2();
        let set_bits = (cache.sets() as u64).ilog2();
        let increment = 1 << (cache_line_bits + set_bits);

        (iterations, increment)
    }
}

struct PageMapReader {
    fd: linux_raw::Fd,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct PageMapEntry(u64);

impl PageMapEntry {
    fn physical_address(self) -> u64 {
        let pfn = (self.0 << 10) >> 10;
        pfn * 4096
    }
}

struct Hex(u64);

impl core::fmt::Debug for Hex {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(fmt, "0x{:x}", self.0)
    }
}

impl core::fmt::Display for PageMapEntry {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        fmt.debug_struct("PagemapEntry")
            .field("physial_address", &Hex(self.physical_address()))
            .field("soft_dirty", &((self.0 >> 55) & 1))
            .field("exclusive", &((self.0 >> 56) & 1))
            .field("shared", &((self.0 >> 61) & 1))
            .field("swapped", &((self.0 >> 62) & 1))
            .field("present", &((self.0 >> 63) & 1))
            .finish()
    }
}

impl PageMapReader {
    fn new(pid: u32) -> Self {
        let path = format!("/proc/{pid}/pagemap\0");
        Self {
            fd: linux_raw::sys_open(core::ffi::CStr::from_bytes_with_nul(path.as_bytes()).unwrap(), linux_raw::O_RDONLY).unwrap(),
        }
    }

    fn read(&self, address: u64) -> PageMapEntry {
        linux_raw::sys_lseek(self.fd.borrow(), (address / 4096 * 8) as i64, linux_raw::SEEK_SET).unwrap();
        let mut xs = [0_u8; 8];
        linux_raw::sys_read(self.fd.borrow(), &mut xs).unwrap();
        let value = u64::from_le_bytes([xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]]);
        PageMapEntry(value)
    }
}

struct Cache {
    is_l1: bool,
    is_physically_indexed: bool,
    index_mask: u64,
    index_bits: u32,
    sets: Vec<Set>,
}

#[derive(Clone)]
struct Set {
    entries: Vec<Entry>,
}

#[derive(Clone)]
struct Entry {
    address: Address,
    timestamp: u32,
    tag: u64,
}

#[derive(Copy, Clone)]
struct Address {
    virt: u64,
    phys: u64,
}

impl Cache {
    fn new(info: raw_cpuid::CacheParameter, is_physically_indexed: bool, is_l1: bool) -> Self {
        Cache {
            is_l1,
            is_physically_indexed,
            sets: vec![
                Set {
                    entries: vec![
                        Entry {
                            address: Address {
                                virt: u64::MAX,
                                phys: u64::MAX
                            },
                            timestamp: 0,
                            tag: u64::MAX,
                        };
                        info.associativity()
                    ]
                };
                info.sets()
            ],
            index_bits: info.sets().ilog2(),
            index_mask: (1 << info.sets().ilog2()) - 1,
        }
    }

    fn decode_address(&self, address: Address) -> (Address, usize, u64) {
        let cacheline_address = Address {
            phys: (address.phys >> 6) << 6,
            virt: (address.virt >> 6) << 6,
        };

        let set_index = (if self.is_physically_indexed { address.phys } else { address.virt } >> 6) & self.index_mask;
        let tag;
        if self.is_l1 {
            assert_eq!(self.index_bits, 6);
            let bit = |b: usize| (address.virt >> b) & 1;

            // https://gruss.cc/files/takeaway.pdf
            tag = (bit(12) ^ bit(27))
                | ((bit(13) ^ bit(26)) << 1)
                | ((bit(14) ^ bit(25)) << 2)
                | ((bit(15) ^ bit(20)) << 3)
                | ((bit(16) ^ bit(21)) << 4)
                | ((bit(17) ^ bit(22)) << 5)
                | ((bit(18) ^ bit(23)) << 6)
                | ((bit(19) ^ bit(24)) << 7);
        } else {
            tag = address.phys >> (self.index_bits + 6);
        }

        (cacheline_address, set_index as usize, tag)
    }

    fn contains(&self, address: Address) -> bool {
        let (_, set_index, tag) = self.decode_address(address);
        self.sets[set_index].entries.iter().any(|entry| entry.tag == tag)
    }

    fn evict(&mut self, address: Address) {
        let (_, set_index, tag) = self.decode_address(address);
        let set = &mut self.sets[set_index];
        if let Some(index) = set.entries.iter().position(|entry| entry.tag == tag) {
            set.entries[index] = Entry {
                address: Address {
                    virt: u64::MAX,
                    phys: u64::MAX,
                },
                timestamp: 0,
                tag: u64::MAX,
            };
        }
    }

    fn insert(&mut self, address: Address) -> Option<Address> {
        let (address, set_index, tag) = self.decode_address(address);
        let set = &mut self.sets[set_index];
        let mut min_entry: usize = 0;
        let mut max_entry: usize = 0;
        let mut found_entry = None;
        for (entry_index, entry) in set.entries.iter().enumerate().skip(1) {
            if entry.timestamp < set.entries[min_entry].timestamp {
                min_entry = entry_index;
            }
            if entry.timestamp > set.entries[max_entry].timestamp {
                max_entry = entry_index;
            }

            if entry.tag == tag {
                found_entry = Some(entry_index);
            }
        }

        let timestamp = set.entries[max_entry].timestamp + 1;
        if let Some(found_entry) = found_entry {
            set.entries[found_entry].timestamp = timestamp;
            None
        } else {
            let old_address = set.entries[min_entry].address;
            set.entries[min_entry] = Entry { address, timestamp, tag };
            Some(old_address)
        }
    }
}

struct CacheSim {
    l1: Cache,
    l2: Cache,
    l3: Cache,
    l1_hit_count: u64,
    l1_miss_count: u64,
    l2_hit_count: u64,
    l2_miss_count: u64,
    l3_hit_count: u64,
    l3_miss_count: u64,
}

impl CacheSim {
    fn new(l1_info: raw_cpuid::CacheParameter, l2_info: raw_cpuid::CacheParameter, l3_info: raw_cpuid::CacheParameter) -> Self {
        CacheSim {
            l1: Cache::new(l1_info, false, true),
            l2: Cache::new(l2_info, false, false),
            l3: Cache::new(l3_info, true, false),
            l1_hit_count: 0,
            l1_miss_count: 0,
            l2_hit_count: 0,
            l2_miss_count: 0,
            l3_hit_count: 0,
            l3_miss_count: 0,
        }
    }

    fn access(&mut self, address: Address) {
        if self.l1.contains(address) {
            self.l1_hit_count += 1;
            self.l1.insert(address);
            return;
        }
        self.l1_miss_count += 1;

        if self.l2.contains(address) {
            self.l2_hit_count += 1;
            self.l2.insert(address);

            if let Some(old_address) = self.l1.insert(address) {
                if let Some(old_address) = self.l3.insert(old_address) {
                    self.l1.evict(old_address);
                    self.l2.evict(old_address);
                }
            }

            return;
        }
        self.l2_miss_count += 1;

        if self.l3.contains(address) {
            self.l3_hit_count += 1;
            self.l3.insert(address);

            if let Some(old_address) = self.l1.insert(address) {
                if let Some(old_address) = self.l3.insert(old_address) {
                    self.l1.evict(old_address);
                    self.l2.evict(old_address);
                }
            }

            if let Some(old_address) = self.l2.insert(address) {
                if let Some(old_address) = self.l3.insert(old_address) {
                    self.l1.evict(old_address);
                    self.l2.evict(old_address);
                }
            }

            return;
        }
        self.l3_miss_count += 1;

        if let Some(old_address) = self.l1.insert(address) {
            if let Some(old_address) = self.l3.insert(old_address) {
                self.l1.evict(old_address);
                self.l2.evict(old_address);
            }
        }

        if let Some(old_address) = self.l2.insert(address) {
            if let Some(old_address) = self.l3.insert(old_address) {
                self.l1.evict(old_address);
                self.l2.evict(old_address);
            }
        }
    }
}

fn main_benchmark_random_cache_misses(isolation_args: IsolationArgs) -> Result<(), Error> {
    use polkavm_common::program::asm::*;
    use polkavm_common::program::Reg::*;

    setup_signal_handler();
    let ctx = Context::new(isolation_args)?;

    let mut pcg_init_code = Vec::new();
    let pcg_state_reg = S0;
    let pcg_tmp_reg = T0;
    let pcg_increment = pcg_init(&mut pcg_init_code, pcg_state_reg, RNG_SEED);
    let repeat_count = 10000;

    ctx.benchmark("test_random_access", |mut code| {
        pcg_rand_u32(code.code, pcg_increment, pcg_state_reg, pcg_tmp_reg, A1);
        code.push(shift_logical_right_imm_64(A1, A1, 6));
        code.push(and_imm(A1, A1, 1024 * 1024 * 1024 - 1));
        code.push(add_64(A1, A1, A0));
        code.push(sub_64(A1, A1, A0));
        code.push(load_indirect_u8(A0, A1, 0x20000 + 0x1000 - 1));
    })
    .on_finished(move |instance| {
        let pid = instance.pid().unwrap();
        let pmr = PageMapReader::new(pid);

        let mut accesses = Vec::new();
        let mut rng = host_rng();
        for _ in 0..repeat_count {
            let mut virtual_address = u64::from(rng.rand_u32());
            virtual_address <<= 6;
            virtual_address &= 1024 * 1024 * 1024 - 1;
            virtual_address += 0x20000 + 0x1000 - 1;

            let physical_address = pmr.read(virtual_address).physical_address();
            accesses.push(Address {
                virt: virtual_address,
                phys: physical_address,
            });
        }

        let mut sim = CacheSim::new(ctx.l1_cache, ctx.l2_cache, ctx.l3_cache);
        for &address in &accesses {
            sim.access(address);
        }

        log::info!("Simulation results:");
        log::info!("  L1 cache hit:  {}", sim.l1_hit_count);
        log::info!("  L2 cache hit:  {}", sim.l2_hit_count);
        log::info!("  L3 cache hit:  {}", sim.l3_hit_count);
        log::info!("  L1 cache miss: {}", sim.l1_miss_count);
        log::info!("  L2 cache miss: {}", sim.l2_miss_count);
        log::info!("  L3 cache miss: {}", sim.l3_miss_count);
    })
    .allocate_hugepages(true)
    .rw_data_size(1024 * 1024 * 1024)
    .setup_code(|mut code| code.extend_from_slice(&pcg_init_code))
    .repeat_code(1)
    .repeat_inner(repeat_count)
    .counters([
        CounterKind::L1DataAccess,
        CounterKind::L1DataMiss,
        CounterKind::L2DataMissFromL1Miss,
        CounterKind::L2DataMissFromHWPF1,
        CounterKind::L2DataMissFromHWPF2,
        CounterKind::L3CacheHit,
        CounterKind::L3CacheMiss,
    ])
    .run();

    Ok(())
}

fn maximize_cache_misses_for_cache(ctx: &Context, kind: CacheKind) -> BenchmarkResult {
    use polkavm_common::program::asm::*;
    use polkavm_common::program::Reg::*;

    // Assuming an L1 cache that is 32 KiB in size and is 8-way set associative we can decompose each address to:
    //   Bits 0..6: offset within the cache line (because 2^6 == 64, and 64 is the cache-line size)
    //   Bits 6..12: index of the set (because 2^6 == 64, and `cache_size / (line_size * associativity) = 32 KiB / (64 * 8) = 32 KiB / 512 = 64 sets`)
    //   Bits 12..: tag
    //
    // The 'offset' is ignored because the smallest cacheable unit is 64-bytes.
    // The 'index' is used to pick a slot in the cache (we have 64 of those slots), where each slot contains 8 entries (because the cache is 8-way associative).
    // The 'tag' is used to find the specific entry in the cache.
    //
    // The 'offset' and 'index' (i.e. the lowest 12 bits of the address) are the same for both the physical and logical addresses, because they're within one page.
    // The 'tag' comes from the *physical* address.
    //
    // struct Cache {
    //     sets: [Set; 64]
    // }
    // struct Set {
    //     entries: [Entry; 8]
    // }
    // struct Entry {
    //     tag: uxx,
    //     cacheline: [u8; 64]
    // }

    let l1_cache = ctx.l1_cache;
    let l2_cache = ctx.l2_cache;
    let l3_cache = ctx.l3_cache;

    let name = match kind {
        CacheKind::L1 => "trigger_l1_misses",
        CacheKind::L2 => "trigger_l2_misses",
        CacheKind::L3 => "trigger_l3_misses",
    };

    let (iterations, increment) = kind.get_config(ctx);

    let mut addresses = Vec::new();
    let mut offset = 0x20000;
    for _ in 0..iterations {
        addresses.push(offset);
        offset += increment;
    }

    let addresses_copy = addresses.clone();
    let addresses_copy2 = addresses.clone();
    log::info!("Page count: {}", addresses.len());

    ctx.benchmark(name, |mut code| {
            for &address in &addresses {
                code.push(load_indirect_u64(A0, A0, address));
            }
        })
        .cost_divisor(iterations)
        .rw_data_size(1024 * 1024 * 1024)
        .allocate_hugepages(matches!(kind, CacheKind::L2 | CacheKind::L3))
        .setup_instance(move |instance| {
            for &address in &addresses_copy {
                instance.zero_memory(address, 4096).unwrap();
            }
        })
        .on_finished(move |instance| {
            let pid = instance.pid().unwrap();

            let mut addresses = Vec::new();
            let pmr = PageMapReader::new(pid);
            log::info!("Accessed addresses:");
            for &address in &addresses_copy2 {
                let pme = pmr.read(u64::from(address));
                let physical_address = pme.physical_address();
                let l1_set = (physical_address >> (l1_cache.coherency_line_size() as u64).ilog2()) & ((1 << l1_cache.sets().ilog2()) - 1);
                let l1_set_count = l1_cache.sets();
                let l2_set = (physical_address >> (l2_cache.coherency_line_size() as u64).ilog2()) & ((1 << l2_cache.sets().ilog2()) - 1);
                let l2_set_count = l2_cache.sets();
                let l3_set = (physical_address >> (l3_cache.coherency_line_size() as u64).ilog2()) & ((1 << l3_cache.sets().ilog2()) - 1);
                let l3_set_count = l3_cache.sets();
                log::info!("  0x{address:07x} (0x{physical_address:010x}) L1 set = {l1_set:>2}/{l1_set_count} L2 set = {l2_set:4>}/{l2_set_count} L3 set = {l3_set:>5}/{l3_set_count}");
                addresses.push(Address {
                    virt: u64::from(address),
                    phys: physical_address
                });
            }

            let mut sim = CacheSim::new(l1_cache, l2_cache, l3_cache);
            for _ in 0..10000 {
                for &address in &addresses {
                    sim.access(address);
                }
            }
            log::info!("Simulation results:");
            log::info!("  L1 cache hit:  {}", sim.l1_hit_count);
            log::info!("  L2 cache hit:  {}", sim.l2_hit_count);
            log::info!("  L3 cache hit:  {}", sim.l3_hit_count);
            log::info!("  L1 cache miss: {}", sim.l1_miss_count);
            log::info!("  L2 cache miss: {}", sim.l2_miss_count);
            log::info!("  L3 cache miss: {}", sim.l3_miss_count);
        })
        .repeat_code(1)
        .repeat_inner(10000)
        .counters([
            CounterKind::L1DataAccess,
            CounterKind::L1DataMiss,
            CounterKind::L2DataMissFromL1Miss,
            CounterKind::L2DataMissFromHWPF1,
            CounterKind::L2DataMissFromHWPF2,
            CounterKind::L3CacheHit,
            CounterKind::L3CacheMiss,
        ])
        .run()
}

fn main_benchmark_maximum_cache_misses(isolation_args: IsolationArgs) -> Result<(), Error> {
    setup_signal_handler();
    let ctx = Context::new(isolation_args)?;
    for kind in [CacheKind::L1, CacheKind::L2, CacheKind::L3] {
        maximize_cache_misses_for_cache(&ctx, kind);
    }

    Ok(())
}

fn main_benchmark_optimistic(isolation_args: IsolationArgs) -> Result<(), Error> {
    use polkavm_common::program::asm::*;
    use polkavm_common::program::Reg::*;

    setup_signal_handler();
    let ctx = Context::new(isolation_args)?;

    let basic_block = {
        ctx.benchmark("basic_block", |mut code| {
            code.push(fallthrough());
        })
        .repeat_code(10000)
        .gas_metering(true)
        .run()
    };

    let branch_hit = {
        let mut n = 0;
        ctx.benchmark("branch_hit", |mut code| {
            code.push(add_imm_64(A1, A1, 1));
            code.push(branch_eq(A1, S1, n + 1));
            n += 1;
        })
        .repeat_code(10000)
        .setup_code(|mut code| {
            code.push(load_imm(S1, 1));
        })
        .counters(vec![CounterKind::BranchesRetired, CounterKind::BranchesRetiredMisprediction])
        .run()
    };

    log::info!("branch_hit: {}", branch_hit.cost_per_operation + basic_block.cost_per_operation);

    {
        ctx.benchmark("l1_hit", |mut code| {
            code.push(load_indirect_u64(A0, A0, 0x20000));
        })
        .repeat_code(10000)
        .counters(vec![
            CounterKind::L1DataAccess,
            CounterKind::L1DataMiss,
            CounterKind::L3CacheHit,
            CounterKind::L3CacheMiss,
        ])
        .rw_data_size(4096)
        .run();
    }
    Ok(())
}

fn main_generate_model(
    isolation_args: IsolationArgs,
    output_model_code: Option<PathBuf>,
    output_model_blob: Option<PathBuf>,
    output_model_json: Option<PathBuf>,
) -> Result<(), Error> {
    use polkavm_common::program::asm::*;
    use polkavm_common::program::Reg::*;

    setup_signal_handler();
    let ctx = Context::new(isolation_args)?;

    let mut model = CostModel::naive();

    let mut pcg_init_code = Vec::new();
    let pcg_state_reg = S0;
    let pcg_tmp_reg = T0;
    let pcg_increment = pcg_init(&mut pcg_init_code, pcg_state_reg, RNG_SEED);

    let [load_imm_and_jump_indirect, jump_indirect] = [false, true].map(|should_load_imm| {
        let copies = 32;
        ctx.benchmark(
            if should_load_imm {
                "load_imm_and_jump_indirect"
            } else {
                "jump_indirect"
            },
            |mut code| {
                code.push(load_imm(A1, 10000));
                code.push(fallthrough());

                for _ in 0..copies {
                    code.push(add_imm_64(A1, A1, -1_i32 as u32));
                    code.push(branch_eq_imm(A1, 0, copies * 2 + 1));
                    pcg_rand_u32(code.code, pcg_increment, pcg_state_reg, pcg_tmp_reg, A0);
                    code.push(and_imm(A0, A0, 1024 * 1024 / 2));
                    code.push(shift_logical_left_imm_64(A0, A0, 1));
                    if should_load_imm {
                        code.push(load_imm_and_jump_indirect(A2, A0, 0xffffffff, 2));
                    } else {
                        code.push(jump_indirect(A0, 2));
                    }
                }
            },
        )
        .cost_divisor(10000)
        .repeat_code(1)
        .setup_code(|mut code| {
            code.extend_from_slice(&pcg_init_code);
        })
        .jump_table({
            let mut rng = host_rng();
            (0..1024 * 1024).map(|_| 1 + rng.rand_range(0..copies) * 2).collect()
        })
        .run()
    });

    model.load_imm_and_jump_indirect = load_imm_and_jump_indirect.cost_per_operation;
    model.jump_indirect = jump_indirect.cost_per_operation;

    model.ecalli = ctx
        .benchmark("ecalli", |mut code| {
            code.push(ecalli(100));
        })
        .hostcall_handler(100, |_| {})
        .repeat_code(10000)
        .run()
        .cost_per_operation;

    macro_rules! division_benchmark {
        ($name:ident) => {{
            let random_full = ctx
                .benchmark(concat!(stringify!($name), "_random_full"), |mut code| {
                    pcg_rand_u32(code.code, pcg_increment, pcg_state_reg, pcg_tmp_reg, A1);
                    pcg_rand_u32(code.code, pcg_increment, pcg_state_reg, pcg_tmp_reg, A2);
                    code.push($name(A0, A1, A2));
                    code.push(add_64(A1, A1, A0));
                })
                .repeat_code(10000)
                .setup_code(|mut code| {
                    code.extend_from_slice(&pcg_init_code);
                })
                .run();

            let random_zero_divisor = ctx
                .benchmark(concat!(stringify!($name), "_random_zero_divisor"), |mut code| {
                    pcg_rand_u32(code.code, pcg_increment, pcg_state_reg, pcg_tmp_reg, A1);
                    code.push(and_imm(A1, A1, 1));
                    code.push(div_signed_64(A0, A0, A1));
                    code.push(add_64(A1, A1, A0));
                })
                .repeat_code(10000)
                .setup_code(|mut code| {
                    code.extend_from_slice(&pcg_init_code);
                })
                .run();

            model.$name = random_full.cost_per_operation.max(random_zero_divisor.cost_per_operation);
        }};
    }

    division_benchmark!(div_signed_64);
    division_benchmark!(div_unsigned_64);
    division_benchmark!(div_signed_32);
    division_benchmark!(div_unsigned_32);
    division_benchmark!(rem_signed_64);
    division_benchmark!(rem_unsigned_64);
    division_benchmark!(rem_signed_32);
    division_benchmark!(rem_unsigned_32);

    let mut branch_miss_cost: Option<u32> = None;
    let mut branch_imm_miss_cost: Option<u32> = None;

    for (branch_kind_name, branch_kind) in [
        ("branch_eq", branch_eq as fn(Reg, Reg, u32) -> Instruction),
        ("branch_not_eq", branch_not_eq),
    ] {
        for arg in [0, 1] {
            let mut n = 0;
            let result = ctx
                .benchmark(format!("{branch_kind_name}_0_{arg}"), |mut code| {
                    code.push(branch_kind(A1, S1, n + 1));
                    n += 1;
                })
                .repeat_code(10000)
                .setup_code(|mut code| {
                    code.push(load_imm(A1, 0));
                    code.push(load_imm(S1, arg));
                })
                .counters(vec![CounterKind::BranchesRetired, CounterKind::BranchesRetiredMisprediction])
                .run();

            assert_eq!(result.counters[&CounterKind::BranchesRetired].med, 10000, "unexpected branch count");
            if result.counters[&CounterKind::BranchesRetiredMisprediction].med >= 9000 {
                branch_miss_cost = branch_miss_cost.max(Some(result.cost_per_operation));
            }
        }
    }

    for (branch_kind_name, branch_kind) in [
        ("branch_eq_imm", branch_eq_imm as fn(Reg, u32, u32) -> Instruction),
        ("branch_not_eq_imm", branch_not_eq_imm),
    ] {
        for arg in [0, 1] {
            let mut n = 0;
            let result = ctx
                .benchmark(format!("{branch_kind_name}_0_{arg}"), |mut code| {
                    code.push(branch_kind(A1, arg, n + 1));
                    n += 1;
                })
                .repeat_code(10000)
                .setup_code(|mut code| {
                    code.push(load_imm(A1, 0));
                })
                .counters(vec![CounterKind::BranchesRetired, CounterKind::BranchesRetiredMisprediction])
                .run();

            assert_eq!(result.counters[&CounterKind::BranchesRetired].med, 10000, "unexpected branch count");
            if result.counters[&CounterKind::BranchesRetiredMisprediction].med >= 9000 {
                branch_imm_miss_cost = branch_imm_miss_cost.max(Some(result.cost_per_operation));
            }
        }
    }

    let Some(branch_miss_cost) = branch_miss_cost else {
        return Err("failed to calculate branch miss cost".into());
    };

    let Some(branch_imm_miss_cost) = branch_imm_miss_cost else {
        return Err("failed to calculate branch_imm miss cost".into());
    };

    let basic_block = {
        ctx.benchmark("basic_block", |mut code| {
            code.push(fallthrough());
        })
        .repeat_code(10000)
        .gas_metering(true)
        .run()
    };

    model.fallthrough = basic_block.cost_per_operation;
    model.trap = basic_block.cost_per_operation;
    model.invalid = model.trap;

    model.branch_greater_or_equal_signed = branch_miss_cost + basic_block.cost_per_operation;
    model.branch_greater_or_equal_unsigned = branch_miss_cost + basic_block.cost_per_operation;
    model.branch_less_signed = branch_miss_cost + basic_block.cost_per_operation;
    model.branch_less_unsigned = branch_miss_cost + basic_block.cost_per_operation;
    model.branch_not_eq = branch_miss_cost + basic_block.cost_per_operation;
    model.branch_eq = branch_miss_cost + basic_block.cost_per_operation;

    model.branch_greater_or_equal_signed_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_greater_or_equal_unsigned_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_greater_signed_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_greater_unsigned_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_less_or_equal_signed_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_less_or_equal_unsigned_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_less_signed_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_less_unsigned_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_not_eq_imm = branch_imm_miss_cost + basic_block.cost_per_operation;
    model.branch_eq_imm = branch_imm_miss_cost + basic_block.cost_per_operation;

    model.jump = {
        let mut n = 0;
        ctx.benchmark("jump", |mut code| {
            code.push(jump(n + 1));
            n += 1;
        })
        .repeat_code(10000)
        .run()
    }
    .cost_per_operation
        + basic_block.cost_per_operation;

    model.load_imm_and_jump = {
        let mut n = 0;
        ctx.benchmark("load_imm_and_jump", |mut code| {
            code.push(load_imm_and_jump(A0, 0xffffffff, n + 1));
            n += 1;
        })
        .repeat_code(10000)
        .run()
    }
    .cost_per_operation
        + basic_block.cost_per_operation;

    // If the one with immediate load is cheaper due to noise then force its cost to be higher.
    model.load_imm_and_jump = model.load_imm_and_jump.max(model.jump);
    model.load_imm_and_jump_indirect = model.load_imm_and_jump_indirect.max(model.jump_indirect);

    let load_cost = maximize_cache_misses_for_cache(&ctx, CacheKind::L3).cost_per_operation;
    model.load_i16 = load_cost;
    model.load_i32 = load_cost;
    model.load_i8 = load_cost;
    model.load_indirect_i16 = load_cost;
    model.load_indirect_i32 = load_cost;
    model.load_indirect_i8 = load_cost;
    model.load_indirect_u16 = load_cost;
    model.load_indirect_u32 = load_cost;
    model.load_indirect_u64 = load_cost;
    model.load_indirect_u8 = load_cost;
    model.load_u16 = load_cost;
    model.load_u32 = load_cost;
    model.load_u64 = load_cost;
    model.load_u8 = load_cost;

    macro_rules! define_simple_benches {
        ($(
            $inst:ident($($arg:expr),+),
        )+) => {
            $({
                let result = ctx.benchmark(stringify!($inst), |mut code| code.push($inst($($arg),+))).repeat_code(100000).rw_data_size(1024 * 1024).run();
                model.$inst = result.cost_per_operation;
            })+
        };
    }

    define_simple_benches! {
        add_32(A0, A0, A1),
        add_64(A0, A0, A1),
        add_imm_32(A0, A0, 0xffffffff),
        add_imm_64(A0, A0, 0xffffffff),
        and(A0, A0, A1),
        and_imm(A0, A0, 0xffffffff),
        and_inverted(A0, A0, A1),
        cmov_if_not_zero(A1, A0, A1),
        cmov_if_not_zero_imm(A0, A0, 0xffffffff),
        cmov_if_zero(A1, A0, A1),
        cmov_if_zero_imm(A0, A0, 0xffffffff),
        count_leading_zero_bits_32(A0, A0),
        count_leading_zero_bits_64(A0, A0),
        count_set_bits_32(A0, A0),
        count_set_bits_64(A0, A0),
        count_trailing_zero_bits_32(A0, A0),
        count_trailing_zero_bits_64(A0, A0),
        load_imm(A0, 0xffffffff),
        load_imm64(A0, 0xffffffffffffffff),
        maximum(A0, A0, A1),
        maximum_unsigned(A0, A0, A1),
        // memset: 1,
        minimum(A0, A0, A1),
        minimum_unsigned(A0, A0, A1),
        move_reg(A0, A1),
        mul_32(A0, A0, A1),
        mul_64(A0, A0, A1),
        mul_imm_32(A0, A0, 0xffffffff),
        mul_imm_64(A0, A0, 0xffffffff),
        mul_upper_signed_signed(A0, A0, A1),
        mul_upper_signed_unsigned(A0, A0, A1),
        mul_upper_unsigned_unsigned(A0, A0, A1),
        negate_and_add_imm_32(A0, A0, 0xffffffff),
        negate_and_add_imm_64(A0, A0, 0xffffffff),
        or(A0, A0, A1),
        or_imm(A0, A0, 0xffffffff),
        or_inverted(A0, A0, A1),
        reverse_byte(A0, A0),
        rotate_left_32(A0, A0, A1),
        rotate_left_64(A0, A0, A1),
        rotate_right_32(A0, A0, A1),
        rotate_right_64(A0, A0, A1),
        rotate_right_imm_32(A0, A0, 0xffffffff),
        rotate_right_imm_64(A0, A0, 0xffffffff),
        rotate_right_imm_alt_32(A0, A0, 0xffffffff),
        rotate_right_imm_alt_64(A0, A0, 0xffffffff),
        // sbrk: 1,
        set_greater_than_signed_imm(A0, A0, 0xffffffff),
        set_greater_than_unsigned_imm(A0, A0, 0xffffffff),
        set_less_than_signed(A0, A0, A1),
        set_less_than_signed_imm(A0, A0, 0xffffffff),
        set_less_than_unsigned(A0, A0, A1),
        set_less_than_unsigned_imm(A0, A0, 0xffffffff),
        shift_arithmetic_right_32(A0, A0, A1),
        shift_arithmetic_right_64(A0, A0, A1),
        shift_arithmetic_right_imm_32(A0, A0, 0xffffffff),
        shift_arithmetic_right_imm_64(A0, A0, 0xffffffff),
        shift_arithmetic_right_imm_alt_32(A0, A0, 0xffffffff),
        shift_arithmetic_right_imm_alt_64(A0, A0, 0xffffffff),
        shift_logical_left_32(A0, A0, A1),
        shift_logical_left_64(A0, A0, A1),
        shift_logical_left_imm_32(A0, A0, 0xffffffff),
        shift_logical_left_imm_64(A0, A0, 0xffffffff),
        shift_logical_left_imm_alt_32(A0, A0, 0xffffffff),
        shift_logical_left_imm_alt_64(A0, A0, 0xffffffff),
        shift_logical_right_32(A0, A0, A1),
        shift_logical_right_64(A0, A0, A1),
        shift_logical_right_imm_32(A0, A0, 0xffffffff),
        shift_logical_right_imm_64(A0, A0, 0xffffffff),
        shift_logical_right_imm_alt_32(A0, A0, 0xffffffff),
        shift_logical_right_imm_alt_64(A0, A0, 0xffffffff),
        sign_extend_16(A0, A0),
        sign_extend_8(A0, A0),
        store_imm_indirect_u16(A0, 0x20000 + 4096 - 1, 0xffffffff),
        store_imm_indirect_u32(A0, 0x20000 + 4096 - 2, 0xffffffff),
        store_imm_indirect_u64(A0, 0x20000 + 4096 - 4, 0xffffffff),
        store_imm_indirect_u8(A0, 0x20000 + 4096, 0xffffffff),
        store_imm_u16(0x20000 + 4096 - 1, 0xffffffff),
        store_imm_u32(0x20000 + 4096 - 2, 0xffffffff),
        store_imm_u64(0x20000 + 4096 - 4, 0xffffffff),
        store_imm_u8(0x20000 + 4096, 0xffffffff),
        store_indirect_u16(A0, A0, 0x20000 + 4096 - 1),
        store_indirect_u32(A0, A0, 0x20000 + 4096 - 2),
        store_indirect_u64(A0, A0, 0x20000 + 4096 - 4),
        store_indirect_u8(A0, A0, 0x20000 + 4096),
        store_u16(A0, 0x20000 + 4096 - 1),
        store_u32(A0, 0x20000 + 4096 - 2),
        store_u64(A0, 0x20000 + 4096 - 4),
        store_u8(A0, 0x20000 + 4096),
        sub_32(A0, A0, A1),
        sub_64(A0, A0, A1),
        xnor(A0, A0, A1),
        xor(A0, A0, A1),
        xor_imm(A0, A0, 0xffffffff),
        zero_extend_16(A0, A0),
    }

    macro_rules! equalize {
        ($lhs:ident, $rhs:ident, $($rem:ident),+) => {
            equalize!($lhs, $rhs);
            equalize!($rhs, $($rem),+);
        };

        ($lhs:ident, $rhs:ident) => {
            if model.$lhs < model.$rhs {
                model.$lhs = model.$rhs;
            } else {
                model.$rhs = model.$lhs;
            }
        };
    }

    equalize! { and, or, xor };
    equalize! { and_imm, or_imm, xor_imm };
    equalize! {
        branch_greater_or_equal_signed,
        branch_greater_or_equal_unsigned,
        branch_less_signed,
        branch_less_unsigned,
        branch_not_eq,
        branch_eq
    }
    equalize! {
        branch_greater_or_equal_signed_imm,
        branch_greater_or_equal_unsigned_imm,
        branch_greater_signed_imm,
        branch_greater_unsigned_imm,
        branch_less_or_equal_signed_imm,
        branch_less_or_equal_unsigned_imm,
        branch_less_signed_imm,
        branch_less_unsigned_imm,
        branch_not_eq_imm,
        branch_eq_imm
    }
    equalize! {
        cmov_if_not_zero,
        cmov_if_zero
    }
    equalize! {
        cmov_if_not_zero_imm,
        cmov_if_zero_imm
    }
    equalize! {
        store_imm_indirect_u16,
        store_imm_indirect_u32,
        store_imm_indirect_u64
    }
    equalize! {
        store_indirect_u16,
        store_indirect_u32,
        store_indirect_u64
    }
    equalize! {
        store_u16,
        store_u32,
        store_u64
    }

    if let Some(output_model_code) = output_model_code {
        log::info!("Writing model code to {}...", output_model_code.display());
        let model_code = serialize_cost_model_to_code(&model);
        std::fs::write(output_model_code, model_code.as_bytes()).unwrap();
    }

    if let Some(output_model_blob) = output_model_blob {
        log::info!("Writing model blob to {}...", output_model_blob.display());
        std::fs::write(output_model_blob, model.serialize()).unwrap();
    }

    if let Some(output_model_json) = output_model_json {
        log::info!("Writing model json to {}...", output_model_json.display());
        let model_json = serialize_cost_model_to_json(&model);
        std::fs::write(output_model_json, model_json.as_bytes()).unwrap();
    }

    Ok(())
}

fn decompress_zstd(mut bytes: &[u8]) -> Vec<u8> {
    use ruzstd::io::Read;
    let mut output = Vec::new();
    let mut fp = ruzstd::streaming_decoder::StreamingDecoder::new(&mut bytes).unwrap();

    let mut buffer = vec![0_u8; 32 * 1024];
    loop {
        let count = fp.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }

        output.extend_from_slice(&buffer);
    }

    output
}

fn pretty_print(number: i64) -> String {
    let number = number.to_string();
    let mut vec = Vec::new();
    for (n, ch) in number.chars().rev().enumerate() {
        if (n % 3 == 0) && n != 0 {
            vec.push('_');
        }
        vec.push(ch);
    }
    vec.reverse();
    vec.into_iter().collect()
}

struct State {
    frames: u64,
}

fn prepare_benchmark(
    benchmark: Benchmark,
    engine: &polkavm::Engine,
    blob: &ProgramBlob,
    step_tracing: bool,
) -> Result<polkavm::InstancePre<State, String>, String> {
    match benchmark {
        Benchmark::Doom => prepare_benchmark_doom(engine, blob, step_tracing),
        _ => prepare_benchmark_generic(engine, blob, step_tracing),
    }
}

fn prepare_benchmark_generic(
    engine: &polkavm::Engine,
    blob: &ProgramBlob,
    step_tracing: bool,
) -> Result<polkavm::InstancePre<State, String>, String> {
    let mut module_config = polkavm::ModuleConfig::default();
    if step_tracing {
        module_config.set_gas_metering(None);
        module_config.set_step_tracing(true);
    } else {
        module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
        module_config.set_step_tracing(false);
    }
    let module = polkavm::Module::from_blob(engine, &module_config, blob.clone())?;
    let linker: polkavm::Linker<State, String> = polkavm::Linker::new();

    Ok(linker.instantiate_pre(&module)?)
}

fn prepare_benchmark_doom(
    engine: &polkavm::Engine,
    blob: &ProgramBlob,
    step_tracing: bool,
) -> Result<polkavm::InstancePre<State, String>, String> {
    const DOOM_WAD: &[u8] = include_bytes!("../../../examples/doom/roms/doom1.wad");

    let mut module_config = polkavm::ModuleConfig::default();
    module_config.set_page_size(16 * 1024);
    if step_tracing {
        module_config.set_gas_metering(None);
        module_config.set_step_tracing(true);
    } else {
        module_config.set_gas_metering(Some(polkavm::GasMeteringKind::Sync));
        module_config.set_step_tracing(false);
    }
    let module = polkavm::Module::from_blob(engine, &module_config, blob.clone())?;
    let mut linker: polkavm::Linker<State, String> = polkavm::Linker::new();

    linker.define_typed(
        "ext_output_video",
        |caller: polkavm::Caller<State>, _address: u32, _width: u32, _height: u32| -> Result<(), String> {
            caller.user_data.frames += 1;
            Ok(())
        },
    )?;

    linker.define_typed(
        "ext_output_audio",
        |_caller: polkavm::Caller<State>, _address: u32, _samples: u32| {},
    )?;

    linker.define_typed("ext_rom_size", |_caller: polkavm::Caller<State>| -> u32 { DOOM_WAD.len() as u32 })?;

    linker.define_typed(
        "ext_rom_read",
        |caller: polkavm::Caller<State>, pointer: u32, offset: u32, length: u32| -> Result<(), String> {
            let chunk = DOOM_WAD
                .get(offset as usize..offset as usize + length as usize)
                .ok_or_else(|| format!("invalid ROM read: offset = 0x{offset:x}, length = {length}"))?;

            caller.instance.write_memory(pointer, chunk).map_err(|err| err.to_string())
        },
    )?;

    linker.define_typed("ext_stdout", |_caller: polkavm::Caller<State>, _buffer: u32, length: u32| -> i32 {
        length as i32
    })?;

    Ok(linker.instantiate_pre(&module)?)
}

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
enum Category {
    Compute,
    DivMul,
    ControlFlow,
    MemoryLoad,
    MemoryStore,
    Ecalli,
}

impl core::fmt::Display for Category {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        let name = match self {
            Category::Compute => "compute",
            Category::DivMul => "div_mul",
            Category::ControlFlow => "control_flow",
            Category::MemoryLoad => "memory_load",
            Category::MemoryStore => "memory_store",
            Category::Ecalli => "ecalli",
        };

        fmt.write_str(name)
    }
}

impl Category {
    #[deny(unreachable_patterns)]
    fn from_opcode(opcode: polkavm::program::Opcode) -> Category {
        use polkavm::program::Opcode::*;
        use Category::*;
        match opcode {
            add_32 => Compute,
            add_64 => Compute,
            add_imm_32 => Compute,
            add_imm_64 => Compute,
            and => Compute,
            and_imm => Compute,
            and_inverted => Compute,
            unlikely => ControlFlow,
            branch_eq => ControlFlow,
            branch_eq_imm => ControlFlow,
            branch_greater_or_equal_signed => ControlFlow,
            branch_greater_or_equal_signed_imm => ControlFlow,
            branch_greater_or_equal_unsigned => ControlFlow,
            branch_greater_or_equal_unsigned_imm => ControlFlow,
            branch_greater_signed_imm => ControlFlow,
            branch_greater_unsigned_imm => ControlFlow,
            branch_less_or_equal_signed_imm => ControlFlow,
            branch_less_or_equal_unsigned_imm => ControlFlow,
            branch_less_signed => ControlFlow,
            branch_less_signed_imm => ControlFlow,
            branch_less_unsigned => ControlFlow,
            branch_less_unsigned_imm => ControlFlow,
            branch_not_eq => ControlFlow,
            branch_not_eq_imm => ControlFlow,
            cmov_if_not_zero => Compute,
            cmov_if_not_zero_imm => Compute,
            cmov_if_zero => Compute,
            cmov_if_zero_imm => Compute,
            count_leading_zero_bits_32 => Compute,
            count_leading_zero_bits_64 => Compute,
            count_set_bits_32 => Compute,
            count_set_bits_64 => Compute,
            count_trailing_zero_bits_32 => Compute,
            count_trailing_zero_bits_64 => Compute,
            div_signed_32 => DivMul,
            div_signed_64 => DivMul,
            div_unsigned_32 => DivMul,
            div_unsigned_64 => DivMul,
            ecalli => Ecalli,
            fallthrough => ControlFlow,
            jump => ControlFlow,
            jump_indirect => ControlFlow,
            load_i16 => MemoryLoad,
            load_i32 => MemoryLoad,
            load_i8 => MemoryLoad,
            load_imm => Compute,
            load_imm64 => Compute,
            load_imm_and_jump => ControlFlow,
            load_imm_and_jump_indirect => ControlFlow,
            load_indirect_i16 => MemoryLoad,
            load_indirect_i32 => MemoryLoad,
            load_indirect_i8 => MemoryLoad,
            load_indirect_u16 => MemoryLoad,
            load_indirect_u32 => MemoryLoad,
            load_indirect_u64 => MemoryLoad,
            load_indirect_u8 => MemoryLoad,
            load_u16 => MemoryLoad,
            load_u32 => MemoryLoad,
            load_u64 => MemoryLoad,
            load_u8 => MemoryLoad,
            maximum => Compute,
            maximum_unsigned => Compute,
            memset => todo!(),
            minimum => Compute,
            minimum_unsigned => Compute,
            move_reg => Compute,
            mul_32 => DivMul,
            mul_64 => DivMul,
            mul_imm_32 => DivMul,
            mul_imm_64 => DivMul,
            mul_upper_signed_signed => DivMul,
            mul_upper_signed_unsigned => DivMul,
            mul_upper_unsigned_unsigned => DivMul,
            negate_and_add_imm_32 => Compute,
            negate_and_add_imm_64 => Compute,
            or => Compute,
            or_imm => Compute,
            or_inverted => Compute,
            rem_signed_32 => DivMul,
            rem_signed_64 => DivMul,
            rem_unsigned_32 => DivMul,
            rem_unsigned_64 => DivMul,
            reverse_byte => Compute,
            rotate_left_32 => Compute,
            rotate_left_64 => Compute,
            rotate_right_32 => Compute,
            rotate_right_64 => Compute,
            rotate_right_imm_32 => Compute,
            rotate_right_imm_64 => Compute,
            rotate_right_imm_alt_32 => Compute,
            rotate_right_imm_alt_64 => Compute,
            sbrk => todo!(),
            set_greater_than_signed_imm => Compute,
            set_greater_than_unsigned_imm => Compute,
            set_less_than_signed => Compute,
            set_less_than_signed_imm => Compute,
            set_less_than_unsigned => Compute,
            set_less_than_unsigned_imm => Compute,
            shift_arithmetic_right_32 => Compute,
            shift_arithmetic_right_64 => Compute,
            shift_arithmetic_right_imm_32 => Compute,
            shift_arithmetic_right_imm_64 => Compute,
            shift_arithmetic_right_imm_alt_32 => Compute,
            shift_arithmetic_right_imm_alt_64 => Compute,
            shift_logical_left_32 => Compute,
            shift_logical_left_64 => Compute,
            shift_logical_left_imm_32 => Compute,
            shift_logical_left_imm_64 => Compute,
            shift_logical_left_imm_alt_32 => Compute,
            shift_logical_left_imm_alt_64 => Compute,
            shift_logical_right_32 => Compute,
            shift_logical_right_64 => Compute,
            shift_logical_right_imm_32 => Compute,
            shift_logical_right_imm_64 => Compute,
            shift_logical_right_imm_alt_32 => Compute,
            shift_logical_right_imm_alt_64 => Compute,
            sign_extend_16 => Compute,
            sign_extend_8 => Compute,
            store_imm_indirect_u16 => MemoryStore,
            store_imm_indirect_u32 => MemoryStore,
            store_imm_indirect_u64 => MemoryStore,
            store_imm_indirect_u8 => MemoryStore,
            store_imm_u16 => MemoryStore,
            store_imm_u32 => MemoryStore,
            store_imm_u64 => MemoryStore,
            store_imm_u8 => MemoryStore,
            store_indirect_u16 => MemoryStore,
            store_indirect_u32 => MemoryStore,
            store_indirect_u64 => MemoryStore,
            store_indirect_u8 => MemoryStore,
            store_u16 => MemoryStore,
            store_u32 => MemoryStore,
            store_u64 => MemoryStore,
            store_u8 => MemoryStore,
            sub_32 => Compute,
            sub_64 => Compute,
            trap => ControlFlow,
            xnor => Compute,
            xor => Compute,
            xor_imm => Compute,
            zero_extend_16 => Compute,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TestcaseJson {
    name: String,
    program: Vec<u8>,
    block_gas_costs: BTreeMap<u32, i64>,
}

fn main_test_cost_model(
    cost_model_path: Option<PathBuf>,
    cache_path: PathBuf,
    benchmark: Benchmark,
    analyze: bool,
    output_testcase: Option<PathBuf>,
) -> Result<(), String> {
    std::fs::create_dir_all(&cache_path).map_err(|error| format!("failed to create '{}': {error}", cache_path.display()))?;

    let mut cost_model: CostModelKind = CostModelKind::Simple(CostModel::naive_ref());
    let mut is_cost_model_full = false;
    if let Some(cost_model_path) = cost_model_path {
        let name = cost_model_path.to_str().unwrap();
        if name == "full-l1-hit" {
            cost_model = CostModelKind::Full(CacheModel::L1Hit);
            is_cost_model_full = true;
        } else if name == "full-l2-hit" {
            cost_model = CostModelKind::Full(CacheModel::L2Hit);
            is_cost_model_full = true;
        } else if name == "full-l3-hit" {
            cost_model = CostModelKind::Full(CacheModel::L3Hit);
            is_cost_model_full = true;
        } else {
            log::info!("Loading cost model from: '{}'", cost_model_path.display());
            let blob =
                std::fs::read(&cost_model_path).map_err(|error| format!("failed to read {}: {}", cost_model_path.display(), error))?;
            if let Some(map) = core::str::from_utf8(&blob).ok().and_then(|blob| serde_json::from_str(blob).ok()) {
                let map: BTreeMap<String, u32> = map;
                cost_model = Arc::new(
                    deserialize_cost_model_from_map(map)
                        .map_err(|error| format!("failed to parse the cost model in {}: {error}", cost_model_path.display()))?,
                )
                .into();
            } else if let Some(new_cost_model) = polkavm::CostModel::deserialize(&blob) {
                cost_model = Arc::new(new_cost_model).into();
            } else {
                return Err(format!("failed to parse the cost model in {}", cost_model_path.display()));
            }
        }
    }

    let blob_path = match benchmark {
        Benchmark::Doom => {
            let doom_blob_path = cache_path.join("doom64.polkavm");
            if !doom_blob_path.exists() {
                std::fs::create_dir_all(&cache_path).map_err(|error| format!("failed to create {}: {}", cache_path.display(), error))?;
                log::info!("Decompressing ELF file...");
                let elf = decompress_zstd(include_bytes!("../../../test-data/doom_64.elf.zst"));
                log::info!("Linking...");
                let mut config = polkavm_linker::Config::default();
                config.set_optimize(true);
                config.set_strip(true);
                let blob = polkavm_linker::program_from_elf(config, &elf)?;
                std::fs::write(&doom_blob_path, &blob)
                    .map_err(|error| format!("failed to write {}: {}", doom_blob_path.display(), error))?;
                log::info!("Blob linked!");
            }

            doom_blob_path
        }
        Benchmark::Pinky | Benchmark::PrimeSieve => {
            let cached_blob_path = cache_path.join(format!("{}.polkavm", benchmark.name()));
            if !cached_blob_path.exists() {
                let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("..")
                    .join("guest-programs")
                    .join("target")
                    .join("riscv64emac-unknown-none-polkavm")
                    .join("release");

                let elf_path = root.join(format!("bench-{}", benchmark.name()));
                let elf = std::fs::read(&elf_path).map_err(|error| format!("failed to read {}: {}", elf_path.display(), error))?;
                log::info!("Linking...");
                let mut config = polkavm_linker::Config::default();
                config.set_optimize(true);
                config.set_strip(true);
                let blob = polkavm_linker::program_from_elf(config, &elf)?;
                std::fs::write(&cached_blob_path, &blob)
                    .map_err(|error| format!("failed to write {}: {}", cached_blob_path.display(), error))?;
                log::info!("Blob linked!");
            }

            cached_blob_path
        }
    };

    let blob = std::fs::read(&blob_path).map_err(|error| format!("failed to read {}: {}", blob_path.display(), error))?;
    let parts = ProgramParts::from_bytes(blob.into())?;
    let blob = ProgramBlob::from_parts(parts.clone())?;

    let mut offset_to_opcode = HashMap::new();
    let mut offset_to_basic_block_number = HashMap::new();
    let mut basic_block_to_offset = Vec::new();
    let mut instruction_counts: BTreeMap<polkavm::program::Opcode, u64> = BTreeMap::new();
    let mut pc_counts: BTreeMap<ProgramCounter, u64> = BTreeMap::new();
    let (ext_initialize, ext_run) = match benchmark {
        Benchmark::Doom => ("ext_initialize", "ext_tick"),
        _ => ("initialize", "run"),
    };

    if let Some(output_path) = output_testcase {
        let mut config = polkavm::Config::from_env()?;
        config.set_allow_experimental(true);
        config.set_backend(Some(polkavm::BackendKind::Interpreter));
        let engine = polkavm::Engine::new(&config)?;
        let instance_pre = prepare_benchmark(benchmark, &engine, &blob, false)?;
        let module = instance_pre.module();
        let mut block_gas_costs = BTreeMap::new();
        let mut is_new_block = true;
        for instruction in blob.instructions_bounded_at(polkavm::program::ISA64_V1, ProgramCounter(0)) {
            if is_new_block {
                let cost = module.calculate_gas_cost_for(instruction.offset).unwrap();
                block_gas_costs.insert(instruction.offset.0, cost);
            }

            is_new_block = instruction.starts_new_basic_block();
        }

        let json = TestcaseJson {
            name: match benchmark {
                Benchmark::Doom => "doom",
                Benchmark::Pinky => "pinky",
                Benchmark::PrimeSieve => "prime-sieve",
            }
            .to_owned(),
            program: parts.code_and_jump_table.to_vec(),
            block_gas_costs,
        };

        let payload = serde_json::to_string(&json).unwrap();
        if !std::fs::read(&output_path)
            .map(|old_payload| old_payload == payload.as_bytes())
            .unwrap_or(false)
        {
            log::info!("Generating {output_path:?}...");
            std::fs::write(output_path, payload).unwrap();
        }

        return Ok(());
    }

    if analyze {
        basic_block_to_offset.push(ProgramCounter(0));
        for instruction in blob.instructions(polkavm::program::ISA64_V1) {
            offset_to_opcode.insert(instruction.offset, instruction.opcode());
            offset_to_basic_block_number.insert(instruction.offset, basic_block_to_offset.len() - 1);
            if instruction.opcode().starts_new_basic_block() {
                basic_block_to_offset.push(instruction.next_offset);
            }
        }

        let instruction_counts_path = cache_path.join(format!("{}-instruction_counts.json", benchmark.name()));
        let pc_counts_path = cache_path.join(format!("{}-pc_counts.json", benchmark.name()));
        if !instruction_counts_path.exists() || !pc_counts_path.exists() {
            if cfg!(debug_assertions) {
                log::error!("ERROR: missing instruction counts; rerun in release mode!");
                std::process::exit(1);
            }
            log::info!("Gathering instruction counts...");
            let mut config = polkavm::Config::from_env()?;
            config.set_allow_experimental(true);
            config.set_backend(Some(polkavm::BackendKind::Interpreter));
            let engine = polkavm::Engine::new(&config)?;
            let instance_pre = prepare_benchmark(benchmark, &engine, &blob, true)?;

            let mut instance = instance_pre.instantiate()?;
            let mut state = State { frames: 0 };
            let mut result = instance.call_typed(&mut state, ext_initialize, ());
            let mut pc_counts: HashMap<u32, u64> = HashMap::new();
            loop {
                if let Some(pc) = instance.program_counter() {
                    *pc_counts.entry(pc.0).or_insert(0) += 1;
                }
                match result {
                    Ok(()) => break,
                    Err(polkavm::CallError::Step) => {
                        result = instance.continue_execution(&mut state);
                    }
                    Err(error) => panic!("unexpected error: {error:?}"),
                }
            }

            for _ in 0..LOOPS {
                result = instance.call_typed(&mut state, ext_run, ());
                loop {
                    if let Some(pc) = instance.program_counter() {
                        *pc_counts.entry(pc.0).or_insert(0) += 1;
                    }
                    match result {
                        Ok(()) => break,
                        Err(polkavm::CallError::Step) => {
                            result = instance.continue_execution(&mut state);
                        }
                        Err(error) => panic!("unexpected error: {error:?}"),
                    }
                }
            }

            let output = serde_json::to_string_pretty(&pc_counts).unwrap();
            std::fs::write(&pc_counts_path, output).map_err(|error| format!("failed to write {}: {}", pc_counts_path.display(), error))?;

            let mut counts = BTreeMap::new();
            for (location, count) in pc_counts {
                let opcode = *offset_to_opcode.get(&ProgramCounter(location)).unwrap();
                *counts.entry(opcode).or_insert(0) += count;
            }

            let counts: BTreeMap<_, _> = counts.into_iter().map(|(opcode, count)| (opcode.to_string(), count)).collect();

            let output = serde_json::to_string_pretty(&counts).unwrap();
            std::fs::write(&instruction_counts_path, output)
                .map_err(|error| format!("failed to write {}: {}", instruction_counts_path.display(), error))?;
        }

        instruction_counts = {
            let instruction_counts: String = std::fs::read_to_string(&instruction_counts_path)
                .map_err(|error| format!("failed to read {}: {}", instruction_counts_path.display(), error))?;
            let instruction_counts: BTreeMap<String, u64> = serde_json::from_str(&instruction_counts)
                .map_err(|error| format!("failed to parse {}: {}", instruction_counts_path.display(), error))?;
            let mut out = BTreeMap::new();
            for (opcode, count) in instruction_counts {
                let opcode = opcode
                    .parse()
                    .map_err(|error| format!("failed to parse {}: {}", instruction_counts_path.display(), error))?;
                out.insert(opcode, count);
            }
            out
        };

        pc_counts = {
            let pc_counts: String = std::fs::read_to_string(&pc_counts_path)
                .map_err(|error| format!("failed to read {}: {}", pc_counts_path.display(), error))?;

            let pc_counts: BTreeMap<u32, u64> =
                serde_json::from_str(&pc_counts).map_err(|error| format!("failed to parse {}: {}", pc_counts_path.display(), error))?;

            pc_counts.into_iter().map(|(pc, count)| (ProgramCounter(pc), count)).collect()
        };
    };

    restart_with_sudo_or_exit();

    let mut tweaks = crate::system::Tweaks::default();
    log::info!("Disabling turbo boost...");
    tweaks.write("/sys/devices/system/cpu/cpufreq/boost", "0")?;

    log::info!("Setting the frequency governor to 'performance'...");
    for cpu in crate::system::list_cpus()?
        .values()
        .flatten()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
    {
        tweaks.write(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"), "performance")?;
    }

    log::info!("Instruction counts:");
    for (&opcode, &count) in &instruction_counts {
        log::info!("  {}: {}", opcode, count);
    }

    log::info!("Preparing module...");
    let mut config = polkavm::Config::from_env()?;
    config.set_allow_experimental(true);
    config.set_backend(Some(polkavm::BackendKind::Compiler));
    config.set_default_cost_model(Some(cost_model));
    let compiler_engine = polkavm::Engine::new(&config)?;
    let instance_pre = prepare_benchmark(benchmark, &compiler_engine, &blob, false)?;

    const INITIAL_GAS: i64 = 100_000_000_000_000;
    const LOOPS: usize = 100;

    log::info!("Running warmup...");
    let cost_model;
    let module;
    {
        let mut instance = instance_pre.instantiate()?;
        instance.set_gas(INITIAL_GAS);
        let mut state = State { frames: 0 };
        instance.call_typed(&mut state, ext_initialize, ()).unwrap();
        for _ in 0..LOOPS {
            instance.call_typed(&mut state, ext_run, ()).unwrap();
        }
        cost_model = instance.module().cost_model().clone();
        module = instance.module().clone();
    }

    log::info!("Running...");
    let mut elapsed_samples = Vec::with_capacity(16);
    let mut final_gas = 0;
    for _ in 0..17 {
        let mut instance = instance_pre.instantiate()?;
        instance.set_gas(INITIAL_GAS);

        let timestamp = std::time::Instant::now();
        let mut state = State { frames: 0 };
        instance.call_typed(&mut state, ext_initialize, ()).unwrap();
        for _ in 0..LOOPS {
            instance.call_typed(&mut state, ext_run, ()).unwrap();
        }
        let elapsed = timestamp.elapsed();
        elapsed_samples.push(elapsed.as_nanos());
        final_gas = instance.gas();
    }

    elapsed_samples.sort();

    let raw_gas_consumed = INITIAL_GAS - final_gas;
    let gas_time_consumed = if is_cost_model_full {
        let frequency = get_frequency(0)?;
        ((raw_gas_consumed as f64 / frequency as f64) * 1_000_000_000_000_f64) as i64
    } else {
        raw_gas_consumed
    };

    log::info!(
        "Gas (raw) consumed: {:>19} ({} -> {})",
        pretty_print(raw_gas_consumed),
        pretty_print(INITIAL_GAS),
        pretty_print(final_gas)
    );

    if is_cost_model_full {
        log::info!("Gas (time) consumed: {:>19}", pretty_print(gas_time_consumed),);
    }

    let wall_time_consumed = elapsed_samples[elapsed_samples.len() / 2] as f64 * 1000.0;
    log::info!("Time expected:       {:>19}", pretty_print(wall_time_consumed as i64));
    log::info!("Overhead: {:.02}x", gas_time_consumed as f64 / wall_time_consumed);

    if analyze {
        let mut format = InstructionFormat::default();
        format.is_64_bit = true;

        if let CostModelKind::Simple(cost_model) = cost_model {
            let mut instruction_counts: Vec<_> = instruction_counts
                .into_iter()
                .map(|(opcode, count)| {
                    let cost = count * u64::from(cost_model.cost_for_opcode(opcode));
                    (opcode, cost, count)
                })
                .collect();

            instruction_counts.sort_by_key(|(_, cost, _)| core::cmp::Reverse(*cost));

            let mut category_counts = BTreeMap::new();
            for &(opcode, cost, count) in &instruction_counts {
                let category = Category::from_opcode(opcode);
                let (ref mut acc_cost, ref mut acc_count) = category_counts.entry(category).or_insert((0, 0));
                *acc_cost += cost;
                *acc_count += count;
            }

            let mut category_counts: Vec<_> = category_counts
                .into_iter()
                .map(|(opcode, (cost, count))| (opcode, cost, count))
                .collect();
            category_counts.sort_by_key(|(_, cost, _)| core::cmp::Reverse(*cost));

            let mut pc_costs: Vec<_> = pc_counts
                .iter()
                .map(|(&pc, &count)| {
                    let opcode = offset_to_opcode.get(&pc).unwrap();
                    let cost = count * u64::from(cost_model.cost_for_opcode(*opcode));
                    (pc, count, cost)
                })
                .collect();

            pc_costs.sort_by_key(|(_, _, cost)| core::cmp::Reverse(*cost));

            let mut basic_block_costs = vec![0; basic_block_to_offset.len()];
            for (&pc, &count) in &pc_counts {
                let basic_block_number = *offset_to_basic_block_number.get(&pc).unwrap();
                let opcode = offset_to_opcode.get(&pc).unwrap();
                let cost = count * u64::from(cost_model.cost_for_opcode(*opcode));
                basic_block_costs[basic_block_number] += cost;
            }
            let mut basic_block_costs: Vec<_> = basic_block_costs.into_iter().enumerate().collect();
            basic_block_costs.retain(|(_, cost)| *cost > 0);
            basic_block_costs.sort_by_key(|(_, cost)| core::cmp::Reverse(*cost));

            log::info!("Top costly instruction categories:");
            for (category, cost, count) in category_counts {
                log::info!(
                    "  {:>36} {:>6} {:>18} ({:>12})",
                    category.to_string(),
                    format!("{:.02}%", (cost as f64 / raw_gas_consumed as f64) * 100.0),
                    pretty_print(cost as i64),
                    pretty_print(count as i64),
                );
            }

            log::info!("Top costly instructions:");
            for (opcode, cost, count) in instruction_counts {
                log::info!(
                    "  {:>36} {:>6} {:>18} ({:>12} x {:>9})",
                    opcode.to_string(),
                    format!("{:.02}%", (cost as f64 / raw_gas_consumed as f64) * 100.0),
                    pretty_print(cost as i64),
                    pretty_print(count as i64),
                    pretty_print(u64::from(cost_model.cost_for_opcode(opcode)) as i64),
                );
            }

            log::info!("Top costly locations:");
            for &(pc, count, cost) in pc_costs.iter().take(100) {
                let opcode = offset_to_opcode.get(&pc).unwrap();
                log::info!(
                    "  {:>6}: {:>15} ({:>11} x {:>6}): {}",
                    pc.0,
                    pretty_print(cost as i64),
                    pretty_print(count as i64),
                    pretty_print(u64::from(cost_model.cost_for_opcode(*opcode)) as i64),
                    blob.instructions_bounded_at(polkavm::program::ISA64_V1, pc)
                        .next()
                        .unwrap()
                        .display(&format),
                );
            }

            let pc_costs: HashMap<_, _> = pc_costs.into_iter().map(|(pc, count, cost)| (pc, (count, cost))).collect();

            log::info!("Top costly basic blocks:");
            for (basic_block_number, cost) in basic_block_costs.into_iter().take(20) {
                let pc = basic_block_to_offset[basic_block_number];
                log::info!(
                    "  @{}: {:>15} ({:.02}%) = {} x {}",
                    basic_block_number,
                    pretty_print(cost as i64),
                    (cost as f64 / raw_gas_consumed as f64) * 100.0,
                    instance_pre.module().calculate_gas_cost_for(pc).unwrap(),
                    pc_counts.get(&pc).unwrap(),
                );

                for instruction in blob.instructions_bounded_at(polkavm::program::ISA64_V1, pc) {
                    log::info!(
                        "    {:>6}: {:>15}: {}",
                        instruction.offset,
                        pretty_print(pc_costs.get(&instruction.offset).copied().unwrap().1 as i64),
                        instruction.display(&format)
                    );
                    if instruction.starts_new_basic_block() {
                        break;
                    }
                }
            }
        } else {
            let mut basic_block_costs: Vec<_> = basic_block_to_offset
                .iter()
                .enumerate()
                .map(|(basic_block_number, &offset)| {
                    let cost_per_one = module.calculate_gas_cost_for(offset).unwrap();
                    let count = pc_counts.get(&offset).copied().unwrap_or(0);
                    let cost = cost_per_one * count as i64;
                    (basic_block_number, offset, cost, cost_per_one, count)
                })
                .collect();

            basic_block_costs.retain(|(_, _, cost, _, _)| *cost > 0);
            basic_block_costs.sort_by_key(|(_, _, cost, _, _)| core::cmp::Reverse(*cost));

            log::info!("Top costly basic blocks:");
            for (basic_block_number, pc, cost, cost_per_one, count) in basic_block_costs.into_iter().take(20) {
                log::info!(
                    "  @{}: {:>15} ({:.02}%) = {} x {}",
                    basic_block_number,
                    pretty_print(cost),
                    (cost as f64 / raw_gas_consumed as f64) * 100.0,
                    cost_per_one,
                    count,
                );

                let mut block_instructions = Vec::new();
                for instruction in blob.instructions_bounded_at(polkavm::program::ISA64_V1, pc) {
                    block_instructions.push(instruction);

                    log::info!("    {:>6}: {}", instruction.offset, instruction.display(&format));
                    if instruction.starts_new_basic_block() {
                        break;
                    }
                }

                if let CostModelKind::Full(cost_model) = cost_model {
                    let (timeline, cycles) = polkavm_common::simulator::timeline_for_instructions(
                        blob.code(),
                        cost_model,
                        &block_instructions,
                        polkavm_common::simulator::TimelineConfig::default(),
                    );
                    log::info!("  Timeline ({cycles} cycles):");
                    for line in timeline.split("\n") {
                        log::info!("    {line}");
                    }
                }
            }
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
struct IsolationArgs {
    /// The CPU on which to run.
    #[clap(short = 'c', long, default_value_t = 8)]
    cpu: usize,

    // Reserve a whole CCD. (Use `lstopo` to see your CPU's topology.)
    /// The CPUs which to isolate.
    #[clap(short = 'r', long, default_value = "8,9,10,11,12,13,14,15", value_delimiter = ',')]
    reserve_cpus: Vec<usize>,

    /// Whether to minimize the measurements' noise by applying various system level tweaks.
    #[clap(
        long,
        default_missing_value("true"),
        default_value("true"),
        num_args(0..=1),
        require_equals(true),
        action = clap::ArgAction::Set,
    )]
    minimize_noise: bool,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Benchmark {
    Doom,
    Pinky,
    PrimeSieve,
}

impl Benchmark {
    fn name(self) -> &'static str {
        match self {
            Benchmark::Doom => "doom",
            Benchmark::Pinky => "pinky",
            Benchmark::PrimeSieve => "prime-sieve",
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version)]
enum Args {
    GenerateModel {
        #[clap(flatten)]
        isolation_args: IsolationArgs,

        #[clap(long)]
        output_model_code: Option<PathBuf>,

        #[clap(long)]
        output_model_blob: Option<PathBuf>,

        #[clap(long)]
        output_model_json: Option<PathBuf>,
    },
    BenchmarkMaximumCacheMisses {
        #[clap(flatten)]
        isolation_args: IsolationArgs,
    },
    BenchmarkRandomCacheMisses {
        #[clap(flatten)]
        isolation_args: IsolationArgs,
    },
    BenchmarkOptimistic {
        #[clap(flatten)]
        isolation_args: IsolationArgs,
    },
    CacheMissBenchmark {
        #[clap(flatten)]
        isolation_args: IsolationArgs,

        #[clap(long, short = 'o')]
        output_chart: Option<PathBuf>,
    },
    TestCostModel {
        #[clap(long)]
        cost_model: Option<PathBuf>,

        #[clap(long)]
        cache_path: PathBuf,

        #[clap(long)]
        benchmark: Benchmark,

        #[clap(long)]
        do_not_analyze: bool,

        #[clap(long)]
        output_testcase: Option<PathBuf>,
    },
}

extern "C" fn on_sigint(_signal: libc::c_int) {
    RUNNING.store(false, Ordering::Relaxed);
}

fn setup_signal_handler() {
    #[allow(clippy::fn_to_numeric_cast_any)]
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as libc::sighandler_t)
    };
}

fn main() {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "info");
    }

    let _ = env_logger::try_init();
    let args = Args::parse();

    let result = match args {
        Args::BenchmarkMaximumCacheMisses { isolation_args } => {
            main_benchmark_maximum_cache_misses(isolation_args).map_err(|error| error.to_string())
        }
        Args::BenchmarkRandomCacheMisses { isolation_args } => {
            main_benchmark_random_cache_misses(isolation_args).map_err(|error| error.to_string())
        }
        Args::BenchmarkOptimistic { isolation_args } => main_benchmark_optimistic(isolation_args).map_err(|error| error.to_string()),
        Args::GenerateModel {
            isolation_args,
            output_model_code,
            output_model_blob,
            output_model_json,
        } => {
            main_generate_model(isolation_args, output_model_code, output_model_blob, output_model_json).map_err(|error| error.to_string())
        }
        Args::CacheMissBenchmark {
            isolation_args,
            output_chart,
        } => main_cache_miss_benchmark(isolation_args, output_chart).map_err(|error| error.to_string()),
        Args::TestCostModel {
            cost_model,
            cache_path,
            benchmark,
            do_not_analyze,
            output_testcase,
        } => main_test_cost_model(cost_model, cache_path, benchmark, !do_not_analyze, output_testcase),
    };

    if let Err(error) = result {
        log::error!("ERROR: {error}");
        std::process::exit(1);
    }
}
