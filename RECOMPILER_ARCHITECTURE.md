# Recompiler architecture

The "recompiler" (a.k.a. compiler) translates PolkaVM guest bytecode into native machine code at module-load time. It targets x86-64 today, and runs the compiled code inside one of two sandboxes (Linux or generic). Below is an end-to-end walkthrough of how that pipeline is wired together.

## 1. Big picture

```
ProgramBlob ──► Module::from_blob ──► compile_module! ──► CompilerVisitor (ParsingVisitor)
                                                            │
                                                            │  per-instruction emit via ArchVisitor (amd64.rs)
                                                            ▼
                                                       Assembler buffer
                                                            │  finish_compilation: fixups, gas costs, trampolines
                                                            ▼
                                                       CompiledModule<S>
                                                            │  spawn_and_load_module
                                                            ▼
                                                  Sandbox instance (Linux | generic)
                                                            │  run() → futex/jump into native code → InterruptKind
                                                            ▼
                                                     Caller (host)
```

Three layers cooperate:

1. **Front-end driver** (`crates/polkavm/src/api.rs`, `compiler.rs`) — owns the compilation lifecycle and the instruction-visitor loop.
2. **Back-end codegen** (`compiler/amd64.rs` + `polkavm-assembler`) — knows about x86-64 instruction encoding, the register map, and the calling-convention boilerplate.
3. **Sandbox runtime** (`sandbox.rs`, `sandbox/{linux,generic}.rs`, `polkavm-zygote`) — owns the address space the compiled code runs in, page-fault recovery, and the host↔guest control transfer.

## 2. The driver loop

Entry points are `Module::new` / `Module::from_blob` in `crates/polkavm/src/api.rs:397,403`. After validating the blob and consulting the module cache, the macro `compile_module!` at `api.rs:525` picks the concrete generic instantiation — `(Sandbox = Linux | Generic) × (Bitness = 32 | 64) × (GasVisitor = simple | Simulator)` — and:

1. Instantiates `CompilerVisitor<'a, S, B, G>` (`compiler.rs:71`), passing the assembler, gas visitor, and program metadata.
2. Calls `blob.visit(&mut visitor)`, which iterates every instruction in the blob and dispatches to the visitor's `ParsingVisitor` impl. Each guest opcode (e.g. `add_32`, `load_u32`, `branch_eq`, `ecalli`) becomes one method call.
3. Finalizes with `CompilerVisitor::finish_compilation` (`compiler.rs:308`) — patches forward jumps, lays out the trampolines, records per-block gas costs and the PC↔native-offset map, and emits the final `CompiledModule<S>`.

Important state held on `CompilerVisitor`:

| Field | Purpose |
| --- | --- |
| `asm: Assembler` | x86-64 instruction buffer (`polkavm-assembler`) |
| `program_counter_to_label: FlatMap` | guest PC → assembler `Label` |
| `gas_visitor: G` | per-instruction cost accumulator |
| pre-allocated labels (`ecall_label`, `trap_label`, `sbrk_label`, `memset_label`, `divrem_label`, `step_label`) | targets for shared trampolines emitted once at the end |

The fact that every handler is dispatched through a single `ParsingVisitor` impl is what lets the compiler stay relatively small (~2 kloc of x86 codegen for the whole ISA): nothing decodes bytecode separately — the upstream `polkavm-common` parser hands typed operands directly to each handler.

## 3. Per-instruction code generation

The actual machine-code emission lives in `crates/polkavm/src/compiler/amd64.rs`. It implements `ArchVisitor` for `CompilerVisitor` and adds an x86-64-specific layer on top of `polkavm-assembler`.

**Register map.** `REG_MAP[16]` at `amd64.rs:52-69` pins each of the 13 guest registers onto a native register; the conversion is `conv_reg(RawReg) → polkavm_common::regmap::to_native_reg`. Two native registers are reserved as scratch (`TMP_REG`, `AUX_TMP_REG`); on the generic sandbox `AUX_TMP_REG` doubles as the guest-memory base pointer (`GENERIC_SANDBOX_MEMORY_REG`).

**Operand addressing.** A macro `load_store_operand!` at `amd64.rs:95-148` expands every load/store into either:
- a direct `[base + disp]` form on Linux (because guest memory sits at a known offset inside the same address space the compiled code lives in), or
- a `[memory_reg + disp]` form on the generic sandbox (where guest memory is reached through an explicit base pointer at `vmctx_ptr - 4096`, see `generic.rs:120`).

This is the only divergence in codegen between the two sandboxes — most handlers are sandbox-agnostic.

**Representative handlers.**

- **Arithmetic.** `add_32` (`amd64.rs:1558`) → `add_generic` (`amd64.rs:1530`): reserve two assembler slots, convert operands, emit `add(RegSize::R32, d, s1, s2)`. Multi-operand arithmetic always reuses the same three-step pattern: reserve, convert, emit.
- **Memory.** `load` (`amd64.rs:345`) expands `load_store_operand!`, then emits a single `mov` of the right width (U8/I8/…/U64). Sign-/zero-extension is part of the load kind itself.
- **Control flow.** `branch` (`amd64.rs:483`) converts both operands, looks up (or lazily creates) the label for the target PC in `program_counter_to_label`, and calls `branch_to_label` (`amd64.rs:180`), which emits a conditional jump and records a `Fixup` for the assembler to resolve at the end.
- **Hostcall.** `ecalli` (`amd64.rs:1354`) stores the current guest PC, the call number and the next PC into the VM context, then `call`s the shared `ecall_label` trampoline. The trampoline (`emit_ecall_trampoline`, `amd64.rs:625`) saves the registers into the VM context, flips the futex to `VMCTX_FUTEX_GUEST_ECALLI`, and parks on it until the host either resumes execution or unwinds via `VMCTX_FUTEX_LONGJUMP`.

**Basic blocks and gas metering.** Block boundaries are managed by `force_start_new_basic_block` (`compiler.rs:470`) and `after_instruction` (`compiler.rs:500-539`). At each block start, when gas metering is on, `emit_gas_metering_stub` (`amd64.rs:942-978`) emits:

```
sub qword [vmctx + gas_offset], i32::MAX   ; the literal is patched to the real block cost
```

In synchronous metering mode, the stub additionally encodes a short backward branch into an illegal opcode so that gas-exhaustion turns into a SIGILL the sandbox can catch. The real per-block cost is written at the end of compilation, when `gas_visitor.take_block_cost()` is called.

**Invalid jumps.** Jump-table entries that don't correspond to a real target are stamped with `JUMP_TABLE_INVALID_ADDRESS = 0xfa6f29540376ba8a` (`compiler.rs:42`) — an address that's guaranteed not to map. Jumping there segfaults, and the sandbox turns the segfault into `InterruptKind::Trap` rather than killing the process.

## 4. The assembler (`polkavm-assembler`)

The recompiler does not synthesise bytes itself; it drives `polkavm-assembler::Assembler` (`crates/polkavm-assembler/src/assembler.rs:12`). `Assembler` holds:

- `code: Vec<u8>` — the emitted bytes,
- `labels: Vec<isize>` — resolved label offsets,
- `fixups: Vec<Fixup>` — forward references waiting to be patched.

A notable design point is the **reserved-slot pattern**: `asm.reserve::<U2>()` returns a `ReservedAssembler<U2>` typed by the number of slots remaining. Each `push(inst)` consumes one slot statically (returns `U1`, then `U0`); `assert_reserved_exactly_as_needed()` ensures every handler pushes exactly what it reserved. That removes a class of "forgot to reserve enough room" bugs from the codegen and lets the assembler keep a single contiguous buffer (no resizing surprises in the middle of an instruction).

Labels declared up-front in `CompilerVisitor::new` (the trampolines) and on-demand at branch targets are resolved during `finish_compilation`.

## 5. Sandbox abstraction

`crates/polkavm/src/sandbox.rs` defines the `Sandbox` trait (`sandbox.rs:88`) along with the helper traits `SandboxConfig`, `SandboxAddressSpace`, and `SandboxProgram`. The two concrete impls — `sandbox/linux.rs` and `sandbox/generic.rs` — differ enormously in implementation but expose the same surface:

- `spawn(global, config, outer)` — produce a new execution context.
- `load_module(program)` — make compiled code visible in this context.
- `run()` → `InterruptKind` — execute from `next_program_counter` until the guest hits a hostcall, trap, breakpoint, or runs out of gas.
- Accessors for guest registers, memory, gas, and PC.

Above the trait, `Instance::spawn_and_load_module` (`sandbox.rs:163`) does the actual wiring: pick the right sandbox kind based on `Config`, hand it the `CompiledModule`, and hold onto the resulting box for the lifetime of the instance.

### 5.1 Linux sandbox — zygote + userfaultfd

This is the fast path on Linux x86-64 and is structurally the most interesting piece of the system.

**Zygote process.** The polkavm-zygote crate (`crates/polkavm-zygote`) builds a tiny standalone binary. The Linux sandbox embeds that binary (`sandbox/linux.rs:799`) and spawns it via `clone`/`execve`. The zygote sets up its address space — code region at `VM_ADDR_NATIVE_CODE`, guest memory, jump table, a shared `VmCtx` (from `polkavm_common::zygote`) holding atomic registers, gas, PC and a futex — then parks itself on the futex waiting for the host to drop work into shared memory. Pre-initialising the zygote once amortises page-table / mmap setup across many spawns.

**Per-instance spawn.** `Sandbox::spawn` (`sandbox/linux.rs:1630`) clones the zygote (copy-on-write) for a new instance. Mapping a freshly compiled module is then mostly an mmap into the shared code region.

**Userfaultfd for dynamic paging.** When dynamic paging is enabled, guest pages are not mapped up front. Instead the host registers the guest memory region with userfaultfd (`linux.rs:2071-2081`, kernel ≥ 6.8). A guest access to an unmapped page traps into the kernel, the kernel forwards the fault to the host process, the host calls `crate::compiler::on_page_fault::<Self>` (`linux.rs:2776`) to either materialise the page (`uffdio_zeropage` at `linux.rs:2868`, `uffdio_writeprotect` at `linux.rs:2884`) or convert the fault into `InterruptKind::Segfault`. The host then resumes the guest. This avoids syscalls on the hot path and gives us copy-on-write/zero-on-demand semantics for free.

This is the reason the `unprivileged_userfaultfd` sysctl shows up in CI — without it, the worker can't register the fd and the recompiler tests fail. The interpreter doesn't use it, which is why the musttail CI job doesn't need that step.

**Run loop.** `Sandbox::run` (`linux.rs:2123`) sets `next_native_program_counter` in `VmCtx`, flips the futex from `IDLE` to `BUSY`, and wakes the worker. The worker executes compiled code directly — no per-instruction syscalls. When the guest needs to talk to the host (hostcall, trap, gas exhaustion, page fault, step), the trampolines in the emitted code flip the futex to one of `VMCTX_FUTEX_GUEST_{ECALLI,SIGNAL,TRAP,NOT_ENOUGH_GAS,STEP,PAGEFAULT}` and park (`linux.rs:2695-2752`). The host's `run` loop wakes, inspects the state and returns the corresponding `InterruptKind`. `VMCTX_FUTEX_LONGJUMP` is used to forcibly abort the guest when the host wants to bail out.

### 5.2 Generic sandbox — same-process signal handling

`sandbox/generic.rs` is the portable fallback (macOS, FreeBSD, Windows in some configurations). It does not fork a separate worker: compiled code runs in the host process, called synchronously. Guest memory sits at a fixed offset from `VmCtx` (`GUEST_MEMORY_TO_VMCTX_OFFSET = -4096`, `generic.rs:120`), reached through the dedicated memory register described above.

Faults and traps use POSIX signals — SIGSEGV for out-of-bounds memory, SIGILL for the gas-overflow trick — caught by signal handlers that convert the context into an `InterruptKind`. This is slower than userfaultfd on Linux (signal delivery and longjmp-style recovery is expensive) but does not require any OS-specific features and is gated behind the `generic-sandbox` Cargo feature.

## 6. Caching: `CompiledModule` and the module cache

`CompiledModule<S>` (`compiler.rs:62-69`) is the artefact handed to a sandbox. It carries:

- `sandbox_program: S::Program` — the assembled bytes plus any sandbox-specific descriptors,
- `native_code_origin: u64`,
- `program_counter_to_machine_code_offset_list` (sorted) and `…_map` (HashMap) — bidirectional guest-PC ↔ native-offset for resumption, breakpoints, and exports,
- `gas_metering_stub_offsets: Vec<u32>` — where every per-block gas stub lives, used by the signal/page-fault recovery path to read the block's cost and update the gas counter.

`crates/polkavm/src/module_cache.rs` keeps two tiers — an active `BTreeMap<ModuleKey, Weak<…>>` of modules still alive somewhere, plus an LRU of evictable ones. The cache key is hashed from `ModuleConfig` plus the blob's content. `Module::from_blob` consults the cache before compiling (`api.rs:450-455`), which is the reason loading the same blob twice is effectively free.

## 7. Gas metering, end to end

Three pieces collaborate:

1. **Compile-time accounting.** `gas_visitor: G` is called for every instruction. `G` is either a cheap per-instruction adder (`GasVisitor`) or the full `Simulator` cost model.
2. **Stub emission.** At each basic-block start, `emit_gas_metering_stub` writes a `sub qword [vmctx+gas_offset], i32::MAX`. The literal is a placeholder; once the block is closed and `gas_visitor.take_block_cost()` is called, the real cost is patched into the stub via the offsets recorded in `gas_metering_stub_offsets`.
3. **Run-time recovery.** When the subtraction underflows, the synchronous-mode stub jumps into the illegal-opcode trick, the kernel delivers SIGILL (generic) or the userfaultfd / signal pipe (Linux) tells the host. `on_page_fault` / `on_signal_trap` look up the stub's offset, restore the gas counter (the subtraction is unwound), and return `InterruptKind::NotEnoughGas` to the caller.

The reason metering lives at block granularity rather than per instruction is that the cost of the subtraction + check is amortised across an entire straight-line block, and the stub itself is only ~10 bytes. The recompiler's per-block model is the same one the interpreter exposes via its block cache, so gas accounting is consistent across backends.

## 8. Where to start reading the code

If you want to follow a single guest instruction all the way through:

1. `api.rs:403` `Module::from_blob` → `api.rs:525` `compile_module!`
2. `compiler.rs:71` `CompilerVisitor` and its `ParsingVisitor` impl around `compiler.rs:614`
3. `compiler/amd64.rs:1530` `add_generic` (or pick any other handler) — see the assembler API in action
4. `compiler.rs:308` `finish_compilation` — see how labels resolve and gas costs get patched
5. `sandbox.rs:163` `Instance::spawn_and_load_module` → `sandbox/linux.rs:1630` `Sandbox::spawn` → `sandbox/linux.rs:2123` `Sandbox::run`
6. For page-fault paths: `sandbox/linux.rs:2752` (`VMCTX_FUTEX_GUEST_PAGEFAULT` handling) → `crate::compiler::on_page_fault`.

Those six stops cover the whole pipeline from "host hands us bytes" to "guest runs and yields a result".

## 9. Testing at 4K page size on macOS

macOS on Apple Silicon runs a **16K** page size natively and it cannot be changed
(the constraint is below the OS). So on a Mac the recompiler's paging / `mprotect`
boundary logic — guard pages, dynamic paging, page-fault resume — is only ever
exercised at 16K granularity, even though the silicon's MMU does support 4K.

To test at a real 4K page size, run the recompiler inside a Linux guest under
Apple's Virtualization.framework. The guest runs a 4K-page kernel on the real
hardware MMU (not software emulation) at near-native speed, so CPU-bound guest
execution is indistinguishable from bare metal — only I/O boundaries pay VM cost.

```
ci/jobs/build-and-test-macos-4k-lima.sh                    # default recompiler tests
ci/jobs/build-and-test-macos-4k-lima.sh <cargo test args>  # custom filter
```

The script boots a dedicated `vz` VM via [Lima](https://lima-vm.io) (`brew install
lima`), provisions the toolchain once, asserts the guest is genuinely 4K, and runs
the generic-sandbox compiler tests against the working tree (build output is
redirected to the guest's local disk since the repo is a read-only mount).
Tear down with `limactl stop polkavm-4k && limactl delete polkavm-4k`.
