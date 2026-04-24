# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolkaVM is a general-purpose RISC-V 64bit based virtual machine. It provides secure sandboxed execution with both an interpreter and a JIT compiler backend.

## Build Commands

```bash
# Build with release optimizations
cargo build --release

# Run all tests (both dev and release)
./ci/jobs/build-and-test.sh

# Run tests for a specific crate
cargo test -p polkavm
cargo test -p polkavm-linker
cargo test -p polkavm-common

# Run tests with all features
cargo test -p polkavm-common --all-features

# Run a single test
cargo test -p polkavm -- test_name

# Linting
./ci/jobs/clippy.sh
cargo clippy -p polkavm

# Formatting
cargo fmt --check --all
cargo fmt --all
```

## Guest Programs

Guest programs (programs that run inside the VM) are in `guest-programs/` and use a separate workspace with nightly Rust:

```bash
cd guest-programs

# Build guest programs (uses custom RISC-V target)
cargo build --release
```

Guest programs require nightly Rust (`nightly-2025-05-10`) with `rust-src` component for `-Z build-std`.

## Architecture

### Core Crates (`crates/`)

- **polkavm**: Main VM crate with `Engine`, `Module`, `Instance` API. Contains:
  - `interpreter.rs`: Cross-platform interpreter backend
  - `compiler.rs`: JIT compiler (x86_64 Linux/macOS/FreeBSD)
  - `sandbox/`: Process isolation for compiled code
  - `api.rs`: Public API types

- **polkavm-common**: Shared types and utilities
  - `program.rs`: Program blob format, instruction definitions, parsing
  - `assembler.rs`: Assembly/disassembly support
  - `abi.rs`: VM ABI definitions (memory map, calling conventions)

- **polkavm-linker**: Offline linker that transforms ELF to PolkaVM format
  - Reads RISC-V ELF files from guest compilation
  - Outputs `.polkavm` program blobs

- **polkavm-zygote**: Linux-only process template for sandboxed execution (separate workspace)

- **polkavm-derive**: Proc macros for guest programs (`#[polkavm_import]`, `#[polkavm_export]`)

### Tools (`tools/`)

- **polkatool**: CLI for inspecting/disassembling `.polkavm` blobs
- **spectool**: Generates test vectors for VM specification
- **gastool**: Gas metering analysis (Linux only)
- **benchtool**: Performance benchmarking suite

### Execution Backends

The VM supports two backends controlled via `Config::set_backend()` or `POLKAVM_BACKEND` env var:
- `interpreter`: Cross-platform, always available
- `compiler`: JIT to native code, requires x86_64 + (Linux | macOS with `generic-sandbox` feature | FreeBSD with `generic-sandbox` feature)

## Testing

```bash
# Full test suite
./ci/jobs/build-and-test.sh

# Run with execution tracing
POLKAVM_TRACE_EXECUTION=1 POLKAVM_ALLOW_INSECURE=1 cargo run -p hello-world-host

# Fuzzing (requires cargo-fuzz)
cd fuzz
cargo fuzz run fuzz_polkavm -- -runs=10000
```

## Key Types

- `Engine`: VM engine instance, configured via `Config`
- `Module`: Compiled program, created from `ProgramBlob`
- `Instance`: Execution instance with memory state
- `Linker`: Links host functions to module imports
- `ProgramBlob`: Serialized program format
