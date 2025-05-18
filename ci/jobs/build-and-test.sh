#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

echo ">> cargo test (debug)"
cargo test --all

echo ">> cargo test (release)"
cargo test --all --release

echo ">> cargo run (examples)"
POLKAVM_TRACE_EXECUTION=1 POLKAVM_ALLOW_INSECURE=1 cargo run -p hello-world-host

echo ">> cargo test (no_std)"
RUSTC_BOOTSTRAP=1 cargo test --no-default-features -p polkavm

echo ">> cargo test (module-cache)"
cargo test --features module-cache -p polkavm

echo ">> cargo test (generic-sandbox)"
cargo test --features generic-sandbox -p polkavm -- \
    tests::compiler_generic_basic_test \
    tests::compiler_generic_simple_test \
    tests::compiler_generic_riscv_ 

echo ">> cargo run generate (spectool)"
cargo run -p spectool generate
