#!/bin/bash

# AArch64 recompiler tests via the generic sandbox. Unlike macOS (16K pages) this host has a 4K
# native page size, so the paging/protection boundary logic is exercised at the production granule.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

echo ">> getconf PAGESIZE"
getconf PAGESIZE

echo ">> cargo test (generic-sandbox)"
cargo test --features generic-sandbox -p polkavm

echo ">> cargo test (assembler)"
cargo test -p polkavm-assembler --features alloc

echo ">> cargo run (examples, compiler, generic)"
POLKAVM_TRACE_EXECUTION=1 POLKAVM_ALLOW_INSECURE=1 POLKAVM_BACKEND=compiler POLKAVM_SANDBOX=generic \
    cargo run -p hello-world-host
