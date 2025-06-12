#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

for PROFILE in "dev" "release"
do
    echo ">> cargo test (main crates, $PROFILE)"
    cargo test --profile $PROFILE -p polkavm
    cargo test --profile $PROFILE -p polkavm-assembler
    cargo test --profile $PROFILE -p polkavm-common
    cargo test --profile $PROFILE -p polkavm-derive
    cargo test --profile $PROFILE -p polkavm-derive-impl
    cargo test --profile $PROFILE -p polkavm-derive-impl-macro
    cargo test --profile $PROFILE -p polkavm-disassembler
    cargo test --profile $PROFILE -p polkavm-linker
    cargo test --profile $PROFILE -p polkavm-linux-raw
    cargo test --profile $PROFILE -p simplealloc

    echo ">> cargo test (examples, $PROFILE)"
    cargo test --profile $PROFILE -p hello-world-host
    cargo test --profile $PROFILE -p doom-host
    cargo test --profile $PROFILE -p quake-host

    echo ">> cargo test (tools, $PROFILE)"
    cargo test --profile $PROFILE -p polkavm-linux-raw-generate
    cargo test --profile $PROFILE -p polkatool
    cargo test --profile $PROFILE -p spectool
    cd tools/benchtool && cargo test --profile $PROFILE && cd ../..
done

echo ">> cargo run (examples)"
POLKAVM_TRACE_EXECUTION=1 POLKAVM_ALLOW_INSECURE=1 cargo run -p hello-world-host

echo ">> cargo test (no_std)"
RUSTC_BOOTSTRAP=1 cargo test --no-default-features -p polkavm

echo ">> cargo test (module-cache)"
cargo test --features module-cache -p polkavm

echo ">> cargo run generate (spectool)"
cargo run -p spectool generate
