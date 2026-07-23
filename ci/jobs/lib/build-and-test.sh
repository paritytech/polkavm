#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../../..

PROFILE="${1:?usage: build-and-test.sh <profile>}"

echo ">> cargo test (main crates, $PROFILE)"
cargo test --profile $PROFILE -p polkavm
cargo test --profile $PROFILE -p polkavm-assembler
cargo test --profile $PROFILE -p polkavm-common
cargo test --profile $PROFILE -p polkavm-common --all-features
cargo test --profile $PROFILE -p polkavm-derive
cargo test --profile $PROFILE -p polkavm-derive-impl
cargo test --profile $PROFILE -p polkavm-derive-impl-macro
cargo test --profile $PROFILE -p polkavm-disassembler
cargo test --profile $PROFILE -p polkavm-linker
cargo test --profile $PROFILE -p polkavm-linux-raw

echo ">> cargo test (examples, $PROFILE)"
cargo test --profile $PROFILE -p hello-world-host
cargo test --profile $PROFILE -p doom-host
cargo test --profile $PROFILE -p quake-host

echo ">> cargo test (tools, $PROFILE)"
cargo test --profile $PROFILE -p polkavm-linux-raw-generate
cargo test --profile $PROFILE -p polkatool
cargo test --profile $PROFILE -p spectool
cd tools/benchtool && cargo test --profile $PROFILE && cd ../..
