#!/usr/bin/env bash

set -euo pipefail

cd "${0%/*}/"
cd ../..

export RUSTFLAGS="-D warnings"

echo ">> cargo clippy (main crates)"
cargo clippy -p polkavm
cargo clippy -p polkavm-assembler
cargo clippy -p polkavm-common
cargo clippy -p polkavm-derive
cargo clippy -p polkavm-derive-impl
cargo clippy -p polkavm-derive-impl-macro
cargo clippy -p polkavm-disassembler
cargo clippy -p polkavm-linker
cargo clippy -p polkavm-linux-raw

echo ">> cargo clippy (examples)"
cargo clippy -p hello-world-host
cargo clippy -p doom-host
cargo clippy -p quake-host

echo ">> cargo clippy (tools)"
cargo clippy -p polkavm-linux-raw-generate
cargo clippy -p polkatool
cargo clippy -p spectool
cd tools/benchtool && cargo clippy --no-default-features && cd ../..

case "$OSTYPE" in
  linux*)
    echo ">> cargo clippy (gastool)"
    cd tools/gastool && cargo clippy && cd ../..

    echo ">> cargo clippy (zygote)"
    cd crates/polkavm-zygote
    cargo clippy --all
    cd ../..
  ;;
esac

echo ">> cargo clippy (guests)"

cd guest-programs
cargo clippy  \
    -Z build-std=core,alloc \
    --target "$PWD/../crates/polkavm-linker/targets/legacy/riscv32emac-unknown-none-polkavm.json" \
    --all

cd ../..
