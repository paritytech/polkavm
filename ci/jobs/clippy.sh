#!/usr/bin/env bash

set -euo pipefail

cd "${0%/*}/"
cd ../..

echo ">> cargo clippy"
RUSTFLAGS="-D warnings" cargo clippy --all

case "$OSTYPE" in
  linux*)
    echo ">> cargo clippy (zygote)"
    cd crates/polkavm-zygote
    RUSTFLAGS="-D warnings" cargo clippy --all
    cd ../..
  ;;
esac

echo ">> cargo clippy (guests)"

cd guest-programs
RUSTFLAGS="-D warnings" \
cargo clippy  \
    -Z build-std=core,alloc \
    --target "$PWD/../crates/polkavm-linker/riscv32emac-unknown-none-polkavm.json" \
    --all

cd ../..
