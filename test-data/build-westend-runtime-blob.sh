#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

POLKADOT_SDK_COMMIT=6a99e45f49a573d3528d941d090501b39ce274d1
TOOLCHAIN=nightly-2026-04-03

BUILD_DIR=/tmp/build-westend-runtime-blob
POLKADOT_SDK_DIR="$BUILD_DIR/polkadot-sdk"
CARGO_TARGET_DIR="$BUILD_DIR/target"
ELF_PATH="$CARGO_TARGET_DIR/release/rbuild/westend-runtime/target/riscv64emac-unknown-none-polkavm/release/westend-runtime-blob"

rustup toolchain install --profile=minimal --component=rust-src "$TOOLCHAIN"

if [ "$(git -C "$POLKADOT_SDK_DIR" rev-parse HEAD 2>/dev/null)" != "$POLKADOT_SDK_COMMIT" ]; then
    rm -rf "$POLKADOT_SDK_DIR"
    git init -q "$POLKADOT_SDK_DIR"
    git -C "$POLKADOT_SDK_DIR" fetch --depth 1 https://github.com/paritytech/polkadot-sdk.git "$POLKADOT_SDK_COMMIT"
    git -C "$POLKADOT_SDK_DIR" checkout -q FETCH_HEAD
fi

echo ">> cargo build (westend-runtime)"
(
    cd "$POLKADOT_SDK_DIR"
    SUBSTRATE_RUNTIME_TARGET=riscv \
    WASM_BUILD_WORKSPACE_HINT="$POLKADOT_SDK_DIR" \
    CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
    cargo "+$TOOLCHAIN" build --release -p westend-runtime
)

zstd -19 -c "$ELF_PATH" > westend-runtime-blob.zst
echo ">> test-data/westend-runtime-blob.zst updated"
