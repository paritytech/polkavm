#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

POLKAVM_CRATES_ROOT="$(pwd)/proxy-crates"
POLKDOT_SDK_COMMIT=c45cc4efc1861f2e5b9ec151914c51dcdfa739b4

LOG_FILE="output.log"
TARGET_DIR=target/run-substrate-node
NODE_RUN_DURATION=100
NODE_BLOCKS_THRESHOLD=10

rustup toolchain install --component=rust-src nightly-2024-11-01-x86_64-unknown-linux-gnu

cd ../..

mkdir -p $TARGET_DIR
cd $TARGET_DIR

if [ ! -d "polkadot-sdk" ]; then
    git clone --depth 1 "https://github.com/paritytech/polkadot-sdk.git"
    git -C polkadot-sdk fetch --depth=1 origin $POLKDOT_SDK_COMMIT
    git -C polkadot-sdk checkout $POLKDOT_SDK_COMMIT
fi

cd polkadot-sdk

echo '[toolchain]' > rust-toolchain.toml
echo 'channel = "nightly-2024-11-01"' >> rust-toolchain.toml

git checkout Cargo.toml
echo "" >> Cargo.toml
echo "[patch.crates-io]" >> Cargo.toml
echo "polkavm-derive018 = { path = \"$POLKAVM_CRATES_ROOT/polkavm-derive018\", package = \"polkavm-derive\" }" >> Cargo.toml
echo "polkavm-derive021 = { path = \"$POLKAVM_CRATES_ROOT/polkavm-derive021\", package = \"polkavm-derive\" }" >> Cargo.toml

SUBSTRATE_RUNTIME_TARGET=riscv \
SUBSTRATE_ENABLE_POLKAVM=1 \
cargo build \
    --config "patch.crates-io.polkavm018.path='$POLKAVM_CRATES_ROOT/polkavm018'" --config "patch.crates-io.polkavm018.package='polkavm'" \
    --config "patch.crates-io.polkavm021.path='$POLKAVM_CRATES_ROOT/polkavm021'" --config "patch.crates-io.polkavm021.package='polkavm'" \
    --config "patch.crates-io.polkavm-linker018.path='$POLKAVM_CRATES_ROOT/polkavm-linker018'" --config "patch.crates-io.polkavm-linker018.package='polkavm-linker'" \
    --config "patch.crates-io.polkavm-linker021.path='$POLKAVM_CRATES_ROOT/polkavm-linker021'" --config "patch.crates-io.polkavm-linker021.package='polkavm-linker'" \
    --release -p staging-node-cli

echo "Running Node in background..."
SUBSTRATE_RUNTIME_TARGET=riscv \
SUBSTRATE_ENABLE_POLKAVM=1 \
./target/release/substrate-node --dev --tmp > "$LOG_FILE" 2>&1 &

CARGO_PID=$!
sleep $NODE_RUN_DURATION

echo "Stopping the cargo process after $NODE_RUN_DURATION seconds..."
kill $CARGO_PID 2>/dev/null || true
wait $CARGO_PID 2>/dev/null || true

if ! grep -qi "Initializing Genesis block" "$LOG_FILE"; then
    echo "Node initialization failed. Please check logs at $LOG_FILE."
    cat "$LOG_FILE"
    exit 1
fi

GENERATED_BLOCK_COUNT=$(grep -ic "Pre-sealed block for proposal at" "$LOG_FILE" || true)

if [ $GENERATED_BLOCK_COUNT -lt $NODE_BLOCKS_THRESHOLD ]; then
    echo "Expected at least $NODE_BLOCKS_THRESHOLD blocks, but only generated $GENERATED_BLOCK_COUNT blocks."
    cat "$LOG_FILE"
    exit 1
fi
