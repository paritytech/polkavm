#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

rustup toolchain install --component=rust-src nightly-2024-11-01-x86_64-unknown-linux-gnu

POLKAVM_CRATES_ROOT="$(pwd)/proxy-crates"

cd ../..

mkdir -p target/test-pallet-revive
cd target/test-pallet-revive

if [ ! -d "polkadot-sdk" ]; then
    git clone --depth 1 "https://github.com/paritytech/polkadot-sdk.git"
fi
cd polkadot-sdk
COMMIT=2700dbf2dda8b7f593447c939e1a26dacdb8ce45
git fetch --depth=1 origin $COMMIT
git checkout $COMMIT

echo '[toolchain]' > rust-toolchain.toml
echo 'channel = "nightly-2024-11-01"' >> rust-toolchain.toml

PALLET_REVIVE_FIXTURES_RUSTUP_TOOLCHAIN=nightly-2024-11-01-x86_64-unknown-linux-gnu \
PALLET_REVIVE_FIXTURES_STRIP=0 \
PALLET_REVIVE_FIXTURES_OPTIMIZE=1 \
cargo test \
    --config "patch.crates-io.polkavm013.path='$POLKAVM_CRATES_ROOT/polkavm013'" --config "patch.crates-io.polkavm013.package='polkavm'" \
    --config "patch.crates-io.polkavm-derive014.path='$POLKAVM_CRATES_ROOT/polkavm-derive014'" --config "patch.crates-io.polkavm-derive014.package='polkavm-derive'" \
    --config "patch.crates-io.polkavm-linker014.path='$POLKAVM_CRATES_ROOT/polkavm-linker014'" --config "patch.crates-io.polkavm-linker014.package='polkavm-linker'" \
    -p pallet-revive
