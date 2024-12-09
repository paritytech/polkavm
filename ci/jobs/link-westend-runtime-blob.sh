#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

# TODO: This should build and run a real node instead of just linking a hardcoded blob.

echo ">> polkatool westend-runtime-blob"
mkdir -p target
zstd -d test-data/westend-runtime-blob.zst -o target/westend-runtime-blob
cargo run --profile=release-lite -p polkatool link target/westend-runtime-blob -o target/westend-runtime-blob.polkavm
