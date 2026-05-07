#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

if [ "$(cat /proc/sys/vm/unprivileged_userfaultfd)" != "1" ]; then
    echo "1" | sudo tee /proc/sys/vm/unprivileged_userfaultfd > /dev/null
fi

echo ">> cargo +nightly build (polkavm, experimental-musttail)"
cargo +nightly build -p polkavm --features experimental-musttail

echo ">> cargo +nightly test (polkavm, experimental-musttail)"
cargo +nightly test -p polkavm --features experimental-musttail

echo ">> cargo +nightly test --release (polkavm, experimental-musttail)"
cargo +nightly test --release -p polkavm --features experimental-musttail
