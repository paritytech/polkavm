#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

echo ">> cargo +nightly test (polkavm, interpreter-musttail-dispatch, interpreter tests only)"
cargo +nightly test -p polkavm --features interpreter-musttail-dispatch -- tests::interpreter_

echo ">> cargo +nightly test --release (polkavm, interpreter-musttail-dispatch, interpreter tests only)"
cargo +nightly test --release -p polkavm --features interpreter-musttail-dispatch -- tests::interpreter_
