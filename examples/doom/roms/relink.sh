#!/usr/bin/bash

set -euo pipefail

zstd -f -d -o /tmp/doom.elf ../../../test-data/doom_64.elf.zst
cargo run -p polkatool link -s /tmp/doom.elf -o doom.polkavm
