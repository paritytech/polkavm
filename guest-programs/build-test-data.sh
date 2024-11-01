#!/bin/bash

set -euo pipefail
cd "${0%/*}/"

function build_test_data() {
    output_path="../test-data/$1_32.elf.zst"
    echo "> Building: '$1' [32 bit] (-> $output_path)"
    RUSTFLAGS="--remap-path-prefix=$(pwd)= --remap-path-prefix=$HOME=~" cargo build -q --profile $2 --bin $1 -p $1 --target=riscv32emac-unknown-none-polkavm.json -Zbuild-std=core,alloc
    zstd -f -q -19 -o $output_path target/riscv32emac-unknown-none-polkavm/$2/$1
    chmod -x $output_path

    output_path="../test-data/$1_64.elf.zst"
    echo "> Building: '$1' [64 bit] (-> $output_path)"
    RUSTFLAGS="--remap-path-prefix=$(pwd)= --remap-path-prefix=$HOME=~" cargo build -q --profile $2 --bin $1 -p $1 --target=riscv64emac-unknown-none-polkavm.json -Zbuild-std=core,alloc
    zstd -f -q -19 -o $output_path target/riscv64emac-unknown-none-polkavm/$2/$1
    chmod -x $output_path
}

build_test_data "bench-pinky" "release"
build_test_data "test-blob" "no-lto"
