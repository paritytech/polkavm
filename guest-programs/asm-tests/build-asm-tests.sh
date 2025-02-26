#!/bin/bash

set -euo pipefail
cd "${0%/*}/"

function build_asm_tests_64bit() {
    output_path="output/$1.elf"
    echo "> Building: '$1' (-> $output_path)"
    riscv64-linux-gnu-as -fPIC -march "rv64ima_zifencei_zbb" -mabi="lp64" $1.S -o $output_path
    chmod -x $output_path
}

build_asm_tests_64bit "reloc_add_sub_64"
build_asm_tests_64bit "reloc_hi_lo_64"

