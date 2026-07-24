#!/bin/sh

# Builds the bench-hash native library in both variants:
#   libbench_hash.so      - portable implementations
#   libbench_hash_simd.so - CPU features enabled at compile time (e.g. AVX2)
#
# Both land in target/x86_64-unknown-linux-gnu/release/, where benchtool's
# `bench-hash` subcommand auto-discovers them as "hash" and "hash-simd".

set -ex
cd "${0%/*}"

out=target/x86_64-unknown-linux-gnu/release

RUSTFLAGS="-C target-cpu=native" cargo build --target=x86_64-unknown-linux-gnu --release --lib -p bench-hash
cp "$out/libbench_hash.so" "$out/libbench_hash_simd.so"

# Rebuild without the CPU features; overwrites libbench_hash.so with the portable variant.
cargo build --target=x86_64-unknown-linux-gnu --release --lib -p bench-hash
