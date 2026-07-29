#!/bin/sh

# Builds the bench-hash host library in both variants:
#   libbench_hash.so        - host portable: runs on any x86-64; crates may
#                             still runtime-dispatch to fast paths (e.g. sha2
#                             uses SHA-NI via cpufeatures when available)
#   libbench_hash_native.so - host native: -C target-cpu=native, all ISA
#                             extensions of the BUILD machine enabled
#                             unconditionally
#
# Both land in target/x86_64-unknown-linux-gnu/release/, where benchtool's
# `bench-hash` subcommand auto-discovers them as "hash" and "hash-native".
#
# WARNING: never copy libbench_hash_native.so between machines - it is
# compiled for the exact CPU it was built on and will crash (SIGILL) on CPUs
# lacking any of its instruction-set extensions. Always rebuild locally.

set -ex
cd "${0%/*}"

out=target/x86_64-unknown-linux-gnu/release

RUSTFLAGS="-C target-cpu=native" cargo build --target=x86_64-unknown-linux-gnu --release --lib -p bench-hash
cp "$out/libbench_hash.so" "$out/libbench_hash_native.so"

# Rebuild without the CPU features; overwrites libbench_hash.so with the portable variant.
cargo build --target=x86_64-unknown-linux-gnu --release --lib -p bench-hash
