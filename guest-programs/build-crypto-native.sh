#!/bin/sh

# Builds each signature benchmark's host library in two variants (same scheme
# as build-hash-native.sh):
#   libbench_<name>.so        - host portable: runs on any x86-64
#   libbench_<name>_native.so - host native: -C target-cpu=native, all ISA
#                               extensions of the BUILD machine enabled
#                               unconditionally
#
# Both land in target/x86_64-unknown-linux-gnu/release/, where benchtool
# auto-discovers them as "<name>" and "<name>-native".
#
# Note: on stable Rust the native variant changes little for this scalar code
# (measured on Zen 3: ecdsa-k256 -7%, everything else within a few percent).
# curve25519-dalek's SIMD/IFMA backends need nightly and are NOT enabled by
# target-cpu=native here - see the report's toolchain note.
#
# WARNING: never copy *_native.so between machines - they are compiled for the
# exact CPU they were built on and will crash (SIGILL) on CPUs lacking any of
# their instruction-set extensions. Always rebuild locally.
#
# Built with the stable toolchain - see the note in build-benchmarks.sh.

set -ex
cd "${0%/*}"

out=target/x86_64-unknown-linux-gnu/release

crates="bench-ed25519 bench-ed25519-zebra bench-sr25519 bench-ecdsa-k256 \
        bench-ecdsa-libsecp bench-recover-k256 bench-recover-libsecp"

for crate in $crates; do
    lib=$(echo "$crate" | tr - _)
    RUSTFLAGS="-C target-cpu=native" cargo +1.86.0 build --target=x86_64-unknown-linux-gnu --release --lib -p "$crate"
    cp "$out/lib$lib.so" "$out/lib${lib}_native.so"
done

# Rebuild without the CPU features; overwrites the portable variants.
for crate in $crates; do
    cargo +1.86.0 build --target=x86_64-unknown-linux-gnu --release --lib -p "$crate"
done
