#!/usr/bin/env bash

# Builds the wide-arithmetic PoC blob sets (see wide-arith-results.md in
# polkadot-sdk/designs/parachain-service-on-jam).
#
# Usage:
#   ./build-wide-arith-blobs.sh [output-dir]
#
# Produces <output-dir>/{off,noredc,redc}/bench-*.polkavm for the five
# signature benches (output-dir defaults to ../../wide-arith-blobs, i.e.
# next to the polkavm checkout).
#
# To actually *benchmark* a set, copy it into the directory benchtool reads:
#   cp <output-dir>/redc/*.polkavm target/riscv64emac-unknown-none-polkavm/release/
# and remember to rebuild benchtool after any ISA change (a stale binary
# decodes the new opcodes as traps). Sanity-check which set is in place with:
#   cargo run -q --release -p opcount -- target/riscv64emac-unknown-none-polkavm/release/bench-ed25519-zebra.polkavm
# (gas per verify: ~837k stock, ~161k noredc, ~133k redc)

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

output_dir="${1:-$PWD/../../wide-arith-blobs}"
target_json="$PWD/../crates/polkavm-linker/targets/legacy/riscv64emac-unknown-none-polkavm.json"
benches="bench-ed25519-zebra bench-ed25519 bench-sr25519 bench-ecdsa-k256 bench-ecdsa-libsecp"

for variant in off noredc redc; do
    case "$variant" in
        off)    flags="";;
        noredc) flags='--cfg curve25519_dalek_backend="pvm"';;
        redc)   flags='--cfg curve25519_dalek_backend="pvm" --cfg pvm_redc';;
    esac

    mkdir -p "$output_dir/$variant"
    for bench in $benches; do
        echo "> Building: $variant/$bench"
        RUSTFLAGS="$flags" cargo build \
            -Z build-std=core,alloc \
            --target "$target_json" \
            -q --release --bin "$bench" -p "$bench"

        # Always relink: --run-only-if-newer would skip the relink when only
        # the flags changed (the PoC #1 footgun).
        (cd .. && cargo run -q -p polkatool link \
            "guest-programs/target/riscv64emac-unknown-none-polkavm/release/$bench" \
            -o "$output_dir/$variant/$bench.polkavm")
    done
done

echo
echo "Blob sets written to: $output_dir"
echo "The k256/libsecp controls must be byte-identical across variants:"
for bench in bench-ecdsa-k256 bench-ecdsa-libsecp; do
    if cmp -s "$output_dir/off/$bench.polkavm" "$output_dir/redc/$bench.polkavm" \
        && cmp -s "$output_dir/off/$bench.polkavm" "$output_dir/noredc/$bench.polkavm"; then
        echo "  $bench: OK (byte-identical)"
    else
        echo "  $bench: MISMATCH - the controls moved, something is wrong!"
        exit 1
    fi
done
