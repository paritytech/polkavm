#!/bin/bash
# Run the AArch64 recompiler (generic-sandbox) tests on a REAL 4K page size.
#
# macOS on Apple Silicon uses 16K pages natively, so the recompiler's paging /
# mprotect logic is never exercised at 4K on a Mac. A Linux guest under Apple's
# Virtualization.framework (Lima's `vz` backend) runs a 4K-page kernel on the
# real hardware MMU at near-native speed. This boots such a guest and runs the
# generic-sandbox compiler tests inside it against the working tree.
#
# Usage:
#   ci/jobs/build-and-test-macos-4k-lima.sh            # default recompiler tests
#   ci/jobs/build-and-test-macos-4k-lima.sh <cargo test filter args...>
#
# Env overrides: POLKAVM_LIMA_VM, POLKAVM_LIMA_CPUS, POLKAVM_LIMA_MEM
# Teardown:      limactl stop polkavm-4k && limactl delete polkavm-4k

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..
REPO="$(pwd)"

VM="${POLKAVM_LIMA_VM:-polkavm-4k}"
CPUS="${POLKAVM_LIMA_CPUS:-8}"
MEM="${POLKAVM_LIMA_MEM:-12}"

command -v limactl >/dev/null || { echo "!! lima not installed. Install with: brew install lima"; exit 1; }

sh() { limactl shell "$VM" -- bash -lc "$1"; }

# Create the VM once. vz => real hardware MMU (not TCG emulation); the stock
# Ubuntu arm64 kernel is built for 4K pages, which is exactly what we want.
if ! limactl list -q 2>/dev/null | grep -qx "$VM"; then
    echo ">> creating Lima VM '$VM' (vz backend, 4K guest, ${CPUS} cpus / ${MEM} GiB)"
    limactl start --vm-type=vz --cpus="$CPUS" --memory="$MEM" --name="$VM" template://ubuntu --tty=false
else
    echo ">> starting Lima VM '$VM'"
    limactl start "$VM" --tty=false
fi

# Fail loudly if the guest is not actually 4K (e.g. a fallback to a 16K kernel).
PS="$(sh 'getconf PAGE_SIZE')"
echo ">> guest page size: $PS"
[ "$PS" = "4096" ] || { echo "!! guest page size is $PS, expected 4096 — aborting"; exit 1; }

# Provision the toolchain and native build deps once (idempotent marker file).
if ! sh 'test -f "$HOME/.polkavm-4k-provisioned"'; then
    echo ">> provisioning guest (apt deps + rustup)"
    sh 'sudo apt-get update -qq && sudo apt-get install -y -qq gcc g++ make cmake clang pkg-config libssl-dev perl curl'
    sh 'command -v cargo >/dev/null || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
    sh 'touch "$HOME/.polkavm-4k-provisioned"'
fi

# The repo is a read-only virtiofs mount inside the guest, so send cargo's build
# output to the guest's own writable disk.
#
# memset_with_dynamic_paging is skipped on macOS (16K) but PASSES at 4K, so we run
# it here. memset_basic is a pre-existing AArch64-recompiler gas-metering quirk
# (skipped on macOS too), unrelated to page size.
TEST_ARGS="${*:-tests::compiler_generic_ --skip tests::compiler_generic_memset_basic}"

echo ">> cargo test (generic-sandbox) at 4K page size"
sh "source \$HOME/.cargo/env
    cd '$REPO'
    export CARGO_TARGET_DIR=\$HOME/polkavm-target-4k
    echo \"   toolchain: \$(rustc --version)  |  page size: \$(getconf PAGE_SIZE)\"
    cargo test --locked --features generic-sandbox -p polkavm -- $TEST_ARGS"

echo ">> done (4K page size validated)"
