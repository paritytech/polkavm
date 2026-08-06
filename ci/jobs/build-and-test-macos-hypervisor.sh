#!/bin/bash

# Hypervisor-sandbox tests. `hv_vm_create` needs the `com.apple.security.hypervisor` entitlement,
# which cargo's test binaries don't carry, so they are ad-hoc codesigned by a cargo test runner.
#
# This needs a real Apple Silicon host: GitHub's macOS runners are themselves VMs and cannot nest
# virtualization, so this job only works on self-hosted hardware.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../..

if [ "$(uname -sm)" != "Darwin arm64" ]; then
    echo ">> skipping: the hypervisor sandbox needs macOS on Apple Silicon (got '$(uname -sm)')"
    exit 0
fi

export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="$(pwd)/ci/hypervisor/sign-and-run.sh"

echo ">> cargo test (hypervisor-sandbox only)"
cargo test --target aarch64-apple-darwin --features hypervisor-sandbox -p polkavm -- tests::aarch64_hypervisor

echo ">> cargo test (hypervisor-sandbox + generic-sandbox)"
cargo test --target aarch64-apple-darwin --features generic-sandbox,hypervisor-sandbox -p polkavm

echo ">> cargo test (AArch64 backend corpus routed through the hypervisor)"
POLKAVM_TEST_HYPERVISOR=1 cargo test --target aarch64-apple-darwin \
    --features generic-sandbox,hypervisor-sandbox -p polkavm -- tests::aarch64_backend
