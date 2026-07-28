#!/bin/sh
# Cargo test-runner: ad-hoc codesign the test binary with the hypervisor
# entitlement, then run it. Lets `cargo test` exercise the Hypervisor sandbox.
set -e
bin="$1"; shift
codesign -s - --entitlements "$(dirname "$0")/hypervisor.entitlements" --force "$bin" >/dev/null 2>&1 || true
exec "$bin" "$@"
