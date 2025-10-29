#!/bin/sh

set -euo pipefail

if [ ! -d ".venv" ]; then
    uv sync
fi

source .venv/bin/activate
uv run python validate-asn-schema.py
