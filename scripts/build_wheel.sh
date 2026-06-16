#!/usr/bin/env bash
# Build a platform wheel for dictum.
#
# Usage:
#   scripts/build_wheel.sh           # wheel only (default)
#   scripts/build_wheel.sh --sdist   # also produce sdist
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo " Building dictum wheel"
echo "============================================"

# scikit-build-core drives CMake → llama.cpp + CrispASR → _native/
exec uv build "$@"
