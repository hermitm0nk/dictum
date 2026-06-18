#!/usr/bin/env bash
# Build a pure-Python platform wheel for dictum.
#
# The wheel no longer bundles native C++ binaries; instead, the daemon and
# `dictum native install` fetch pinned llama.cpp + CrispASR releases from
# GitHub on first run into $XDG_DATA_HOME/dictum/native/.
#
# Usage:
#   scripts/build_wheel.sh           # wheel only (default)
#   scripts/build_wheel.sh --sdist   # also produce sdist
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo " Building dictum wheel (pure Python)"
echo "============================================"

exec uv build "$@"
