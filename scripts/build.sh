#!/usr/bin/env bash
# Build all native dependencies (llama.cpp + CrispASR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-Release}"

echo "============================================"
echo " Building native deps ($BUILD_TYPE)"
echo "============================================"

echo ""
echo "--- llama.cpp ---"
bash "$SCRIPT_DIR/build-llama.sh" "$BUILD_TYPE"

echo ""
echo "--- CrispASR ---"
bash "$SCRIPT_DIR/build-crispasr.sh" "$BUILD_TYPE"

echo ""
echo "============================================"
echo " All builds complete."
echo " Run 'scripts/deploy.sh' to install binaries."
echo "============================================"
