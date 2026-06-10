#!/usr/bin/env bash
# Deploy built native binaries to ~/.dictum/bin/ and libs to ~/.dictum/lib/.
# llama.cpp and CrispASR each get their own lib subdirectory to avoid ggml conflicts.
# Binaries are built with rpath so they find their libs without LD_LIBRARY_PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
INSTALL_DIR="${DICTUM_BIN_DIR:-$HOME/.dictum/bin}"
LIB_DIR="${DICTUM_LIB_DIR:-$HOME/.dictum/lib}"

LLAMA_BIN="$BUILD_DIR/llama.cpp/build/bin"
CRISPASR_BIN="$BUILD_DIR/CrispASR/build/bin"
CRISPASR_LIB="$BUILD_DIR/CrispASR/build/src"
CRISPASR_GGML="$BUILD_DIR/CrispASR/build/ggml/src"

LLAMA_LIB="$LIB_DIR/llama"
CRISPASR_LIBDIR="$LIB_DIR/crispasr"

mkdir -p "$INSTALL_DIR" "$LLAMA_LIB" "$CRISPASR_LIBDIR"

deploy_llama() {
    echo ":: Deploying llama-server …"
    if [ ! -f "$LLAMA_BIN/llama-server" ]; then
        echo "ERROR: $LLAMA_BIN/llama-server not found. Run scripts/build-llama.sh first." >&2
        return 1
    fi

    cp -f "$LLAMA_BIN/llama-server" "$INSTALL_DIR/llama-server"

    # Copy all shared libraries to dedicated subdir
    local count=0
    for so in "$LLAMA_BIN"/lib*.so*; do
        [ -e "$so" ] || continue
        cp -f "$so" "$LLAMA_LIB/"
        count=$((count + 1))
    done

    echo "   llama-server + $count libs → $LLAMA_LIB/"
}

deploy_crispasr() {
    echo ":: Deploying parakeet-main (crispasr) …"
    if [ ! -f "$CRISPASR_BIN/crispasr" ]; then
        echo "ERROR: $CRISPASR_BIN/crispasr not found. Run scripts/build-crispasr.sh first." >&2
        return 1
    fi

    # Deploy as parakeet-main (name expected by asr.py)
    cp -f "$CRISPASR_BIN/crispasr" "$INSTALL_DIR/parakeet-main"

    local count=0
    # CrispASR shared libraries
    for so in "$CRISPASR_LIB"/lib*.so*; do
        [ -e "$so" ] || continue
        cp -f "$so" "$CRISPASR_LIBDIR/"
        count=$((count + 1))
    done

    # CrispASR's own ggml libs (different version from llama.cpp's)
    for so in "$CRISPASR_GGML"/lib*.so*; do
        [ -e "$so" ] || continue
        cp -f "$so" "$CRISPASR_LIBDIR/"
        count=$((count + 1))
    done

    echo "   parakeet-main + $count libs → $CRISPASR_LIBDIR/"
}

deploy_llama
deploy_crispasr

echo ""
echo ":: Deployed to:"
echo "   Binaries: $INSTALL_DIR/"
echo "   llama libs: $LLAMA_LIB/"
echo "   crispasr libs: $CRISPASR_LIBDIR/"
echo ""
ls -lh "$INSTALL_DIR/llama-server" "$INSTALL_DIR/parakeet-main" 2>/dev/null
echo ""
echo ":: Done. Binaries have rpath set — no LD_LIBRARY_PATH needed."
