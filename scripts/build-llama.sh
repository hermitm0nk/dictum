#!/usr/bin/env bash
# Build llama.cpp from source with Vulkan support (no CUDA).
# Sources are cloned into build/llama.cpp (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC_DIR="$BUILD_DIR/llama.cpp"
BUILD_TYPE="${1:-Release}"
NPROC="$(nproc)"

clone() {
    if [ -d "$SRC_DIR/.git" ]; then
        echo ":: Updating llama.cpp …"
        git -C "$SRC_DIR" pull --ff-only || true
    else
        echo ":: Cloning llama.cpp …"
        mkdir -p "$BUILD_DIR"
        git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
    fi
}

build() {
    echo ":: Configuring llama.cpp ($BUILD_TYPE, Vulkan) …"
    cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGML_VULKAN=ON \
        -DGGML_CUDA=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib/llama'

    echo ":: Building llama-server ($NPROC jobs) …"
    cmake --build "$SRC_DIR/build" -j "$NPROC" --target llama-server
}

clone
build

echo ":: Build complete. Binaries in $SRC_DIR/build/bin/"
ls -lh "$SRC_DIR/build/bin/llama-server" "$SRC_DIR/build/bin/"lib*.so* 2>/dev/null
