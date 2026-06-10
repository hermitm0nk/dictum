#!/usr/bin/env bash
# Build CrispASR (parakeet ASR) from source.
# Sources are cloned into build/CrispASR (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SRC_DIR="$BUILD_DIR/CrispASR"
BUILD_TYPE="${1:-Release}"
NPROC="$(nproc)"

clone() {
    if [ -d "$SRC_DIR/.git" ]; then
        echo ":: Updating CrispASR …"
        git -C "$SRC_DIR" pull --ff-only || true
    else
        echo ":: Cloning CrispASR …"
        mkdir -p "$BUILD_DIR"
        git clone --depth=1 https://github.com/CrispStrobe/CrispASR.git "$SRC_DIR"
    fi
}

build() {
    echo ":: Configuring CrispASR ($BUILD_TYPE) …"
    cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCRISPASR_BUILD_EXAMPLES=ON \
        -DCRISPASR_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib/crispasr'

    echo ":: Building crispasr ($NPROC jobs) …"
    cmake --build "$SRC_DIR/build" -j "$NPROC" --target crispasr
}

clone
build

echo ":: Build complete. Binary in $SRC_DIR/build/bin/"
ls -lh "$SRC_DIR/build/bin/crispasr" "$SRC_DIR/build/src/"lib*.so* 2>/dev/null
