"""Hatchling build hook: build llama.cpp + CrispASR, stage into src/dictum/{bin,lib}."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class DictumBuildHook(BuildHookInterface):
    """Custom build hook to compile native dependencies and stage them."""

    def initialize(self, version, build_data):
        """Called before building - compile and stage native artifacts."""
        self._build_native()

    def _run(self, cmd, cwd=None, env=None):
        print(f":: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=cwd, env=env or os.environ, check=True)

    def _build_native(self):
        ROOT = Path(__file__).parent
        SRC_DIR = ROOT / "src" / "dictum"
        BIN_DIR = SRC_DIR / "bin"
        LLAMA_LIB_DIR = SRC_DIR / "lib" / "llama"
        CRISPASR_LIB_DIR = SRC_DIR / "lib" / "crispasr"

        BUILD_DIR = ROOT / "build"
        LLAMA_SRC = BUILD_DIR / "llama.cpp"
        CRISPASR_SRC = BUILD_DIR / "CrispASR"

        NPROC = os.cpu_count() or 4

        def clone_or_update(repo_url, dest):
            if (dest / ".git").exists():
                self._run(["git", "-C", str(dest), "pull", "--ff-only"])
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._run(["git", "clone", "--depth=1", repo_url, str(dest)])

        def prepare_dirs():
            for d in (BIN_DIR, LLAMA_LIB_DIR, CRISPASR_LIB_DIR):
                d.mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)

        def build_llama():
            print(":: Building llama.cpp (Vulkan, no CUDA)...")
            clone_or_update("https://github.com/ggml-org/llama.cpp.git", LLAMA_SRC)

            build_dir = LLAMA_SRC / "build"
            self._run([
                "cmake", "-S", str(LLAMA_SRC), "-B", str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_VULKAN=ON",
                "-DGGML_CUDA=OFF",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_UI=OFF",
                "-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib/llama'",
            ])
            self._run(["cmake", "--build", str(build_dir), "-j", str(NPROC), "--target", "llama-server"])

            bin_dir = build_dir / "bin"
            for so in bin_dir.glob("lib*.so*"):
                shutil.copy2(so, LLAMA_LIB_DIR / so.name)
            shutil.copy2(bin_dir / "llama-server", BIN_DIR / "llama-server")
            (BIN_DIR / "llama-server").chmod(0o755)

        def build_crispasr():
            print(":: Building CrispASR (Parakeet)...")
            clone_or_update("https://github.com/CrispStrobe/CrispASR.git", CRISPASR_SRC)

            build_dir = CRISPASR_SRC / "build"
            self._run([
                "cmake", "-S", str(CRISPASR_SRC), "-B", str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCRISPASR_BUILD_EXAMPLES=ON",
                "-DCRISPASR_BUILD_TESTS=OFF",
                "-DGGML_VULKAN=OFF",
                "-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib/crispasr'",
            ])
            self._run(["cmake", "--build", str(build_dir), "-j", str(NPROC), "--target", "crispasr-cli"])

            crispasr_bin = build_dir / "bin" / "crispasr"
            shutil.copy2(crispasr_bin, BIN_DIR / "parakeet-main")
            (BIN_DIR / "parakeet-main").chmod(0o755)

            for so in (build_dir / "src").glob("lib*.so*"):
                shutil.copy2(so, CRISPASR_LIB_DIR / so.name)
            for so in (build_dir / "ggml" / "src").glob("lib*.so*"):
                shutil.copy2(so, CRISPASR_LIB_DIR / so.name)

        prepare_dirs()
        build_llama()
        build_crispasr()
        print(":: Native artifacts staged for wheel packaging")