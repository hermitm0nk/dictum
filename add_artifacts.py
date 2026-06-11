#!/usr/bin/env python3
"""Post-process wheel to add native artifacts."""
import zipfile
import os
from pathlib import Path

WHEEL_PATH = Path("dist/dictum-0.1.0-py3-none-any.whl")
SRC_DIR = Path("src/dictum")

# Read existing files to avoid duplicates
with zipfile.ZipFile(WHEEL_PATH, 'r') as whl:
    existing = set(whl.namelist())

with zipfile.ZipFile(WHEEL_PATH, 'a', zipfile.ZIP_DEFLATED) as whl:
    # Add bin/
    for f in (SRC_DIR / "bin").rglob("*"):
        if f.is_file():
            arcname = f"dictum/{f.relative_to(SRC_DIR)}"
            if arcname not in existing:
                whl.write(f, arcname)
                print(f"Added: {arcname}")
            else:
                print(f"Skipped (exists): {arcname}")
    
    # Add lib/
    for f in (SRC_DIR / "lib").rglob("*"):
        if f.is_file():
            # Skip libwhisper.so - it's just an alias of libcrispasr.so
            if "libwhisper" in f.name:
                print(f"Skipped (alias): dictum/{f.relative_to(SRC_DIR)}")
                continue
            # Skip old-version .so.N.N.N files - only SONAMEs are loaded
            if ".0.0.1" in f.name or ".0.0.7" in f.name:
                print(f"Skipped (old version): dictum/{f.relative_to(SRC_DIR)}")
                continue
            arcname = f"dictum/{f.relative_to(SRC_DIR)}"
            if arcname not in existing:
                whl.write(f, arcname)
                print(f"Added: {arcname}")
            else:
                print(f"Skipped (exists): {arcname}")

print("Done!")