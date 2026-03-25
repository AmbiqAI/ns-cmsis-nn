#!/usr/bin/env python3
"""Download release-build dependencies into a repo-owned cache directory."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


CMSIS_REPO = "https://github.com/ARM-software/CMSIS_5.git"
ETHOS_REPO = "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", required=True, help="Directory that will hold downloaded dependencies")
    parser.add_argument("--skip-gcc", action="store_true", help="Do not download the Arm GNU toolchain")
    return parser.parse_args()


def ensure_git_clone(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth=1", url, str(destination)], check=True)


def ensure_gcc(downloads_dir: Path) -> None:
    gcc_dir = downloads_dir / "arm_gcc_download"
    if (gcc_dir / "bin" / "arm-none-eabi-gcc").exists():
        return

    machine = platform.machine().lower()
    if machine == "x86_64":
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz"
    elif machine in {"aarch64", "arm64"}:
        gcc_url = "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz"
    else:
        raise SystemExit(f"Unsupported machine architecture for GCC download: {machine}")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "arm-gcc.tar.xz"
        urllib.request.urlretrieve(gcc_url, archive_path)
        gcc_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="r:xz") as tar:
            members = tar.getmembers()
            prefix = members[0].name.split("/", 1)[0]
            for member in members:
                if member.name == prefix:
                    continue
                if member.name.startswith(prefix + "/"):
                    member.name = member.name[len(prefix) + 1 :]
                tar.extract(member, path=gcc_dir)


def main() -> int:
    args = parse_args()
    downloads_dir = Path(args.downloads_dir).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    ensure_git_clone(CMSIS_REPO, downloads_dir / "CMSIS_5")
    ensure_git_clone(ETHOS_REPO, downloads_dir / "ethos-u-core-platform")

    if not args.skip_gcc:
        ensure_gcc(downloads_dir)

    print(f"Prepared release dependencies in {downloads_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
