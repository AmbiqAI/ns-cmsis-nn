#!/usr/bin/env python3
"""Download release-build dependencies into a repo-owned cache directory."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import tempfile
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


CMSIS_DEPENDENCY = {
    "name": "CMSIS_5",
    "url": "https://github.com/ARM-software/CMSIS_5.git",
    "ref": "55b19837f5703e418ca37894d5745b1dc05e4c91",
}

ETHOS_U_CORE_PLATFORM_DEPENDENCY = {
    "name": "ethos-u-core-platform",
    "url": "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git",
    "ref": "02d02901842fb26c077adc1061c2650d89f9f0e5",
}

GCC_DOWNLOADS = {
    "x86_64": {
        "url": "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz",
        "sha256": "62a63b981fe391a9cbad7ef51b17e49aeaa3e7b0d029b36ca1e9c3b2a9b78823",
    },
    "aarch64": {
        "url": "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz",
        "sha256": "87330bab085dd8749d4ed0ad633674b9dc48b237b61069e3b481abd364d0a684",
    },
    "arm64": {
        "url": "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-aarch64-arm-none-eabi.tar.xz",
        "sha256": "87330bab085dd8749d4ed0ad633674b9dc48b237b61069e3b481abd364d0a684",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", required=True, help="Directory that will hold downloaded dependencies")
    parser.add_argument("--skip-gcc", action="store_true", help="Do not download the Arm GNU toolchain")
    return parser.parse_args()


def run(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def ensure_git_checkout(dependency: dict[str, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        subprocess.run(["git", "clone", dependency["url"], str(destination)], check=True)
    elif not (destination / ".git").exists():
        raise SystemExit(f"Existing dependency path is not a git checkout: {destination}")

    subprocess.run(["git", "fetch", "--tags", "origin"], cwd=destination, check=True)
    subprocess.run(["git", "fetch", "origin", dependency["ref"]], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", dependency["ref"]], cwd=destination, check=True)

    resolved_ref = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if resolved_ref != dependency["ref"]:
        raise SystemExit(
            f"Dependency {dependency['name']} resolved to {resolved_ref}, expected {dependency['ref']}"
        )


def verify_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
        )


def safe_extract_tar(archive_path: Path, destination: Path, strip_components: int = 0) -> None:
    destination = destination.resolve()

    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute():
                raise SystemExit(f"Archive member has an absolute path: {member.name}")

            parts = [part for part in member_path.parts if part not in ("", ".")]
            if any(part == ".." for part in parts):
                raise SystemExit(f"Archive member escapes extraction root: {member.name}")
            if len(parts) <= strip_components:
                continue

            relative_parts = parts[strip_components:]
            target_path = destination.joinpath(*relative_parts)
            resolved_target = target_path.resolve()
            if resolved_target != destination and destination not in resolved_target.parents:
                raise SystemExit(f"Archive member escapes extraction root: {member.name}")

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                target_path.chmod(member.mode & 0o777)
                continue

            if member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SystemExit(f"Archive member could not be read: {member.name}")
                with extracted, target_path.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                target_path.chmod(member.mode & 0o777)
                continue

            raise SystemExit(f"Unsupported archive member type for {member.name}")


def ensure_gcc(downloads_dir: Path) -> None:
    gcc_dir = downloads_dir / "arm_gcc_download"
    if (gcc_dir / "bin" / "arm-none-eabi-gcc").exists():
        return

    machine = platform.machine().lower()
    gcc_download = GCC_DOWNLOADS.get(machine)
    if gcc_download is None:
        raise SystemExit(f"Unsupported machine architecture for GCC download: {machine}")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "arm-gcc.tar.xz"
        staging_dir = Path(temp_dir) / "arm-gcc"
        urllib.request.urlretrieve(gcc_download["url"], archive_path)
        verify_sha256(archive_path, gcc_download["sha256"])
        staging_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_tar(archive_path, staging_dir, strip_components=1)

        if gcc_dir.exists():
            shutil.rmtree(gcc_dir)
        shutil.move(str(staging_dir), str(gcc_dir))

    if not (gcc_dir / "bin" / "arm-none-eabi-gcc").exists():
        raise SystemExit(f"Downloaded GCC toolchain is missing arm-none-eabi-gcc: {gcc_dir}")


def main() -> int:
    args = parse_args()
    downloads_dir = Path(args.downloads_dir).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    ensure_git_checkout(CMSIS_DEPENDENCY, downloads_dir / "CMSIS_5")
    ensure_git_checkout(ETHOS_U_CORE_PLATFORM_DEPENDENCY, downloads_dir / "ethos-u-core-platform")

    if not args.skip_gcc:
        ensure_gcc(downloads_dir)

    print(f"Prepared release dependencies in {downloads_dir}")
    print(f"CMSIS_5 pinned to {CMSIS_DEPENDENCY['ref']}")
    print(f"ethos-u-core-platform pinned to {ETHOS_U_CORE_PLATFORM_DEPENDENCY['ref']}")
    if not args.skip_gcc:
        print(f"Arm GNU toolchain pinned to {GCC_DOWNLOADS[platform.machine().lower()]['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
