#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path


SUPPORTED_ARCHES = {
    "cortex-m0": {"arch_label": "cm0", "target_cpu": "cortex-m0"},
    "cortex-m4+fp": {"arch_label": "cm4", "target_cpu": "cortex-m4"},
    "cortex-m55": {"arch_label": "cm55", "target_cpu": "cortex-m55"},
}

SUPPORTED_TOOLCHAINS = {"gcc", "armclang", "llvm-et-arm"}
HEADER_AFFECTING_DEFINES: list[str] = []
OPTIMIZATION = "Ofast"
VISIBILITY = "hidden-by-default"


def _variant_metadata(arch: str, arch_label: str, target_cpu: str, toolchain: str) -> dict[str, object]:
    if arch not in SUPPORTED_ARCHES:
        raise SystemExit(f"Unsupported arch: {arch}")
    expected = SUPPORTED_ARCHES[arch]
    if expected["arch_label"] != arch_label:
        raise SystemExit(f"arch_label mismatch for {arch}: expected {expected['arch_label']}, got {arch_label}")
    if expected["target_cpu"] != target_cpu:
        raise SystemExit(f"target_cpu mismatch for {arch}: expected {expected['target_cpu']}, got {target_cpu}")
    if toolchain not in SUPPORTED_TOOLCHAINS:
        raise SystemExit(f"Unsupported toolchain: {toolchain}")

    if toolchain == "gcc":
        if target_cpu == "cortex-m0":
            float_abi = "soft"
            feature_level = "none"
            isa = "armv6-m"
        elif target_cpu == "cortex-m4":
            float_abi = "hard"
            feature_level = "fpv4-sp-d16"
            isa = "armv7e-m"
        else:
            float_abi = "hard"
            feature_level = "mve-fp"
            isa = "armv8.1-m.main"
    elif toolchain == "armclang":
        if target_cpu == "cortex-m0":
            float_abi = "soft"
            feature_level = "none"
            isa = "armv6-m"
        elif target_cpu == "cortex-m4":
            float_abi = "hard"
            feature_level = "fpv4-sp-d16"
            isa = "armv7e-m"
        else:
            float_abi = "hard"
            feature_level = "mve-fp"
            isa = "armv8.1-m.main"
    else:
        if target_cpu == "cortex-m0":
            float_abi = "soft"
            feature_level = "none"
            isa = "armv6-m"
        elif target_cpu == "cortex-m4":
            float_abi = "softfp"
            feature_level = "fpv4-sp-d16"
            isa = "armv7e-m"
        else:
            float_abi = "softfp"
            feature_level = "mve-fp"
            isa = "armv8.1-m.main"

    filename = (
        f"libns-cmsis-nn-{arch_label}-{target_cpu}-{toolchain}-{float_abi}-"
        f"{feature_level}-{OPTIMIZATION.lower()}-hidden.a"
    )
    return {
        "schema": 2,
        "product": "ns-cmsis-nn-staticlib",
        "public_arch": arch,
        "arch_label": arch_label,
        "target_cpu": target_cpu,
        "toolchain": toolchain,
        "float_abi": float_abi,
        "feature_level": feature_level,
        "isa": isa,
        "optimization": OPTIMIZATION,
        "visibility": VISIBILITY,
        "header_affecting_defines": HEADER_AFFECTING_DEFINES,
        "file": filename,
    }


def _variant_key(metadata: dict[str, object]) -> tuple[object, ...]:
    return (
        metadata["public_arch"],
        metadata["arch_label"],
        metadata["target_cpu"],
        metadata["toolchain"],
        metadata["float_abi"],
        metadata["feature_level"],
        metadata["optimization"],
        metadata["visibility"],
        tuple(metadata["header_affecting_defines"]),
    )


def _public_identity_key(metadata: dict[str, object]) -> tuple[object, ...]:
    return (
        metadata["public_arch"],
        metadata["arch_label"],
        metadata["target_cpu"],
        metadata["toolchain"],
    )


def _expected_variants(archs: list[str], toolchains: list[str]) -> list[dict[str, object]]:
    variants = []
    for arch in archs:
        resolved = SUPPORTED_ARCHES.get(arch)
        if resolved is None:
            raise SystemExit(f"Unsupported arch in matrix: {arch}")
        for toolchain in toolchains:
            variants.append(_variant_metadata(arch, resolved["arch_label"], resolved["target_cpu"], toolchain))
    return variants


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cmd_variant_filename(args: argparse.Namespace) -> int:
    payload = _variant_metadata(args.arch, args.arch_label, args.target_cpu, args.toolchain)
    print(payload["file"])
    return 0


def cmd_write_variant_metadata(args: argparse.Namespace) -> int:
    payload = _variant_metadata(args.arch, args.arch_label, args.target_cpu, args.toolchain)
    payload["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(Path(args.output), payload)
    return 0


def cmd_assemble_staticlib_bundle(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root).resolve()
    outdir = Path(args.outdir).resolve()
    archs = [item for item in args.archs.split(",") if item]
    toolchains = [item for item in args.toolchains.split(",") if item]
    expected_variants = _expected_variants(archs, toolchains)
    expected_by_key = {_variant_key(item): item for item in expected_variants}

    metadata_files = sorted(artifact_root.rglob("libns-cmsis-nn-*.json"))
    actual_by_key: dict[tuple[object, ...], tuple[dict[str, object], Path]] = {}
    public_identity_map: dict[tuple[object, ...], tuple[tuple[object, ...], Path]] = {}

    for metadata_path in metadata_files:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in metadata file {metadata_path}: {exc}") from exc
        if payload.get("product") != "ns-cmsis-nn-staticlib":
            continue
        key = _variant_key(payload)
        public_key = _public_identity_key(payload)
        if key in actual_by_key:
            first_path = actual_by_key[key][1]
            raise SystemExit(f"Duplicate staticlib variant metadata for {payload['file']}: {first_path} and {metadata_path}")
        if public_key in public_identity_map and public_identity_map[public_key][0] != key:
            first_path = public_identity_map[public_key][1]
            raise SystemExit(
                f"Conflicting ABI metadata for public variant {public_key}: {first_path} and {metadata_path}"
            )
        public_identity_map[public_key] = (key, metadata_path)
        actual_by_key[key] = (payload, metadata_path)

    missing = [item for key, item in expected_by_key.items() if key not in actual_by_key]
    if missing:
        missing_names = ", ".join(item["file"] for item in missing)
        raise SystemExit(f"Missing expected static library variants: {missing_names}")

    bundle_dir = outdir / f"ns-cmsis-nn-staticlibs-{args.tag}"
    zip_path = outdir / f"ns-cmsis-nn-staticlibs-{args.tag}.zip"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    if zip_path.exists():
        zip_path.unlink()
    (bundle_dir / "lib").mkdir(parents=True, exist_ok=True)

    libraries: list[dict[str, object]] = []
    for variant in sorted(expected_variants, key=lambda item: str(item["file"])):
        payload, metadata_path = actual_by_key[_variant_key(variant)]
        archive_path = metadata_path.with_name(str(payload["file"]))
        if not archive_path.is_file():
            raise SystemExit(f"Archive missing for {metadata_path}: {archive_path}")
        shutil.copy2(archive_path, bundle_dir / "lib" / archive_path.name)
        libraries.append(payload)

    manifest = {
        "schema": 2,
        "product": "ns-cmsis-nn-staticlibs",
        "tag": args.tag,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bundle_policy": {
            "optimization": OPTIMIZATION,
            "visibility": VISIBILITY,
            "header_affecting_defines": HEADER_AFFECTING_DEFINES,
        },
        "libraries": libraries,
    }
    _write_json(bundle_dir / "MANIFEST.json", manifest)
    return 0


def cmd_write_zephyr_manifest(args: argparse.Namespace) -> int:
    payload = {
        "schema": 1,
        "product": "ns-cmsis-nn-zephyr-module",
        "tag": args.tag,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "module_name": "ns-cmsis-nn",
        "distribution_model": "source-module",
        "required_paths": [
            "zephyr/module.yml",
            "zephyr/CMakeLists.txt",
            "zephyr/Kconfig",
            "Source",
            "Include",
        ],
    }
    _write_json(Path(args.output), payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release metadata helpers for ns-cmsis-nn packaging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    variant_filename = subparsers.add_parser("variant-filename")
    for target in (variant_filename,):
        target.add_argument("--arch", required=True)
        target.add_argument("--arch-label", required=True)
        target.add_argument("--target-cpu", required=True)
        target.add_argument("--toolchain", required=True)
    variant_filename.set_defaults(func=cmd_variant_filename)

    write_variant = subparsers.add_parser("write-variant-metadata")
    write_variant.add_argument("--arch", required=True)
    write_variant.add_argument("--arch-label", required=True)
    write_variant.add_argument("--target-cpu", required=True)
    write_variant.add_argument("--toolchain", required=True)
    write_variant.add_argument("--output", required=True)
    write_variant.set_defaults(func=cmd_write_variant_metadata)

    assemble = subparsers.add_parser("assemble-staticlib-bundle")
    assemble.add_argument("--artifact-root", required=True)
    assemble.add_argument("--outdir", required=True)
    assemble.add_argument("--tag", required=True)
    assemble.add_argument("--archs", required=True)
    assemble.add_argument("--toolchains", required=True)
    assemble.set_defaults(func=cmd_assemble_staticlib_bundle)

    zephyr_manifest = subparsers.add_parser("write-zephyr-manifest")
    zephyr_manifest.add_argument("--tag", required=True)
    zephyr_manifest.add_argument("--output", required=True)
    zephyr_manifest.set_defaults(func=cmd_write_zephyr_manifest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
