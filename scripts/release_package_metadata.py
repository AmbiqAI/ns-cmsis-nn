#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path


SUPPORTED_ARCHES = {
    "cortex-m0": {
        "arch_label": "cm0",
        "target_cpu": "cortex-m0",
        "zephyr_cpu_symbols": ["CONFIG_CPU_CORTEX_M0", "CONFIG_CPU_CORTEX_M0PLUS"],
        "requires_fpu": False,
    },
    "cortex-m4+fp": {
        "arch_label": "cm4",
        "target_cpu": "cortex-m4",
        "zephyr_cpu_symbols": ["CONFIG_CPU_CORTEX_M4"],
        "requires_fpu": True,
    },
    "cortex-m55": {
        "arch_label": "cm55",
        "target_cpu": "cortex-m55",
        "zephyr_cpu_symbols": ["CONFIG_CPU_CORTEX_M55"],
        "requires_fpu": True,
    },
}

SUPPORTED_TOOLCHAINS = {
    "gcc": {
        "cmake_compiler_ids": ["GNU"],
        "zephyr_toolchain_variants": ["gnuarmemb", "cross-compile"],
    },
    "armclang": {
        "cmake_compiler_ids": ["ARMClang"],
        "zephyr_toolchain_variants": ["armclang"],
    },
    "llvm-et-arm": {
        "cmake_compiler_ids": ["Clang"],
        "zephyr_toolchain_variants": ["llvm", "llvm-et-arm"],
    },
}

HEADER_AFFECTING_DEFINES: list[str] = []
OPTIMIZATION = "Ofast"
VISIBILITY = "hidden-by-default"
ABI_VERSION = 1
PACKAGE_SCHEMA = 1
VARIANT_SCHEMA = 3
MINIMUM_CONSUMER_METADATA_SCHEMA = 1


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_package_cmake_support(package_dir: Path, repo_root: Path) -> None:
    cmake_dir = package_dir / "cmake"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmake_src = repo_root / "cmake" / "package"
    for filename in (
        "ns-cmsis-nn-config.cmake",
        "ns-cmsis-nn-config-version.cmake",
        "ns-cmsis-nn-select-variant.cmake",
    ):
        shutil.copy2(cmake_src / filename, cmake_dir / filename)


def _resolve_variant_properties(target_cpu: str, toolchain: str) -> tuple[str, str, str]:
    if toolchain in {"gcc", "armclang"}:
        if target_cpu == "cortex-m0":
            return "soft", "none", "armv6-m"
        if target_cpu == "cortex-m4":
            return "hard", "fpv4-sp-d16", "armv7e-m"
        return "hard", "mve-fp", "armv8.1-m.main"

    if target_cpu == "cortex-m0":
        return "soft", "none", "armv6-m"
    if target_cpu == "cortex-m4":
        return "softfp", "fpv4-sp-d16", "armv7e-m"
    return "softfp", "mve-fp", "armv8.1-m.main"


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

    float_abi, feature_level, isa = _resolve_variant_properties(target_cpu, toolchain)
    filename = (
        f"libns-cmsis-nn-{arch_label}-{target_cpu}-{toolchain}-{float_abi}-"
        f"{feature_level}-{OPTIMIZATION.lower()}.a"
    )
    return {
        "schema": VARIANT_SCHEMA,
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
        "abi_version": ABI_VERSION,
        "minimum_consumer_metadata_schema": MINIMUM_CONSUMER_METADATA_SCHEMA,
        "zephyr_compat": {
            "toolchain_variants": SUPPORTED_TOOLCHAINS[toolchain]["zephyr_toolchain_variants"],
            "cpu_kconfig_any_of": expected["zephyr_cpu_symbols"],
            "requires_fpu": expected["requires_fpu"],
        },
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
        metadata["abi_version"],
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


def _load_matrix_metadata(
    artifact_root: Path, archs: list[str], toolchains: list[str]
) -> list[tuple[dict[str, object], Path]]:
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

    resolved: list[tuple[dict[str, object], Path]] = []
    for variant in sorted(expected_variants, key=lambda item: str(item["file"])):
        payload, metadata_path = actual_by_key[_variant_key(variant)]
        archive_path = metadata_path.with_name(str(payload["file"]))
        if not archive_path.is_file():
            raise SystemExit(f"Archive missing for {metadata_path}: {archive_path}")
        payload = dict(payload)
        payload["library_sha256"] = _sha256(archive_path)
        resolved.append((payload, archive_path))
    return resolved


def _package_version_from_tag(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def _base_provenance(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_commit": args.source_commit,
        "source_ref": args.source_ref,
        "build_system": "github-actions" if args.workflow_name or args.workflow_run_id else "local",
        "workflow_name": args.workflow_name,
        "workflow_run_id": args.workflow_run_id,
    }


def _base_package_manifest(
    *,
    product: str,
    package_type: str,
    tag: str,
    args: argparse.Namespace,
    variants: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": PACKAGE_SCHEMA,
        "product": product,
        "package_type": package_type,
        "package_version": _package_version_from_tag(tag),
        "release_tag": tag,
        "generated_at_utc": _now_utc(),
        "abi_version": ABI_VERSION,
        "source_commit": args.source_commit,
        "source_ref": args.source_ref,
        "visibility_policy": VISIBILITY,
        "header_set_version": _package_version_from_tag(tag),
        "supported_variants": variants,
        "checksums": {"algorithm": "sha256", "file": "checksums.txt"},
        "provenance": _base_provenance(args),
        "toolchain_policy": {
            "supported_toolchains": sorted(SUPPORTED_TOOLCHAINS.keys()),
        },
        "consumption": {
            "minimum_consumer_metadata_schema": MINIMUM_CONSUMER_METADATA_SCHEMA,
            "public_headers_dir": "include",
            "libraries_dir": "lib",
        },
    }


def _copy_common_package_files(package_dir: Path, repo_root: Path, readme_src: Path, readme_name: str) -> None:
    include_src = repo_root / "Include"
    if not include_src.is_dir():
        raise SystemExit(f"Public header directory not found: {include_src}")
    _copy_tree(include_src, package_dir / "include")

    license_src = repo_root / "LICENSES"
    if license_src.is_dir():
        _copy_tree(license_src, package_dir / "LICENSES")

    shutil.copy2(readme_src, package_dir / readme_name)


def _write_checksums(package_dir: Path, output: Path) -> None:
    entries: list[tuple[str, str]] = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.resolve() != output.resolve():
            entries.append((_sha256(path), path.relative_to(package_dir).as_posix()))
    output.write_text("".join(f"{digest}  {rel}\n" for digest, rel in entries), encoding="utf-8")


def _validate_package(
    package_dir: Path,
    package_type: str,
    *,
    require_binary_only: bool,
) -> None:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Package manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_fields = [
        "schema",
        "product",
        "package_type",
        "package_version",
        "release_tag",
        "generated_at_utc",
        "abi_version",
        "source_commit",
        "source_ref",
        "visibility_policy",
        "header_set_version",
        "supported_variants",
        "checksums",
        "provenance",
        "toolchain_policy",
        "consumption",
    ]
    for field in required_fields:
        if field not in manifest:
            raise SystemExit(f"Manifest missing required field '{field}' in {manifest_path}")

    if manifest["package_type"] != package_type:
        raise SystemExit(
            f"Manifest package_type mismatch for {package_dir}: expected {package_type}, got {manifest['package_type']}"
        )

    required_paths = {
        "binary-sdk": [
            package_dir / "include",
            package_dir / "lib",
            package_dir / "cmake" / "ns-cmsis-nn-config.cmake",
            package_dir / "cmake" / "ns-cmsis-nn-config-version.cmake",
            package_dir / "cmake" / "ns-cmsis-nn-select-variant.cmake",
            package_dir / "README.md",
            package_dir / "checksums.txt",
        ],
        "zephyr-binary": [
            package_dir / "include",
            package_dir / "lib",
            package_dir / "zephyr" / "module.yml",
            package_dir / "zephyr" / "CMakeLists.txt",
            package_dir / "zephyr" / "Kconfig",
            package_dir / "README-ZEPHYR.md",
            package_dir / "checksums.txt",
        ],
    }[package_type]

    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"Required package path missing: {path}")

    if require_binary_only:
        blocked_dirs = [package_dir / "Source", package_dir / "Tests"]
        for path in blocked_dirs:
            if path.exists():
                raise SystemExit(f"Customer package must not include {path.relative_to(package_dir)}")

        blocked_suffixes = {".c", ".cc", ".cpp", ".cxx"}
        for path in package_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in blocked_suffixes:
                raise SystemExit(f"Customer package must not include source file: {path.relative_to(package_dir)}")

def cmd_variant_filename(args: argparse.Namespace) -> int:
    payload = _variant_metadata(args.arch, args.arch_label, args.target_cpu, args.toolchain)
    print(payload["file"])
    return 0


def cmd_write_variant_metadata(args: argparse.Namespace) -> int:
    payload = _variant_metadata(args.arch, args.arch_label, args.target_cpu, args.toolchain)
    payload["generated_at_utc"] = _now_utc()
    payload["source_commit"] = args.source_commit
    payload["source_ref"] = args.source_ref
    _write_json(Path(args.output), payload)
    return 0


def cmd_assemble_staticlib_bundle(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root).resolve()
    outdir = Path(args.outdir).resolve()
    repo_root = Path(args.repo_root).resolve()
    archs = [item for item in args.archs.split(",") if item]
    toolchains = [item for item in args.toolchains.split(",") if item]
    resolved = _load_matrix_metadata(artifact_root, archs, toolchains)

    bundle_dir = outdir / f"ns-cmsis-nn-binary-sdk-{args.tag}"
    zip_path = outdir / f"ns-cmsis-nn-binary-sdk-{args.tag}.zip"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    if zip_path.exists():
        zip_path.unlink()

    (bundle_dir / "lib").mkdir(parents=True, exist_ok=True)
    _copy_common_package_files(
        bundle_dir, repo_root, repo_root / "Documentation" / "binary_sdk_release.md", "README.md"
    )
    _copy_package_cmake_support(bundle_dir, repo_root)

    variants: list[dict[str, object]] = []
    for payload, archive_path in resolved:
        shutil.copy2(archive_path, bundle_dir / "lib" / archive_path.name)
        variants.append(payload)

    manifest = _base_package_manifest(
        product="ns-cmsis-nn-binary-sdk",
        package_type="binary-sdk",
        tag=args.tag,
        args=args,
        variants=variants,
    )
    _write_json(bundle_dir / "manifest.json", manifest)
    _write_checksums(bundle_dir, bundle_dir / "checksums.txt")
    return 0


def cmd_assemble_zephyr_binary_package(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root).resolve()
    outdir = Path(args.outdir).resolve()
    repo_root = Path(args.repo_root).resolve()
    archs = [item for item in args.archs.split(",") if item]
    toolchains = [item for item in args.toolchains.split(",") if item]
    resolved = _load_matrix_metadata(artifact_root, archs, toolchains)

    package_dir = outdir / f"ns-cmsis-nn-zephyr-binary-{args.tag}"
    zip_path = outdir / f"ns-cmsis-nn-zephyr-binary-{args.tag}.zip"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    if zip_path.exists():
        zip_path.unlink()

    (package_dir / "lib").mkdir(parents=True, exist_ok=True)
    _copy_common_package_files(
        package_dir, repo_root, repo_root / "Documentation" / "zephyr_module_release.md", "README-ZEPHYR.md"
    )
    _copy_tree(repo_root / "zephyr", package_dir / "zephyr")
    _copy_package_cmake_support(package_dir, repo_root)

    variants: list[dict[str, object]] = []
    for payload, archive_path in resolved:
        shutil.copy2(archive_path, package_dir / "lib" / archive_path.name)
        variants.append(payload)

    manifest = _base_package_manifest(
        product="ns-cmsis-nn-zephyr-binary",
        package_type="zephyr-binary",
        tag=args.tag,
        args=args,
        variants=variants,
    )
    manifest["zephyr"] = {
        "module_name": "ns-cmsis-nn",
        "supported_toolchains": {
            toolchain: data["zephyr_toolchain_variants"] for toolchain, data in SUPPORTED_TOOLCHAINS.items()
        },
        "selection_rules": {
            "cpu": {arch: data["zephyr_cpu_symbols"] for arch, data in SUPPORTED_ARCHES.items()},
            "requires_fpu": {
                arch: data["requires_fpu"] for arch, data in SUPPORTED_ARCHES.items()
            },
        },
        "unsupported_conditions": [
            "toolchain family not in supported_toolchains",
            "cpu target not in supported cpu matrix",
            "FPU not enabled for variants that require hard-float binaries",
        ],
        "cmake_entrypoint": "zephyr/CMakeLists.txt",
        "kconfig_entrypoint": "zephyr/Kconfig",
    }
    _write_json(package_dir / "manifest.json", manifest)
    _write_checksums(package_dir, package_dir / "checksums.txt")
    return 0


def cmd_write_checksums(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir).resolve()
    output = Path(args.output).resolve()
    _write_checksums(package_dir, output)
    return 0


def cmd_validate_package(args: argparse.Namespace) -> int:
    _validate_package(
        Path(args.package_dir).resolve(),
        args.package_type,
        require_binary_only=args.require_binary_only,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release metadata helpers for ns-cmsis-nn packaging.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
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
    write_variant.add_argument("--source-commit", default="local")
    write_variant.add_argument("--source-ref", default="local")
    write_variant.set_defaults(func=cmd_write_variant_metadata)

    for name, func in (
        ("assemble-staticlib-bundle", cmd_assemble_staticlib_bundle),
        ("assemble-zephyr-binary-package", cmd_assemble_zephyr_binary_package),
    ):
        assemble = subparsers.add_parser(name)
        assemble.add_argument("--artifact-root", required=True)
        assemble.add_argument("--outdir", required=True)
        assemble.add_argument("--tag", required=True)
        assemble.add_argument("--archs", required=True)
        assemble.add_argument("--toolchains", required=True)
        assemble.add_argument("--source-commit", default="local")
        assemble.add_argument("--source-ref", default="local")
        assemble.add_argument("--workflow-name", default="")
        assemble.add_argument("--workflow-run-id", default="")
        assemble.set_defaults(func=func)

    checksums = subparsers.add_parser("write-checksums")
    checksums.add_argument("--package-dir", required=True)
    checksums.add_argument("--output", required=True)
    checksums.set_defaults(func=cmd_write_checksums)

    validate = subparsers.add_parser("validate-package")
    validate.add_argument("--package-dir", required=True)
    validate.add_argument("--package-type", required=True, choices=["binary-sdk", "zephyr-binary"])
    validate.add_argument("--require-binary-only", action="store_true")
    validate.set_defaults(func=cmd_validate_package)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
