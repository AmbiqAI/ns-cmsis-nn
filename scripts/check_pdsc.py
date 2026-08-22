#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# CMSIS-Pack manifest contract test for Ambiq.NS-CMSIS-NN.pdsc.
#
# Mirrors the SSoT contract (#165) and the Zephyr / NSX wiring tests
# (#163 / #164) — pure-Python, no Open-CMSIS-Pack tooling required, runs
# in <1s. Pins the invariants the heliaRT consumer relies on:
#
#   1. Pack identity (schemaVersion, name, vendor) matches the heliaRT
#      contract.
#   2. Component identity (Cclass / Cgroup / Csub / Cvendor) matches the
#      heliaRT contract.
#   3. Version sync — the latest <release version>, the component
#      Cversion, the markers in Include/arm_nn_types.h, and the NSX
#      module manifest version all agree. Cross-checked against
#      .release-please-manifest.json so a manual bump that forgets one
#      of these fails CI.
#   4. License plumbing — every <license name=...> in the pdsc points at
#      a file that exists, and the repo top-level LICENSE / LICENSES/
#      files are all declared.
#   5. File existence — every <file name="..."/> path is on disk.
#   6. Coverage cross-check — every Source/**/*.c that exists in the
#      repo is enumerated under <files> with category="source", so
#      adding a kernel without updating the pack fails CI (replaces the
#      old check_pdsc.sh diff).
#   7. Float dtype gating — every float source the pdsc ships collapses
#      to an empty translation unit when its dtype is off. The pdsc lists
#      them unconditionally while cmake/ns_cmsis_nn.cmake lists them only
#      under the matching ARM_NN_ENABLE_F32/F16 block, so an ungated
#      kernel builds fine in CI and breaks pack consumers (#264).
#   8. extra-files annotation coverage — release-please's own generic
#      updater (generic.js) silently leaves a line untouched if nothing
#      on it matches the scope's value regex, and its manifest strategy
#      only *warns* (does not fail) about an extra-files path that does
#      not exist. Both mean a broken or inert extra-files entry ships
#      silently, which is the exact failure class this script exists to
#      catch elsewhere. Assert every release-please-config.json
#      extra-files entry exists, carries at least one working
#      x-release-please-* annotation (or is on the explicit
#      LITERAL_ONLY_EXTRA_FILES allowlist), and that every annotated /
#      allowlisted value agrees with the canonical arm_nn_types.h version.
#   9. SSoT/pdsc source-list agreement — the set of sources
#      cmake/ns_cmsis_nn.cmake can resolve (unioned over every dtype
#      gate) equals the set the pdsc ships, modulo an explicit
#      allowlist. The two consumption paths are independent, so a file
#      that is in the pdsc but not the SSoT builds for CMSIS-Pack
#      consumers and link-errors for CMake / Zephyr / NSX consumers
#      (#268), and a file in the SSoT but not the pdsc does the
#      reverse.
#

from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PDSC = REPO / "Ambiq.NS-CMSIS-NN.pdsc"
SSOT_CMAKE = REPO / "cmake" / "ns_cmsis_nn.cmake"
TYPES_H = REPO / "Include" / "arm_nn_types.h"
NSX_MODULE = REPO / "nsx" / "nsx-module.yaml"
RP_MANIFEST = REPO / ".release-please-manifest.json"
RP_CONFIG = REPO / "release-please-config.json"

# release-please's generic updater (src/updaters/generic.js) matches these
# same patterns: x-release-please-<scope> anywhere on a line selects that
# line for replacement, using a bare \d+\.\d+\.\d+ for "version" and a bare
# \d+ for "major"/"minor"/"patch". Mirrored here (not imported — this is a
# pure-Python script with no Node dependency) so this check fails the same
# way the real updater would silently no-op.
RP_INLINE_SCOPE_RE = re.compile(
    r"x-release-please-(major|minor|patch|version-date|version|date)"
)
# Block form: content between a x-release-please-start-<scope> line and the
# next x-release-please-end line is scope-replaced line by line (used where
# the annotation and its value can't share a line without also being on a
# line that isn't safe to put an inline comment on -- e.g. inside a Kconfig
# `help` block, which has no comment syntax of its own; see zephyr/Kconfig's
# NS_CMSIS_NN_PREBUILT_PATH entry). Checked with lower priority than the
# inline regex, same as generic.js: a line only opens/continues a block if
# it didn't already match the inline regex.
RP_BLOCK_START_RE = re.compile(
    r"x-release-please-start-(major|minor|patch|version-date|version|date)"
)
RP_BLOCK_END_RE = re.compile(r"x-release-please-end")
RP_VERSION_TRIPLET_RE = re.compile(r"\d+\.\d+\.\d+")
RP_BARE_INT_RE = re.compile(r"\d+\b")

# Extra-files entries whose version literal cannot carry an
# x-release-please annotation, mapped to a regex whose first capture group
# is the literal to check against the canonical version. Today this is
# just docs/guides/toolchains.md's manifest.json illustration: JSON has no
# comment syntax, and wrapping it in an x-release-please-start-version/-end
# block would also rewrite the unrelated ATfE compiler version a few lines
# below ("toolchain": {"version": "19.1.5"}), since the block updater
# rewrites the first semver-shaped string on *every* line it spans. The
# regex is anchored to exactly two leading spaces so it matches only the
# top-level manifest.json `"version"` field, not the nested toolchain one
# (four leading spaces) — if that example's fields are ever reordered or
# re-indented, the anchor (correctly) stops matching and this check fails
# loudly instead of silently checking the wrong field.
LITERAL_ONLY_EXTRA_FILES: dict[str, re.Pattern[str]] = {
    "docs/guides/toolchains.md": re.compile(r'^ {2}"version": "([^"]+)",?$', re.MULTILINE),
}

EXPECTED_PACK = {
    "schemaVersion": "1.7.36",
    "name": "NS-CMSIS-NN",
    "vendor": "Ambiq",
}

# The heliaRT consumer pins this exact 4-tuple. Renaming any of them is
# a breaking change for downstream packs and must be a deliberate decision.
EXPECTED_COMPONENT = {
    "Cclass": "Machine Learning",
    "Cgroup": "NN Lib",
    "Csub": "heliaCORE",
    "Cvendor": "Ambiq",
}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def parse_pdsc_text() -> tuple[ET.Element, str]:
    text = PDSC.read_text()
    return ET.fromstring(text), text


def check_pack_identity(pkg: ET.Element) -> None:
    if pkg.tag != "package":
        fail(f"root element is <{pkg.tag}>, expected <package>")
        return
    actual_schema = pkg.attrib.get("schemaVersion", "")
    if actual_schema != EXPECTED_PACK["schemaVersion"]:
        fail(
            f"schemaVersion='{actual_schema}', expected "
            f"'{EXPECTED_PACK['schemaVersion']}'"
        )
    for tag in ("name", "vendor"):
        node = pkg.find(tag)
        actual = (node.text or "").strip() if node is not None else None
        if actual != EXPECTED_PACK[tag]:
            fail(f"<{tag}>='{actual}', expected '{EXPECTED_PACK[tag]}'")


def find_component(pkg: ET.Element) -> ET.Element | None:
    comps = pkg.find("components")
    if comps is None:
        fail("missing <components> element")
        return None
    for c in comps.findall("component"):
        if all(c.attrib.get(k) == v for k, v in EXPECTED_COMPONENT.items()):
            return c
    fail(
        "no <component> with the expected heliaCORE identity "
        f"({EXPECTED_COMPONENT}); component renames are breaking — bump "
        "the pack major and update heliaRT before changing any of these."
    )
    return None


def parse_header_version(text: str) -> tuple[str, str, str] | None:
    """Pull (major, minor, patch) strings out of arm_nn_types.h's
    NS_CMSIS_NN_VERSION_* macros. Shared by check_versions() (pdsc/nsx/
    manifest sync) and check_extra_files_annotations() (extra-files
    cross-check), so both agree on exactly one canonical source."""
    m_major = re.search(r"NS_CMSIS_NN_VERSION_MAJOR\s*\((\d+)\)", text)
    m_minor = re.search(r"NS_CMSIS_NN_VERSION_MINOR\s*\((\d+)\)", text)
    m_patch = re.search(r"NS_CMSIS_NN_VERSION_PATCH\s*\((\d+)\)", text)
    if m_major and m_minor and m_patch:
        return m_major.group(1), m_minor.group(1), m_patch.group(1)
    return None


def check_versions(pkg: ET.Element, comp: ET.Element | None) -> None:
    # Latest release version (first <release> under <releases>).
    releases = pkg.find("releases")
    rel_ver: str | None = None
    if releases is not None:
        first = releases.find("release")
        if first is not None:
            rel_ver = first.attrib.get("version")
    if not rel_ver:
        fail("could not find a <release version=...> element")

    # Component Cversion.
    comp_ver = comp.attrib.get("Cversion") if comp is not None else None
    if not comp_ver:
        fail("component is missing the Cversion attribute")

    # arm_nn_types.h version markers.
    th = TYPES_H.read_text()
    header_parts = parse_header_version(th)
    m_rev = re.search(r'\$Revision:\s*"v(\d+\.\d+\.\d+)"', th)
    header_ver: str | None = None
    if header_parts:
        header_ver = ".".join(header_parts)
    else:
        fail("Include/arm_nn_types.h is missing one of the NS_CMSIS_NN_VERSION_* markers")

    # Upstream Arm headers may carry an Arm-local revision marker such as
    # `$Revision: V.3.6.1`; NS-CMSIS-NN's package version is tracked by the
    # explicit NS_CMSIS_NN_VERSION_* macros instead.
    rev_ver = m_rev.group(1) if m_rev else header_ver

    # NSX module manifest version.
    nsx_ver: str | None = None
    try:
        nsx_text = NSX_MODULE.read_text()
        m_nsx = re.search(
            r'^\s*version:\s*["\']?([^"\'\s#]+)', nsx_text, re.MULTILINE
        )
        if m_nsx:
            nsx_ver = m_nsx.group(1)
        else:
            fail("nsx/nsx-module.yaml is missing module.version")
    except Exception as e:
        fail(f"could not read {NSX_MODULE.relative_to(REPO)}: {e}")

    # release-please manifest (canonical source of truth used by the bot).
    try:
        manifest = json.loads(RP_MANIFEST.read_text())
        manifest_ver = manifest.get(".")
    except Exception as e:
        manifest_ver = None
        fail(f"could not read {RP_MANIFEST.relative_to(REPO)}: {e}")

    # All five must agree.
    versions = {
        "<release version>": rel_ver,
        "component Cversion": comp_ver,
        "arm_nn_types.h NS_CMSIS_NN_VERSION_*": header_ver,
        "arm_nn_types.h $Revision or NS_CMSIS_NN_VERSION_*": rev_ver,
        "nsx/nsx-module.yaml module.version": nsx_ver,
        ".release-please-manifest.json": manifest_ver,
    }
    unique = {v for v in versions.values() if v}
    if len(unique) > 1:
        detail = ", ".join(f"{k}={v!r}" for k, v in versions.items())
        fail(f"version markers disagree — release-please bump incomplete: {detail}")


def check_licenses(pkg: ET.Element) -> None:
    seen: set[str] = set()
    for lic in pkg.iter("license"):
        name = lic.attrib.get("name")
        if not name:
            continue
        seen.add(name)
        p = REPO / name
        if not p.is_file():
            fail(f"<license name='{name}'/> does not exist on disk")
    # Both top-level license files must be declared.
    for required in ("LICENSE", "LICENSES/Apache-2.0.txt"):
        if required not in seen:
            fail(f"pdsc does not declare a <license> entry for {required}")


def collect_file_entries(comp: ET.Element | None) -> list[tuple[str, str]]:
    if comp is None:
        return []
    out: list[tuple[str, str]] = []
    for files in comp.findall("files"):
        for f in files.findall("file"):
            name = f.attrib.get("name") or ""
            cat = f.attrib.get("category") or ""
            out.append((cat, name))
    return out


def check_file_existence(entries: list[tuple[str, str]]) -> None:
    # Documentation/html/ is generated by Doxygen at pack-build time
    # (.gitignored), so a clean checkout doesn't have it. We still want to
    # validate the path is well-formed, but skip the on-disk check for
    # generated artefacts.
    GENERATED_PREFIXES = ("Documentation/html/",)
    for cat, name in entries:
        if not name:
            fail(f"<file category='{cat}'/> is missing the 'name' attribute")
            continue
        if any(name.startswith(p) for p in GENERATED_PREFIXES):
            continue
        if not (REPO / name).is_file():
            fail(f"<file name='{name}'/> not found on disk")


def tracked_source_files() -> list[str] | None:
    """Repo-relative paths of every tracked Source/**/*.c.

    Use `git ls-files` rather than a filesystem glob: matches the legacy
    check_pdsc.sh behaviour, and avoids false-positives from untracked
    .c files left around during local development. Returns None (after
    recording a failure) when git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "Source/"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        fail(f"`git ls-files Source/` failed: {e}")
        return None
    return sorted(p for p in out.splitlines() if p.endswith(".c"))


def check_source_coverage(entries: list[tuple[str, str]]) -> None:
    listed = sorted(name for cat, name in entries if cat == "source")
    actual = tracked_source_files()
    if actual is None:
        return
    missing_in_pdsc = set(actual) - set(listed)
    extra_in_pdsc = set(listed) - set(actual)
    for m in sorted(missing_in_pdsc):
        fail(f"Source/ file not enumerated in pdsc: {m}")
    for x in sorted(extra_in_pdsc):
        fail(f"pdsc references nonexistent Source/ file: {x}")


# --- float dtype gating -------------------------------------------------
#
# Every float kernel the pdsc ships must collapse to an empty translation
# unit when its dtype is disabled. The pdsc lists float sources
# unconditionally, so a CMSIS-Pack consumer (and the module.mk glob) hands
# them to the compiler even in an integer-only build; cmake/ns_cmsis_nn.cmake
# adds them only under the matching ARM_NN_ENABLE_F32/F16 block, which is why
# a missing gate is invisible to the CMake build and to CI until a pack
# consumer trips over it.

# Matches the dtype token anywhere in the basename, not just as a suffix:
# arm_convolve_f16_fast_small_kernel.c is as much a float source as
# arm_convolve_f16.c, and a suffix-only pattern would wave it through.
FLOAT_SRC_RE = re.compile(r"(?:^|_)(?:f16|f32|fp16|fp32)(?:_|\.c$)")
GATE_RE = re.compile(r"^\s*#\s*if\s+ARM_NN_ENABLE_(F16|F32)\b")
COND_OPEN_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
COND_CLOSE_RE = re.compile(r"^\s*#\s*endif\b")
PP_RE = re.compile(r"^\s*#")

# Frozen legacy allowlist. Do not extend it for new kernels: a new float
# source must carry its own ARM_NN_ENABLE_F32/F16 gate.
#   - the two *_fp16.c files predate the gate and self-guard on
#     ARM_FLOAT16_SUPPORTED / MVE instead;
#   - QuantizationFunctions/ takes float32_t across an otherwise integer
#     API by design, so it is built in integer-only configurations.
GATE_EXEMPT_FILES = {
    "Source/BasicMathFunctions/arm_elementwise_add_fp16.c",
    "Source/FullyConnectedFunctions/arm_fully_connected_fp16.c",
}
GATE_EXEMPT_DIRS = ("Source/QuantizationFunctions/",)


def _code_line_numbers(lines: list[str]) -> list[int]:
    """Indices of file-scope C code lines, ignoring comments and directives."""
    out: list[int] = []
    in_block = False
    for i, raw in enumerate(lines):
        line = raw
        if in_block:
            end = line.find("*/")
            if end < 0:
                continue
            line = line[end + 2 :]
            in_block = False
        while True:
            start = line.find("/*")
            if start < 0:
                break
            end = line.find("*/", start + 2)
            if end < 0:
                line = line[:start]
                in_block = True
                break
            line = line[:start] + line[end + 2 :]
        line = re.sub(r"//.*", "", line).strip()
        if not line or PP_RE.match(line):
            continue
        out.append(i)
    return out


def check_float_source_gating(entries: list[tuple[str, str]]) -> None:
    for cat, name in entries:
        base = name.rsplit("/", 1)[-1]
        if cat != "source" or not name or not FLOAT_SRC_RE.search(base):
            continue
        if name in GATE_EXEMPT_FILES or name.startswith(GATE_EXEMPT_DIRS):
            continue
        path = REPO / name
        if not path.is_file():
            continue  # already reported by check_file_existence
        lines = path.read_text(encoding="utf-8").splitlines()

        gate_idx = next((i for i, ln in enumerate(lines) if GATE_RE.match(ln)), None)
        if gate_idx is None:
            fail(
                f"{name}: float source has no `#if ARM_NN_ENABLE_F16/F32` gate — it "
                f"would not compile when its dtype is disabled (the pdsc builds it "
                f"unconditionally)"
            )
            continue

        expected = "F16" if re.search(r"(?:^|_)f?p?16(?:_|\.c$)", base) else "F32"
        actual = GATE_RE.match(lines[gate_idx]).group(1)
        if actual != expected:
            fail(f"{name}: gated on ARM_NN_ENABLE_{actual}, expected ARM_NN_ENABLE_{expected}")
            continue

        # Locate the #endif that closes the gate.
        depth = 0
        close_idx = None
        for i in range(gate_idx, len(lines)):
            if COND_OPEN_RE.match(lines[i]):
                depth += 1
            elif COND_CLOSE_RE.match(lines[i]):
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        if close_idx is None:
            fail(f"{name}: `#if ARM_NN_ENABLE_{expected}` is never closed by a matching #endif")
            continue

        stray = [i for i in _code_line_numbers(lines) if i < gate_idx or i > close_idx]
        if stray:
            fail(
                f"{name}: code at line {stray[0] + 1} sits outside the "
                f"ARM_NN_ENABLE_{expected} gate (lines {gate_idx + 1}-{close_idx + 1}); "
                f"the file must be an empty translation unit when the dtype is off"
            )


def _annotated_value(line: str, scope: str) -> str | None:
    """The substring release-please's generic updater would substitute in
    for this (line, scope) pair, mirroring VERSION_REGEX / MAJOR_VERSION_REGEX
    / SINGLE_VERSION_REGEX from generic.js exactly. None means the real
    updater's `line.replace(...)` finds nothing to replace and leaves the
    line untouched -- for an inline annotation that is always a bug; inside
    a block it is normal for most lines (most of a block's lines have no
    version-shaped content at all, e.g. Kconfig help prose)."""
    if scope == "version":
        m = RP_VERSION_TRIPLET_RE.search(line)
        return m.group(0) if m else None
    if scope in ("major", "minor", "patch"):
        m = RP_BARE_INT_RE.search(line)
        return m.group(0) if m else None
    return "<unchecked: version-date/date not used in this repo>"


def _check_annotated_value(
    rel: str, lineno: int, scope: str, value: str, canonical: dict[str, str], canonical_full: str
) -> None:
    if scope == "version":
        expected = canonical_full
    elif scope in ("major", "minor", "patch"):
        expected = canonical[scope]
    else:
        return  # version-date / date: nothing in this repo to cross-check against
    if value != expected:
        fail(
            f"{rel}:{lineno}: annotated {scope} {value!r} != canonical "
            f"{expected!r} (Include/arm_nn_types.h)"
        )


def check_extra_files_annotations() -> None:
    """Guard release-please-config.json's `extra-files` list against the
    two silent failure modes release-please itself has: a path that no
    longer exists (manifest.js only warns), and a listed file that
    carries no matching annotation, so the generic updater quietly
    changes zero lines in it forever (generic.js just leaves such lines
    alone — no warning, no error). Also cross-checks every annotated
    value, plus every LITERAL_ONLY_EXTRA_FILES literal, against the
    canonical version in Include/arm_nn_types.h, so a hand-bumped docs
    literal that drifts from the real version fails here instead of
    shipping quietly.
    """
    try:
        config = json.loads(RP_CONFIG.read_text())
    except Exception as e:
        fail(f"could not read {RP_CONFIG.relative_to(REPO)}: {e}")
        return

    canonical_parts = parse_header_version(TYPES_H.read_text()) if TYPES_H.is_file() else None
    if canonical_parts is None:
        fail(
            "cannot cross-check extra-files versions: "
            "Include/arm_nn_types.h NS_CMSIS_NN_VERSION_* markers unreadable"
        )
        return
    canonical = dict(zip(("major", "minor", "patch"), canonical_parts))
    canonical_full = ".".join(canonical_parts)

    # Normalize both entry forms release-please accepts: the object form
    # used everywhere in this repo ({"type": ..., "path": ...}), and the
    # bare-string form (path only, updater picked by file extension).
    # Existence is checked for every entry regardless of form; the
    # annotation/literal checks below only apply to the object form with
    # "type": "generic", since that is the only form this repo uses and
    # the only one that means "look for x-release-please comments" --
    # the jsonpath-based updaters (json/yaml/toml/xml, object form with a
    # "type" other than "generic") and the bare-string, extension-sniffed
    # form both use a different, non-comment mechanism this check does
    # not pin.
    extra_files: list[tuple[str, str | None]] = []  # (path, type or None)
    for pkg_cfg in config.get("packages", {}).values():
        for entry in pkg_cfg.get("extra-files", []):
            if isinstance(entry, dict):
                rel = entry.get("path")
                if not rel:
                    fail(f"extra-files entry missing 'path': {entry!r}")
                    continue
                extra_files.append((rel, entry.get("type")))
            elif isinstance(entry, str):
                extra_files.append((entry, None))
            else:
                fail(f"extra-files entry is neither a path string nor an object: {entry!r}")

    if not extra_files:
        fail(f"{RP_CONFIG.relative_to(REPO)} has no extra-files entries to check")
        return

    for rel, entry_type in extra_files:
        path = REPO / rel
        if not path.is_file():
            fail(f"extra-files entry does not exist on disk: {rel}")
            continue
        if entry_type != "generic":
            continue  # not a comment-annotation-based updater; nothing more to check here
        text = path.read_text(encoding="utf-8")

        if rel in LITERAL_ONLY_EXTRA_FILES:
            pattern = LITERAL_ONLY_EXTRA_FILES[rel]
            m = pattern.search(text)
            if not m:
                fail(
                    f"{rel}: on the literal-only allowlist but pattern "
                    f"{pattern.pattern!r} found no match — did the file move "
                    "or get reformatted?"
                )
                continue
            literal = m.group(1)
            if literal != canonical_full:
                fail(
                    f"{rel}: literal version {literal!r} does not match canonical "
                    f"{canonical_full!r} (Include/arm_nn_types.h) — this file has no "
                    "working annotation, so it must be bumped by hand at release time"
                )
            continue

        # Mirrors generic.js's own per-line state machine: an inline
        # annotation always wins over an open block; otherwise, once inside
        # a block, every line (including the x-release-please-end line
        # itself) is scope-replaced until the end marker; otherwise look for
        # a block-start marker (that line itself is never scope-replaced).
        annotations_found = 0
        block_scope: str | None = None
        block_started_at: int | None = None
        for lineno, line in enumerate(text.splitlines(), start=1):
            inline_match = RP_INLINE_SCOPE_RE.search(line)
            if inline_match:
                scope = inline_match.group(1)
                value = _annotated_value(line, scope)
                if value is None:
                    fail(
                        f"{rel}:{lineno}: x-release-please-{scope} annotation but no "
                        "value on the line — release-please's generic updater will "
                        "silently skip this line"
                    )
                    continue
                annotations_found += 1
                _check_annotated_value(rel, lineno, scope, value, canonical, canonical_full)
                continue

            if block_scope is not None:
                value = _annotated_value(line, block_scope)
                if value is not None:
                    annotations_found += 1
                    _check_annotated_value(rel, lineno, block_scope, value, canonical, canonical_full)
                if RP_BLOCK_END_RE.search(line):
                    block_scope = None
                    block_started_at = None
                continue

            block_start_match = RP_BLOCK_START_RE.search(line)
            if block_start_match:
                block_scope = block_start_match.group(1)
                block_started_at = lineno
            # else: an ordinary line outside any annotation, nothing to do.

        if block_scope is not None:
            fail(
                f"{rel}:{block_started_at}: x-release-please-start-{block_scope} block "
                "is never closed with a matching x-release-please-end — release-please "
                "would keep scope-replacing every line through end of file"
            )

        if annotations_found == 0:
            fail(
                f"{rel}: listed in extra-files but carries zero x-release-please "
                "annotations and is not on LITERAL_ONLY_EXTRA_FILES — release-please "
                "will never touch this file, so it will drift silently. Either add a "
                "matching x-release-please-* annotation or add it to "
                "LITERAL_ONLY_EXTRA_FILES with a pattern that pins its literal."
            )


# --- SSoT (cmake) vs pdsc source-list agreement ------------------------
#
# `Ambiq.NS-CMSIS-NN.pdsc` and `cmake/ns_cmsis_nn.cmake` are two
# independent enumerations of the same source tree: the pdsc feeds
# CMSIS-Pack consumers, the SSoT feeds every in-repo CMake build plus
# the Zephyr and NSX modules. Nothing structural keeps them in step, and
# each direction of drift fails silently for exactly one audience:
#
#   - in the pdsc, not in the SSoT: pack consumers can call the kernel,
#     CMake/Zephyr/NSX consumers get an undefined reference at link time
#     (#268: arm_fully_connected_fp16, arm_nn_vec_mat_mult_t_fp16,
#     arm_softmax_u8 shipped that way);
#   - in the SSoT, not in the pdsc: the CMake build compiles a file the
#     pack never ships (and, since check_source_coverage() pins the pdsc
#     to `git ls-files Source/`, usually means the SSoT names a file that
#     does not exist — a typo that `file(GLOB)` swallows but an explicit
#     `extras` entry turns into a hard CMake error).
#
# The comparison is made against the *union* over every dtype gate
# (ARM_NN_ENABLE_F32 and ARM_NN_ENABLE_F16 both treated as on), because
# the question here is reachability under some flag combination, not
# under one particular build. Gate *correctness* — that a float source
# is under the matching gate — is check_float_source_gating()'s job.

# Files intentionally listed in the pdsc but not built by the SSoT (or
# vice versa), each mapped to the reason. Empty on purpose: an entry
# here is a documented exception, not a parking spot. Adding one means
# the file is reachable for one consumer and not the other, so say why
# and link an issue.
SSOT_PDSC_ALLOWLIST: dict[str, str] = {}

# The SSoT is CMake, so reading it means a (small) CMake parse. Only the
# `_ns_cmsis_nn_group_def` function is interpreted, and only the four
# constructs it uses: `if/elseif(group STREQUAL "...")` to select a
# branch, and `set`/`list(APPEND ...)` on subdir/patterns/extras. Nested
# `if(ARM_NN_ENABLE_F32/F16)` blocks are deliberately *not* evaluated —
# their bodies are folded in unconditionally to build the union. If the
# SSoT ever grows a construct this does not model, the group-coverage
# assertion below fails loudly rather than silently under-reporting.
CMAKE_CMD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CMAKE_STRING_RE = re.compile(r'"([^"]*)"')
CMAKE_GROUP_RE = re.compile(r'group\s+STREQUAL\s+"([^"]+)"')
CMAKE_COMMENT_RE = re.compile(r"#.*")
SSOT_LIST_VARS = ("subdir", "patterns", "extras")


def _cmake_commands(text: str) -> list[tuple[str, str]]:
    """(command name, raw argument text) for each invocation in <text>."""
    out: list[tuple[str, str]] = []
    pos = 0
    while True:
        m = CMAKE_CMD_RE.search(text, pos)
        if not m:
            return out
        depth = 1
        i = m.end()
        while i < len(text) and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth:
            return out  # unbalanced; caller's coverage assertion will catch it
        out.append((m.group(1), text[m.end() : i - 1]))
        pos = m.end()


def _cmake_function_body(text: str, name: str) -> str | None:
    m = re.search(rf"^function\(\s*{re.escape(name)}\b", text, re.MULTILINE)
    if not m:
        return None
    end = re.search(r"^endfunction\(\)", text[m.start() :], re.MULTILINE)
    if not end:
        return None
    return text[m.start() : m.start() + end.start()]


def parse_ssot_sources() -> tuple[dict[str, set[str]], list[str]] | None:
    """Repo-relative sources each SSoT group resolves to, plus the SSoT's
    own canonical group list. Returns None (after recording a failure)
    when the SSoT cannot be read or parsed."""
    if not SSOT_CMAKE.is_file():
        fail(f"{SSOT_CMAKE.relative_to(REPO)} not found")
        return None
    text = CMAKE_COMMENT_RE.sub("", SSOT_CMAKE.read_text(encoding="utf-8"))

    declared: list[str] = []
    for cmd, args in _cmake_commands(text):
        toks = args.split()
        if cmd == "set" and toks and toks[0] == "_NS_CMSIS_NN_GROUPS":
            declared = toks[1:]
            break
    if not declared:
        fail("could not parse _NS_CMSIS_NN_GROUPS from the SSoT cmake module")
        return None

    body = _cmake_function_body(text, "_ns_cmsis_nn_group_def")
    if body is None:
        fail("could not locate _ns_cmsis_nn_group_def() in the SSoT cmake module")
        return None

    # group -> {"subdir": [...], "patterns": [...], "extras": [...]}
    defs: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    for cmd, args in _cmake_commands(body):
        toks = args.split()
        if cmd in ("if", "elseif"):
            m = CMAKE_GROUP_RE.search(args)
            if m:
                current = m.group(1)
                defs.setdefault(current, {k: [] for k in SSOT_LIST_VARS})
            # A nested if(ARM_NN_ENABLE_*) does not change the current
            # group: its body is folded into the union.
            continue
        if current is None:
            continue
        values = [v for v in CMAKE_STRING_RE.findall(args) if v]
        if cmd == "set" and toks and toks[0] in SSOT_LIST_VARS:
            defs[current][toks[0]] = values
        elif cmd == "list" and len(toks) >= 2 and toks[0] == "APPEND" and toks[1] in SSOT_LIST_VARS:
            defs[current][toks[1]].extend(values)

    missing = [g for g in declared if g not in defs]
    if missing:
        fail(
            "SSoT parse incomplete — no _ns_cmsis_nn_group_def branch found for "
            f"group(s) {', '.join(missing)}; cmake/ns_cmsis_nn.cmake likely grew a "
            "construct scripts/check_pdsc.py does not model"
        )
        return None

    tracked = tracked_source_files()
    if tracked is None:
        return None

    resolved: dict[str, set[str]] = {}
    for group in declared:
        subdirs = defs[group]["subdir"]
        if len(subdirs) != 1:
            fail(f"SSoT group '{group}' does not set exactly one subdir (got {subdirs})")
            return None
        prefix = f"Source/{subdirs[0]}/"
        hit: set[str] = set()
        for path in tracked:
            if not path.startswith(prefix):
                continue
            base = path[len(prefix) :]
            if "/" in base:
                continue
            if any(fnmatch(base, pat) for pat in defs[group]["patterns"]):
                hit.add(path)
        hit.update(prefix + extra for extra in defs[group]["extras"])
        resolved[group] = hit
    return resolved, declared


def check_ssot_pdsc_agreement(entries: list[tuple[str, str]]) -> None:
    parsed = parse_ssot_sources()
    if parsed is None:
        return
    resolved, _declared = parsed
    ssot = set().union(*resolved.values()) if resolved else set()
    listed = {name for cat, name in entries if cat == "source" and name}

    for name in sorted(listed - ssot - set(SSOT_PDSC_ALLOWLIST)):
        fail(
            f"{name}: shipped by the pdsc but unreachable from "
            "cmake/ns_cmsis_nn.cmake under any dtype gate — CMake / Zephyr / NSX "
            "consumers link-error on its public symbols (#268). Add it to the "
            "matching group in the SSoT, or document the exception in "
            "SSOT_PDSC_ALLOWLIST."
        )
    for name in sorted(ssot - listed - set(SSOT_PDSC_ALLOWLIST)):
        fail(
            f"{name}: referenced by cmake/ns_cmsis_nn.cmake but not shipped by the "
            "pdsc. If the file does not exist, the SSoT `extras` entry is a typo and "
            "will fail the CMake build; otherwise add it to the pdsc or document the "
            "exception in SSOT_PDSC_ALLOWLIST."
        )
    for name in sorted(SSOT_PDSC_ALLOWLIST):
        if name in listed and name in ssot:
            fail(
                f"{name}: on SSOT_PDSC_ALLOWLIST but the pdsc and the SSoT now agree "
                "about it — drop the allowlist entry so the exception cannot outlive "
                "its reason."
            )


def main() -> int:
    if not PDSC.is_file():
        fail(f"{PDSC.relative_to(REPO)} not found")
        report()
        return 1

    pkg, _text = parse_pdsc_text()
    check_pack_identity(pkg)
    comp = find_component(pkg)
    check_versions(pkg, comp)
    check_licenses(pkg)
    entries = collect_file_entries(comp)
    check_file_existence(entries)
    check_source_coverage(entries)
    check_float_source_gating(entries)
    check_ssot_pdsc_agreement(entries)
    check_extra_files_annotations()

    report()
    return 1 if failures else 0


def report() -> None:
    if failures:
        print("PDSC contract FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print(
            "PDSC contract OK: pack/component identity, versions in sync, "
            "NSX module version synced, licenses declared, all <file> paths exist, "
            "Source/ coverage complete, float sources dtype-gated, "
            "pdsc and cmake/ns_cmsis_nn.cmake source lists agree, "
            "extra-files annotations live and in sync."
        )


if __name__ == "__main__":
    sys.exit(main())
