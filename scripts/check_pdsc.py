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
#      reverse. Union-equality cannot see a *misplaced* gate, so each
#      ARM_NN_ENABLE_F32/F16 combination is also resolved separately
#      and every dtype-tagged source must be reachable in exactly the
#      configurations that enable its own dtype. The SSoT parser
#      models a closed set of CMake constructs and fails on anything
#      else rather than skipping it — a guard that silently ignores
#      `list(REMOVE_ITEM ...)` reports drift as clean.
#  10. Registered unit-test suites exist and their relative includes
#      resolve — every suite registered with `add_subdirectory(TestCases/...)`
#      in Tests/UnitTest/CMakeLists.txt exists on disk, and every
#      path-shaped `#include "..."` in its tracked sources resolves to a
#      tracked file. 36 float suites were registered against
#      `../TestData/<name>/test_data.h` paths that their generators do
#      produce, but only into a gitignored `TestData/` tree — the data was
#      never checked in, so the suites were unbuildable in every checkout.
#      No PR-gating job builds the float suites either way
#      (ARM_NN_ENABLE_F32/F16 default OFF in the legacy build), so they
#      looked like coverage for years while being uncompilable — which is
#      how a real transpose-conv output-shift bug survived to a release
#      (#253, #256).
#

from __future__ import annotations

import json
import os
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
# source must carry its own ARM_NN_ENABLE_F32/F16 gate. It is empty since the
# legacy *_fp16.c sources were either removed or gated like every other float
# source; QuantizationFunctions/ takes float32_t across an otherwise integer
# API by design, so it is built in integer-only configurations.
GATE_EXEMPT_FILES: set[str] = set()
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
#     (#268: arm_softmax_u8 and two since-removed legacy fp16 sources
#     shipped that way);
#   - in the SSoT, not in the pdsc: the CMake build compiles a file the
#     pack never ships (and, since check_source_coverage() pins the pdsc
#     to `git ls-files Source/`, usually means the SSoT names a file that
#     does not exist — a typo that `file(GLOB)` swallows but an explicit
#     `extras` entry turns into a hard CMake error).
#
# Two assertions are made, because set-equality alone is not enough:
#
#   A. Reachability. The union of what the SSoT resolves over every
#      ARM_NN_ENABLE_F32/F16 combination equals the pdsc's source set.
#   B. Gate placement. Every dtype-tagged source is reachable in exactly
#      the configurations where its own dtype is enabled. Union-equality
#      cannot see a *misplaced* gate — moving arm_softmax_f32.c under
#      `if(ARM_NN_ENABLE_F16)` keeps the union identical while giving an
#      F32-only consumer a fresh #268 — so each configuration is resolved
#      separately and checked against the file's dtype tag.
#
# (B) overlaps check_float_source_gating() only superficially: that check
# reads the *source file's* `#if ARM_NN_ENABLE_*` so the translation unit
# collapses for pack consumers; this one reads the *SSoT's* gate so the
# file is handed to the compiler in the right configurations. A file can
# pass either one while failing the other.
#
# Out of scope, deliberately: the pdsc has no float toggle of its own
# (it ships every source unconditionally), so pack-side reachability is a
# different problem and this check cannot see #273.

# Files intentionally listed in the pdsc but not built by the SSoT (or
# vice versa), each mapped to the reason. Empty on purpose: an entry
# here is a documented exception, not a parking spot. Adding one means
# the file is reachable for one consumer and not the other, so say why
# and link an issue.
SSOT_PDSC_ALLOWLIST: dict[str, str] = {}

# The four float configurations the SSoT distinguishes, as
# (ARM_NN_ENABLE_F32, ARM_NN_ENABLE_F16). Every one is a shipped build:
# integer-only is the default, and m55-f16-mvef / m55-f32-mvef are
# separate CI targets.
SSOT_CONFIGS: dict[str, tuple[bool, bool]] = {
    "integer-only": (False, False),
    "F32-only": (True, False),
    "F16-only": (False, True),
    "F32+F16": (True, True),
}
SSOT_ALL_CONFIGS = frozenset(SSOT_CONFIGS)

# The SSoT is CMake, so reading it means a (small) CMake parse. Only the
# `_ns_cmsis_nn_group_def` function is interpreted, and only over a
# CLOSED set of constructs: `if/elseif(group STREQUAL "...")` to select a
# branch, `if(ARM_NN_ENABLE_F32 [OR ARM_NN_ENABLE_F16])` to gate one, and
# `set` / `list(APPEND ...)` on subdir/patterns/extras. Anything else —
# another `list()` subcommand such as REMOVE_ITEM, an unrecognized
# command, an unmodelled condition, an argument that is not a
# double-quoted literal — is a hard failure, never a silent skip. That
# distinction is the whole point: a parser that quietly ignores
# `list(REMOVE_ITEM extras ...)` reports a removed file as still
# reachable, which is exactly the drift this check exists to catch.
CMAKE_CMD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CMAKE_STRING_RE = re.compile(r'"([^"]*)"')
CMAKE_GROUP_RE = re.compile(r'group\s+STREQUAL\s+"([^"]+)"')
CMAKE_COMMENT_RE = re.compile(r"#.*")
SSOT_LIST_VARS = ("subdir", "patterns", "extras")
# Commands the group_def parser knows how to interpret or safely ignore.
# `message` is the FATAL_ERROR in the unknown-group else-branch.
SSOT_IGNORED_COMMANDS = frozenset({"function", "endfunction", "message"})
SSOT_ALLOWED_COMMANDS = SSOT_IGNORED_COMMANDS | frozenset(
    {"if", "elseif", "else", "endif", "set", "list"}
)
SSOT_GATE_ATOM_RE = re.compile(r"^ARM_NN_ENABLE_(F16|F32)$")


def _cmake_commands(text: str, what: str) -> list[tuple[str, str]] | None:
    """(command name, raw argument text) for each invocation in <text>.

    CMake commands do not nest, so scanning resumes after each closing
    paren. Returns None (after recording a failure) on unbalanced
    parentheses, whether from a stray paren or from a `#` inside a quoted
    string (comment stripping is not quote-aware).
    """
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
            fail(
                f"{what}: unbalanced parentheses after `{m.group(1)}(` — the SSoT "
                "parser cannot read this file. Look for a stray or unmatched "
                "parenthesis, or a '#' inside a double-quoted string (comment "
                "stripping is not quote-aware)."
            )
            return None
        out.append((m.group(1), text[m.end() : i - 1]))
        pos = i


def _cmake_function_body(text: str, name: str) -> str | None:
    m = re.search(rf"^function\(\s*{re.escape(name)}\b", text, re.MULTILINE)
    if not m:
        return None
    end = re.search(r"^endfunction\(\)", text[m.start() :], re.MULTILINE)
    if not end:
        return None
    return text[m.start() : m.start() + end.start()]


def _ssot_gate_configs(cond: str, group: str) -> frozenset[str] | None:
    """Configurations in which an `if(<cond>)` inside a group branch is
    live. Only `ARM_NN_ENABLE_F32`/`ARM_NN_ENABLE_F16`, optionally
    OR-joined, are modelled."""
    toks = cond.split()
    atoms: list[str] = []
    ok = bool(toks)
    for i, tok in enumerate(toks):
        if i % 2 == 0:
            m = SSOT_GATE_ATOM_RE.match(tok)
            if not m:
                ok = False
                break
            atoms.append(m.group(1).lower())
        elif tok != "OR":
            ok = False
            break
    if not ok or len(toks) % 2 == 0:
        fail(
            f"unmodelled condition in group '{group}': `if({cond.strip()})`. The SSoT "
            "parser understands only ARM_NN_ENABLE_F32 / ARM_NN_ENABLE_F16, optionally "
            "OR-joined; teach scripts/check_pdsc.py the new condition rather than "
            "leaving its sources unchecked."
        )
        return None
    return frozenset(
        name
        for name, (f32, f16) in SSOT_CONFIGS.items()
        if any(f32 if atom == "f32" else f16 for atom in atoms)
    )


def _ssot_quoted_values(group: str, var: str, remainder: str) -> list[str] | None:
    """Double-quoted literals in <remainder>. Returns None (after
    recording a failure) if there is argument text but no quoted literal
    in it — resolving that to "nothing" would silently drop real sources
    and then misreport an innocent, reachable file as pdsc-only."""
    raw = CMAKE_STRING_RE.findall(remainder)
    if not raw and remainder.strip():
        fail(
            f"unquoted or variable argument in group '{group}': the SSoT parser reads "
            f"only double-quoted literals, but `{var}` is given `{remainder.strip()}`. "
            "Spell source names as \"...\" literals in cmake/ns_cmsis_nn.cmake."
        )
        return None
    return [v for v in raw if v]


def _parse_ssot_defs(body: str) -> dict[str, dict[str, list]] | None:
    """group -> {subdir/patterns/extras: [(value, live_configs), ...]}."""
    commands = _cmake_commands(body, "_ns_cmsis_nn_group_def")
    if commands is None:
        return None

    defs: dict[str, dict[str, list]] = {}
    current: str | None = None
    gates: list[frozenset[str]] = []

    for cmd, args in commands:
        if cmd in SSOT_IGNORED_COMMANDS:
            continue
        if cmd not in SSOT_ALLOWED_COMMANDS:
            fail(
                f"unrecognized command `{cmd}(...)` in _ns_cmsis_nn_group_def "
                f"(group '{current}') — the SSoT parser models a closed set of "
                "constructs and will not guess at this one."
            )
            return None

        if cmd in ("if", "elseif"):
            m = CMAKE_GROUP_RE.search(args)
            if m:
                current = m.group(1)
                gates = []
                defs.setdefault(current, {k: [] for k in SSOT_LIST_VARS})
                continue
            if cmd == "elseif":
                # Modelling this would mean tracking mutual exclusion with
                # the sibling branch and popping the leaked gate at the
                # single endif(). Treating it as a nested if() does
                # neither: it AND-s the two conditions and lets the gate
                # outlive the block, which resolves later appends under a
                # gate that is not there. Refuse instead.
                fail(
                    f"`elseif({args.strip()})` inside a group branch is not modelled — "
                    "the SSoT parser cannot tell which branch is live; use independent "
                    "if(...) blocks or teach scripts/check_pdsc.py."
                )
                return None
            if current is None:
                fail(f"`if({args.strip()})` outside any group branch is not modelled")
                return None
            live = _ssot_gate_configs(args, current)
            if live is None:
                return None
            gates.append((gates[-1] if gates else SSOT_ALL_CONFIGS) & live)
            continue

        if cmd == "endif":
            if gates:
                gates.pop()
            else:
                current = None
            continue

        if cmd == "else":
            if gates:
                fail(
                    f"`else()` inside a dtype gate in group '{current}' is not "
                    "modelled — the SSoT parser cannot tell which branch is live."
                )
                return None
            current = None
            continue

        toks = args.split()
        live_now = gates[-1] if gates else SSOT_ALL_CONFIGS

        if cmd == "set":
            if not toks or toks[0].startswith("${"):
                continue  # the PARENT_SCOPE hand-back at the end of the function
            if toks[0] not in SSOT_LIST_VARS:
                fail(
                    f"`set({toks[0]} ...)` in _ns_cmsis_nn_group_def (group "
                    f"'{current}') targets a variable the SSoT parser does not model."
                )
                return None
            if current is None:
                continue  # the subdir/patterns/extras reset at the top of the function
            if gates:
                fail(
                    f"`set({toks[0]} ...)` inside a dtype gate in group '{current}' is "
                    "not modelled — it replaces rather than appends, so its effect "
                    "depends on which gates are on. Use list(APPEND ...)."
                )
                return None
            remainder = args.split(None, 1)[1] if len(toks) > 1 else ""
            values = _ssot_quoted_values(current, toks[0], remainder)
            if values is None:
                return None
            defs[current][toks[0]] = [(v, live_now) for v in values]
            continue

        # cmd == "list"
        if len(toks) < 2 or toks[0] != "APPEND" or toks[1] not in SSOT_LIST_VARS:
            detail = toks[0] if toks else "<empty>"
            fail(
                f"`list({detail} ...)` in _ns_cmsis_nn_group_def (group '{current}') — "
                "the SSoT parser models only list(APPEND patterns|extras \"...\"). "
                "Any other subcommand (REMOVE_ITEM, FILTER, ...) changes the resolved "
                "source set in a way it would otherwise silently miss."
            )
            return None
        if current is None:
            fail("`list(APPEND ...)` outside any group branch is not modelled")
            return None
        remainder = args.split(None, 2)[2] if len(toks) > 2 else ""
        values = _ssot_quoted_values(current, toks[1], remainder)
        if values is None:
            return None
        defs[current][toks[1]].extend((v, live_now) for v in values)

    return defs


def _ssot_dtype_tag(basename: str, dtypes: list[str]) -> str | None:
    """The dtype tag _ns_cmsis_nn_filter_dtypes() would assign, including
    its `_fp16` -> f16 special case. Order matters: the CMake loop breaks
    on the first hit, so arm_quantize_f32_s8.c tags as s8, not f32.

    <dtypes> is read from the SSoT, but this algorithm is a hand mirror of
    the CMake one and nothing links them — a second special case added
    there would silently divorce the two and make assertion B skip the
    affected files. test_filter_dtypes_special_cases_are_mirrored in
    scripts/tests/test_check_pdsc_ssot.py is the canary for that."""
    if re.search(r"_fp16([._]|$)", basename):
        return "f16"
    for dt in dtypes:
        if re.search(rf"_{dt}([._]|$)", basename):
            return dt
    return None


def parse_ssot_sources() -> tuple[dict[str, set[str]], list[str]] | None:
    """(sources reachable in each SSOT_CONFIGS configuration, dtype tags).
    Returns None (after recording a failure) when the SSoT cannot be read
    or fully parsed — never a partial resolution."""
    if not SSOT_CMAKE.is_file():
        fail(f"{SSOT_CMAKE.relative_to(REPO)} not found")
        return None
    text = CMAKE_COMMENT_RE.sub("", SSOT_CMAKE.read_text(encoding="utf-8"))

    top = _cmake_commands(text, SSOT_CMAKE.name)
    if top is None:
        return None
    declared: list[str] = []
    dtypes: list[str] = []
    for cmd, args in top:
        toks = args.split()
        if cmd == "set" and toks and toks[0] == "_NS_CMSIS_NN_GROUPS":
            declared = toks[1:]
        elif cmd == "set" and toks and toks[0] == "_NS_CMSIS_NN_DTYPES":
            dtypes = toks[1:]
    if not declared or not dtypes:
        fail(
            "could not parse _NS_CMSIS_NN_GROUPS / _NS_CMSIS_NN_DTYPES from the SSoT "
            "cmake module"
        )
        return None

    body = _cmake_function_body(text, "_ns_cmsis_nn_group_def")
    if body is None:
        fail("could not locate _ns_cmsis_nn_group_def() in the SSoT cmake module")
        return None

    defs = _parse_ssot_defs(body)
    if defs is None:
        return None

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

    per_config: dict[str, set[str]] = {name: set() for name in SSOT_CONFIGS}
    for group in declared:
        subdirs = [v for v, _live in defs[group]["subdir"]]
        if len(subdirs) != 1:
            fail(f"SSoT group '{group}' does not set exactly one subdir (got {subdirs})")
            return None
        prefix = f"Source/{subdirs[0]}/"
        candidates = [
            (p, p[len(prefix) :])
            for p in tracked
            if p.startswith(prefix) and "/" not in p[len(prefix) :]
        ]
        for config in SSOT_CONFIGS:
            patterns = [v for v, live in defs[group]["patterns"] if config in live]
            extras = [v for v, live in defs[group]["extras"] if config in live]
            for path, base in candidates:
                if any(fnmatch(base, pat) for pat in patterns):
                    per_config[config].add(path)
            per_config[config].update(prefix + extra for extra in extras)
    return per_config, dtypes


def check_ssot_pdsc_agreement(entries: list[tuple[str, str]]) -> None:
    parsed = parse_ssot_sources()
    if parsed is None:
        return
    per_config, dtypes = parsed
    ssot = set().union(*per_config.values())
    listed = {name for cat, name in entries if cat == "source" and name}

    # A. Reachability under some configuration.
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

    # B. Gate placement: reachable in exactly the configurations that
    #    enable the file's own dtype.
    for name in sorted(ssot):
        if name in SSOT_PDSC_ALLOWLIST:
            continue
        tag = _ssot_dtype_tag(name.rsplit("/", 1)[-1], dtypes)
        if tag not in ("f16", "f32"):
            continue
        for config, (f32_on, f16_on) in SSOT_CONFIGS.items():
            enabled = f32_on if tag == "f32" else f16_on
            if (name in per_config[config]) == enabled:
                continue
            if enabled:
                fail(
                    f"{name}: unreachable from cmake/ns_cmsis_nn.cmake in the "
                    f"{config} configuration even though ARM_NN_ENABLE_"
                    f"{tag.upper()} is on there — it sits under the wrong "
                    "ARM_NN_ENABLE_* gate, so that build link-errors on its symbols."
                )
            else:
                fail(
                    f"{name}: reachable from cmake/ns_cmsis_nn.cmake in the {config} "
                    f"configuration even though ARM_NN_ENABLE_{tag.upper()} is off "
                    "there — it sits under the wrong ARM_NN_ENABLE_* gate, so that "
                    "build compiles a translation unit its dtype does not support."
                )


# --- unit-test suite registration vs on-disk test data -----------------
#
# Tests/UnitTest/CMakeLists.txt registers each suite with
# `add_subdirectory(TestCases/<suite>)`. Registration is the only thing
# that makes a suite look like coverage, and nothing checks that the
# suite can actually be compiled:
#
#   - the float suites are registered under `if(ARM_NN_ENABLE_F32)` /
#     `if(ARM_NN_ENABLE_F16)`, and no PR-gating job turns either flag on
#     (the legacy build defaults both OFF), so a float suite is never
#     configured, never compiled, and never run in CI;
#   - 36 of them included `../TestData/<name>/test_data.h` paths that
#     their `*_settings_flt.py` generators do produce, but only into a
#     gitignored `TestData/` tree — the data was never force-added, so
#     the paths did not resolve in any checkout and the suites could not
#     compile at all, generator or not.
#
# The cost is not hypothetical: those registrations were counted as
# transpose-conv float coverage while the shipped kernel had an
# output-shift bug, which is what #253 found and #256 swept up. A
# registered-but-unbuildable suite is worse than no suite, because it
# stops anyone from writing the real one.
#
# Two things are asserted, both cheap and both textual:
#
#   A. Every registered TestCases/<suite> directory exists.
#      Deleting a suite without unregistering it is a hard CMake error
#      that only surfaces in a configure nobody runs.
#   B. Every path-shaped `#include "..."` in a registered suite's tracked
#      sources resolves to a tracked file, relative to the including
#      file. "Path-shaped" means the include contains a `/`, which is
#      what distinguishes `"../TestData/foo/test_data.h"` and
#      `"../test_arm_reduce_sum_f32.c"` (relative, must resolve on disk)
#      from `"unity.h"` (bare name, legitimately found on the compiler's
#      include path, which this script cannot and should not model).
#
# Generated files are invisible to (B) by construction: the check reads
# `git ls-files`, and Unity's TestRunner stubs are produced at configure
# time and untracked. That is deliberate — a guard that trusted the
# working tree would pass locally after a configure and fail on a clean
# CI checkout, or vice versa.

UNIT_TEST_CMAKE = REPO / "Tests" / "UnitTest" / "CMakeLists.txt"
ADD_SUBDIR_RE = re.compile(r"^\s*add_subdirectory\(([^)]*)\)", re.MULTILINE)
QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
SUITE_SOURCE_SUFFIXES = (".c", ".h", ".cpp", ".hpp")

# `add_subdirectory` arguments in Tests/UnitTest/CMakeLists.txt that are
# not test suites: the library build itself (a `${CMAKE_CURRENT_SOURCE_DIR}`
# path) and the vendored Unity checkout. Anything else that is neither one
# of these nor a `TestCases/...` suite is a hard failure rather than a
# silent skip — the whole point of this check is that an unexamined
# registration is how the #256 suites hid.
NON_SUITE_SUBDIRS = frozenset({"Unity"})

# TEMPORARY ALLOWLIST — REMOVE WITH #236.
#
# ############################################################
# #  Exactly one suite, and it is not a precedent. Do not add #
# #  to this dict. A suite that cannot compile gets deleted   #
# #  (#256's disposition), not allowlisted.                   #
# ############################################################
#
# test_arm_convolve_f16 is broken in precisely the way this check exists
# to catch: dangling `../TestData/...` includes, exactly like the suites
# #256 deleted. It was left in place by the #256 sweep only because PR
# #236 is open against that exact directory, and deleting it underneath
# an in-flight PR trades one avoidable mess for another. Its fate rides
# with #236 — whichever way that PR lands, this entry and (if #236 does
# not fix the suite) the directory itself must go with it. If #236 closes
# unmerged, delete the suite and this entry.
#
# Keyed by the exact resolved include path rather than by suite name, so
# this stays a snapshot of what was already broken when #256 landed and
# cannot silently absorb something new: PR #236 is expected to add its
# own float datasets, and if it adds a `../TestData/...` include without
# checking the data in, that path is not in this dict and check #10 fires
# on it — which mechanically enforces what #236's own review already
# requires. Paths that #236 fixes (checked the data in) stop being
# dangling and are caught as stale entries below; paths #236 leaves
# broken stay allowlisted under their existing entry.
#
# Enumerated from Tests/UnitTest/TestCases/test_arm_convolve_f16/test_arm_convolve_f16.c
# as of this PR — see that file for the current list if this ever needs
# re-deriving.
_CONVOLVE_F16_ALLOWLIST_REASON = (
    "dangling ../TestData include, same as the suites #256 deleted; "
    "excluded from that sweep only because PR #236 has test_arm_convolve_f16 "
    "open. Delete this entry (and the suite, unless #236 repairs it) when "
    "#236 lands; if #236 closes unmerged, delete the suite and this entry."
)
UNBUILDABLE_SUITE_ALLOWLIST: dict[str, dict[str, str]] = {
    "test_arm_convolve_f16": {
        path: _CONVOLVE_F16_ALLOWLIST_REASON
        for path in (
            "Tests/UnitTest/TestCases/TestData/conv_1x1_stride2_nhwc_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_basic_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_basic_nhwc_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_k3_opt_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_k3_opt_nhwc_tuned_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_k5_opt_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_k5_opt_nhwc_tuned_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_kernel_2x2_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_kernel_3x3_pad1_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1x1_basic_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1x1_stride_x_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1x1_stride_x_y_1_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1x1_stride_x_y_2_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1x1_stride_x_y_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_1_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_2_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_3_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_4_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_5_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_6_generic_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_7_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_1xn_8_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_2x2_dilation_5x5_input_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_2x2_dilation_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_2x3_dilation_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_3x2_dilation_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_3x3_dilation_5x5_input_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_basic_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_conv_2_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_conv_3_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_conv_4_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_conv_5_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_dilation_golden_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_out_activation_f16/test_data.h",
            "Tests/UnitTest/TestCases/TestData/conv_match_stride2pad1_f16/test_data.h",
        )
    },
}


def tracked_repo_files() -> set[str] | None:
    """Every path `git ls-files` reports, as a set of repo-relative
    strings. Used instead of the working tree so this check sees what a
    clean CI checkout sees, not what a local configure left behind."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        fail(f"`git ls-files` failed: {e}")
        return None
    return set(out.splitlines())


def registered_unit_test_suites() -> list[str] | None:
    """Suite names from `add_subdirectory(TestCases/<suite>)`, in file
    order. Returns None (after recording a failure) on an
    `add_subdirectory` argument this check does not model."""
    if not UNIT_TEST_CMAKE.is_file():
        fail(f"{UNIT_TEST_CMAKE.relative_to(REPO)} not found")
        return None
    text = CMAKE_COMMENT_RE.sub("", UNIT_TEST_CMAKE.read_text(encoding="utf-8"))

    suites: list[str] = []
    for m in ADD_SUBDIR_RE.finditer(text):
        arg = m.group(1).split()[0] if m.group(1).split() else ""
        if not arg or arg.startswith("${"):
            continue  # the library build: add_subdirectory(${...}/../.. cmsis-nn)
        if arg in NON_SUITE_SUBDIRS:
            continue
        if not arg.startswith("TestCases/"):
            fail(
                f"{UNIT_TEST_CMAKE.relative_to(REPO)}: unmodelled "
                f"`add_subdirectory({arg})` — this check knows how to validate "
                "`TestCases/<suite>` registrations and nothing else. Teach "
                "scripts/check_pdsc.py about the new form rather than leaving it "
                "unchecked."
            )
            return None
        suites.append(arg[len("TestCases/") :].rstrip("/"))

    if not suites:
        fail(
            f"{UNIT_TEST_CMAKE.relative_to(REPO)}: no "
            "`add_subdirectory(TestCases/...)` registrations found — either the "
            "file changed shape or this check stopped seeing any suites at all."
        )
        return None
    return suites


def check_unit_test_suite_data() -> None:
    suites = registered_unit_test_suites()
    if suites is None:
        return
    tracked = tracked_repo_files()
    if tracked is None:
        return

    # Bucket tracked files by suite so each suite is a dict lookup rather
    # than a scan of ~30k paths.
    by_suite: dict[str, list[str]] = {}
    prefix = "Tests/UnitTest/TestCases/"
    for path in tracked:
        if not path.startswith(prefix) or not path.endswith(SUITE_SOURCE_SUFFIXES):
            continue
        rest = path[len(prefix) :]
        head, sep, _ = rest.partition("/")
        if sep:
            by_suite.setdefault(head, []).append(path)

    # Per suite, the set of currently-dangling resolved include paths —
    # used below to find UNBUILDABLE_SUITE_ALLOWLIST entries that are
    # stale (their path no longer dangles, because the data was checked
    # in or the include was removed) as distinct from entries that are
    # still covering a real gap.
    dangling: dict[str, set[str]] = {}
    for suite in suites:
        suite_dir = f"{prefix}{suite}"
        sources = sorted(by_suite.get(suite, []))
        allowlisted_paths = UNBUILDABLE_SUITE_ALLOWLIST.get(suite, {})
        if not sources:
            if suite not in UNBUILDABLE_SUITE_ALLOWLIST:
                fail(
                    f"Tests/UnitTest/CMakeLists.txt registers "
                    f"`add_subdirectory(TestCases/{suite})` but {suite_dir}/ has no "
                    "tracked source files — CMake fails to configure the unit tests "
                    "at all. Drop the registration if the suite was deleted."
                )
            continue

        for src in sources:
            src_dir = src.rsplit("/", 1)[0]
            try:
                text = (REPO / src).read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                # `src` came from `git ls-files`, so it is tracked, but a
                # locally deleted/renamed working-tree file (or a broken
                # symlink) would otherwise surface as an uncaught
                # traceback instead of a normal fail() report.
                fail(f"{src}: could not be read ({e}), but is a tracked source of "
                     f"TestCases/{suite}, which is registered in "
                     "Tests/UnitTest/CMakeLists.txt.")
                continue
            for inc in QUOTED_INCLUDE_RE.findall(text):
                if "/" not in inc:
                    continue  # bare name: resolved off the include path, not relative
                resolved = os.path.normpath(f"{src_dir}/{inc}")
                if resolved in tracked:
                    continue
                dangling.setdefault(suite, set()).add(resolved)
                if resolved in allowlisted_paths:
                    continue
                fail(
                    f"{src}: #include \"{inc}\" does not resolve ({resolved} is not "
                    f"tracked), but TestCases/{suite} is registered in "
                    "Tests/UnitTest/CMakeLists.txt. A registered suite that cannot "
                    "compile reads as coverage and is not — see #256, where 36 such "
                    "float suites hid a shipped transpose-conv bug. Either check the "
                    "data in (the `<case>_data.h` convention) or delete the suite and "
                    "its registration. If this path is expected to stay broken for a "
                    "documented reason, it needs its own UNBUILDABLE_SUITE_ALLOWLIST "
                    "entry — an unrelated existing entry for this suite does not cover "
                    "a new path."
                )

    registered = set(suites)
    for suite, entries in sorted(UNBUILDABLE_SUITE_ALLOWLIST.items()):
        if suite not in registered:
            fail(
                f"TestCases/{suite} is on UNBUILDABLE_SUITE_ALLOWLIST but is no longer "
                "registered in Tests/UnitTest/CMakeLists.txt — drop the allowlist "
                "entry so the exception cannot outlive its reason "
                f"({next(iter(entries.values()), 'no entries')})"
            )
            continue
        still_dangling = dangling.get(suite, set())
        stale_paths = sorted(p for p in entries if p not in still_dangling)
        if stale_paths:
            fail(
                f"TestCases/{suite}: UNBUILDABLE_SUITE_ALLOWLIST entries for "
                f"{len(stale_paths)} path(s) no longer dangle (checked in, or no "
                f"longer referenced) — drop them: {', '.join(stale_paths)}"
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
    check_unit_test_suite_data()

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
            "pdsc and cmake/ns_cmsis_nn.cmake source lists agree with "
            "dtype gates correctly placed, "
            "extra-files annotations live and in sync, "
            "Tests/UnitTest/CMakeLists.txt registered suites exist and their "
            "relative includes resolve."
        )


if __name__ == "__main__":
    sys.exit(main())
