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
#      Cversion, and the markers in Include/arm_nn_types.h all agree.
#      Cross-checked against .release-please-manifest.json so a manual
#      bump that forgets one of these fails CI.
#   4. License plumbing — every <license name=...> in the pdsc points at
#      a file that exists, and the repo top-level LICENSE / LICENSES/
#      files are all declared.
#   5. File existence — every <file name="..."/> path is on disk.
#   6. Coverage cross-check — every Source/**/*.c that exists in the
#      repo is enumerated under <files> with category="source", so
#      adding a kernel without updating the pack fails CI (replaces the
#      old check_pdsc.sh diff).
#

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PDSC = REPO / "Ambiq.NS-CMSIS-NN.pdsc"
TYPES_H = REPO / "Include" / "arm_nn_types.h"
RP_MANIFEST = REPO / ".release-please-manifest.json"

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
    m_major = re.search(r"NS_CMSIS_NN_VERSION_MAJOR\s*\((\d+)\)", th)
    m_minor = re.search(r"NS_CMSIS_NN_VERSION_MINOR\s*\((\d+)\)", th)
    m_patch = re.search(r"NS_CMSIS_NN_VERSION_PATCH\s*\((\d+)\)", th)
    m_rev = re.search(r'\$Revision:\s*"v(\d+\.\d+\.\d+)"', th)
    header_ver: str | None = None
    if m_major and m_minor and m_patch:
        header_ver = f"{m_major.group(1)}.{m_minor.group(1)}.{m_patch.group(1)}"
    else:
        fail("Include/arm_nn_types.h is missing one of the NS_CMSIS_NN_VERSION_* markers")

    rev_ver = m_rev.group(1) if m_rev else None
    if not rev_ver:
        fail('Include/arm_nn_types.h is missing the `$Revision: "vX.Y.Z"` marker')

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
        "arm_nn_types.h $Revision": rev_ver,
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


def check_source_coverage(entries: list[tuple[str, str]]) -> None:
    listed = sorted(name for cat, name in entries if cat == "source")
    actual = sorted(
        str(p.relative_to(REPO))
        for p in REPO.glob("Source/**/*.c")
        if p.is_file()
    )
    missing_in_pdsc = set(actual) - set(listed)
    extra_in_pdsc = set(listed) - set(actual)
    for m in sorted(missing_in_pdsc):
        fail(f"Source/ file not enumerated in pdsc: {m}")
    for x in sorted(extra_in_pdsc):
        fail(f"pdsc references nonexistent Source/ file: {x}")


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
            "licenses declared, all <file> paths exist, "
            "Source/ coverage complete."
        )


if __name__ == "__main__":
    sys.exit(main())
