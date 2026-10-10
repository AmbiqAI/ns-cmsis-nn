#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# No struct or union in the headers may declare an enum-typed field. An enum's size follows the compiler's convention
# (arm-none-eabi-gcc: short enums; ATfE, armclang and host compilers: int), so such a field gives the record a different
# layout or field width in a caller and a library built with different conventions, and one of them misreads it. See
# #693 and #764.
#
# The headers are parsed by clang (CLANG, default "clang"), so every way of declaring a field is seen as the compiler
# sees it: through typedef aliases, in nested and anonymous records, in unions, as arrays and bitfields. Pointers to
# an enum are allowed; their size does not depend on the enum's.

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE_DIR = REPO / "Include"
# Headers that the others expect to be included first
FIRST = ("arm_nnfunctions.h", "arm_nnsupportfunctions.h")
CLANG_ARGS = ("-fsyntax-only", "-w", "-Wno-error=implicit-function-declaration", "-DARM_NN_ENABLE_F16=1",
              "-DARM_NN_ENABLE_F32=1", "-Xclang", "-ast-dump=json")


def scalar_type(name: str) -> str:
    """A type name with qualifiers and array bounds removed."""
    name = re.sub(r"(\[\d*\])+$", "", name).strip()
    return re.sub(r"^((const|volatile)\s+)+", "", re.sub(r"\s+(const|volatile)$", "", name))


def names_enum(type_node: dict) -> bool:
    """Whether a type node of the AST resolves to an enum (or an array of one) rather than a pointer or function."""
    kind = type_node.get("kind", "")
    if kind in ("PointerType", "FunctionProtoType", "FunctionNoProtoType", "BlockPointerType"):
        return False
    return kind == "EnumType" or any(names_enum(child) for child in type_node.get("inner", []))


def main(include_dir: Path = INCLUDE_DIR) -> int:
    headers = sorted(include_dir.rglob("*.h"), key=lambda p: (p.name not in FIRST, FIRST.index(p.name) if p.name in FIRST
                                                           else 0, str(p)))
    if not headers:
        print(f"no headers found under {include_dir}", file=sys.stderr)
        return 1

    clang = os.environ.get("CLANG", "clang")
    with tempfile.TemporaryDirectory() as tmp:
        tu = Path(tmp) / "all_headers.c"
        tu.write_text("".join(f'#include "{h.resolve()}"\n' for h in headers), encoding="utf-8")
        try:
            result = subprocess.run([clang, *CLANG_ARGS, f"-I{include_dir}", str(tu)], capture_output=True, text=True)
        except FileNotFoundError:
            print(f"{clang} not found; set CLANG to a clang executable", file=sys.stderr)
            return 1
    if result.returncode != 0:
        print(f"{clang} could not parse the headers:\n{result.stderr}", file=sys.stderr)
        return 1

    ast = json.loads(result.stdout)
    root = Path(include_dir).resolve()
    # Typedefs that resolve to an enum, by name and by id. Decided from the type nodes, not from printed type names,
    # which differ between clang releases.
    enum_typedefs: set[str] = set()
    # A "typedef struct { ... } name;" record is anonymous; name it after its typedef
    record_names: dict[str, str] = {}

    def owned_tags(node: dict, typedef: str) -> None:
        for key in ("ownedTagDecl", "decl"):
            if "id" in node.get(key, {}):
                record_names.setdefault(node[key]["id"], typedef)
        for child in node.get("inner", []):
            owned_tags(child, typedef)

    for decl in ast.get("inner", []):
        if decl.get("kind") == "TypedefDecl":
            owned_tags(decl, decl["name"])

    fields: list[tuple[str, str, str, dict]] = []  # (header, record, field, clang type)
    records = set()
    current_file = ""

    def walk(node: dict, record: str) -> None:
        nonlocal current_file
        # clang's JSON names a file only when it differs from the previous location's
        for loc in (node.get("loc", {}), node.get("loc", {}).get("expansionLoc", {}),
                    node.get("range", {}).get("begin", {}), node.get("range", {}).get("begin", {}).get("expansionLoc", {})):
            if "file" in loc:
                current_file = loc["file"]
        kind = node.get("kind")
        in_headers = Path(current_file).resolve().is_relative_to(root) if current_file else False
        if kind == "TypedefDecl":
            if any(names_enum(child) for child in node.get("inner", [])):
                enum_typedefs.update((node["name"], node["id"]))
        elif kind == "RecordDecl" and in_headers and node.get("completeDefinition"):
            record = node.get("name") or record_names.get(node["id"]) or f"anonymous {node['tagUsed']} in {record}"
            records.add(node["id"])
        elif kind == "FieldDecl" and in_headers:
            fields.append((current_file, record, node.get("name", "(unnamed)"), node["type"]))
        for child in node.get("inner", []):
            walk(child, record)

    walk(ast, "")

    def is_enum(type_json: dict) -> bool:
        names = [scalar_type(type_json[key]) for key in ("qualType", "desugaredQualType") if key in type_json]
        if any("*" in name or "(" in name.replace("(unnamed at", "").replace("(anonymous at", "") for name in names):
            return False
        return (type_json.get("typeAliasDeclId") in enum_typedefs or
                any(name.startswith("enum ") or name in enum_typedefs for name in names))

    if not fields:
        print("no struct or union fields found under the headers; the check would pass vacuously", file=sys.stderr)
        return 1

    failures = [f"{Path(h).resolve().relative_to(root.parent)}: {record} field {field} has enum type "
                f"{type_['qualType']}; use int32_t and name the enum in the field's comment"
                for h, record, field, type_ in fields if is_enum(type_)]
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"Public struct enum check OK: {len(headers)} headers, {len(records)} records, {len(fields)} fields, "
          "no enum-typed field.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else INCLUDE_DIR))
