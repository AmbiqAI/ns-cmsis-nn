#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# No struct in the public headers may declare an enum-typed field. An enum's size follows the compiler's convention
# (arm-none-eabi-gcc: short enums; ATfE, armclang and host compilers: int), so such a field gives the struct a different
# layout in a caller and a library built with different conventions, and one of them misreads it. See #693 and #764.

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE_DIR = REPO / "Include"

ENUM_TYPEDEF = re.compile(r"typedef\s+enum\s*\w*\s*\{[^}]*\}\s*(\w+)\s*;", re.S)
STRUCT_TYPEDEF = re.compile(r"typedef\s+struct\s*\w*\s*\{(?P<body>[^}]*)\}\s*(?P<name>\w+)\s*;", re.S)
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def main(include_dir: Path = INCLUDE_DIR) -> int:
    headers = sorted(include_dir.rglob("*.h"))
    if not headers:
        print(f"no headers found under {include_dir}", file=sys.stderr)
        return 1

    texts = {path: COMMENT.sub(" ", path.read_text(encoding="utf-8")) for path in headers}
    enums = {name for text in texts.values() for name in ENUM_TYPEDEF.findall(text)}
    if not enums:
        print("no enum typedefs found; the check would pass vacuously", file=sys.stderr)
        return 1

    field = re.compile(r"(?:^|[;{\s])(?:const\s+)?(" + "|".join(sorted(enums)) + r")\s+\w+\s*(?:\[[^\]]*\])?\s*;")
    failures = []
    for path, text in texts.items():
        for struct in STRUCT_TYPEDEF.finditer(text):
            for match in field.finditer(struct.group("body")):
                failures.append(f"{path.relative_to(include_dir.parent)}: struct {struct.group('name')} has a field of enum type "
                                f"{match.group(1)}; use int32_t and name the enum in the field's comment")

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"Public struct enum check OK: {len(headers)} headers, {len(enums)} enum types, no enum-typed struct fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else INCLUDE_DIR))
