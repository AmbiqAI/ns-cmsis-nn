#!/usr/bin/env python3
"""Generate an annotated public include tree for release builds."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HEADER_GUARD_RE = re.compile(r"^(#define\s+\w+)\s*$", re.MULTILINE)
SYMBOL_RE = re.compile(r"\b(arm_[A-Za-z0-9_]+)\s*\(")

UMBRELLA_HEADER = """/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CMSIS_NN_UMBRELLA_H
#define CMSIS_NN_UMBRELLA_H

#include "public_api.h"
#include "arm_nn_math_types.h"
#include "arm_nn_types.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nnfunctions.h"

#endif /* CMSIS_NN_UMBRELLA_H */
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="Path to public_symbols.txt")
    parser.add_argument("--hdr-dir", required=True, help="Source header tree to copy")
    parser.add_argument("--out-dir", required=True, help="Destination header tree")
    parser.add_argument("--public-api", required=True, help="Path to release/public_api.h")
    return parser.parse_args()


def load_symbols(path: Path) -> list[str]:
    symbols: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!exclude "):
            continue
        if line.startswith("arm_"):
            symbols.append(line)
    if not symbols:
        raise ValueError(f"No annotatable function symbols found in {path}")
    return symbols


def clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def copy_header_tree(hdr_dir: Path, out_dir: Path) -> list[Path]:
    shutil.copytree(hdr_dir, out_dir, dirs_exist_ok=True)
    return sorted(path for path in out_dir.rglob("*.h") if path.name not in {"public_api.h", "cmsis_nn.h"})


def copy_public_api(public_api: Path, out_dir: Path) -> None:
    shutil.copy2(public_api, out_dir / "public_api.h")


def inject_public_api_include(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if '#include "public_api.h"' in text:
        return

    match = HEADER_GUARD_RE.search(text)
    if match:
        text = text[: match.end()] + '\n\n#include "public_api.h"' + text[match.end() :]
    else:
        text = '#include "public_api.h"\n' + text

    path.write_text(text, encoding="utf-8")


def annotate_symbols(path: Path, symbols: set[str]) -> set[str]:
    matched: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    brace_depth = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if brace_depth > 0:
            brace_depth += line.count("{") - line.count("}")
            continue
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "}")):
            brace_depth += line.count("{") - line.count("}")
            continue
        if "CMSIS_API" in line:
            brace_depth += line.count("{") - line.count("}")
            continue
        if "__STATIC" in line or "static inline" in line or "static __inline" in line:
            brace_depth += line.count("{") - line.count("}")
            continue

        candidates = [symbol for symbol in SYMBOL_RE.findall(line) if symbol in symbols]
        if not candidates:
            brace_depth += line.count("{") - line.count("}")
            continue

        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = f"{indent}CMSIS_API {line.lstrip()}"
        matched.update(candidates)
        brace_depth += line.count("{") - line.count("}")

    path.write_text("".join(lines), encoding="utf-8")
    return matched


def write_umbrella_header(out_dir: Path) -> None:
    (out_dir / "cmsis_nn.h").write_text(UMBRELLA_HEADER, encoding="utf-8")


def main() -> int:
    args = parse_args()

    symbols_path = Path(args.symbols).resolve()
    header_dir = Path(args.hdr_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    public_api = Path(args.public_api).resolve()

    if not header_dir.is_dir():
        raise SystemExit(f"Header directory not found: {header_dir}")
    if not public_api.is_file():
        raise SystemExit(f"Public API header not found: {public_api}")

    symbols = load_symbols(symbols_path)

    clean_output_dir(out_dir)
    headers = copy_header_tree(header_dir, out_dir)
    copy_public_api(public_api, out_dir)
    write_umbrella_header(out_dir)
    headers.append(out_dir / "cmsis_nn.h")

    matched: set[str] = set()
    symbol_set = set(symbols)
    for header in headers:
        inject_public_api_include(header)
        matched.update(annotate_symbols(header, symbol_set))

    print(f"Generated annotated headers in {out_dir} ({len(matched)} annotated symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
