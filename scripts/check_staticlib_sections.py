#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
"""Check that a prebuilt ns-cmsis-nn archive was compiled with one section per function.

A --gc-sections link can drop an unused kernel only when each function sits in its own
.text.<name> section. This reads the archive and every ELF member directly (no toolchain
binutils, so it works the same for gcc, atfe and armclang archives) and fails when:

  - any member keeps code in a plain ".text" section of non-zero size, or
  - the multi-function reference object has fewer than two ".text.*" sections.

Usage: check_staticlib_sections.py <archive.a> [--reference <member-name-substring>]
Exit 0 = pass, 1 = check failed, 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import struct
import sys

AR_MAGIC = b"!<arch>\n"
DEFAULT_REFERENCE = "arm_convolve_get_buffer_sizes_s8.c"


def ar_members(data: bytes):
    """Yield (name, payload) for the regular members of a GNU or BSD ar archive."""
    if not data.startswith(AR_MAGIC):
        raise ValueError("not an ar archive")
    pos = len(AR_MAGIC)
    long_names = b""
    while pos + 60 <= len(data):
        header = data[pos : pos + 60]
        if header[58:60] != b"`\n":
            raise ValueError(f"bad member header at offset {pos}")
        raw_name = header[0:16].rstrip(b" ")
        size = int(header[48:58].strip() or b"0")
        body = data[pos + 60 : pos + 60 + size]
        pos += 60 + size + (size & 1)
        if raw_name == b"//":
            long_names = body
            continue
        if raw_name in (b"/", b"/SYM64/", b"__.SYMDEF", b"__.SYMDEF SORTED"):
            continue
        if raw_name.startswith(b"#1/"):
            name_len = int(raw_name[3:])
            name, body = body[:name_len].rstrip(b"\0"), body[name_len:]
        elif raw_name.startswith(b"/") and raw_name[1:].isdigit():
            start = int(raw_name[1:])
            end = long_names.find(b"\n", start)
            name = long_names[start:end].rstrip(b"/")
        else:
            name = raw_name.rstrip(b"/")
        yield name.decode("utf-8", "replace"), body


def elf_sections(obj: bytes):
    """Return [(name, size)] for the section headers of a little-endian ELF object."""
    if obj[:4] != b"\x7fELF":
        return None
    elf_class, elf_data = obj[4], obj[5]
    if elf_data != 1:
        raise ValueError("big-endian ELF is not expected here")
    if elf_class == 1:
        shoff, = struct.unpack_from("<I", obj, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", obj, 0x2E)
        name_fmt, size_off, size_fmt, off_off, off_fmt = "<I", 0x14, "<I", 0x10, "<I"
    elif elf_class == 2:
        shoff, = struct.unpack_from("<Q", obj, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", obj, 0x3A)
        name_fmt, size_off, size_fmt, off_off, off_fmt = "<I", 0x20, "<Q", 0x18, "<Q"
    else:
        raise ValueError("unknown ELF class")
    headers = []
    for i in range(shnum):
        base = shoff + i * shentsize
        name_idx, = struct.unpack_from(name_fmt, obj, base)
        offset, = struct.unpack_from(off_fmt, obj, base + off_off)
        size, = struct.unpack_from(size_fmt, obj, base + size_off)
        headers.append((name_idx, offset, size))
    strtab_off, strtab_size = headers[shstrndx][1], headers[shstrndx][2]
    strtab = obj[strtab_off : strtab_off + strtab_size]
    sections = []
    for name_idx, _, size in headers:
        end = strtab.find(b"\0", name_idx)
        sections.append((strtab[name_idx:end].decode("utf-8", "replace"), size))
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="substring of the multi-function member that must have >= 2 .text.* sections")
    args = parser.parse_args()
    try:
        with open(args.archive, "rb") as handle:
            members = list(ar_members(handle.read()))
    except (OSError, ValueError) as exc:
        print(f"check_staticlib_sections: cannot read {args.archive}: {exc}", file=sys.stderr)
        return 2

    failures = []
    objects = 0
    function_sections = 0
    reference_count = None
    for name, body in members:
        sections = elf_sections(body)
        if sections is None:
            continue
        objects += 1
        per_function = sum(1 for sec, _ in sections if sec.startswith(".text."))
        function_sections += per_function
        plain = [size for sec, size in sections if sec == ".text" and size > 0]
        if plain:
            failures.append(f"{name}: {plain[0]} bytes of code in a plain .text section")
        if args.reference in name:
            reference_count = per_function

    if reference_count is None:
        failures.append(f"reference member '{args.reference}' not found in the archive")
    elif reference_count < 2:
        failures.append(f"reference member '{args.reference}' has {reference_count} .text.* section(s), expected >= 2")

    if failures:
        print(f"check_staticlib_sections: {args.archive} was not built with -ffunction-sections:", file=sys.stderr)
        for line in failures[:20]:
            print(f"  {line}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        return 1
    print(f"check_staticlib_sections: OK: {objects} objects, {function_sections} .text.* sections, "
          f"reference '{args.reference}' has {reference_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
