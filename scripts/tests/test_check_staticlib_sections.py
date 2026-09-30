#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
"""Mutation tests for scripts/check_staticlib_sections.py.

The release and dry-run workflows only ever run the checker on archives that pass, so these cases prove it still
fails: each builds a small ar archive of hand-made ELF objects and checks the exit code (0 pass, 1 check failed,
2 unreadable input).
"""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_staticlib_sections.py"
REFERENCE = "arm_convolve_get_buffer_sizes_s8.c.o"


def elf(sections: list[tuple[str, int]]) -> bytes:
    """A little-endian ELF32 relocatable with the given (name, size) sections, plus the null and .shstrtab ones."""
    names = [""] + [name for name, _ in sections] + [".shstrtab"]
    strtab = b""
    offsets = []
    for name in names:
        offsets.append(len(strtab))
        strtab += name.encode() + b"\0"
    header_size = 52
    shoff = header_size + len(strtab)
    ident = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header = ident + struct.pack("<HHIIIIIHHHHHH", 1, 40, 1, 0, 0, shoff, 0, 52, 0, 0, 40, len(names), len(names) - 1)
    shdrs = b""
    sizes = [0] + [size for _, size in sections] + [len(strtab)]
    for index, (name_off, size) in enumerate(zip(offsets, sizes)):
        offset = header_size if index == len(names) - 1 else 0
        shdrs += struct.pack("<IIIIIIIIII", name_off, 0 if index == 0 else 1, 0, 0, offset, size, 0, 0, 1, 0)
    return header + strtab + shdrs


def ar_header(raw_name: bytes, size: int) -> bytes:
    fields = raw_name.ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6) + b"644".ljust(8)
    return fields + str(size).encode().ljust(10) + b"`\n"


def archive(*members: tuple[str, bytes]) -> bytes:
    """A GNU ar archive; names longer than 15 characters go through the // long-name table."""
    table = b""
    out = b""
    for name, body in members:
        if len(name) > 15:
            raw = b"/" + str(len(table)).encode()
            table += name.encode() + b"/\n"
        else:
            raw = name.encode() + b"/"
        out += ar_header(raw, len(body)) + body + (b"\n" if len(body) & 1 else b"")
    head = ar_header(b"//", len(table)) + table + (b"\n" if len(table) & 1 else b"") if table else b""
    return b"!<arch>\n" + head + out


GOOD_REFERENCE = elf([(".text", 0), (".text.arm_convolve_s8_get_buffer_size", 20), (".text.arm_convolve_1x1", 8)])
GOOD_OTHER = elf([(".text", 0), (".text.arm_relu_s8", 12)])


class CheckStaticlibSectionsTest(unittest.TestCase):
    def run_full(self, data: bytes, *extra: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.a"
            path.write_bytes(data)
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(path), *extra], capture_output=True, text=True, check=False
            )

    def run_checker(self, data: bytes, *extra: str) -> int:
        return self.run_full(data, *extra).returncode

    def test_sectioned_archive_passes(self):
        self.assertEqual(0, self.run_checker(archive((REFERENCE, GOOD_REFERENCE), ("relu.c.o", GOOD_OTHER))))

    def test_armclang_member_name_passes(self):
        self.assertEqual(0, self.run_checker(archive(("arm_convolve_get_buffer_sizes_s8.o", GOOD_REFERENCE))))

    def test_bsd_long_name_passes(self):
        name = REFERENCE.encode()
        header = f"#1/{len(name)}".encode().ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
        body = name + GOOD_REFERENCE
        header += b"644".ljust(8) + str(len(body)).encode().ljust(10) + b"`\n"
        data = b"!<arch>\n" + header + body + (b"\n" if len(body) & 1 else b"")
        self.assertEqual(0, self.run_checker(data))

    def test_plain_text_with_code_fails(self):
        plain = elf([(".text", 64), (".text.arm_relu_s8", 12)])
        self.assertEqual(1, self.run_checker(archive((REFERENCE, GOOD_REFERENCE), ("relu.c.o", plain))))

    def test_reference_with_one_section_fails(self):
        single = elf([(".text", 0), (".text.arm_convolve_s8_get_buffer_size", 20)])
        self.assertEqual(1, self.run_checker(archive((REFERENCE, single))))

    def test_missing_reference_fails(self):
        result = self.run_full(archive(("relu.c.o", GOOD_OTHER)))
        self.assertEqual(1, result.returncode)
        self.assertIn("not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_truncated_elf_member_is_unreadable(self):
        self.assertEqual(2, self.run_checker(archive((REFERENCE, GOOD_REFERENCE[:40]))))

    def test_partial_trailing_header_is_unreadable(self):
        data = archive((REFERENCE, GOOD_REFERENCE)) + b"junk.o/".ljust(30)
        self.assertEqual(2, self.run_checker(data))

    def test_member_past_end_is_unreadable(self):
        data = archive((REFERENCE, GOOD_REFERENCE))
        # The header declares the whole object but the file ends 20 bytes into it.
        truncated = ar_header(b"relu.c.o/", len(GOOD_OTHER)) + GOOD_OTHER[:20]
        self.assertEqual(2, self.run_checker(data + truncated))

    def test_odd_final_member_without_pad_byte_is_unreadable(self):
        odd = GOOD_OTHER + (b"\0" if len(GOOD_OTHER) % 2 == 0 else b"")
        data = archive((REFERENCE, GOOD_REFERENCE), ("relu.c.o", odd))
        self.assertEqual(0, self.run_checker(data))
        # The archive ends where the final member's pad byte should be.
        self.assertEqual(2, self.run_checker(data[:-1]))

    def test_not_an_archive_is_unreadable(self):
        self.assertEqual(2, self.run_checker(b"not an archive"))

    def test_missing_file_is_unreadable(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "/nonexistent/lib.a"], capture_output=True, text=True, check=False
        )
        self.assertEqual(2, result.returncode)


if __name__ == "__main__":
    unittest.main()
