#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_public_struct_enums.py (#764): the check catches an enum-typed field, accepts the
# int32_t form, ignores enums named only in comments, fails loudly when discovery finds nothing, and the real headers
# pass.
#
# Run with: python3 scripts/tests/test_check_public_struct_enums.py

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_public_struct_enums.py"

ENUM = "typedef enum\n{\n    FMT_A = 0,\n    FMT_B = 1,\n} my_format;\n"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_public_struct_enums", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CheckPublicStructEnums(unittest.TestCase):
    def run_on(self, header: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "Include"
            include.mkdir()
            (include / "types.h").write_text(header, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return load_checker().main(include)

    def test_enum_field_is_caught(self):
        self.assertEqual(1, self.run_on(ENUM + "typedef struct\n{\n    int32_t a;\n    my_format fmt;\n} params;\n"))

    def test_const_enum_field_is_caught(self):
        self.assertEqual(1, self.run_on(ENUM + "typedef struct\n{\n    const my_format fmt;\n} params;\n"))

    def test_int32_field_naming_the_enum_passes(self):
        self.assertEqual(0, self.run_on(ENUM + "typedef struct\n{\n    int32_t fmt; /**< A my_format value. */\n} params;\n"))

    def test_enum_named_in_a_comment_only_passes(self):
        self.assertEqual(0, self.run_on(ENUM + "typedef struct\n{\n    // my_format fmt;\n    int32_t fmt;\n} params;\n"))

    def test_no_headers_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(1, load_checker().main(Path(tmp)))

    def test_no_enums_fails_loudly(self):
        self.assertEqual(1, self.run_on("typedef struct\n{\n    int32_t a;\n} params;\n"))

    def test_repo_is_clean(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, load_checker().main())


if __name__ == "__main__":
    unittest.main()
