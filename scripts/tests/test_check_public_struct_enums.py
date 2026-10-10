#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_public_struct_enums.py (#764): the check catches an enum-typed field in each form it
# covers (listed in the script), accepts the int32_t form and pointers to an enum, fails loudly when the headers do not
# parse or hold no fields, and the real headers pass. Needs clang (CLANG, default "clang").
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

PRELUDE = "#include <stdint.h>\ntypedef enum\n{\n    FMT_A = 0,\n    FMT_B = 1,\n} my_format;\n"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_public_struct_enums", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CheckPublicStructEnums(unittest.TestCase):
    def run_on(self, header: str, parent: str = "repo") -> int:
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / parent / "Include"
            include.mkdir(parents=True)
            (include / "types.h").write_text(header, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return load_checker().main(include)

    def test_plain_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    int32_t a;\n    my_format fmt;\n} params;\n"))

    def test_const_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    const my_format fmt;\n} params;\n"))

    def test_trailing_const_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    my_format const fmt;\n} params;\n"))

    def test_alias_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef my_format my_format_alias;\ntypedef struct\n{\n    my_format_alias fmt;\n} params;\n"))

    def test_tagged_enum_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "enum tag_format { TAG_A };\nstruct params\n{\n    enum tag_format fmt;\n};\n"))

    def test_second_declarator_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    int32_t a, b;\n    my_format c, d;\n} params;\n"))

    def test_bitfield_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    my_format fmt : 4;\n} params;\n"))

    def test_array_2d_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    my_format fmt[2][3];\n} params;\n"))

    def test_after_nested_struct_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    struct\n    {\n        int32_t w;\n    } inner;\n    my_format fmt;\n} params;\n"))

    def test_in_nested_struct_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    struct\n    {\n        my_format fmt;\n    } inner;\n} params;\n"))

    def test_union_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef union\n{\n    int32_t a;\n    my_format fmt;\n} params;\n"))

    def test_attribute_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct __attribute__((aligned(8)))\n{\n    my_format fmt;\n} params;\n"))

    def test_declarator_list_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "struct params\n{\n    my_format fmt;\n} p, *pp;\n"))

    def test_inline_anonymous_enum_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "struct params\n{\n    enum\n    {\n        XA,\n        XB\n    } fmt;\n};\n"))

    def test_inline_anonymous_enum_array_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "struct params\n{\n    enum\n    {\n        XA\n    } fmt[2];\n};\n"))

    def test_inline_anonymous_enum_under_a_path_with_parentheses_is_caught(self):
        header = PRELUDE + "struct params\n{\n    enum\n    {\n        XA\n    } fmt;\n};\n"
        self.assertEqual(1, self.run_on(header, parent="nn (copy) *"))

    def test_typeof_typedef_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    typeof(my_format) fmt;\n} params;\n"))

    def test_typeof_tagged_enum_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "enum tag_format { TAG_A };\nstruct params\n{\n    __typeof__(enum tag_format) fmt;\n};\n"))

    def test_atomic_specifier_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    _Atomic(my_format) fmt;\n} params;\n"))

    def test_atomic_qualifier_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    _Atomic my_format fmt;\n} params;\n"))

    def test_typedef_of_enum_array_is_caught(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef my_format my_formats[2];\ntypedef struct\n{\n    my_formats fmt;\n} params;\n"))

    def test_pointer_to_anonymous_enum_passes(self):
        self.assertEqual(0, self.run_on(PRELUDE + "struct params\n{\n    enum\n    {\n        XA\n    } *fmt;\n};\n"))

    def test_typedef_of_enum_pointer_passes(self):
        self.assertEqual(0, self.run_on(PRELUDE + "typedef my_format *my_format_ptr;\ntypedef struct\n{\n    my_format_ptr fmt;\n} params;\n"))

    def test_shadowed_enum_typedef_name_passes(self):
        header = (PRELUDE + "static inline int local_only(void)\n{\n    typedef enum\n    {\n        ZA\n    } z_t;\n"
                  "    return (int)sizeof(z_t);\n}\ntypedef int32_t z_t;\ntypedef struct\n{\n    z_t fmt;\n} params;\n")
        self.assertEqual(0, self.run_on(header))

    def test_int32_field_naming_the_enum_passes(self):
        self.assertEqual(0, self.run_on(PRELUDE + "typedef struct\n{\n    int32_t fmt; /**< A my_format value. */\n} params;\n"))

    def test_enum_named_in_a_comment_only_passes(self):
        self.assertEqual(0, self.run_on(PRELUDE + "typedef struct\n{\n    // my_format fmt;\n    int32_t fmt;\n} params;\n"))

    def test_pointer_to_enum_passes(self):
        self.assertEqual(0, self.run_on(PRELUDE + "typedef struct\n{\n    const my_format *fmt;\n} params;\n"))

    def test_no_headers_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(1, load_checker().main(Path(tmp)))

    def test_no_fields_fails_loudly(self):
        self.assertEqual(1, self.run_on(PRELUDE))

    def test_unparsable_headers_fail_loudly(self):
        self.assertEqual(1, self.run_on(PRELUDE + "typedef struct\n{\n    undeclared_type fmt;\n} params;\n"))

    def test_repo_is_clean(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, load_checker().main())


if __name__ == "__main__":
    unittest.main()
