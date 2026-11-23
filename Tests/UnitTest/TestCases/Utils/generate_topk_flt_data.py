# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Generate the arm_topk_v2_f32 and arm_topk_v2_f16 goldens.

Each block is an NHWC tensor whose innermost extent is reduced, with the indices and value
bits of its top k for several k. The reference decodes every element to a Python float
(binary64, exact for both widths) and sorts each row with

    sorted(range(c), key=lambda i: (isnan(x[i]), -x[i], i))

so it is independent of the kernel's integer keys: numbers descending, equal numbers
(+0 and -0 included) by ascending index, every NaN after every number, NaNs by ascending
index (they compare unordered, so the stable sort keeps their index order). The value
written for an index is that element's input bits.

The float32 block `litert` holds the eight rows on which ai_edge_litert 2.2.0's TOPK_V2
was observed at k = 1, 3 and 8; its expected outputs are the interpreter's, written below
as literals, and generation fails if the reference does not reproduce them.

--check regenerates in memory and exits 1 if a checked-in header differs.
"""
import argparse
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERPRETER = "ai_edge_litert 2.2.0"


class Width:
    def __init__(self, name, fmt, digits, ctype):
        self.name = name
        self.fmt = fmt
        self.digits = digits
        self.ctype = ctype

    def value(self, bits):
        return struct.unpack("<" + self.fmt, struct.pack("<" + ("I" if self.fmt == "f" else "H"), bits))[0]


F32 = Width("f32", "f", 8, "uint32_t")
F16 = Width("f16", "e", 4, "uint16_t")


def f32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f16(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def reference(width, bits, dims, k):
    """The first k of each row's order: (indices, value bits), row-major."""
    rows = dims[0] * dims[1] * dims[2]
    c = dims[3]
    indices = []
    values = []
    for r in range(rows):
        row = bits[r * c:(r + 1) * c]
        x = [width.value(b) for b in row]
        order = sorted(range(c), key=lambda i: (math.isnan(x[i]), -x[i], i))
        indices += order[:k]
        values += [row[i] for i in order[:k]]
    return indices, values


# ai_edge_litert 2.2.0, TOPK_V2 over a [8, 8] float32 tensor with a constant k, one
# model per k (built with heliaAOT's tests/unit/utils/litert_model_builder.py). Rows:
# all equal; a three-way tie at the top; a tie straddling the k-th place; both zeros;
# a NaN first; a NaN in the middle; six NaNs and two numbers; -inf everywhere but one 0.
NINF32 = 0xFF800000
LITERT_ROWS = [
    [f32(2.5)] * 8,
    [f32(v) for v in (0, 5, 2, 5, 1, 5, 0, 0)],
    [f32(v) for v in (9, 3, 7, 3, 3, 1, 3, 0)],
    [f32(0.0), f32(-0.0), f32(0.0), f32(-0.0), f32(-1.0), f32(-1.0), f32(-1.0), f32(-1.0)],
    [0x7FC00000] + [f32(v) for v in (3, 1, 4, 1, 5, 9, 2)],
    [f32(3), f32(1), f32(4), 0xFFC00001, f32(1), f32(5), f32(9), f32(2)],
    [0x7FC00000, 0xFFC00000, f32(1.0), 0x7F800001, 0xFF812345, f32(-2.0), 0x7FFFFFFF, 0xFFFFFFFF],
    [NINF32, NINF32, NINF32, f32(0.0), NINF32, NINF32, NINF32, NINF32],
]
LITERT_INDICES = {
    1: [[0], [1], [0], [0], [6], [6], [2], [3]],
    3: [[0, 1, 2], [1, 3, 5], [0, 2, 1], [0, 1, 2], [6, 5, 3], [6, 5, 2], [2, 5, 0], [3, 0, 1]],
    8: [[0, 1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 2, 4, 0, 6, 7], [0, 2, 1, 3, 4, 6, 5, 7], [0, 1, 2, 3, 4, 5, 6, 7],
        [6, 5, 3, 1, 7, 2, 4, 0], [6, 5, 2, 0, 7, 1, 4, 3], [2, 5, 0, 1, 3, 4, 6, 7], [3, 0, 1, 2, 4, 5, 6, 7]],
}
LITERT_VALUES = {
    1: [[0x40200000], [0x40A00000], [0x41100000], [0x00000000], [0x41100000], [0x41100000], [0x3F800000],
        [0x00000000]],
    3: [[0x40200000, 0x40200000, 0x40200000], [0x40A00000, 0x40A00000, 0x40A00000],
        [0x41100000, 0x40E00000, 0x40400000], [0x00000000, 0x80000000, 0x00000000],
        [0x41100000, 0x40A00000, 0x40800000], [0x41100000, 0x40A00000, 0x40800000],
        [0x3F800000, 0xC0000000, 0x7FC00000], [0x00000000, 0xFF800000, 0xFF800000]],
    8: [[0x40200000] * 8,
        [0x40A00000, 0x40A00000, 0x40A00000, 0x40000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000],
        [0x41100000, 0x40E00000, 0x40400000, 0x40400000, 0x40400000, 0x40400000, 0x3F800000, 0x00000000],
        [0x00000000, 0x80000000, 0x00000000, 0x80000000, 0xBF800000, 0xBF800000, 0xBF800000, 0xBF800000],
        [0x41100000, 0x40A00000, 0x40800000, 0x40400000, 0x40000000, 0x3F800000, 0x3F800000, 0x7FC00000],
        [0x41100000, 0x40A00000, 0x40800000, 0x40400000, 0x40000000, 0x3F800000, 0x3F800000, 0xFFC00001],
        [0x3F800000, 0xC0000000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF812345, 0x7FFFFFFF, 0xFFFFFFFF],
        [0x00000000] + [0xFF800000] * 7],
}

# The same eight rows in binary16 (NaN payloads narrowed by hand; the reference alone
# decides these outputs, since the interpreter has no float16 TOPK_V2).
NINF16 = 0xFC00
LITERT_ROWS_F16 = [
    [f16(2.5)] * 8,
    [f16(v) for v in (0, 5, 2, 5, 1, 5, 0, 0)],
    [f16(v) for v in (9, 3, 7, 3, 3, 1, 3, 0)],
    [f16(0.0), f16(-0.0), f16(0.0), f16(-0.0), f16(-1.0), f16(-1.0), f16(-1.0), f16(-1.0)],
    [0x7E00] + [f16(v) for v in (3, 1, 4, 1, 5, 9, 2)],
    [f16(3), f16(1), f16(4), 0xFE01, f16(1), f16(5), f16(9), f16(2)],
    [0x7E00, 0xFE00, f16(1.0), 0x7C01, 0xFC45, f16(-2.0), 0x7FFF, 0xFFFF],
    [NINF16, NINF16, NINF16, f16(0.0), NINF16, NINF16, NINF16, NINF16],
]

SPECIALS_F32 = [
    [0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF],
    [0x00800000, 0x80800000, 0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800001, 0xFFC00000],
    [0x80000001, 0x00000001, 0x80000001, 0x00000001, 0x80000000, 0x00000000, 0x007FFFFF, 0x807FFFFF],
    [0x7FFFFFFF, 0x7F800000, 0xFF800000, 0xFFFFFFFF, 0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x7FC00001],
]
SPECIALS_F16 = [
    [0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x0001, 0x8001, 0x03FF, 0x83FF],
    [0x0400, 0x8400, 0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C01, 0xFE00],
    [0x8001, 0x0001, 0x8001, 0x0001, 0x8000, 0x0000, 0x03FF, 0x83FF],
    [0x7FFF, 0x7C00, 0xFC00, 0xFFFF, 0x7BFF, 0x7BFF, 0xFBFF, 0x7E01],
]

# Pools for the random block: repeats make ties at every place, NaNs of both kinds and
# signs, both zeros, both infinities, subnormals and the largest finite values.
POOL_F32 = [0x3F800000, 0x3F800000, 0xBF800000, 0x40000000, 0x00000000, 0x80000000, 0x7FC00000, 0xFF800001,
            0x7F800000, 0xFF800000, 0x00000001, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0x40000000]
POOL_F16 = [0x3C00, 0x3C00, 0xBC00, 0x4000, 0x0000, 0x8000, 0x7E00, 0xFC01,
            0x7C00, 0xFC00, 0x0001, 0x8001, 0x7BFF, 0xFBFF, 0x3C00, 0x4000]


def random_rows(pool, count):
    state = 0x2545F491
    out = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(pool[(state >> 16) % len(pool)])
    return out


def blocks(width):
    """[(name, dims, input bits, [(k, indices, values)])] for one width."""
    flat = lambda rows: [b for row in rows for b in row]  # noqa: E731
    if width is F32:
        litert, specials, pool = LITERT_ROWS, SPECIALS_F32, POOL_F32
    else:
        litert, specials, pool = LITERT_ROWS_F16, SPECIALS_F16, POOL_F16
    plan = [
        ("litert", [1, 1, 8, 8], flat(litert), [1, 3, 8]),
        ("specials", [1, 2, 2, 8], flat(specials), [1, 2, 5, 8]),
        ("random", [2, 3, 2, 11], random_rows(pool, 2 * 3 * 2 * 11), [1, 3, 5, 11]),
    ]
    out = []
    for name, dims, bits, ks in plan:
        cases = []
        for k in ks:
            indices, values = reference(width, bits, dims, k)
            if width is F32 and name == "litert":
                want_i = [i for row in LITERT_INDICES[k] for i in row]
                want_v = [v for row in LITERT_VALUES[k] for v in row]
                if (indices, values) != (want_i, want_v):
                    raise SystemExit(f"the reference does not reproduce {INTERPRETER} at k = {k}")
            cases.append((k, indices, values))
        out.append((name, dims, bits, cases))
    return out


def rows(items, fmt, per_line):
    return ["    " + ", ".join(fmt.format(x) for x in items[i:i + per_line]) + ","
            for i in range(0, len(items), per_line)]


def header(width):
    hexfmt = "0x{:0%dX}u" % width.digits
    lines = ["/*",
             " * SPDX-FileCopyrightText: 2026 Ambiq",
             " * SPDX-License-Identifier: Apache-2.0",
             " *",
             " * Generated by ../Utils/generate_topk_flt_data.py; do not edit. TOPK_V2 goldens for",
             f" * arm_topk_v2_{width.name}: each block is an NHWC input reduced over its innermost extent,",
             " * with the indices and value bits of the first k of each row's order for several k.",
             " */",
             "#pragma once",
             "",
             "#include <stdint.h>",
             ""]
    table = []
    for name, dims, bits, cases in blocks(width):
        if width is F32 and name == "litert":
            lines += [f"/* The {INTERPRETER} observation: the expected outputs of this block are the",
                      " * interpreter's TOPK_V2 indices and value bits, as literals; the reference reproduces",
                      " * them. */"]
        elif name == "litert":
            lines += ["/* The rows the interpreter was observed on, in binary16; the expected outputs are the",
                      " * reference's (the interpreter has no float16 TOPK_V2). */"]
        lines += [f"static const {width.ctype} tk_{name}_input[{len(bits)}] = {{"]
        lines += rows(bits, hexfmt, 8)
        lines += ["};"]
        for k, indices, values in cases:
            lines += [f"static const int32_t tk_{name}_k{k}_indices[{len(indices)}] = {{"]
            lines += rows(indices, "{}", 16)
            lines += ["};", f"static const {width.ctype} tk_{name}_k{k}_values[{len(values)}] = {{"]
            lines += rows(values, hexfmt, 8)
            lines += ["};"]
            dim_text = ", ".join(str(d) for d in dims)
            table.append(f"    {{{{{dim_text}}}, {k}, tk_{name}_input, tk_{name}_k{k}_indices, "
                         f"tk_{name}_k{k}_values}},")
        lines += [""]
    lines += ["typedef struct",
              "{",
              "    int32_t dims[4];",
              "    int32_t k;",
              f"    const {width.ctype} *input;",
              "    const int32_t *indices;",
              f"    const {width.ctype} *values;",
              "} tk_golden_case;",
              "",
              f"#define TK_GOLDEN_CASES {len(table)}",
              "static const tk_golden_case tk_golden_cases[TK_GOLDEN_CASES] = {"]
    lines += table
    lines += ["};", ""]
    if width is F32:
        lines += [f"/* {INTERPRETER}'s cases are the first three of tk_golden_cases. */",
                  "#define TK_LITERT_CASES 3", ""]
    return "\n".join(lines)


def outputs():
    root = HERE.parent
    return {
        root / "test_arm_topk_v2_f32" / "topk_v2_f32_data.h": header(F32),
        root / "test_arm_topk_v2_f16" / "topk_v2_f16_data.h": header(F16),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a checked-in header differs")
    args = parser.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    for path in stale:
        print(f"stale: {path.relative_to(HERE.parent)}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
