#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-shot generator for ``arm_nn_tanh_lut384_f32`` in
``Source/NNSupportFunctions/arm_nntables_flt.c``.

The table samples ``tanh(x)`` on a uniform grid over ``x in [0, XMAX]`` with
``SEGMENTS`` interpolation segments, storing ``SEGMENTS + 1`` entries so that
interpolation can safely read ``lut[idx + 1]``.

Grid-spacing invariant: ``XMAX / SEGMENTS == 6 / 384 == 4 / 256 == 1 / 64``.
The previous 257-entry table over ``[0, 4]`` used the same spacing, so entries
``0..256`` of this table are bit-identical to it and the index multiplier
(``SEGMENTS / XMAX == 64``) is unchanged.

Values are evaluated in double precision and printed with ``%.9f``; the C
compiler rounds the decimal literal to float32. This reproduces the original
table exactly -- do not switch to ``numpy.float32``/``tanhf``, which differs in
the last printed digit for most entries.

Usage:
    python3 scripts/gen_tanh_lut_f32.py          # print the table block
    python3 scripts/gen_tanh_lut_f32.py --check <file>   # verify a C file's block
"""

import argparse
import math
import re
import sys

XMAX = 6.0
SEGMENTS = 384
PER_LINE = 8
SYMBOL = "arm_nn_tanh_lut384_f32"


def values():
    """Yield the SEGMENTS + 1 table values as formatted C float literals."""
    for i in range(SEGMENTS + 1):
        yield "%.9ff" % math.tanh(XMAX * i / SEGMENTS)


def block():
    """Return the full C definition block for the table."""
    lits = list(values())
    lines = [
        "/*",
        " * LUT for tanh(x) sampled over x in [0, %g]." % XMAX,
        " * Generation formula:",
        " *   %s[i] = tanh(%.1f * i / %d.0)" % (SYMBOL, XMAX, SEGMENTS),
        " * Regenerate with scripts/gen_tanh_lut_f32.py.",
        " */",
        "const float32_t %s[%d] = {" % (SYMBOL, SEGMENTS + 1),
    ]
    for start in range(0, len(lits), PER_LINE):
        chunk = lits[start : start + PER_LINE]
        lines.append("    " + " ".join(v + "," for v in chunk))
    lines.append("};")
    return "\n".join(lines) + "\n"


def check(path):
    """Verify that ``path`` contains the expected table values."""
    text = open(path, "r", encoding="utf-8").read()
    match = re.search(
        r"const\s+float32_t\s+%s\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};" % SYMBOL,
        text,
        re.S,
    )
    if match is None:
        print("FAIL: %s not found in %s" % (SYMBOL, path))
        return 1
    found = [tok.strip() for tok in match.group(2).split(",") if tok.strip()]
    expected = list(values())
    if int(match.group(1)) != SEGMENTS + 1:
        print("FAIL: declared length %s != %d" % (match.group(1), SEGMENTS + 1))
        return 1
    if found != expected:
        bad = [i for i in range(min(len(found), len(expected))) if found[i] != expected[i]]
        print("FAIL: %d entries differ (count %d vs %d), first: %s" % (len(bad), len(found), len(expected), bad[:5]))
        return 1
    print("OK: %s matches the generator (%d entries)" % (SYMBOL, len(found)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", metavar="FILE", help="verify an existing C file instead of printing")
    args = parser.parse_args()
    if args.check:
        return check(args.check)
    sys.stdout.write(block())
    return 0


if __name__ == "__main__":
    sys.exit(main())
