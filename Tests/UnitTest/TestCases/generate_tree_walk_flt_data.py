# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Generate the arm_nn_tree_walk_f32 goldens: numpy only, seeded, no scikit-learn.

Each table is built here as the semantic columns a walk file holds (feature, threshold,
missing_left, left, right, roots, with a child below N a split and N..N+L-1 a leaf),
and numpy_walk answers from those columns. The kernel's packed record is written out
separately by packed_table and the walk never reads it, so the expected leaves share
no code and no leaf encoding with the C. --check regenerates in memory and exits 1 if
the checked-in header differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SEED = 20261104
RANDOM_TABLES = 64
RANDOM_ROWS = 32
MAX_NODES = 50
MAX_TREES = 8
MAX_FEATURES = 6
OUTPUT = Path(__file__).parent / "test_arm_nn_tree_walk_f32" / "tree_walk_f32_data.h"

POSITIVE_INF = 0x7F800000
NEGATIVE_INF = 0xFF800000
# Quiet, negative quiet, signalling, largest payload, negative signalling.
NANS = (0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0xFF800001)
MISSING_LEFT = 0x8000

BOUNDARY_KINDS = ("equal", "up", "down")
# Breadth first: 0 -> 1, 2; 1 -> 3, 4; 2 -> 5, 6; the leaves below are 0..7 in order.
BOUNDARY_THRESHOLDS = np.array(
    [3.0, 2.5, 0.1, -1e-30, 1e30, np.finfo(np.float32).max, -np.finfo(np.float32).max],
    dtype=np.float32,
)
BOUNDARY_LEFT = (1, 3, 5, 7 + 0, 7 + 2, 7 + 4, 7 + 6)
BOUNDARY_RIGHT = (2, 4, 6, 7 + 1, 7 + 3, 7 + 5, 7 + 7)
BOUNDARY_PATHS = {0: (), 1: ((0, "left"),), 2: ((0, "right"),), 3: ((0, "left"), (1, "left")),
                  4: ((0, "left"), (1, "right")), 5: ((0, "right"), (2, "left")),
                  6: ((0, "right"), (2, "right"))}
# A feature the walk has no business reading holds this: a misread row misroutes.
UNREAD = np.float32(1234.5)


def bits_of(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def values_of(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def numpy_walk(table, row):
    """The leaf index each tree reaches, from the semantic columns."""
    splits = len(table["feature"])
    leaves = []
    for root in table["roots"]:
        node = int(root)
        while node < splits:
            value = row[table["feature"][node]]
            if np.isnan(value):
                go_left = table["missing_left"][node] == 1
            else:
                go_left = bool(value <= table["threshold"][node])
            node = int(table["left"][node] if go_left else table["right"][node])
        leaves.append(node - splits)
    return leaves


def random_table(rng, splits, trees, features):
    """Trees grown at random open slots over shuffled split and leaf ids."""
    node_ids = rng.permutation(splits)
    sizes = np.ones(trees, dtype=np.int64)
    for _ in range(splits - trees):
        sizes[rng.integers(trees)] += 1
    leaves = splits + trees
    leaf_ids = iter(splits + rng.permutation(leaves))
    left = np.zeros(splits, dtype=np.int32)
    right = np.zeros(splits, dtype=np.int32)
    roots = []
    start = 0
    for size in sizes:
        members = [int(node) for node in node_ids[start:start + size]]
        start += size
        roots.append(members[0])
        slots = [(members[0], left), (members[0], right)]
        for member in members[1:]:
            parent, side = slots.pop(rng.integers(len(slots)))
            side[parent] = member
            slots += [(member, left), (member, right)]
        for parent, side in slots:
            side[parent] = next(leaf_ids)
    thresholds = np.where(
        rng.random(splits) < 0.25,
        rng.integers(-8, 9, splits).astype(np.float64),
        rng.normal(0.0, 10.0, splits),
    ).astype(np.float32)
    return {
        "feature": rng.integers(0, features, splits).astype(np.int32),
        "threshold": thresholds,
        "missing_left": rng.integers(0, 2, splits).astype(np.int32),
        "left": left,
        "right": right,
        "roots": np.array(roots, dtype=np.int32),
        "leaf_count": leaves,
        "feature_count": features,
    }


def random_row(rng, table):
    """One row in bits: ordinary values, thresholds and their neighbours, NaNs, infinities."""
    features = table["feature_count"]
    row = bits_of(rng.normal(0.0, 10.0, features).astype(np.float32)).copy()
    for index in range(features):
        draw = rng.random()
        if draw < 0.10:
            row[index] = NANS[rng.integers(len(NANS))]
        elif draw < 0.15:
            row[index] = POSITIVE_INF if rng.integers(2) else NEGATIVE_INF
        elif draw < 0.45:
            on_this_feature = table["threshold"][table["feature"] == index]
            if on_this_feature.size:
                threshold = on_this_feature[rng.integers(on_this_feature.size)]
                direction = np.float32(rng.choice([np.inf, -np.inf]))
                if draw < 0.30:
                    row[index] = bits_of(threshold)
                else:
                    row[index] = bits_of(np.nextafter(threshold, direction))
    return row


def packed_table(table):
    """The kernel's record, as a writer packs it: bit 15 the NaN route, a leaf -(leaf + 1)."""
    splits = len(table["feature"])
    feature = table["feature"] | np.where(table["missing_left"] == 1, MISSING_LEFT, 0)
    left = np.where(table["left"] >= splits, -(table["left"] - splits + 1), table["left"])
    right = np.where(table["right"] >= splits, -(table["right"] - splits + 1), table["right"])
    return feature, bits_of(table["threshold"]), left, right


def random_block(rng):
    records, features, thresholds, lefts, rights, roots, rows, leaves = [], [], [], [], [], [], [], []
    routes = {"nan_left": 0, "nan_right": 0, "equal": 0}
    for number in range(RANDOM_TABLES):
        if number == 0:
            shape = (1, 1, 1)
        elif number == 1:
            shape = (MAX_NODES, MAX_TREES, MAX_FEATURES)
        else:
            nodes = int(rng.integers(1, MAX_NODES + 1))
            shape = (nodes, int(rng.integers(1, min(MAX_TREES, nodes) + 1)), int(rng.integers(1, MAX_FEATURES + 1)))
        table = random_table(rng, *shape)
        records.append(
            (len(features), shape[0], table["leaf_count"], shape[1], shape[2], len(roots), len(rows), len(leaves))
        )
        packed = packed_table(table)
        features += packed[0].tolist()
        thresholds += packed[1].tolist()
        lefts += packed[2].tolist()
        rights += packed[3].tolist()
        roots += table["roots"].tolist()
        for _ in range(RANDOM_ROWS):
            row = random_row(rng, table)
            rows += row.tolist()
            values = values_of(row)
            leaves += numpy_walk(table, values)
            for split in range(shape[0]):
                value = values[table["feature"][split]]
                if np.isnan(value):
                    routes["nan_left" if table["missing_left"][split] else "nan_right"] += 1
                elif value == table["threshold"][split]:
                    routes["equal"] += 1
    # The rows must reach what they are drawn for, or the block proves less than it says.
    if min(routes.values()) == 0:
        raise SystemExit(f"a random route was never drawn: {routes}")
    return records, features, thresholds, lefts, rights, roots, rows, leaves


def boundary_table():
    splits = len(BOUNDARY_THRESHOLDS)
    return {
        "feature": np.arange(splits, dtype=np.int32),
        "threshold": BOUNDARY_THRESHOLDS,
        "missing_left": (np.arange(splits, dtype=np.int32) + 1) % 2,
        "left": np.array(BOUNDARY_LEFT, dtype=np.int32),
        "right": np.array(BOUNDARY_RIGHT, dtype=np.int32),
        "roots": np.array([0], dtype=np.int32),
        "leaf_count": splits + 1,
        "feature_count": splits,
    }


def boundary_block():
    """A row per split per kind: its feature at the threshold or next to it, the path held open."""
    table = boundary_table()
    rows, leaves, splits, kinds = [], [], [], []
    for split, threshold in enumerate(BOUNDARY_THRESHOLDS):
        for kind, name in enumerate(BOUNDARY_KINDS):
            row = np.full(len(BOUNDARY_THRESHOLDS), UNREAD, dtype=np.float32)
            for ancestor, side in BOUNDARY_PATHS[split]:
                offset = np.float32(-1.0) if side == "left" else np.float32(1.0)
                row[ancestor] = BOUNDARY_THRESHOLDS[ancestor] + offset
            if name == "equal":
                row[split] = threshold
            else:
                direction = np.float32(np.inf if name == "up" else -np.inf)
                with np.errstate(over="ignore"):  # The neighbour of the largest float is the infinity.
                    row[split] = np.nextafter(threshold, direction)
            rows += bits_of(row).tolist()
            leaves += numpy_walk(table, row)
            splits.append(split)
            kinds.append(kind)
    return table, rows, leaves, splits, kinds


def array(ctype, name, values, form, per_line=8):
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        lines.append("    " + ", ".join(form(v) for v in values[start:start + per_line]) + ",")
    lines.append("};")
    return lines


def u32(value):
    return f"0x{int(value):08X}u"


def u16(value):
    return f"0x{int(value):04X}u"


def plain(value):
    return str(int(value))


def render():
    rng = np.random.default_rng(SEED)
    records, features, thresholds, lefts, rights, roots, rows, leaves = random_block(rng)
    table, b_rows, b_leaves, b_splits, b_kinds = boundary_block()
    b_features, b_thresholds, b_left, b_right = packed_table(table)
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: 2026 Ambiq",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * Generated by ../generate_tree_walk_flt_data.py; do not edit. Goldens for arm_nn_tree_walk_f32:",
        f" * numpy Generator(PCG64) seed {SEED}, the expected leaves from the script's own walk over the",
        " * semantic columns. Floats are written as their bits, so NaN payloads and neighbours survive.",
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define TW_RANDOM_TABLES {RANDOM_TABLES}",
        f"#define TW_RANDOM_ROWS {RANDOM_ROWS}",
        f"#define TW_RANDOM_NODES {len(features)}",
        f"#define TW_RANDOM_ROOTS {len(roots)}",
        f"#define TW_RANDOM_ROW_VALUES {len(rows)}",
        f"#define TW_RANDOM_LEAVES {len(leaves)}",
        f"#define TW_BOUNDARY_SPLITS {len(BOUNDARY_THRESHOLDS)}",
        f"#define TW_BOUNDARY_ROWS {len(b_leaves)}",
        "",
        "/* Offsets into the flat arrays below, and each table's counts. */",
        "typedef struct",
        "{",
        "    int32_t node_start;",
        "    int32_t node_count;",
        "    int32_t leaf_count;",
        "    int32_t tree_count;",
        "    int32_t feature_count;",
        "    int32_t root_start;",
        "    int32_t row_start;",
        "    int32_t leaf_start;",
        "} tw_random_record;",
        "",
        f"static const tw_random_record tw_random_records[{len(records)}] = {{",
    ]
    lines += ["    {" + ", ".join(plain(v) for v in record) + "}," for record in records]
    lines.append("};")
    lines += array("uint16_t", "tw_random_feature", features, u16, 12)
    lines += array("uint32_t", "tw_random_threshold", thresholds, u32)
    lines += array("int16_t", "tw_random_left", lefts, plain, 16)
    lines += array("int16_t", "tw_random_right", rights, plain, 16)
    lines += array("uint32_t", "tw_random_roots", roots, plain, 16)
    lines += array("uint32_t", "tw_random_rows", rows, u32)
    lines += array("int32_t", "tw_random_expected", leaves, plain, 16)
    lines.append("")
    lines.append("/* The 7-split table of the boundary rows; kinds: 0 x == t, 1 next above, 2 next below. */")
    lines += array("uint16_t", "tw_boundary_feature", b_features.tolist(), u16, 12)
    lines += array("uint32_t", "tw_boundary_threshold", b_thresholds.tolist(), u32)
    lines += array("int16_t", "tw_boundary_left", b_left.tolist(), plain, 16)
    lines += array("int16_t", "tw_boundary_right", b_right.tolist(), plain, 16)
    lines += array("uint32_t", "tw_boundary_row_bits", b_rows, u32)
    lines += array("int32_t", "tw_boundary_expected", b_leaves, plain, 16)
    lines += array("int32_t", "tw_boundary_split", b_splits, plain, 16)
    lines += array("int32_t", "tw_boundary_kind", b_kinds, plain, 16)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the checked-in header differs")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(f"stale: {OUTPUT}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
