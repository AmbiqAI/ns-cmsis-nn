# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Generate the arm_nn_tree_walk_values_f32 golden set from scikit-learn fits.

scikit-learn is needed here, at generation time only; the suite reads the checked-in header. Each fit is one of the
five tree regressor classes (DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
GradientBoostingRegressor) on a bundled or seeded synthetic dataset, from one tree to 100. For each fit the script
writes the packed table the entry reads, its leaf values, float32 rows and, for each row and tree, two answers taken
from the estimator alone: the word of float32(tree_.value[apply(row)]) and the packed leaf of the node apply() names.

The table is the estimator's trees as they are: thresholds rounded toward -inf to float32, because scikit-learn
compares a float32 feature with a float64 threshold and x <= t holds for a float32 x exactly when x <= the largest
float32 not above t; a NaN goes the way missing_go_to_left sends it; a tree whose root is a leaf gets one split with
that leaf on both sides. Leaf l's value is float32(tree_.value[node, 0, 0]) of the node packed as leaf l, rounded
once from float64 (a gradient-boosted tree's raw leaf value, before the learning rate). Rows are the held-out rows,
then training rows, then copies of the first rows with one feature in turn set to NaN, so every fit's walk meets a
NaN at its splits whether or not it was fitted with missing values.

The script walks its packed table by the entry's rules and compares each tree's leaf with apply(). A (row, tree)
pair where they differ is not dropped: it is listed in the header with its cause (the split where the two paths part,
the row's float32 feature, the float64 threshold and its float32 rounding) and the suite requires the entry to differ
from apply() at exactly the listed pairs. --check regenerates in memory and exits 1 if the checked-in header differs
(the same scikit-learn and numpy releases are needed for that; the header names them).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes, make_friedman1, make_regression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

OUTPUT = Path(__file__).parent / "test_arm_nn_tree_walk_values_f32" / "tree_walk_values_f32_data.h"
MISSING_LEFT = 0x8000
SEED = 20261111
NAN_ROWS = 1  # rows copied with one feature at a time set to NaN, per fit


def dataset(name):
    """float32 X and float64 y of one dataset; the synthetic ones are seeded."""
    if name == "diabetes":
        x, y = load_diabetes(return_X_y=True)
    elif name in ("friedman", "friedman-nan"):
        x, y = make_friedman1(n_samples=400, n_features=7, noise=0.5, random_state=SEED)
        if name == "friedman-nan":
            mask = np.random.default_rng(SEED).random(x.shape) < 0.06
            x = np.where(mask, np.nan, x)
    elif name == "linear":
        # Large targets of both signs, so leaf values span several binades and cross zero.
        x, y = make_regression(n_samples=400, n_features=5, n_informative=4, noise=3.0, bias=-20.0,
                               random_state=SEED)
        y = y * 1e3
    else:
        raise ValueError(name)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float64)


# (name, data, estimator, rows kept before the NaN rows). Every estimator is seeded and single-threaded.
FITS = (
    ("dt_diabetes", "diabetes", lambda: DecisionTreeRegressor(random_state=0), 80),
    ("dt_friedman_nan", "friedman-nan", lambda: DecisionTreeRegressor(max_depth=8, random_state=0), 80),
    ("xt_friedman", "friedman", lambda: ExtraTreeRegressor(random_state=0), 80),
    ("xt_linear", "linear", lambda: ExtraTreeRegressor(max_depth=10, random_state=0), 60),
    ("rf1_diabetes", "diabetes", lambda: RandomForestRegressor(n_estimators=1, random_state=0), 40),
    ("rf5_linear", "linear", lambda: RandomForestRegressor(n_estimators=5, max_depth=6, random_state=0), 60),
    ("rf16_friedman_nan", "friedman-nan",
     lambda: RandomForestRegressor(n_estimators=16, max_depth=4, random_state=0), 60),
    ("rf100_diabetes", "diabetes", lambda: RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0), 24),
    ("et10_friedman_nan", "friedman-nan", lambda: ExtraTreesRegressor(n_estimators=10, max_depth=5, random_state=0), 60),
    ("et100_linear", "linear", lambda: ExtraTreesRegressor(n_estimators=100, max_depth=3, random_state=0), 20),
    ("gb3_diabetes", "diabetes", lambda: GradientBoostingRegressor(n_estimators=3, random_state=0), 60),
    ("gb50_friedman", "friedman", lambda: GradientBoostingRegressor(n_estimators=50, random_state=0), 24),
    ("gb100_linear", "linear",
     lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.2, random_state=0), 20),
)


def threshold_f32(t):
    """The largest float32 not above the float64 threshold t."""
    low = np.float32(t)
    if np.float64(low) > t:
        low = np.nextafter(low, np.float32(-np.inf))
    return low


class Packed:
    """The trees of one fit as the entry's record: split columns, roots, leaf values and, per tree, the maps from
    scikit-learn nodes to packed splits and leaves."""

    def __init__(self):
        self.feature, self.threshold, self.left, self.right, self.roots, self.values = [], [], [], [], [], []
        self.split_of = []  # per tree: {sklearn split node: packed split index}
        self.leaf_of = []  # per tree: {sklearn leaf node: packed leaf index}

    def add(self, tree):
        t = tree.tree_
        splits, leaves = {}, {}
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                leaves[node] = len(self.values)
                self.values.append(np.float32(t.value[node, 0, 0]))
        base = len(self.feature)
        for node in range(t.node_count):
            if t.children_left[node] != -1:
                splits[node] = base + len(splits)

        def code(child):
            return splits[child] if child in splits else -(leaves[child] + 1)

        if not splits:
            self.roots.append(len(self.feature))
            self.feature.append(0)
            self.threshold.append(np.float32(0.0))
            self.left.append(code(0))
            self.right.append(code(0))
        else:
            self.roots.append(splits[0])
            for node in sorted(splits, key=splits.get):
                missing = int(t.missing_go_to_left[node]) == 1
                self.feature.append(int(t.feature[node]) | (MISSING_LEFT if missing else 0))
                self.threshold.append(threshold_f32(float(t.threshold[node])))
                self.left.append(code(int(t.children_left[node])))
                self.right.append(code(int(t.children_right[node])))
        self.split_of.append(splits)
        self.leaf_of.append(leaves)

    def walk(self, row, tree):
        """The packed leaf tree `tree` reaches and the packed splits on its way, by the entry's rules."""
        node, path = self.roots[tree], []
        while True:
            path.append(node)
            f = self.feature[node] & 0x7FFF
            x = row[f]
            go_left = bool(self.feature[node] & MISSING_LEFT) if np.isnan(x) else bool(x <= self.threshold[node])
            child = self.left[node] if go_left else self.right[node]
            if child < 0:
                return -child - 1, path
            node = child


def cause(packed, estimator_tree, t, row):
    """The first split of the estimator's decision path where the packed walk goes the other way, as text."""
    tree = estimator_tree.tree_
    path = list(estimator_tree.decision_path(row.reshape(1, -1)).indices)
    for node, following in zip(path, path[1:]):
        s = packed.split_of[t][node]
        f = packed.feature[s] & 0x7FFF
        x = row[f]
        ours = bool(packed.feature[s] & MISSING_LEFT) if np.isnan(x) else bool(x <= packed.threshold[s])
        if ours != (following == int(tree.children_left[node])):
            return (f"split {s}, feature {f}: x = {float(x)!r}, float64 threshold {float(tree.threshold[node])!r}, "
                    f"float32 {float(packed.threshold[s])!r}, missing_go_to_left {int(tree.missing_go_to_left[node])}")
    return "no split of the decision path parts the walks"


def fit_one(name, data, make, keep):
    x, y = dataset(data)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=SEED)
    estimator = make().fit(x_train, y_train)
    rows = np.concatenate([x_test, x_train])[:keep]
    nan_rows = []
    for r in range(NAN_ROWS):
        for f in range(x.shape[1]):
            copy = rows[r].copy()
            copy[f] = np.nan
            nan_rows.append(copy)
    rows = np.concatenate([rows, np.asarray(nan_rows, dtype=np.float32)])
    if isinstance(estimator, GradientBoostingRegressor):
        trees = list(estimator.estimators_[:, 0])
    elif hasattr(estimator, "estimators_"):
        trees = list(estimator.estimators_)
    else:
        trees = [estimator]
    packed = Packed()
    for tree in trees:
        packed.add(tree)
    tree_count = len(trees)

    # The estimator's answers: per row and tree, apply()'s node, its value as float32 bits, and its packed leaf.
    applied = np.stack([tree.apply(rows) for tree in trees], axis=1)
    expected = np.zeros((len(rows), tree_count), dtype=np.float32)
    apply_leaf = np.zeros((len(rows), tree_count), dtype=np.int64)
    for t, tree in enumerate(trees):
        expected[:, t] = tree.tree_.value[applied[:, t], 0, 0].astype(np.float32)
        apply_leaf[:, t] = [packed.leaf_of[t][int(n)] for n in applied[:, t]]
        assert np.array_equal(np.asarray(packed.values, dtype=np.float32)[apply_leaf[:, t]].view(np.uint32),
                              expected[:, t].view(np.uint32)), (name, t)

    differing = []
    for r, row in enumerate(rows):
        for t in range(tree_count):
            leaf, _ = packed.walk(row, t)
            if leaf != apply_leaf[r, t]:
                differing.append((r, t, leaf, int(apply_leaf[r, t]), cause(packed, trees[t], t, row)))
    distinct = len(set(np.asarray(packed.values, dtype=np.float32).view(np.uint32).tolist()))
    return {
        "name": name, "data": data, "estimator": type(estimator).__name__, "trees": tree_count,
        "features": x.shape[1], "packed": packed, "rows": rows, "expected": expected, "apply_leaf": apply_leaf,
        "differing": differing, "distinct": distinct,
        "depth": max(int(tree.get_depth()) for tree in trees),
    }


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def array(ctype, name, values, form, per_line):
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        lines.append("    " + ", ".join(form(v) for v in values[start:start + per_line]) + ",")
    lines.append("};")
    return lines


def render():
    fits = [fit_one(*fit) for fit in FITS]
    feature, threshold, left, right, roots, values, rows, expected, apply_leaf = [], [], [], [], [], [], [], [], []
    records, differing, table = [], [], []
    for fit in fits:
        packed = fit["packed"]
        n = len(fit["rows"])
        records.append((len(feature), len(packed.feature), len(packed.values), fit["trees"], fit["features"],
                        len(roots), len(values), len(rows), n, len(expected), len(differing), len(fit["differing"])))
        feature += packed.feature
        threshold += bits(packed.threshold).tolist()
        left += packed.left
        right += packed.right
        roots += packed.roots
        values += bits(packed.values).tolist()
        rows += bits(fit["rows"].reshape(-1)).tolist()
        expected += bits(fit["expected"].reshape(-1)).tolist()
        apply_leaf += fit["apply_leaf"].reshape(-1).tolist()
        differing += [d[:4] for d in fit["differing"]]
        rows_differing = len({d[0] for d in fit["differing"]})
        table.append(f" * {fit['name']:<18} {fit['estimator']:<25} T={fit['trees']:<3} depth={fit['depth']:<2} "
                     f"leaves={len(packed.values):<4} rows={n:<3} leaf differs: rows={rows_differing} "
                     f"pairs={len(fit['differing'])}")
        for r, t, leaf, applied, why in fit["differing"]:
            table.append(f" *     row {r} tree {t}: walk leaf {leaf}, apply leaf {applied}: {why}")
    total_pairs = sum(len(f["rows"]) * f["trees"] for f in fits)
    total_rows = sum(len(f["rows"]) for f in fits)
    table.append(f" * total: {len(fits)} fits, {total_rows} rows, {total_pairs} (row, tree) pairs, "
                 f"{len({(i, d[0]) for i, f in enumerate(fits) for d in f['differing']})} rows and "
                 f"{len(differing)} pairs whose leaf differs from apply()")
    assert len(feature) <= 32767 and len(values) <= 32768
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: 2026 Ambiq",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * Generated by ../generate_tree_walk_values_flt_data.py; do not edit. The golden set of",
        f" * arm_nn_tree_walk_values_f32: scikit-learn {sklearn.__version__}, numpy {np.__version__}, seed {SEED}.",
        " * Per fit: the packed table, its leaf values, float32 rows and, per row and tree, the word of",
        " * float32(tree_.value[apply(row)]) and the packed leaf of apply()'s node. Floats are written as their bits.",
        " *",
        " * Fit               estimator                 trees, deepest tree, leaves, rows, and the rows and (row, tree)",
        " * pairs where the script's walk of the packed table reaches another leaf than apply(), each with its cause:",
    ] + table + [
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define TVG_FITS {len(fits)}",
        f"#define TVG_DIFFERING {len(differing)}",
        f"#define TVG_MOST_NODES {max(len(f['packed'].feature) for f in fits)}",
        f"#define TVG_MOST_LEAVES {max(len(f['packed'].values) for f in fits)}",
        f"#define TVG_MOST_TREES {max(f['trees'] for f in fits)}",
        f"#define TVG_MOST_FEATURES {max(f['features'] for f in fits)}",
        "",
        "/* Offsets into the flat arrays below and each fit's counts: row_value_start indexes tvg_rows, expect_start",
        " * tvg_expected and tvg_apply_leaf (row-major, row r and tree t at r x tree_count + t), and differ_start and",
        " * differ_count tvg_differing. */",
        "typedef struct",
        "{",
        "    const char *name;",
        "    int32_t node_start;",
        "    int32_t node_count;",
        "    int32_t leaf_count;",
        "    int32_t tree_count;",
        "    int32_t feature_count;",
        "    int32_t root_start;",
        "    int32_t leaf_value_start;",
        "    int32_t row_value_start;",
        "    int32_t row_count;",
        "    int32_t expect_start;",
        "    int32_t differ_start;",
        "    int32_t differ_count;",
        "} tvg_fit;",
        "",
        "/* A (row, tree) pair where the script's walk differs from apply(): the row in its fit, the tree, the walk's",
        " * packed leaf, apply()'s. */",
        "typedef struct",
        "{",
        "    int32_t row;",
        "    int32_t tree;",
        "    int32_t walk_leaf;",
        "    int32_t apply_leaf;",
        "} tvg_differ;",
        "",
        f"static const tvg_fit tvg_fits[{len(fits)}] = {{",
    ]
    for fit, record in zip(fits, records):
        lines.append(f"    {{\"{fit['name']}\", " + ", ".join(str(int(v)) for v in record) + "},")
    lines.append("};")
    if differing:
        lines.append(f"static const tvg_differ tvg_differing[{len(differing)}] = {{")
        lines += [f"    {{{r}, {t}, {a}, {b}}}," for r, t, a, b in differing]
        lines.append("};")
    else:
        lines.append("static const tvg_differ tvg_differing[1] = {{0, 0, 0, 0}};")
    lines += array("uint16_t", "tvg_feature", feature, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tvg_threshold", threshold, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("int16_t", "tvg_left", left, lambda v: str(int(v)), 16)
    lines += array("int16_t", "tvg_right", right, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tvg_roots", roots, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tvg_leaf_values", values, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("uint32_t", "tvg_rows", rows, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("uint32_t", "tvg_expected", expected, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("uint16_t", "tvg_apply_leaf", apply_leaf, lambda v: str(int(v)), 16)
    return "\n".join(lines) + "\n", table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the checked-in header differs")
    args = parser.parse_args()
    text, table = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(f"stale: {OUTPUT}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text)
    print("\n".join(line[3:] for line in table))
    print(f"{len(text)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
