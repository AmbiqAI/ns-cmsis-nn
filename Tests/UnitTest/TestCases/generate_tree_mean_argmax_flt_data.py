# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Generate the arm_nn_tree_walk_mean_argmax_f32 golden set from scikit-learn fits.

scikit-learn is needed here, at generation time only; the suite reads the checked-in header. Each fit is one of the
five tree classifier classes (DecisionTreeClassifier, ExtraTreeClassifier, RandomForestClassifier,
ExtraTreesClassifier, AdaBoostClassifier) on a bundled or seeded synthetic dataset, from one tree to 100. For each fit
the script writes the packed table the entry reads, its leaf rows, float32 rows and, beside each row, the index of
the estimator's own predict in classes_.

The table is the estimator's trees as they are: thresholds rounded toward -inf to float32, because scikit-learn
compares a float32 feature with a float64 threshold and x <= t holds for a float32 x exactly when x <= the largest
float32 not above t; a NaN goes the way missing_go_to_left sends it; a tree whose root is a leaf gets one split with
that leaf on both sides. The script walks its packed table and asserts that every tree of every row reaches the leaf
apply() names, so a walk defect cannot pass as a tie. Leaf rows: a tree's or a forest's class fractions (normalised
in float64, rounded once); an AdaBoost model's weight times the one-hot of the class its tree predicts, at the one
power of two that puts the largest weight in [0.5, 1).

The script mirrors the entry's arithmetic in numpy float32 (tree 0's row, then one add per later tree in tree order,
one multiply by the float32 reciprocal of the tree count, the first maximum with NaN never winning) and lists every
row where that class differs from predict, with its cause: the float64 margin between the two classes against the
float32 scores' errors from the float64 ones. --check regenerates in memory and exits 1 if the checked-in header
differs (the same scikit-learn and numpy releases are needed for that; the header names them).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, make_classification
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

OUTPUT = Path(__file__).parent / "test_arm_nn_tree_walk_mean_argmax_f32" / "tree_mean_argmax_f32_data.h"
MISSING_LEFT = 0x8000
SEED = 20261111


def dataset(name):
    """float32 X and integer y of one dataset; the synthetic ones are seeded."""
    if name == "iris":
        x, y = load_iris(return_X_y=True)
    elif name == "wine":
        x, y = load_wine(return_X_y=True)
    elif name == "cancer":
        x, y = load_breast_cancer(return_X_y=True)
    elif name == "digits":
        x, y = load_digits(return_X_y=True)
    elif name in ("blobs5", "blobs5-nan"):
        x, y = make_classification(n_samples=600, n_features=6, n_informative=4, n_redundant=1, n_classes=5,
                                   n_clusters_per_class=1, class_sep=0.8, random_state=SEED)
        if name == "blobs5-nan":
            mask = np.random.default_rng(SEED).random(x.shape) < 0.06
            x = np.where(mask, np.nan, x)
    elif name == "ties":
        # Points on a coarse grid with labels drawn at random: the leaves stay impure at every depth, so a forest's
        # class fractions meet at exact rational ties (1/3 and 1/3, 2/5 and 2/5) that float32 cannot hold.
        rng = np.random.default_rng(SEED + 1)
        x = rng.integers(0, 4, (400, 3)).astype(np.float64)
        y = rng.integers(0, 3, 400)
    else:
        raise ValueError(name)
    return np.asarray(x, dtype=np.float32), np.asarray(y)


# (name, data, estimator, rows kept). Every estimator is seeded and single-threaded.
FITS = (
    ("dt_iris", "iris", lambda: DecisionTreeClassifier(random_state=0), 120),
    ("xt_wine", "wine", lambda: ExtraTreeClassifier(random_state=0), 80),
    ("dt_blobs5_nan", "blobs5-nan", lambda: DecisionTreeClassifier(max_depth=8, random_state=0), 120),
    ("rf1_cancer", "cancer", lambda: RandomForestClassifier(n_estimators=1, random_state=0), 40),
    ("rf5_wine", "wine", lambda: RandomForestClassifier(n_estimators=5, random_state=0), 80),
    ("rf16_blobs5_nan", "blobs5-nan", lambda: RandomForestClassifier(n_estimators=16, max_depth=4, random_state=0), 120),
    ("rf16_digits", "digits", lambda: RandomForestClassifier(n_estimators=16, max_depth=4, random_state=0), 24),
    ("rf100_iris", "iris", lambda: RandomForestClassifier(n_estimators=100, random_state=0), 120),
    ("rf100_cancer", "cancer", lambda: RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0), 40),
    ("rf3_ties", "ties", lambda: RandomForestClassifier(n_estimators=3, max_depth=4, random_state=0), 150),
    ("rf7_ties", "ties", lambda: RandomForestClassifier(n_estimators=7, max_depth=3, random_state=0), 150),
    ("et10_iris", "iris", lambda: ExtraTreesClassifier(n_estimators=10, random_state=0), 120),
    ("et100_wine", "wine", lambda: ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=0), 60),
    ("et16_blobs5", "blobs5", lambda: ExtraTreesClassifier(n_estimators=16, max_depth=4, random_state=0), 120),
    ("ada50_iris", "iris", lambda: AdaBoostClassifier(n_estimators=50, random_state=0), 120),
    ("ada100_cancer", "cancer", lambda: AdaBoostClassifier(n_estimators=100, random_state=0), 40),
    ("ada16_wine", "wine", lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=16,
                                                      random_state=0), 80),
)


def threshold_f32(t):
    """The largest float32 not above the float64 threshold t."""
    low = np.float32(t)
    if np.float64(low) > t:
        low = np.nextafter(low, np.float32(-np.inf))
    return low


class Packed:
    """The trees of one fit as the entry's record: split columns, roots and, per tree, sklearn node -> code."""

    def __init__(self):
        self.feature, self.threshold, self.left, self.right, self.roots = [], [], [], [], []
        self.leaf_count = 0
        self.leaf_of = []  # per tree: {sklearn leaf node: packed leaf index}

    def add(self, tree):
        t = tree.tree_
        splits = {}
        leaves = {}
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                leaves[node] = self.leaf_count
                self.leaf_count += 1
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
        self.leaf_of.append(leaves)

    def walk(self, row):
        """The packed leaf each tree reaches, by the entry's rules read from the packed columns."""
        out = []
        for root in self.roots:
            node = root
            while True:
                f = self.feature[node] & 0x7FFF
                x = row[f]
                go_left = bool(self.feature[node] & MISSING_LEFT) if np.isnan(x) else bool(x <= self.threshold[node])
                child = self.left[node] if go_left else self.right[node]
                if child < 0:
                    out.append(-child - 1)
                    break
                node = child
        return out


def fit_one(name, data, make, keep):
    x, y = dataset(data)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    estimator = make().fit(x_train, y_train)
    rows = np.concatenate([x_test, x_train])[:keep]
    classes = estimator.classes_
    count = len(classes)
    trees = list(estimator.estimators_) if hasattr(estimator, "estimators_") else [estimator]
    packed = Packed()
    for tree in trees:
        packed.add(tree)
    tree_count = len(trees)

    # Leaf rows, float64 then rounded once.
    rows64 = np.zeros((packed.leaf_count, count))
    if isinstance(estimator, AdaBoostClassifier):
        weights = np.asarray(estimator.estimator_weights_[:tree_count], dtype=np.float64)
        scale = np.ldexp(1.0, -int(np.frexp(weights.max())[1]))
        assert 0.5 <= weights.max() * scale < 1.0
        for t, tree in enumerate(trees):
            for node, leaf in packed.leaf_of[t].items():
                rows64[leaf, int(np.argmax(tree.tree_.value[node, 0]))] = weights[t] * scale
    else:
        for t, tree in enumerate(trees):
            for node, leaf in packed.leaf_of[t].items():
                value = np.asarray(tree.tree_.value[node, 0], dtype=np.float64)
                rows64[leaf] = value / value.sum()
    rows32 = rows64.astype(np.float32)

    # Each tree's apply() against the packed walk, and the float64 scores predict decides on.
    applied = np.stack([tree.apply(rows) for tree in trees], axis=1)
    predict = np.searchsorted(classes, estimator.predict(rows))
    assert np.all(classes[predict] == estimator.predict(rows))
    scores64 = np.zeros((len(rows), count))
    mirror = np.zeros(len(rows), dtype=np.int64)
    scores32 = np.zeros((len(rows), count), dtype=np.float32)
    reciprocal = np.float32(np.float32(1.0) / np.float32(tree_count))
    for r, row in enumerate(rows):
        leaves = packed.walk(row)
        assert leaves == [packed.leaf_of[t][int(applied[r, t])] for t in range(tree_count)], (name, r)
        scores64[r] = rows64[leaves].sum(axis=0) / tree_count
        sums = rows32[leaves[0]].copy()
        for leaf in leaves[1:]:
            sums = (sums + rows32[leaf]).astype(np.float32)
        scores32[r] = (sums * reciprocal).astype(np.float32)
        mirror[r] = first_maximum(scores32[r])
    if isinstance(estimator, AdaBoostClassifier):
        assert np.array_equal(np.argmax(scores64, axis=1), predict), name
    else:
        assert np.allclose(scores64, estimator.predict_proba(rows), rtol=0, atol=1e-12), name

    ordered = np.sort(scores64, axis=1)
    closest = float(np.min(ordered[:, -1] - ordered[:, -2]))
    differing = []
    for r in np.flatnonzero(mirror != predict):
        p, k = int(predict[r]), int(mirror[r])
        margin = scores64[r, p] - scores64[r, k]
        errors = abs(np.float64(scores32[r, p]) - scores64[r, p]) + abs(np.float64(scores32[r, k]) - scores64[r, k])
        assert 0.0 <= margin <= errors, (name, r, margin, errors)
        differing.append((int(r), p, k, margin, errors, "exact tie" if margin == 0.0 else "near tie"))
    return {
        "name": name, "data": data, "estimator": type(estimator).__name__, "trees": tree_count, "classes": count,
        "features": x.shape[1], "packed": packed, "leaf_rows": rows32, "rows": rows, "predict": predict,
        "mirror": mirror, "differing": differing, "closest": closest,
    }


def first_maximum(values):
    best, index = None, 0
    for c, v in enumerate(values):
        if not np.isnan(v) and (best is None or v > best):
            best, index = v, c
    return index


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
    feature, threshold, left, right, roots, leaf_rows, rows, predict, mirror = [], [], [], [], [], [], [], [], []
    records, differing, table = [], [], []
    for fit in fits:
        packed = fit["packed"]
        records.append((len(feature), len(packed.feature), packed.leaf_count, fit["trees"], fit["features"],
                        len(roots), len(leaf_rows), fit["classes"], len(predict), len(rows), len(fit["rows"]),
                        len(differing),
                        len(fit["differing"])))
        feature += packed.feature
        threshold += bits(packed.threshold).tolist()
        left += packed.left
        right += packed.right
        roots += packed.roots
        leaf_rows += bits(fit["leaf_rows"].reshape(-1)).tolist()
        rows += bits(fit["rows"].reshape(-1)).tolist()
        predict += fit["predict"].tolist()
        mirror += fit["mirror"].tolist()
        differing += [d[:3] for d in fit["differing"]]
        n = len(fit["rows"])
        table.append(f" * {fit['name']:<16} {fit['estimator']:<23} T={fit['trees']:<3} C={fit['classes']:<2} "
                     f"rows={n:<3} agree={n - len(fit['differing']):<3} differ={len(fit['differing'])} "
                     f"closest={fit['closest']:.2e}")
        for r, p, k, margin, errors, cause in fit["differing"]:
            table.append(f" *     row {r}: predict {p}, entry {k}, {cause}: float64 margin {margin:.3e} <= "
                         f"float32 score errors {errors:.3e}")
    assert len(feature) < 32768 * 8
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: 2026 Ambiq",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * Generated by ../generate_tree_mean_argmax_flt_data.py; do not edit. The golden set of",
        f" * arm_nn_tree_walk_mean_argmax_f32: scikit-learn {sklearn.__version__}, numpy {np.__version__}, seed {SEED}.",
        " * Per fit: the packed table, its leaf rows, float32 rows, the index in classes_ of the estimator's own",
        " * predict, and the class of the script's float32 mirror of the entry. Floats are written as their bits.",
        " *",
        " * Fit             estimator               trees, classes, rows, rows agreeing and differing with predict,",
        " * the smallest float64 margin between a row's top two classes; each differing row with its cause:",
    ] + table + [
        " */",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define TG_FITS {len(fits)}",
        f"#define TG_DIFFERING {len(differing)}",
        f"#define TG_MOST_NODES {max(len(f['packed'].feature) for f in fits)}",
        f"#define TG_MOST_LEAF_VALUES {max(f['packed'].leaf_count * f['classes'] for f in fits)}",
        f"#define TG_MOST_TREES {max(f['trees'] for f in fits)}",
        f"#define TG_MOST_FEATURES {max(f['features'] for f in fits)}",
        f"#define TG_MOST_CLASSES {max(f['classes'] for f in fits)}",
        "",
        "/* Offsets into the flat arrays below and each fit's counts: row_start indexes tg_predict and tg_mirror,",
        " * row_value_start tg_rows, and differ_start and differ_count tg_differing. */",
        "typedef struct",
        "{",
        "    const char *name;",
        "    int32_t node_start;",
        "    int32_t node_count;",
        "    int32_t leaf_count;",
        "    int32_t tree_count;",
        "    int32_t feature_count;",
        "    int32_t root_start;",
        "    int32_t leaf_row_start;",
        "    int32_t class_count;",
        "    int32_t row_start;",
        "    int32_t row_value_start;",
        "    int32_t row_count;",
        "    int32_t differ_start;",
        "    int32_t differ_count;",
        "} tg_fit;",
        "",
        "/* A row where the mirror's class differs from predict: the row in its fit, predict's class, the entry's. */",
        "typedef struct",
        "{",
        "    int32_t row;",
        "    int32_t predict;",
        "    int32_t entry;",
        "} tg_differ;",
        "",
        f"static const tg_fit tg_fits[{len(fits)}] = {{",
    ]
    for fit, record in zip(fits, records):
        lines.append(f"    {{\"{fit['name']}\", " + ", ".join(str(int(v)) for v in record) + "},")
    lines.append("};")
    if differing:
        lines.append(f"static const tg_differ tg_differing[{len(differing)}] = {{")
        lines += [f"    {{{r}, {p}, {k}}}," for r, p, k in differing]
        lines.append("};")
    else:
        lines.append("static const tg_differ tg_differing[1] = {{0, 0, 0}};")
    lines += array("uint16_t", "tg_feature", feature, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tg_threshold", threshold, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("int16_t", "tg_left", left, lambda v: str(int(v)), 16)
    lines += array("int16_t", "tg_right", right, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tg_roots", roots, lambda v: str(int(v)), 16)
    lines += array("uint32_t", "tg_leaf_rows", leaf_rows, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("uint32_t", "tg_rows", rows, lambda v: f"0x{int(v):08X}u", 8)
    lines += array("uint8_t", "tg_predict", predict, lambda v: str(int(v)), 32)
    lines += array("uint8_t", "tg_mirror", mirror, lambda v: str(int(v)), 32)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
