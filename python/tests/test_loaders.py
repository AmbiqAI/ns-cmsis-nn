# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""load_signatures and load_implementations on the checkout, and the corruptions each must refuse."""

import hashlib
import json

import pytest

from conftest import ROOT
from ns_cmsis_nn_contracts import (ContractError, ContractSchemaError, load_implementations, load_signatures)
from ns_cmsis_nn_contracts import implementations as impl_module
from ns_cmsis_nn_contracts import signatures as sig_module

SIGS = sig_module.RELPATH
IMPLS = impl_module.RELPATH


def edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document, indent=2) + "\n")


def reseal(root):
    """Point the manifest at the (edited) signature export so only the intended corruption is left."""
    digest = hashlib.sha256((root / SIGS).read_bytes()).hexdigest()
    edit(root / IMPLS, lambda d: d.__setitem__("signature_sha256", digest))


def test_checkout_loads():
    signatures = load_signatures(ROOT)
    assert signatures.present and len(signatures.functions) > 500
    assert signatures.sha256 == hashlib.sha256((ROOT / SIGS).read_bytes()).hexdigest()
    decl = signatures.require("arm_vector_sum_s8")
    assert [p.name for p in decl.params][:3] == ["vector_sum_buf", "vector_cols", "vector_rows"]
    impls = load_implementations(ROOT, signatures)
    assert impls.planner_abi == 1
    assert impls.by_route("arm_fully_connected_wrapper_s8", "fc_wrapper_s8:per_channel").id == \
        "fully_connected_per_channel_s8"
    assert impls.by_direct_entry("arm_max_pool_s8").id == "max_pool_s8"
    packed = impls.require("fully_connected_per_channel_packed_s8")
    assert [s.symbol for s in packed.prepare] == ["arm_vector_sum_s8",
                                                  "arm_fully_connected_per_channel_packed_s8_pack"]
    assert packed.buffer("packed_weights").format == "fc_per_channel_packed_s8/1"
    assert impls.require("max_pool_s8").buffer("ctx").size_query is None


def test_lookups_fail_closed():
    signatures = load_signatures(ROOT)
    impls = load_implementations(ROOT, signatures)
    with pytest.raises(ContractError, match="not in the signature export"):
        signatures.require("arm_no_such_kernel")
    with pytest.raises(ContractError, match="no such implementation"):
        impls.require("no_such_impl")
    with pytest.raises(ContractError, match="no implementation"):
        impls.by_route("arm_fully_connected_wrapper_s8", "fc_wrapper_s8:nope")
    with pytest.raises(ContractError, match="0 direct implementations"):
        impls.by_direct_entry("arm_fully_connected_wrapper_s8")
    with pytest.raises(KeyError):
        impls.require("max_pool_s8").buffer("nope")


def test_absent_export(tmp_path):
    assert not load_signatures(tmp_path).present
    assert not load_signatures(None).present
    with pytest.raises(ContractError, match="none"):
        load_implementations(tmp_path, load_signatures(tmp_path))


@pytest.mark.parametrize("change, error, match", [
    (lambda d: d.__setitem__("schema", "ns-cmsis-nn/kernel-contracts/2"), ContractSchemaError, "schema"),
    (lambda d: d.__setitem__("extra", 1), ContractError, "exactly"),
    (lambda d: d.__setitem__("functions", []), ContractError, "no functions"),
    (lambda d: d["functions"].append(dict(d["functions"][0])), ContractError, "appears twice"),
    (lambda d: d["functions"][0].pop("guards"), ContractError, "keys"),
    (lambda d: d["functions"][0].__setitem__("line", 0), ContractError, "positive integer"),
    (lambda d: d["functions"][0].__setitem__("header", "Include/arm_nn_types.h"), ContractError, "public"),
    (lambda d: d["functions"][0].__setitem__("returns", "struct foo"), ContractError, "vocabulary"),
    (lambda d: d["functions"][0]["params"][0].__setitem__("direction", "inout"), ContractError, "direction"),
    (lambda d: d["functions"][0]["params"][0].__setitem__("extent", "3"), ContractError, "extent"),
    (lambda d: d["functions"][0]["params"].append(dict(d["functions"][0]["params"][0])), ContractError, "duplicate"),
    (lambda d: d["functions"][0].__setitem__("name", "arm_not_declared_anywhere"), ContractError, "stale"),
])
def test_signature_corruption(tree, change, error, match):
    edit(tree / SIGS, change)
    with pytest.raises(error, match=match):
        load_signatures(tree)


def test_signature_unparseable(tree):
    (tree / SIGS).write_text("{")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_signatures(tree)


def test_signature_header_missing(tree):
    header = json.loads((tree / SIGS).read_text())["functions"][0]["header"]
    (tree / header).unlink()
    with pytest.raises(ContractError, match="cannot be read"):
        load_signatures(tree)


def test_declaration_inside_a_comment_does_not_count(tree):
    header = tree / "Include" / "arm_nnfunctions.h"
    text = header.read_text()
    declaration = "arm_cmsis_nn_status arm_vector_sum_s8("
    assert text.count(declaration) == 1 and text.count("arm_vector_sum_s8(") > 1
    header.write_text(text.replace(declaration, "arm_cmsis_nn_status arm_vector_sum_s8_renamed("))
    with pytest.raises(ContractError, match="arm_vector_sum_s8"):
        load_signatures(tree)


def test_manifest_stale_against_signatures(tree):
    edit(tree / SIGS, lambda d: d["functions"].sort(key=lambda f: f["name"], reverse=True))
    with pytest.raises(ContractError, match="stale"):
        load_implementations(tree, load_signatures(tree))


def first(d, impl_id="fully_connected_per_channel_packed_s8"):
    return next(i for i in d["implementations"] if i["id"] == impl_id)


@pytest.mark.parametrize("change, error, match", [
    (lambda d: d.__setitem__("schema", "ns-cmsis-nn/kernel-implementations/2"), ContractSchemaError, "schema"),
    (lambda d: d.pop("planner_abi"), ContractError, "keys"),
    (lambda d: d.__setitem__("planner_abi", 0), ContractError, "planner_abi"),
    (lambda d: d.__setitem__("planner_abi", True), ContractError, "planner_abi"),
    (lambda d: d.__setitem__("library_version", "7.47"), ContractError, "library_version"),
    (lambda d: d.__setitem__("implementations", []), ContractError, "no implementations"),
    (lambda d: d["implementations"].append(dict(d["implementations"][0])), ContractError, "appears twice"),
    (lambda d: first(d).pop("since"), ContractError, "keys"),
    (lambda d: first(d).__setitem__("id", "Bad-Id"), ContractError, "lower_snake_case"),
    (lambda d: first(d).__setitem__("op", "teleport"), ContractError, "op"),
    (lambda d: first(d).__setitem__("entry", "arm_no_such_kernel"), ContractError, "not a public function"),
    (lambda d: first(d).__setitem__("requires", ["mvei", "quantum"]), ContractError, "unknown feature"),
    (lambda d: first(d).__setitem__("requires", "mvei"), ContractError, "list of feature"),
    (lambda d: first(d)["buffers"][1].__setitem__("param", "no_such_param"), ContractError, "no parameter"),
    (lambda d: first(d)["buffers"][1].__setitem__("lifetime", "forever"), ContractError, "lifetime"),
    (lambda d: first(d)["buffers"][1].__setitem__("align", 0), ContractError, "align"),
    (lambda d: first(d)["buffers"][1]["size_query"].__setitem__("symbol", "arm_nope"), ContractError,
     "not a public function"),
    (lambda d: first(d)["buffers"].append(dict(first(d)["buffers"][0])), ContractError, "duplicate buffer"),
    (lambda d: first(d)["prepare"][0].__setitem__("fills", "nowhere"), ContractError, "fills"),
    (lambda d: first(d)["prepare"][0].__setitem__("symbol", "arm_nope"), ContractError, "not a public function"),
    (lambda d: first(d)["numerics"].__setitem__("claim", "close_enough"), ContractError, "claim"),
    (lambda d: first(d).__setitem__("since", "next"), ContractError, "since"),
    (lambda d: first(d, "fully_connected_s8")["routed_by"].__setitem__("route", "per_tensor"), ContractError,
     "'<wrapper>:<route>'"),
    (lambda d: first(d, "fully_connected_s8")["routed_by"].__setitem__("route", "fc_wrapper_s8:per_channel"),
     ContractError, "same wrapper route"),
    (lambda d: first(d, "depthwise_conv_s8_opt_planar")["eligibility"].__setitem__("predicate", "arm_nope"),
     ContractError, "not a public function"),
])
def test_manifest_corruption(tree, change, error, match):
    edit(tree / IMPLS, change)
    with pytest.raises(error, match=match):
        load_implementations(tree, load_signatures(tree))


def test_manifest_unparseable(tree):
    (tree / IMPLS).write_text("[")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_implementations(tree, load_signatures(tree))


def test_manifest_symbol_removed_from_signatures(tree):
    edit(tree / SIGS, lambda d: d.__setitem__(
        "functions", [f for f in d["functions"] if f["name"] != "arm_fully_connected_per_channel_packed_s8_pack"]))
    reseal(tree)
    with pytest.raises(ContractError, match="arm_fully_connected_per_channel_packed_s8_pack"):
        load_implementations(tree, load_signatures(tree))
