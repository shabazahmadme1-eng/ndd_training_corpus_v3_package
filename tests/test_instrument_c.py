"""Instrument C tests: the margin plumbing behind the >= 0.05 Tier 2 gate.

Uses the synthetic fixture and an untrained model, so no checkpoint is needed and
nothing here is a scientific number. What is guarded: the gate arithmetic
(margin = clean - scrambled), the negative-control classifier, seed handling, and
that a scramble which cannot act is reported rather than passed off as a null.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from mipo.data import FeatureStore, Metadata, TargetTransform
from mipo.instrument_c import GATE, predict_batches, run_instrument_c, structure_blind
from mipo.model import MIPO
from mipo.smoke import fixture
from mipo.corpus import read_corpus
from mipo.train import loader


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("ic")
    corpus, resources, cache, _, config_path = fixture(out)
    config = json.loads(Path(config_path).read_text())
    rows = read_corpus(corpus)
    store = FeatureStore(resources, cache, allow_synthetic=True)
    metadata = Metadata(None, {"<unknown>": 0})
    transform = TargetTransform().fit(rows)
    batches = loader(rows, store, transform, metadata, config)
    torch.manual_seed(0)
    model = MIPO(store.manifest["dim"], len(metadata.vocab), config)
    return model, batches, config


def test_structure_blind_truth_table():
    assert structure_blind({"model": "esm_mlp"})
    assert structure_blind({"model": "mipo", "structure": False})
    assert not structure_blind({"model": "mipo"})
    assert not structure_blind({"model": "mipo", "structure": True})


def test_clean_prediction_is_deterministic(setup):
    model, batches, _ = setup
    first, info = predict_batches(model, batches, "cpu")
    second, _ = predict_batches(model, batches, "cpu")
    assert info["attempted"] == 0, "no scramble was requested"
    pd.testing.assert_frame_equal(first, second)


def test_scramble_reports_rewire_counts(setup):
    model, batches, _ = setup
    _, chain = predict_batches(model, batches, "cpu", scramble=True, seed=0)
    _, full = predict_batches(model, batches, "cpu", scramble=False, seed=0)
    # Full mode rewires every masked edge; chain-preserved only the far ones.
    assert full["attempted"] == full["masked"] > 0
    assert chain["attempted"] <= full["attempted"]
    assert full["success_rate"] == pytest.approx(1.0)


def test_margin_is_clean_minus_scrambled_and_gate_is_explicit(setup):
    model, batches, _ = setup
    summary, per_seed, per_assay = run_instrument_c(model, batches, "cpu", seeds=(0, 1))
    assert summary["gate_threshold"] == GATE
    for mode in ["chain_preserved", "full", "edge_ablated"]:
        assert mode in summary
        rows = per_seed[per_seed["mode"] == mode]
        assert not rows.empty
        for row in rows.itertuples():
            if row.margin is not None and pd.notna(row.margin):
                assert row.margin == pytest.approx(row.clean_macro - row.scrambled_macro)
    assert summary["gate_passed"] == (summary["chain_preserved"]["margin_mean"] >= GATE)
    assert str(GATE) in summary["gate_statement"]
    assert set(per_assay["mode"]) == {"chain_preserved", "full", "edge_ablated"}
    # Two seeds plus the seed-independent ablation.
    assert len(per_seed) == 2 * 2 + 1


def test_edge_ablation_runs_once_and_is_seed_independent(setup):
    model, batches, _ = setup
    first, _, _ = run_instrument_c(model, batches, "cpu", seeds=(0,))
    second, _, _ = run_instrument_c(model, batches, "cpu", seeds=(7,))
    assert (first["edge_ablated"]["margin_mean"]
            == pytest.approx(second["edge_ablated"]["margin_mean"]))
    assert first["edge_ablated"]["undefined_seeds"] == 0


def test_a_model_that_ignores_geometry_yields_no_margin(setup):
    """An esm_mlp-shaped model never reads edges, so every margin must vanish.

    This is the instrument's own control: a nonzero margin here would mean the
    scramble is reaching the prediction through something other than geometry.
    """
    _, batches, config = setup
    store = batches.dataset.store
    torch.manual_seed(0)
    blind = MIPO(store.manifest["dim"], 1, {**config, "model": "esm_mlp", "structure": False})
    summary, _, _ = run_instrument_c(blind, batches, "cpu", seeds=(0, 1))
    assert structure_blind({**config, "model": "esm_mlp"})
    for mode in ["chain_preserved", "full", "edge_ablated"]:
        assert summary[mode]["margin_mean"] == pytest.approx(0.0, abs=1e-9)
    assert not summary["gate_passed"]
