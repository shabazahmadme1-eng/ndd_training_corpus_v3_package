"""Regressions for the changes aimed at function prediction rather than folding.

Each test pins one defect found in the first multi-gene PTEN run: pooled target scales,
patience spent before the final curriculum stage, batches dominated by stability genes,
mechanisms that cannot be told apart, and a trained model that loses the zero-shot signal.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mipo.common import save_json
from mipo.corpus import read_corpus
from mipo.data import BalancedAssayBatches, Metadata, TargetTransform, VariantDataset, collate
from mipo.losses import contrast_term, loss_function
from mipo.model import MIPO
from mipo.smoke import fixture
from mipo.splits import make_splits


def two_task_corpus(tmp_path):
    """Fixture corpus plus, for every gene, activity and abundance over the same substitutions.

    The fixture's own assay is collapsed consensus, which contrast pairing excludes, so the
    pair that the sampler is meant to find is the added activity/abundance one.
    """
    paths = fixture(tmp_path)
    rows = pd.read_csv(paths[0])
    extra = []
    for name, kind in [("synthetic_activity", "activity"), ("synthetic_abundance", "abundance")]:
        part = rows.copy()
        part["assay_id"], part["assay_type"], part["score_kind"] = name, kind, "assay_score"
        # Same ranks at most positions, reversed at position 3: a mechanism-specific site.
        flip = (part.ref_pos == 3) & (kind == "activity")
        part["score_value"] = np.where(flip, 1-part.score_value, part.score_value)+.01
        extra.append(part)
    corpus = tmp_path/"two_task.csv"
    pd.concat([rows]+extra, ignore_index=True).to_csv(corpus, index=False)
    splits = tmp_path/"two_task_splits.json"
    make_splits(corpus, splits)
    return corpus, paths[1], paths[2], splits, paths[4]


def dataset(corpus, resources, cache, config, **kwargs):
    from mipo.data import FeatureStore
    d = read_corpus(corpus)
    store = FeatureStore(resources, cache, allow_synthetic=True)
    transform = TargetTransform().fit(d)
    return d, VariantDataset(d, store, transform, Metadata().fit(d), config, **kwargs)


def test_each_assay_is_scaled_on_its_own_spread_not_the_pooled_task():
    """A pooled task scale turned a millions-scale assay into 1e8 standard deviations."""
    rows = pd.DataFrame({"gene": ["A"]*40+["B"]*40, "task": ["binding"]*80,
                         "assay_key": ["A::small"]*40+["B::huge"]*40,
                         "score_value": list(np.linspace(0, 1, 40))+list(np.linspace(0, 1e8, 40))})
    transform = TargetTransform().fit(rows)
    z = transform.transform(rows.score_value, rows.task, rows.assay_key)
    assert np.abs(z).max() < 5
    # An assay the training data never saw falls back to the task's median scale.
    unseen = transform.transform([0., 1.], ["binding", "binding"], ["C::held_out"]*2)
    assert np.isfinite(unseen).all()
    assert transform.lookup("binding", "C::held_out") == transform.state["tasks"]["binding"]
    assert transform.lookup("binding", "B::huge")["scale"] > 1e6


def test_transform_round_trips_and_clips_training_targets_only(tmp_path):
    paths = fixture(tmp_path)
    config = json.loads(paths[4].read_text())
    d, train_ds = dataset(paths[0], paths[1], paths[2], config, clip_targets=.5)
    _, eval_ds = dataset(paths[0], paths[1], paths[2], config)
    assert np.abs(train_ds.targets).max() == pytest.approx(.5)
    assert np.abs(eval_ds.targets).max() > .5
    assert train_ds.clipped > 0
    restored = train_ds.transform.inverse(eval_ds.targets, d.task, d.assay_key)
    assert np.allclose(restored, d.score_value.to_numpy(), atol=1e-5)


def test_task_balanced_sampling_lifts_rare_tasks_off_the_floor(tmp_path):
    """One gene holds activity and twenty hold stability, as the real corpus does."""
    rows = pd.concat([pd.DataFrame({"gene": f"S{i}", "assay_key": f"S{i}::a", "task": "stability",
                                    "variant_key": [f"S{i}:V{j}" for j in range(20)]}) for i in range(20)]
                     + [pd.DataFrame({"gene": "F", "assay_key": "F::a", "task": "activity",
                                      "variant_key": [f"F:V{j}" for j in range(20)]})], ignore_index=True)
    share = {}
    for balanced in [False, True]:
        sampler = BalancedAssayBatches(rows, 4, 400, 0, task_balanced=balanced)
        drawn = [rows.task.iloc[batch].iloc[0] for batch in sampler]
        share[balanced] = drawn.count("activity")/len(drawn)
    assert share[False] < .1
    assert share[True] > .4


def test_contrast_batches_pair_the_same_variant_across_two_assays(tmp_path):
    corpus, resources, cache, _, config_path = two_task_corpus(tmp_path)
    config = json.loads(config_path.read_text())
    _, ds = dataset(corpus, resources, cache, config)
    sampler = BalancedAssayBatches(ds.rows, 8, 40, 0, contrast_fraction=1.)
    assert sampler.pairs
    for batch in sampler:
        rows = ds.rows.iloc[batch]
        assert rows.task.nunique() == 2
        assert rows.groupby("variant_key").assay_key.nunique().eq(2).all()
        assert rows.gene.nunique() == 1


def test_contrast_loss_rewards_separating_two_assays_of_one_variant():
    def batch(mu):
        return {"variant_id": torch.tensor([0, 0, 1, 1]), "group": torch.tensor([0, 1, 0, 1]),
                "y": torch.tensor([0., 0., 0., 2.])}, torch.tensor(mu)
    b, separated = batch([0., 0., 0., 2.])  # Second variant's assays disagree, as its target does.
    _, merged = batch([0., 0., 1., 1.])     # One damage score for both assays.
    assert contrast_term(separated, b) < contrast_term(merged, b)
    # A batch whose variants are each measured once contributes nothing.
    alone = {"variant_id": torch.tensor([0, 1]), "group": torch.tensor([0, 0]), "y": torch.tensor([0., 1.])}
    assert contrast_term(torch.tensor([0., 1.]), alone) == 0


def test_contrast_term_enters_the_loss_only_when_weighted(tmp_path):
    corpus, resources, cache, _, config_path = two_task_corpus(tmp_path)
    config = json.loads(config_path.read_text())
    _, ds = dataset(corpus, resources, cache, config)
    sampler = BalancedAssayBatches(ds.rows, 8, 1, 0, contrast_fraction=1.)
    b = collate([ds[i] for i in next(iter(sampler))])
    model = MIPO(32, 1, config)
    output = model(b)
    off, parts_off = loss_function(output, b, contrast_weight=0.)
    on, parts_on = loss_function(output, b, contrast_weight=1.)
    assert parts_off["contrast"] == 0
    assert parts_on["contrast"] > 0
    assert on > off
    on.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_llr_residual_keeps_the_zero_shot_signal_in_the_prediction(tmp_path):
    paths = fixture(tmp_path)
    config = {**json.loads(paths[4].read_text()), "use_llr": True, "llr_residual": True}
    _, ds = dataset(paths[0], paths[1], paths[2], config)
    b = collate([ds[i] for i in range(4)])
    model = MIPO(32, 1, config).eval()
    with torch.no_grad():
        before = model(b)["mu"]
        model.llr_weight.fill_(2.)
        after = model(b)["mu"]
    expected = before+2*b["llr"][:, 0]*b["llr"][:, 1]
    assert torch.allclose(after, expected, atol=1e-5)
    # A missing cached score contributes nothing, whatever the coefficient.
    absent = dict(b)
    absent["llr"] = b["llr"].clone()
    absent["llr"][:, 1] = 0.
    with torch.no_grad():
        model.llr_weight.zero_()
        baseline = model(absent)["mu"]
        model.llr_weight.fill_(7.)
        model.llr_bias.fill_(3.)
        assert torch.allclose(model(absent)["mu"], baseline, atol=1e-6)
    # The residual anchors on the cached LLR whether or not LLR is also a fusion input.
    assert MIPO(32, 1, {**config, "use_llr": False, "llr_residual": True}).llr_residual is True


def test_curriculum_patience_starts_at_the_final_stage(tmp_path, monkeypatch):
    """Earlier stages used to exhaust patience, so the final stage ran one epoch."""
    import importlib
    trainer = importlib.import_module("mipo.train")
    paths = fixture(tmp_path)
    config = json.loads(paths[4].read_text())
    config.update(patience=1, steps_per_epoch=2, stages=[{"name": "early", "epochs": 3, "tiers": ["*"]},
                                                         {"name": "final", "epochs": 3, "tiers": ["*"]}])
    save_json(paths[4], config)
    original = trainer.metrics
    scores = [.9, .8, .7, .1, .2, .3]  # Falling in the early stage, rising in the final one.

    def scripted(data):
        assays, summary = original(data)
        if scores:  # Calls beyond the epoch loop are the calibration/test evaluation.
            summary["macro_gene_spearman"] = scores.pop(0)
        return assays, summary
    monkeypatch.setattr(trainer, "metrics", scripted)
    run = tmp_path/"run"
    trainer.train(*paths[:3], paths[3], paths[4], out=run, allow_synthetic=True)
    history = json.loads((run/"history.json").read_text())
    status = json.loads((run/"training_status.json").read_text())
    assert [h["selecting"] for h in history] == [False]*3+[True]*3
    assert status["epochs_run"] == 6
    assert status["best_epoch"] == 5
    assert status["best_validation_spearman"] == pytest.approx(.3)
