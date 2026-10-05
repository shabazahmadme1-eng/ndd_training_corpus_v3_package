"""Focused tests for Tier 1 per-branch ranking loss: probes + branch_term + training wiring."""
import json

import torch

from mipo.common import save_json
from mipo.losses import branch_term, loss_function, pairwise_ranking
from mipo.model import MIPO
from mipo.smoke import fixture
from mipo.splits import expand_holdouts


def _batch(n=8):
    return {"y": torch.arange(n, dtype=torch.float32), "group": torch.zeros(n, dtype=torch.long),
            "row_id": torch.arange(n), "node_mask": torch.ones(n, 4, dtype=torch.bool),
            "aux_mask": torch.zeros(n, 4, 2, dtype=torch.bool),
            "aux_target": torch.zeros(n, 4, 2)}


def _output(n=8):
    return {"mu": torch.zeros(n), "sigma": torch.ones(n),
            "field": torch.zeros(n, 4, 8), "physical_proxies": torch.zeros(n, 4, 2),
            "branch_mu": torch.randn(n, 4)}


def test_probes_exist_only_when_branch_weight_on():
    plain = MIPO(1280, 4, {"model": "mipo"})
    assert plain.branch_probes is None
    assert not any("branch_probes" in k for k in plain.state_dict())
    branched = MIPO(1280, 4, {"model": "mipo", "branch_weight": 0.2})
    assert len(branched.branch_probes) == 4


def test_pairwise_ranking_prefers_correct_order():
    batch = _batch()
    w = torch.ones(8)
    good = pairwise_ranking(torch.arange(8, dtype=torch.float32), batch, w)
    bad = pairwise_ranking(torch.arange(8, dtype=torch.float32).flip(0), batch, w)
    assert good.item() < bad.item()


def test_branch_term_averages_branches_and_matches_joint_recipe():
    torch.manual_seed(0)
    batch = _batch()
    w = torch.ones(8)
    mus = torch.randn(8, 4)
    expected = sum(pairwise_ranking(mus[:, i], batch, w) for i in range(4)) / 4
    assert branch_term(mus, batch, w).item() == expected.item()


def test_branch_weight_gates_total_and_missing_probes_safe():
    batch = _batch()
    base, _ = loss_function(_output(), batch)
    total, parts = loss_function(_output(), batch, branch_weight=0.2)
    assert parts["branch"].item() > 0
    assert abs((total - base).item() - 0.2 * parts["branch"].item()) < 1e-6
    no_probes = _output()
    no_probes["branch_mu"] = None
    _, parts = loss_function(no_probes, batch, branch_weight=0.2)
    assert parts["branch"].item() == 0.0


def test_effective_loss_weights_consumes_every_knob():
    from mipo.train import effective_loss_weights
    assert effective_loss_weights({}) == (.2, 1e-5, .1, 0., 0., 0.)
    w = effective_loss_weights({"ranking_weight": .5, "field_weight": 1e-4, "auxiliary_weight": .3,
                                "contrast_weight": .2, "balance_weight": 1e-2, "branch_weight": .5})
    assert w == (.5, 1e-4, .3, .2, 1e-2, .5)


def test_branch_loss_flows_through_training(tmp_path):
    import mipo.train as trainer
    paths = fixture(tmp_path)
    expand_holdouts(paths[0], paths[3], tmp_path / "multi.json", 2, 1)
    config = json.loads(paths[-1].read_text())
    config.update(epochs=2, patience=5, min_validation_genes=2, branch_weight=0.2,
                  steps_per_epoch=4, batch_size=4)
    save_json(paths[-1], config)
    run = tmp_path / "run"
    trainer.train(*paths[:3], tmp_path / "multi.json", paths[-1], out=run, allow_synthetic=True)
    history = json.loads((run / "history.json").read_text())
    assert history[0]["loss_components"]["branch"] > 0
    assert all(v == v for v in [history[0]["loss_components"]["branch"]])  # finite
    state = torch.load(run / "best.pt", weights_only=True)
    assert any("branch_probes" in k for k in state["model"])
