"""Focused tests for fusion-revival knobs: Switch gate init + load-balancing loss."""
import math

import torch

from mipo.losses import loss_function
from mipo.model import MIPO


def _output(gates=None, n=8):
    out = {"mu": torch.zeros(n), "sigma": torch.ones(n),
           "field": torch.zeros(n, 4, 8), "physical_proxies": torch.zeros(n, 4, 2)}
    if gates is not None:
        out["gates"] = gates
    return out


def _batch(n=8):
    return {"y": torch.randn(n), "group": torch.zeros(n, dtype=torch.long),
            "row_id": torch.arange(n), "node_mask": torch.ones(n, 4, dtype=torch.bool),
            "aux_mask": torch.zeros(n, 4, 2, dtype=torch.bool),
            "aux_target": torch.zeros(n, 4, 2)}


def test_switch_gate_init():
    torch.manual_seed(0)
    default = MIPO(1280, 4, {"model": "mipo"})
    torch.manual_seed(0)
    switched = MIPO(1280, 4, {"model": "mipo", "fusion_gate_init": "switch"})
    w = switched.fusion_gate[-1].weight
    assert switched.fusion_gate[-1].bias.eq(0).all()
    assert w.std().item() == torch.std(w).item()  # sanity: finite
    assert abs(w.std().item() - math.sqrt(0.1 / w.shape[1])) < 0.3 * math.sqrt(0.1 / w.shape[1])
    assert not torch.equal(w, default.fusion_gate[-1].weight)
    assert default.fusion_gate[-1].bias.ne(0).any()  # default path untouched
    torch.manual_seed(0)
    mlp = MIPO(1280, 4, {"model": "esm_mlp", "fusion_gate_init": "switch"})
    assert mlp.fusion_gate[-1].bias.eq(0).all()


def test_balance_loss_uniformity():
    n = 8
    uniform = torch.full((n, 4), 0.25)
    _, parts = loss_function(_output(uniform), _batch(n), balance_weight=1.0)
    assert parts["balance"].item() == 1.0
    peaked = torch.zeros(n, 4)
    peaked[:, 2] = 1.0
    _, parts = loss_function(_output(peaked), _batch(n), balance_weight=1.0)
    assert parts["balance"].item() == 4.0


def test_balance_weight_gates_total_and_missing_gates_safe():
    n = 8
    gates = torch.softmax(torch.randn(n, 4), -1)
    batch = _batch(n)
    base, _ = loss_function(_output(gates), batch)
    total, parts = loss_function(_output(gates), batch, balance_weight=0.01)
    assert abs((total - base).item() - 0.01 * parts["balance"].item()) < 1e-6
    _, parts = loss_function(_output(None), _batch(n), balance_weight=0.01)
    assert parts["balance"].item() == 0.0
