"""Synthetic-batch unit tests for the eval-time graph scrambles.

Guards the confirmation gate's primary number: a silently invalid scramble would
feed a plausible-looking but meaningless margin into the >= 0.05 bar.
"""
import torch

from mipo.instruments import rewire_edges, scramble_batch
from mipo.model import MIPO


def _batch(B=2, N=24, K=6, C=2):
    g = torch.Generator().manual_seed(0)
    ids = torch.stack([torch.sort(torch.randperm(60, generator=g)[:N]).values for _ in range(B)])
    return {
        "ids": ids,
        "edge": torch.randint(0, N, (B, N, K), generator=g),
        "edge_mask": torch.rand(B, N, K, generator=g) > 0.2,
        "edge_features": torch.randn(B, N, K, 5, generator=g),
        "unit": torch.randn(B, C, N, K, 3, generator=g),
    }


def _tgt(ids, idx):
    return ids[:, :, None].expand_as(idx).gather(1, idx.clamp_min(0))


def test_rewire_shapes_unchanged_and_single_target_per_edge():
    b = _batch()
    new_edge, rw, info = rewire_edges(b["edge"], b["ids"], b["edge_mask"])
    assert new_edge.shape == b["edge"].shape == rw.shape == (2, 24, 6)
    assert new_edge.dtype == b["edge"].dtype
    assert bool(((new_edge >= 0) & (new_edge < 24)).all())  # exactly one valid target per edge


def test_rewire_keeps_close_edges_bit_identical():
    b = _batch()
    new_edge, rw, info = rewire_edges(b["edge"], b["ids"], b["edge_mask"], radius=8)
    assert info["succeeded"] == int(rw.sum()) and info["success_rate"] > 0.9
    tgt = _tgt(b["ids"], b["edge"])
    close = ((tgt - b["ids"][:, :, None]).abs() <= 8) & b["edge_mask"]
    assert bool((new_edge[close] == b["edge"][close]).all())
    assert not bool((rw & close).any())


def test_rewire_lands_far_from_own_source():
    torch.manual_seed(0)
    for trial in range(5):
        b = _batch()
        new_edge, rw, _ = rewire_edges(b["edge"], b["ids"], b["edge_mask"], radius=8)
        tgt = _tgt(b["ids"], new_edge)
        sep = (tgt - b["ids"][:, :, None]).abs()
        assert bool((sep[rw] > 8).all()), f"trial {trial}: rewired edge too close"


def test_scramble_batch_modes_and_node_features_untouched():
    b = _batch()
    esm = torch.randn(2, 24, 8)
    full, finfo = scramble_batch({**b, "esm": esm}, chain_preserved=False)
    assert full["edge"].shape == b["edge"].shape
    assert full["unit"].shape == b["unit"].shape
    assert full["edge_features"].shape == b["edge_features"].shape
    assert torch.equal(full["esm"], esm)
    chain, cinfo = scramble_batch(dict(b), chain_preserved=True)
    assert cinfo["success_rate"] > 0.9
    tgt = _tgt(b["ids"], b["edge"])
    close = ((tgt - b["ids"][:, :, None]).abs() <= 8) & b["edge_mask"]
    assert bool((chain["edge"][close] == b["edge"][close]).all())
    assert not torch.equal(chain["edge_features"][~close & b["edge_mask"]],
                            b["edge_features"][~close & b["edge_mask"]])


def test_full_mode_rewires_every_masked_edge():
    b = _batch()
    new_edge, rw, info = rewire_edges(b["edge"], b["ids"], b["edge_mask"], radius=-1)
    assert info["attempted"] == int(b["edge_mask"].sum())
    assert bool((rw == b["edge_mask"]).all())
    assert info["success_rate"] == 1.0


def test_scramble_is_deterministic_for_fixed_seed():
    b = _batch()
    first, _ = scramble_batch(dict(b), chain_preserved=True, seed=7)
    second, _ = scramble_batch(dict(b), chain_preserved=True, seed=7)
    for k in ["edge", "edge_features", "unit"]:
        assert torch.equal(first[k], second[k])
    third, _ = scramble_batch(dict(b), chain_preserved=True, seed=8)
    assert not torch.equal(first["edge"], third["edge"])


def test_fusion_pieces_and_branch_probes_share_one_width():
    model = MIPO(32, 4, {"model": "mipo", "hidden": 16, "branch_weight": 0.2,
                         "sequence_layers": 1, "field_layers": 1})
    (total, widths) = model.fusion[0].in_features, {p.in_features for p in model.branch_probes}
    assert total % 4 == 0 and widths == {total // 4}  # four equal pieces; slicing is aligned
