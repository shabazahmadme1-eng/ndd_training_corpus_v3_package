"""Eval-time graph instruments: topology scrambles with per-edge 2D semantics.

All ops stay 2D (B, N, K); nothing here relies on broadcasting, so a shape bug
raises instead of silently promoting. Used by the structure-vs-leakage controls.
"""
import torch


def rewire_edges(edge, ids, edge_mask, radius=8, tries=5, generator=None):
    """Rewire edges whose endpoints are far apart in sequence; keep close ones bit-identical.

    edge: (B, N, K) int64 node indices, node-major (edge[b, i, k] leaves node i).
    ids: (B, N) int64 residue numbers (-1 padding). Returns (new_edge, rewired_mask,
    info) where info reports attempted/succeeded counts; every rewired edge lands
    > radius from its own source, verified post-loop. radius < 0 rewires every
    masked edge (full topology scramble).
    """
    B, N, K = edge.shape
    ids3 = ids[:, :, None].expand(B, N, K)
    tgt = ids3.gather(1, edge.clamp_min(0))
    src = ids[:, :, None].expand(B, N, K)
    rw = ((tgt - src).abs() > radius) & edge_mask & (src >= 0)
    rand = torch.randint(0, N, edge.shape, dtype=edge.dtype, device=edge.device, generator=generator)
    rtgt = ids3.gather(1, rand.clamp(0, N - 1))
    far = ((rtgt - src).abs() > radius) | ~rw
    for _ in range(tries):
        if bool(far.all()):
            break
        new = torch.randint(0, N, edge.shape, dtype=edge.dtype, device=edge.device, generator=generator)
        ntgt = ids3.gather(1, new.clamp(0, N - 1))
        take = ((ntgt - src).abs() > radius) & ~far
        rand = torch.where(take, new, rand)
        rtgt = torch.where(take, ntgt, rtgt)
        far = ((rtgt - src).abs() > radius) | ~rw
    ok = ((rtgt - src).abs() > radius) & (rtgt >= 0)
    keep = ~rw | ~ok
    rewired = ~keep
    info = {"attempted": int(rw.sum()), "succeeded": int(rewired.sum()),
            "masked": int(edge_mask.sum()),
            "success_rate": float(rewired.sum()) / max(1, int(rw.sum()))}
    return torch.where(keep, edge, rand), rewired, info


def scramble_batch(batch, chain_preserved=True, radius=8, seed=0):
    """Return a copy of a collated batch with graph topology corrupted, plus rewire info.

    chain_preserved=True keeps backbone edges (|dseq| <= radius) intact and rewires
    the rest (geometry-specific control); False rewires every masked edge (topology
    control). Node features are never touched. Shapes and dtypes are unchanged.

    Notes: the seed fixes all RNG (one pattern per seed, repeated across same-shape
    batches — a repeated-pattern null, not i.i.d.). Edge-feature noise is a single
    shared slot permutation, so under a slot-permutation-equivariant aggregator only
    the masked-slot substitution corrupts; if full-scramble margins look
    suspiciously small, suspect this first.
    """
    gen = torch.Generator(device=batch["edge"].device).manual_seed(seed)
    out = dict(batch)
    edge, mask = batch["edge"], batch["edge_mask"]
    new_edge, rw, info = rewire_edges(edge, batch["ids"], mask,
                                      radius=radius if chain_preserved else -1, generator=gen)
    out["edge"] = new_edge
    K = batch["edge_features"].shape[2]
    out["unit"] = torch.where(
        rw[:, None, :, :, None].expand_as(batch["unit"]),
        torch.randn(batch["unit"].shape, dtype=batch["unit"].dtype,
                    device=batch["unit"].device, generator=gen),
        batch["unit"])
    perm = torch.randperm(K, generator=gen)
    noise = batch["edge_features"][:, :, perm, :]
    out["edge_features"] = torch.where(
        rw[:, :, :, None].expand_as(batch["edge_features"]), noise, batch["edge_features"])
    return out, info
