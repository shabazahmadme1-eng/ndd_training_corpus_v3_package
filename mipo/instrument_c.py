"""Instrument C: does the structure leg carry signal, or is it decoration?

The confirmation gate (frozen in the reviews): build Tier 2 only if the
chain-preserved scramble margin is >= 0.05. This module measures that margin on a
trained checkpoint's own test split, using `mipo.instruments.scramble_batch`.

Margin = clean macro_gene_spearman - scrambled macro_gene_spearman, so a model
that reads geometry loses accuracy when geometry is corrupted and a model that
ignores it does not. macro_gene_spearman is the project's selection metric, so
the margin is denominated in the quantity the runs were chosen by.

Two modes, both reported:
  chain_preserved=True  - backbone edges (|dseq| <= radius) kept bit-identical,
                          the rest rewired. This is the GATE: it isolates
                          geometry beyond what sequence adjacency already gives.
  chain_preserved=False - every masked edge rewired (topology control). Expected
                          to be the larger of the two; if it is not, read the
                          slot-permutation caveat in instruments.scramble_batch
                          before trusting either number.

Seeds: scramble_batch fixes all RNG from one seed, so a single seed is one
corruption pattern repeated across same-shape batches, not an i.i.d. draw. The
margin is therefore averaged over several seeds and the spread is reported; a
gate verdict from one seed is not meaningful.

Scrambling happens on the CPU batch before the move to device. scramble_batch
builds its generator on the batch's device and calls torch.randperm without a
device argument, which raises on a CUDA generator; scrambling first keeps the
instrument identical on GPU and CPU runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .common import TASKS, to_device
from .instruments import scramble_batch
from .metrics import metrics

GATE = 0.05


def predict_batches(model, batches, device, scramble=None, seed=0, radius=8, ablate_edges=False):
    """Mirror of train.predict, optionally corrupting each batch's topology first.

    scramble=None scores the batch as loaded; True/False select chain-preserved
    or full rewiring. Rewire statistics are accumulated so a silently ineffective
    scramble (success_rate ~ 0, nothing attempted) cannot pass as a null result.

    ablate_edges zeroes every edge weight instead of rewiring, which silences
    message passing rather than corrupting it. Its margin is the ceiling on what
    any scramble can show, and it is a diagnostic, never the gate.
    """
    model.eval()
    frames, info = [], {"attempted": 0, "succeeded": 0, "masked": 0}
    with torch.inference_mode():
        for b in batches:
            if scramble is not None:
                b, stats = scramble_batch(b, chain_preserved=scramble, radius=radius, seed=seed)
                for key in info:
                    info[key] += stats[key]
            if ablate_edges:
                b = dict(b)
                b["edge_weight"] = torch.zeros_like(b["edge_weight"])
            b = to_device(b, device)
            out = model(b)
            frames.append(pd.DataFrame({
                "row_id": b["row_id"].cpu().numpy(), "y": b["y"].cpu().numpy(),
                "mu": out["mu"].float().cpu().numpy(),
                "sigma": out["sigma"].float().cpu().numpy()}))
    p = pd.concat(frames, ignore_index=True).merge(
        batches.dataset.rows[["row_id", "gene", "assay_key", "variant_key", "task", "score_value"]],
        on="row_id", validate="one_to_one")
    p["task_seen_in_training"] = p.task.map(dict(zip(TASKS, model.seen_tasks.cpu().tolist())))
    info["success_rate"] = info["succeeded"] / max(1, info["attempted"])
    return p, info


def macro(predictions):
    """The project's selection metric for one prediction frame."""
    _, summary = metrics(predictions)
    return summary["macro_gene_spearman"]


def per_assay_spearman(predictions):
    assays, _ = metrics(predictions)
    return assays.set_index(["gene", "assay_key"]).spearman


def run_instrument_c(model, batches, device, seeds=(0, 1, 2, 3, 4), radius=8):
    """Clean vs scrambled margins over several corruption seeds.

    Returns (summary dict, per-seed DataFrame, per-assay DataFrame). The per-assay
    frame is paired on (gene, assay_key) against the clean run, so the direction of
    the effect can be counted instead of inferred from one pooled number.
    """
    clean, _ = predict_batches(model, batches, device)
    clean_macro = macro(clean)
    if clean_macro is None:
        raise ValueError("Clean macro_gene_spearman is undefined (an assay has no "
                         "defined Spearman); the margin would be meaningless")
    clean_assays = per_assay_spearman(clean)

    rows, assay_deltas = [], []
    for chain_preserved in (True, False):
        mode = "chain_preserved" if chain_preserved else "full"
        for seed in seeds:
            scrambled, info = predict_batches(model, batches, device,
                                              scramble=chain_preserved, seed=seed, radius=radius)
            value = macro(scrambled)
            rows.append({"mode": mode, "seed": seed, "clean_macro": clean_macro,
                         "scrambled_macro": value,
                         "margin": None if value is None else clean_macro - value,
                         "rewire_attempted": info["attempted"], "rewire_succeeded": info["succeeded"],
                         "rewire_success_rate": info["success_rate"],
                         "masked_edges": info["masked"],
                         "mu_shift_mean_abs": float(np.abs(scrambled.mu - clean.mu).mean())})
            delta = (clean_assays - per_assay_spearman(scrambled)).rename("delta").reset_index()
            delta["mode"], delta["seed"] = mode, seed
            assay_deltas.append(delta)

    # Ceiling diagnostic: silencing message passing outright. Seed-independent.
    ablated, _ = predict_batches(model, batches, device, ablate_edges=True)
    ablated_macro = macro(ablated)
    rows.append({"mode": "edge_ablated", "seed": None, "clean_macro": clean_macro,
                 "scrambled_macro": ablated_macro,
                 "margin": None if ablated_macro is None else clean_macro - ablated_macro,
                 "rewire_attempted": 0, "rewire_succeeded": 0, "rewire_success_rate": float("nan"),
                 "masked_edges": 0,
                 "mu_shift_mean_abs": float(np.abs(ablated.mu - clean.mu).mean())})
    ablated_delta = (clean_assays - per_assay_spearman(ablated)).rename("delta").reset_index()
    ablated_delta["mode"], ablated_delta["seed"] = "edge_ablated", None
    assay_deltas.append(ablated_delta)

    per_seed = pd.DataFrame(rows)
    per_assay = pd.concat(assay_deltas, ignore_index=True)
    summary = {"clean_macro_gene_spearman": clean_macro, "gate_threshold": GATE,
               "seeds": list(seeds), "radius": radius, "rows_scored": len(clean)}
    for mode, group in per_seed.groupby("mode"):
        margins = group.margin.dropna()
        detail = per_assay[(per_assay["mode"] == mode) & per_assay.delta.notna()]
        summary[mode] = {
            "margin_mean": float(margins.mean()) if len(margins) else None,
            "margin_min": float(margins.min()) if len(margins) else None,
            "margin_max": float(margins.max()) if len(margins) else None,
            "margin_std": float(margins.std(ddof=0)) if len(margins) else None,
            "undefined_seeds": int(group.margin.isna().sum()),
            "rewire_success_rate": float(group.rewire_success_rate.mean()),
            "assays_degraded": int((detail.delta > 0).sum()),
            "assays_compared": int(len(detail)),
            "mu_shift_mean_abs": float(group.mu_shift_mean_abs.mean())}
    gate = summary["chain_preserved"]["margin_mean"]
    summary["gate_passed"] = bool(gate is not None and gate >= GATE)
    summary["gate_statement"] = (
        f"chain-preserved margin {gate:.4f} "
        f"{'>=' if summary['gate_passed'] else '<'} {GATE} -> Tier 2 gate "
        f"{'OPEN' if summary['gate_passed'] else 'NOT met on instrument C'}"
        if gate is not None else "chain-preserved margin undefined")
    return summary, per_seed, per_assay


def structure_blind(config):
    """True when the config cannot read geometry, so its margin must be ~0.

    esm_mlp skips the field layers outright and `structure: False` zeroes the
    confidence mask, which drops every edge from topology and distances. Such a
    run is the instrument's negative control, not a candidate for the gate.
    """
    return config.get("model") == "esm_mlp" or not config.get("structure", True)
