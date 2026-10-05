"""FireProt homologue-free ddG benchmark: eval-only transfer test (Tier 1.5a).

BENCHMARK ONLY. These rows must never touch training; the builders here write a
separate corpus + resources tree, never into v4/.

Conventions: ddG > 0 = destabilizing. Every scorer is oriented so higher = more
stable (anchor columns arrive ddG-like and are negated). Metrics: pooled Spearman
on per-protein z-scored values + direction accuracy of sign(score_z) against
sign(-ddG_z). Per-protein Spearman is underpowered (median n 9, singletons exist).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .common import AA, save_json
from .corpus import read_corpus
from .data import FeatureStore, Metadata, TargetTransform

FIELD_PROBE_INDEX = 1  # pieces = [local, field_pool, global, impulse] (model.py forward)


def safe_seq1(resname):
    """Three-letter code to one-letter; unknown/modified residues become X, never crash."""
    from Bio.SeqUtils import seq1
    try:
        letter = seq1(resname)
    except KeyError:
        return "X"
    return letter if len(letter) == 1 and letter in AA + "X" else "X"


def best_offset(chain_seq, full_seq):
    """Offset placing chain_seq onto full_seq maximizing residue matches.

    Chain position j maps to full position j + offset. X never matches.
    Returns (offset, matches).
    """
    best = (0, -1)
    for offset in range(-(len(chain_seq) - 1), len(full_seq)):
        lo = max(0, offset)
        hi = min(len(full_seq), len(chain_seq) + offset)
        matches = sum(1 for i in range(lo, hi)
                      if full_seq[i] == chain_seq[i - offset] and chain_seq[i - offset] != "X")
        if matches > best[1]:
            best = (offset, matches)
    return best


def parse_chain(pdb_path, chain_id):
    """Per-model ordered residues: list over models of [(aa, xyz or None)]."""
    from Bio.PDB import PDBParser
    models = []
    for model in PDBParser(QUIET=True).get_structure("protein", str(pdb_path)):
        chains = [c for c in model if c.id == chain_id]
        if len(chains) != 1:
            raise ValueError(f"{pdb_path}: chain {chain_id} found {len(chains)} times in a model")
        residues = []
        for residue in chains[0]:
            if residue.id[0] != " ":
                continue
            xyz = np.asarray(residue["CA"].coord, np.float32) if "CA" in residue else None
            residues.append((safe_seq1(residue.resname), xyz))
        models.append(residues)
    if not models:
        raise ValueError(f"{pdb_path}: no coordinate models")
    return models


def build_structure_arrays(pdb_paths, chain_id, sequence, min_identity=0.9):
    """CA coords aligned to the full sequence from the first PDB that clears the bar.

    Uncovered positions get zero coords and zero confidence, which the loader's
    confidence mask already excludes from topology and distances. Returns
    (coords, confidence, source_pdb, attempts) with coords None when no PDB clears
    the bar; the caller then ships sequence-only (rows still scorable).
    """
    attempts = []
    fallback = []
    for pdb_path in pdb_paths:
        try:
            models = parse_chain(pdb_path, chain_id)
        except (ValueError, KeyError) as error:
            attempts.append({"pdb": Path(pdb_path).name, "status": "parse_failed", "reason": str(error)[:120]})
            fallback.append(pdb_path)
            continue
        chain_seq = "".join(aa for aa, _ in models[0])
        known = sum(1 for a in chain_seq if a != "X")
        offset, matches = best_offset(chain_seq, sequence)
        identity = matches / max(1, known)
        if identity < min_identity:
            attempts.append({"pdb": Path(pdb_path).name, "status": "alignment_rejected",
                             "identity": round(identity, 4), "chain": chain_id,
                             "chain_len": len(chain_seq), "seq_len": len(sequence)})
            continue
        # Overhanging tags are clipped: only aligned span positions take coords.
        coords, conf = place_coords(models, offset, sequence)
        covered = int((conf[0] > 0).sum())
        attempts.append({"pdb": Path(pdb_path).name, "status": "accepted", "identity": round(identity, 4),
                         "chain": chain_id, "conformers": len(models),
                         "covered": covered, "seq_len": len(sequence)})
        return coords, conf, Path(pdb_path).name, attempts
    # Second pass: the listed chain may be mislabeled; accept an alternate chain
    # only if exactly one chain clears the bar and every other scores below 0.5.
    for pdb_path in fallback:
        try:
            hit = unique_chain(pdb_path, sequence, min_identity)
        except (ValueError, KeyError) as error:
            attempts.append({"pdb": Path(pdb_path).name, "status": "fallback_failed", "reason": str(error)[:120]})
            continue
        if hit is None:
            attempts.append({"pdb": Path(pdb_path).name, "status": "fallback_ambiguous"})
            continue
        alt, models, offset, identity, runner_up = hit
        coords, conf = place_coords(models, offset, sequence)
        covered = int((conf[0] > 0).sum())
        attempts.append({"pdb": Path(pdb_path).name, "status": "accepted_alt_chain", "identity": round(identity, 4),
                         "chain": alt, "runner_up_identity": round(runner_up, 4),
                         "conformers": len(models), "covered": covered, "seq_len": len(sequence)})
        return coords, conf, Path(pdb_path).name, attempts
    return None, None, None, attempts


def place_coords(models, offset, sequence):
    n_models, length = len(models), len(sequence)
    coords = np.zeros((n_models, length, 3), np.float32)
    conf = np.zeros((n_models, length), np.float32)
    for m, residues in enumerate(models):
        for j, (aa, xyz) in enumerate(residues):
            i = j + offset
            if xyz is not None and 0 <= i < length and sequence[i] == aa:
                coords[m, i] = xyz
                conf[m, i] = 1.
    return coords, conf


def unique_chain(pdb_path, sequence, min_identity=0.9, runner_cap=0.5):
    """(chain_id, models, offset, identity, runner_up) or None when ambiguous."""
    from Bio.PDB import PDBParser
    scored = []
    for model in PDBParser(QUIET=True).get_structure("protein", str(pdb_path)):
        for chain in model:
            residues = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                xyz = np.asarray(residue["CA"].coord, np.float32) if "CA" in residue else None
                residues.append((safe_seq1(residue.resname), xyz))
            seq = "".join(aa for aa, _ in residues)
            known = sum(1 for a in seq if a != "X")
            if not known:
                continue
            offset, matches = best_offset(seq, sequence)
            scored.append((chain.id, matches / known, offset))
        break  # chain identity comes from the first model only
    if not scored:
        return None
    scored.sort(key=lambda t: t[1], reverse=True)
    (top_id, top_score, top_offset), runner = scored[0], scored[1][1] if len(scored) > 1 else 0.
    if top_score >= min_identity and runner < runner_cap:
        return top_id, parse_chain(pdb_path, top_id), top_offset, top_score, runner
    return None


def zscore(values):
    values = np.asarray(values, float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return None
    return (values - values.mean()) / std


def pooled_metrics(df, score_cols, group_col="gene", target_col="ddG"):
    """Pooled z-Spearman vs stability (-ddG) + direction accuracy per scorer.

    Groups with < 2 rows or zero target variance are excluded (reported, not
    silently kept). Direction accuracy excludes rows where either z-score is
    exactly zero. Constant scores yield NaN Spearman, never a crash.
    """
    per_group, excluded = [], []
    for name, group in df.groupby(group_col):
        if len(group) < 2:
            excluded.append({"gene": name, "n": len(group), "reason": "singleton"})
            continue
        tz = zscore(group[target_col].to_numpy())
        if tz is None:
            excluded.append({"gene": name, "n": len(group), "reason": "zero_target_variance"})
            continue
        row = {"gene": name, "n": len(group), "stability_z": -tz}
        for col in score_cols:
            row[col] = zscore(group[col].to_numpy())
        per_group.append(row)
    metrics = {}
    for col in score_cols:
        kept = [r for r in per_group if r[col] is not None]
        s = np.concatenate([r[col] for r in kept]) if kept else np.array([])
        t = np.concatenate([r["stability_z"] for r in kept]) if kept else np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(s, t).statistic) if len(s) > 2 and np.std(s) > 0 else float("nan")
        keep = (s != 0) & (t != 0)
        acc = float((((s[keep] > 0) == (t[keep] > 0)).mean())) if keep.sum() else float("nan")
        metrics[col] = {"pooled_z_spearman": rho, "direction_accuracy": acc,
                        "n_rows": int(keep.sum()), "n_pooled": int(len(s))}
    metrics["_pooled_groups"] = len(per_group)
    metrics["_excluded_groups"] = excluded
    return metrics


def sign_check(df, score_col="llr", target_col="ddG", k=3):
    """The 3 most stabilizing rows must outscore the 3 most destabilizing on LLR.

    Catches a flipped stability convention before any number is trusted. LLR is
    the check because it is assay-free; model scores are what is under test.
    """
    ordered = df.sort_values(target_col)
    stab = ordered.head(k)
    destab = ordered.tail(k)
    if len(ordered) < 2 * k:
        raise ValueError("Too few rows for the sign check")
    a, b = float(stab[score_col].mean()), float(destab[score_col].mean())
    if not a > b:
        raise ValueError(f"SIGN CHECK FAILED: mean {score_col} of stabilizing rows ({a:.4f}) "
                         f"<= destabilizing rows ({b:.4f}); convention flipped or LLR noise")
    return {"stabilizing_mean": a, "destabilizing_mean": b, "k": k}


def score_checkpoint(checkpoint, corpus_csv, resources, cache, metadata_path, device=None, batch_size=64,
                   require_probes=True):
    """Score every row with mu, cached LLR and the field-probe scalar.

    The assay head sees task=stability with an unseen assay key, so the meta
    embedding is unknown tokens: no benchmark label enters the prediction. Rows
    keep their corpus columns plus llr/mu/field_probe.

    require_probes=True (default) preserves the original contract: probeless
    checkpoints raise. Pass require_probes=False only to score a surviving
    branch_weight=0 config (Tier 1 failed); field_probe is then NaN by
    construction and the pooled table shows it as missing, never as zero.
    """
    from torch.utils.data import DataLoader

    from .data import VariantDataset, collate
    from .train import restore
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, state = restore(str(checkpoint), device)
    probeless = model.branch_probes is None or len(model.branch_probes) != 4
    if probeless and require_probes:
        raise ValueError("Checkpoint has no per-branch probes; score a branch_weight > 0 run (v3+) "
                         "or pass require_probes=False to score llr+mu only")
    if probeless:
        print("FIELD PROBE UNAVAILABLE: checkpoint has no branch probes; "
              "field_probe will be NaN (missing, not zero)", flush=True)
    config = dict(state["provenance"]["config"], batch_size=batch_size)
    rows = read_corpus(corpus_csv)
    store = FeatureStore(str(resources), str(cache))
    transform = TargetTransform(state["transform"])
    metadata = Metadata(str(metadata_path), state["metadata_vocab"])
    ds = VariantDataset(rows, store, transform, metadata, config)
    batches = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    model.eval()
    frames = []
    with torch.inference_mode():
        for b in batches:
            present = b["llr"][:, 1]
            if not bool(present.all()):
                missing = int((~present.bool()).sum())
                raise ValueError(f"{missing} rows lack cached LLR; extract masked LLR for every mutated site")
            from .common import to_device
            out = model(to_device(b, device))
            if probeless:
                probe = np.full(len(b["row_id"]), np.nan)
            else:
                probe = out["branch_mu"][:, FIELD_PROBE_INDEX].float().cpu().numpy()
            frames.append(pd.DataFrame({
                "row_id": b["row_id"].numpy(),
                "llr": (b["llr"][:, 0] * 10).numpy(),
                "mu": out["mu"].float().cpu().numpy(),
                "field_probe": probe}))
    scores = pd.concat(frames, ignore_index=True).merge(
        rows[["row_id", "gene", "assay_key", "variant_key", "wt", "ref_pos", "mut", "score_value"]],
        on="row_id", validate="one_to_one")
    return scores.rename(columns={"score_value": "ddG"})
