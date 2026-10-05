"""Tier 1.5a FireProt benchmark: mapping, metrics, sign check, scoring plumbing."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mipo.corpus import read_corpus
from mipo.fireprot import (best_offset, build_structure_arrays, pooled_metrics,
                           score_checkpoint, sign_check)

PDB_TINY = """\
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 50.00           C
ATOM      2  CA  CYS A   2      11.000  10.000  10.000  1.00 50.00           C
ATOM      3  CA  ASP A   3      12.000  10.000  10.000  1.00 50.00           C
ATOM      4  CA  GLU A   4      13.000  10.000  10.000  1.00 50.00           C
END
"""


def test_best_offset_exact_and_overhang():
    assert best_offset("ACDE", "XXACDEYY") == (2, 4)
    assert best_offset("ACDE", "ACDE") == (0, 4)
    offset, matches = best_offset("TTACDETT", "ACDE")
    assert matches == 4 and offset == -2  # tags overhang, counted matches only


def test_best_offset_ignores_x():
    offset, matches = best_offset("AXDE", "ACDE")
    assert (offset, matches) == (0, 3)


def test_build_structure_arrays_tiny(tmp_path):
    pdb = tmp_path / "t.pdb"
    pdb.write_text(PDB_TINY)
    coords, conf, source, attempts = build_structure_arrays([pdb], "A", "ZZACDEZZ")
    assert source == "t.pdb" and coords.shape == (1, 8, 3) and conf.shape == (1, 8)
    assert conf[0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert np.isfinite(coords).all()


def test_build_structure_arrays_rejects_and_reports(tmp_path):
    pdb = tmp_path / "t.pdb"
    pdb.write_text(PDB_TINY)
    coords, conf, source, attempts = build_structure_arrays([pdb], "A", "WWWWWWWW")
    assert coords is None and source is None
    assert attempts and attempts[0]["status"] == "alignment_rejected"


def test_pooled_metrics_perfect_and_direction():
    df = pd.DataFrame({"gene": ["a"] * 6 + ["b"] * 6,
                       "ddG": [3, 2, 1, -1, -2, -3] * 2,
                       "good": [-3, -2, -1, 1, 2, 3] * 2,   # stability direction
                       "bad": [3, 2, 1, -1, -2, -3] * 2})   # ddG direction
    m = pooled_metrics(df, ["good", "bad"])
    assert m["good"]["pooled_z_spearman"] == pytest.approx(1.)
    assert m["good"]["direction_accuracy"] == pytest.approx(1.)
    assert m["bad"]["pooled_z_spearman"] == pytest.approx(-1.)
    assert m["bad"]["direction_accuracy"] == pytest.approx(0.)
    assert m["_pooled_groups"] == 2 and m["_excluded_groups"] == []


def test_pooled_metrics_excludes_singleton_and_flat():
    df = pd.DataFrame({"gene": ["a"] * 4 + ["solo", "flat", "flat"],
                       "ddG": [3, 1, -1, -3, 9, 2, 2],
                       "s": [1, 2, 3, 4, 5, 6, 7]})
    m = pooled_metrics(df, ["s"])
    reasons = sorted(e["reason"] for e in m["_excluded_groups"])
    assert reasons == ["singleton", "zero_target_variance"]
    assert m["_pooled_groups"] == 1
    assert np.isfinite(m["s"]["pooled_z_spearman"])


def test_pooled_metrics_constant_scores_nan_not_crash():
    df = pd.DataFrame({"gene": ["a"] * 4, "ddG": [3, 1, -1, -3], "s": [7, 7, 7, 7]})
    m = pooled_metrics(df, ["s"])
    assert np.isnan(m["s"]["pooled_z_spearman"]) and np.isnan(m["s"]["direction_accuracy"])


def test_sign_check_pass_and_fail():
    df = pd.DataFrame({"ddG": [-3, -2, -1, 0, 1, 2, 3],
                       "llr": [3, 2, 1, 0, -1, -2, -3]})
    assert sign_check(df)["stabilizing_mean"] > 0
    with pytest.raises(ValueError, match="SIGN CHECK FAILED"):
        sign_check(df.assign(llr=-df.llr))


def _fake_tree(tmp_path):
    """Two-gene FireProt-like tree: corpus, resources, cache, metadata, checkpoint."""
    from mipo.common import save_json, seq_key
    seqs = {"FP_1AAA": "ACDEFGHIKLMNPQRSTVWYACDEFGHIK", "FP_2BBB": "MNPQRSTVWYACDEFGHIKLMNPQRSTVW"}
    resources, cache = tmp_path / "res", tmp_path / "cache"
    (resources / "structures").mkdir(parents=True)
    records = {}
    for gene, seq in seqs.items():
        key = seq_key(seq)
        records[gene] = {"gene": gene, "protein_reference_id": gene, "sequence": seq,
                         "key": key, "length": len(seq)}
        folder = cache / key
        folder.mkdir(parents=True)
        rng = np.random.default_rng(hash(gene) % 2**32)
        np.save(folder / "esm.npy", rng.normal(size=(len(seq), 32)).astype(np.float16))
        np.save(folder / "llr.npy", rng.normal(size=(len(seq), 20)).astype(np.float32))
        save_json(folder / "sequence.json", records[gene])
    save_json(resources / "sequences.json", records)
    save_json(cache / "manifest.json", {"dim": 32, "model": "fake-test", "schema": 1})
    rows = []
    for gene, seq in seqs.items():
        for pos in (2, 5, 8, 11):
            rows.append({"gene": gene, "protein_reference_id": gene, "wt": seq[pos - 1],
                         "ref_pos": pos, "mut": "A" if seq[pos - 1] != "A" else "G",
                         "score_value": float(pos), "score_kind": "ddG", "assay_id": "fireprot_ddg",
                         "assay_type": "stability", "supervision_tier": "benchmark"})
    corpus = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(corpus, index=False)
    meta = tmp_path / "assay_metadata.csv"
    pd.DataFrame([{"assay_key": f"{g}::fireprot_ddg", "task": "stability", "score_kind": "ddG",
                   "host": "u", "selection": "u", "system": "u", "treatment": "u", "region": "u"}
                  for g in seqs]).to_csv(meta, index=False)
    return corpus, resources, cache, meta


def _fake_checkpoint(tmp_path):
    from mipo.model import MIPO
    config = {"hidden": 32, "model": "mipo", "branch_weight": 0.5, "batch_size": 8,
              "window_radius": 5, "max_nodes": 64}
    torch.manual_seed(0)
    model = MIPO(32, 1, config)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(),
                "provenance": {"config": config, "feature_manifest": {"dim": 32}},
                "transform": {"version": 2, "tasks": {"stability": {"center": 0., "scale": 1.}},
                              "assays": {}},
                "metadata_vocab": {"<unknown>": 0}}, ckpt)
    return ckpt


def test_score_checkpoint_end_to_end_cpu(tmp_path):
    corpus, resources, cache, meta = _fake_tree(tmp_path)
    ckpt = _fake_checkpoint(tmp_path)
    assert len(read_corpus(corpus)) == 8
    scores = score_checkpoint(ckpt, corpus, resources, cache, meta, device="cpu", batch_size=4)
    assert len(scores) == 8
    assert {"gene", "ddG", "llr", "mu", "field_probe"} <= set(scores)
    assert np.isfinite(scores[["llr", "mu", "field_probe"]].to_numpy()).all()
    assert scores.gene.nunique() == 2


def test_score_checkpoint_rejects_probeless_ckpt(tmp_path):
    from mipo.model import MIPO
    corpus, resources, cache, meta = _fake_tree(tmp_path)
    config = {"hidden": 32, "model": "mipo", "branch_weight": 0., "batch_size": 8,
              "window_radius": 5, "max_nodes": 64}
    torch.manual_seed(0)
    ckpt = tmp_path / "noprobe.pt"
    torch.save({"model": MIPO(32, 1, config).state_dict(),
                "provenance": {"config": config, "feature_manifest": {"dim": 32}},
                "transform": {"version": 2, "tasks": {"stability": {"center": 0., "scale": 1.}},
                              "assays": {}},
                "metadata_vocab": {"<unknown>": 0}}, ckpt)
    with pytest.raises(ValueError, match="no per-branch probes"):
        score_checkpoint(ckpt, corpus, resources, cache, meta, device="cpu", batch_size=4)


def test_score_checkpoint_probeless_opt_in_nan_not_zero(tmp_path):
    from mipo.model import MIPO
    corpus, resources, cache, meta = _fake_tree(tmp_path)
    config = {"hidden": 32, "model": "mipo", "branch_weight": 0., "batch_size": 8,
              "window_radius": 5, "max_nodes": 64}
    torch.manual_seed(0)
    ckpt = tmp_path / "noprobe.pt"
    torch.save({"model": MIPO(32, 1, config).state_dict(),
                "provenance": {"config": config, "feature_manifest": {"dim": 32}},
                "transform": {"version": 2, "tasks": {"stability": {"center": 0., "scale": 1.}},
                              "assays": {}},
                "metadata_vocab": {"<unknown>": 0}}, ckpt)
    scores = score_checkpoint(ckpt, corpus, resources, cache, meta, device="cpu",
                              batch_size=4, require_probes=False)
    assert len(scores) == 8
    assert scores.field_probe.isna().all()
    assert np.isfinite(scores[["llr", "mu"]].to_numpy()).all()


def _colab_result_tag():
    """Load result_tag from the Colab cell source without executing the cell."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts/colab_fireprot_15a.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "result_tag":
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), ns)
            return ns["result_tag"]
    raise AssertionError("result_tag missing from scripts/colab_fireprot_15a.py")


def test_colab_result_tags_unique_across_models(tmp_path):
    tag = _colab_result_tag()
    pilot = tmp_path / "contrast_pilot_v1"
    ckpts = [pilot / model / "seed_42" / fold / "best.pt"
             for model in ("esm_mlp", "mipo") for fold in ("fold_12", "fold_57")]
    tags = [tag(c, pilot) for c in ckpts]
    assert len(set(tags)) == len(tags) == 4  # old parent.parent tag gave 2: silent overwrite
