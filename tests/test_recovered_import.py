"""Regressions for scripts/import_recovered_assays.py.

Two invariants the recovered-source import must never break:
  1. Source measurements are carried through unchanged -- replicate SNVs are averaged and
     nothing else (no rank transform, no z-scoring, no sign flip, no clamping).
  2. SPOP's orientation (recorded rho = -1) is handled by *checking* the recorded direction,
     not by rewriting the source values to match the collapsed consensus.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "import_recovered_assays.py"


def _load():
    spec = importlib.util.spec_from_file_location("import_recovered_assays", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


imp = _load()


def test_source_scores_average_replicates_and_never_transform_values(tmp_path, monkeypatch):
    urn = "urn:mavedb:00000001-a-1"
    table = pd.DataFrame({
        "hgvs_pro": ["p.Ala10Gly", "p.Ala10Gly", "p.Asp20Glu", "p.Gly30Gly"],
        "score": [1.0, 3.0, -5.0, 0.7],  # A10G replicated; D20E negative; G30G synonymous.
    })
    monkeypatch.setattr(imp, "SOURCES", tmp_path)
    table.to_csv(tmp_path / "urn_mavedb_00000001-a-1.csv", index=False)

    grouped, problem = imp.source_scores(urn)
    assert problem is None
    grouped = grouped.set_index(["wt", "ref_pos", "mut"])
    # Replicates are the plain arithmetic mean -- not a rank or a z-score.
    assert grouped.loc[("A", 10, "G"), "score_value"] == pytest.approx(2.0)
    # A negative source value is preserved as-is: no sign flip, no clamping to [0, 1].
    assert grouped.loc[("D", 20, "E"), "score_value"] == pytest.approx(-5.0)
    # Synonymous changes are dropped, not scored.
    assert ("G", 30, "G") not in grouped.index


def test_negative_orientation_is_checked_not_applied_to_source_values():
    seq = "M" + "ACDEFGHIKL" * 3
    sequences = {"ref": {"sequence": seq}}
    positions = [2, 5, 8, 11]
    source = pd.DataFrame({
        "wt": [seq[p - 1] for p in positions],
        "ref_pos": positions,
        "mut": ["G", "W", "R", "P"],
        "score_value": [0.1, 0.5, 0.9, 1.3],
    })
    consensus = source.copy()
    consensus["score_value"] = source["score_value"].to_numpy()[::-1]  # exact reversal -> rho = -1
    before = source.score_value.tolist()

    # SPOP is recorded as inverted_relative_to_source (rho = -1): a negative record is accepted.
    assert imp.verify(source, consensus, sequences, "ref", -1.0, 0.01) is None
    # The same data recorded as +1 must be rejected -- the direction is enforced, not ignored.
    rejected = imp.verify(source, consensus, sequences, "ref", +1.0, 0.01)
    assert rejected and "rank match" in rejected
    # verify never rewrites the source measurements to match the consensus orientation.
    assert source.score_value.tolist() == before
