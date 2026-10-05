"""Score a checkpoint on the FireProt benchmark (Tier 1.5a, eval-only).

Scorers: cached LLR zero-shot, model mu under the stability task embedding with
an unseen assay key, and the assay-free field-probe scalar. Optional --anchors
CSV joins published ddG-direction predictors (higher = more destabilizing) on
(gene, wt, ref_pos, mut). Metrics: pooled z-Spearman vs stability + direction
accuracy. A sign check on LLR extremes runs before any number is trusted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mipo.common import save_json
from mipo.fireprot import pooled_metrics, score_checkpoint, sign_check


def clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--metadata", required=True, help="TRAINING assay_metadata.csv (frozen vocab)")
    parser.add_argument("--anchors", default=None, help="CSV with gene,wt,ref_pos,mut + ddG-direction columns")
    parser.add_argument("--allow-probeless", action="store_true",
                        help="Score branch_weight=0 checkpoints with field_probe=NaN (missing, not zero)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scores = score_checkpoint(args.checkpoint, args.corpus, args.resources, args.cache,
                              args.metadata, args.device, args.batch_size,
                              require_probes=not args.allow_probeless)
    scorers = ["llr", "mu", "field_probe"]
    anchor_cols = []
    if args.anchors:
        anchors = pd.read_csv(args.anchors)
        anchor_cols = [c for c in anchors if c not in {"gene", "wt", "ref_pos", "mut"}]
        before = len(scores)
        scores = scores.merge(anchors, on=["gene", "wt", "ref_pos", "mut"], how="left",
                              validate="one_to_one")
        assert len(scores) == before
        for col in anchor_cols:  # ddG direction in, stability direction out
            scores[col] = -scores[col]
        missing = int(scores[anchor_cols].isna().any(axis=1).sum())
        print(f"anchors joined: {len(anchor_cols)} methods, {missing} rows without anchor values")
        scores = scores.dropna(subset=anchor_cols)
        scorers += anchor_cols
    scores.to_csv(out / "fireprot_scores.csv", index=False)

    check = sign_check(scores)
    print(f"SIGN CHECK PASS: stabilizing LLR mean {check['stabilizing_mean']:.4f} > "
          f"destabilizing {check['destabilizing_mean']:.4f} (k={check['k']})")
    metrics = pooled_metrics(scores, scorers)
    save_json(out / "fireprot_metrics.json", clean(metrics))
    table = pd.DataFrame({k: v for k, v in metrics.items() if not k.startswith("_")}).T
    print(table.round(4).to_string())
    print(f"pooled groups: {metrics['_pooled_groups']}; excluded: {metrics['_excluded_groups']}")
    print(json.dumps({"sign_check": check, "scorers": scorers}, indent=2))


if __name__ == "__main__":
    main()
