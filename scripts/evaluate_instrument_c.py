"""Run instrument C on trained checkpoints: the Tier 2 confirmation gate.

Measures the chain-preserved scramble margin (clean minus scrambled
macro_gene_spearman) on each checkpoint's own test split, over several corruption
seeds. Gate: >= 0.05. A structure-blind run (esm_mlp, or structure=False) is
scored too and must come back at ~0; that is the instrument's negative control,
and a nonzero margin there means the instrument is measuring something else.

Provenance is checked exactly as mipo.train.evaluate does, so a margin can never
be computed against a corpus, metadata, feature cache or structure set other than
the one the checkpoint was trained on.

Usage:
  python scripts/evaluate_instrument_c.py --runs <pilot>/runs --corpus ... \
      --resources ... --cache ... --metadata ... --out <dir> [--seeds 0,1,2,3,4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mipo.common import digest, save_json
from mipo.corpus import read_corpus
from mipo.data import FeatureStore, Metadata, TargetTransform
from mipo.instrument_c import GATE, run_instrument_c, structure_blind
from mipo.train import feature_fingerprints, loader, restore


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def check_provenance(provenance, corpus, resources, metadata_path, store, rows):
    """Same contract as train.evaluate: refuse to score against shifted inputs."""
    if digest(corpus) != provenance["corpus_sha256"]:
        raise ValueError("Corpus differs from training provenance")
    if (digest(metadata_path) if metadata_path else None) != provenance["metadata_sha256"]:
        raise ValueError("Metadata differs from training metadata")
    if store.manifest != provenance["feature_manifest"]:
        raise ValueError("Feature manifest differs from training")
    if digest(Path(resources) / "sequences.json") != provenance["sequence_sha256"]:
        raise ValueError("Sequence reference differs from training")
    if feature_fingerprints(store, rows.protein_reference_id.unique()) != provenance["feature_files"]:
        raise ValueError("Feature/structure file contents changed since training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", help="Directory scanned recursively for best.pt")
    source.add_argument("--checkpoint", help="A single best.pt")
    for key in ["corpus", "resources", "cache", "out"]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--metadata", help="TRAINING assay_metadata.csv (frozen vocab); omit only "
                        "for a run trained without one, as train.evaluate allows")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Permit smoke features; never for a scientific number")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated scramble seeds")
    parser.add_argument("--radius", type=int, default=8,
                        help="Backbone window kept intact in chain-preserved mode")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = ([Path(args.checkpoint)] if args.checkpoint
                   else sorted(Path(args.runs).rglob("best.pt")))
    if not checkpoints:
        raise SystemExit(f"No best.pt under {args.runs}")
    print(f"{len(checkpoints)} checkpoint(s); scramble seeds {seeds}; gate {GATE}", flush=True)

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    rows = read_corpus(args.corpus)
    store = FeatureStore(args.resources, args.cache, args.allow_synthetic)
    verdicts = []
    for checkpoint in checkpoints:
        model, state = restore(str(checkpoint), device)
        provenance = state["provenance"]
        check_provenance(provenance, args.corpus, args.resources, args.metadata, store, rows)
        config = dict(provenance["config"])
        if args.batch_size:
            config["batch_size"] = args.batch_size
        fold = provenance["fold"]
        test = rows[rows.gene.isin(fold["test"])]
        if test.empty:
            raise ValueError(f"{checkpoint}: no test rows for this fold")
        batches = loader(test, store, TargetTransform(state["transform"]),
                         Metadata(args.metadata, state["metadata_vocab"]), config)
        blind = structure_blind(config)
        tag = "_".join(checkpoint.parent.relative_to(
            Path(args.runs) if args.runs else checkpoint.parent.parent).parts) or checkpoint.parent.name
        print(f"\n=== {tag} | model={config.get('model', 'mipo')} "
              f"structure={config.get('structure', True)} | test genes {sorted(fold['test'])} "
              f"| {len(test)} rows{' | NEGATIVE CONTROL' if blind else ''}", flush=True)

        summary, per_seed, per_assay = run_instrument_c(model, batches, device, seeds, args.radius)
        summary.update({"checkpoint": str(checkpoint), "tag": tag,
                        "model": config.get("model", "mipo"),
                        "structure": config.get("structure", True),
                        "structure_blind_negative_control": blind,
                        "test_genes": sorted(fold["test"])})
        run_out = out / tag
        run_out.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(run_out / "instrument_c_per_seed.csv", index=False)
        per_assay.to_csv(run_out / "instrument_c_per_assay.csv", index=False)
        save_json(run_out / "instrument_c.json", clean_json(summary))

        chain, full = summary["chain_preserved"], summary["full"]
        ablated = summary["edge_ablated"]
        print(f"  clean macro_gene_spearman {summary['clean_macro_gene_spearman']:.4f}")
        for name, block in [("chain-preserved (GATE)", chain), ("full scramble", full),
                            ("edge-ablated ceiling", ablated)]:
            mean = block["margin_mean"]
            print(f"  {name:<22} margin {mean if mean is None else round(mean, 4)} "
                  f"(min {block['margin_min'] if block['margin_min'] is None else round(block['margin_min'], 4)}, "
                  f"max {block['margin_max'] if block['margin_max'] is None else round(block['margin_max'], 4)})  "
                  f"rewired {block['rewire_success_rate']:.2f}  "
                  f"assays degraded {block['assays_degraded']}/{block['assays_compared']}  "
                  f"|dmu| {block['mu_shift_mean_abs']:.4f}")
        print(f"  {summary['gate_statement']}")
        ceiling = ablated["margin_mean"]
        if not blind and ceiling is not None and abs(ceiling) < 0.005:
            print("  NOTE: silencing every edge barely moves the metric, so this run "
                  "does not use geometry at all. The gate is UNINFORMATIVE here rather "
                  "than failed: no scramble could have produced a margin.", flush=True)
        if blind and chain["margin_mean"] is not None and abs(chain["margin_mean"]) > 0.005:
            print("  WARNING: structure-blind control moved; the instrument is not "
                  "acting through geometry alone. Investigate before reading any gate.", flush=True)
        if (full["margin_mean"] is not None and chain["margin_mean"] is not None
                and not blind and full["margin_mean"] < chain["margin_mean"]):
            print("  WARNING: full scramble margin below chain-preserved; see the "
                  "slot-permutation caveat in instruments.scramble_batch.", flush=True)
        verdicts.append(summary)

    table = pd.DataFrame([{
        "tag": v["tag"], "model": v["model"], "control": v["structure_blind_negative_control"],
        "clean": round(v["clean_macro_gene_spearman"], 4),
        "chain_margin": None if v["chain_preserved"]["margin_mean"] is None
        else round(v["chain_preserved"]["margin_mean"], 4),
        "chain_std": None if v["chain_preserved"]["margin_std"] is None
        else round(v["chain_preserved"]["margin_std"], 4),
        "full_margin": None if v["full"]["margin_mean"] is None
        else round(v["full"]["margin_mean"], 4),
        "ablated_ceiling": None if v["edge_ablated"]["margin_mean"] is None
        else round(v["edge_ablated"]["margin_mean"], 4),
        "gate_passed": v["gate_passed"]} for v in verdicts])
    table.to_csv(out / "instrument_c_summary.csv", index=False)
    save_json(out / "instrument_c_summary.json", clean_json(verdicts))
    print("\n" + table.to_string(index=False))

    graded = [v for v in verdicts if not v["structure_blind_negative_control"]]
    passed = [v["tag"] for v in graded if v["gate_passed"]]
    print(f"\nGATE: {len(passed)}/{len(graded)} structure-aware checkpoint(s) reach "
          f"margin >= {GATE}" + (f": {passed}" if passed else ""))
    print(json.dumps({"out": str(out), "seeds": list(seeds), "gate": GATE}, indent=2))


if __name__ == "__main__":
    main()
