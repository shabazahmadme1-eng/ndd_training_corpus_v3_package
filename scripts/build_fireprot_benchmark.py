"""Build the eval-only FireProt benchmark tree: corpus + sequences + structures.

Reads external_data/fireprot_benchmark/ (frozen 2511-row homologue-free split).
Writes a standalone tree the eval script consumes; nothing here enters v4/ or
any training corpus. Replicate variants are mean-averaged (reported, with max
spread); structures come from the listed PDBs or fall back to sequence-only.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mipo.common import save_json, seq_key
from mipo.corpus import read_corpus
from mipo.fireprot import build_structure_arrays

ROOT = Path(__file__).resolve().parents[1]


def gene_name(pdb_id):
    return "FP_" + str(pdb_id).replace("|", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", default=str(ROOT / "external_data/fireprot_benchmark/fireprot_benchmark.csv"))
    parser.add_argument("--pdbs", default=str(ROOT / "external_data/fireprot_benchmark/pdbs"))
    parser.add_argument("--out", default=str(ROOT / "fireprot_eval"))
    parser.add_argument("--exclude-pdbs", default="",
                        help="Comma-separated pdb_id values to drop before building (dedupe gate)")
    args = parser.parse_args()
    out = Path(args.out)
    resources = out / "resources_fireprot"
    structures = resources / "structures"
    structures.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.benchmark_csv)
    n_in = len(d)
    excluded = [p for p in args.exclude_pdbs.split(",") if p]
    if excluded:
        missing = [p for p in excluded if p not in set(d.pdb_id)]
        if missing:
            raise ValueError(f"exclude-pdbs not in benchmark input: {missing}")
        d = d[~d.pdb_id.isin(excluded)].copy()
    groups = d.drop_duplicates("pdb_id").set_index("pdb_id")
    sequences, struct_report, corpus_rows = {}, {}, []
    n_averaged, max_spread = 0, 0.
    for pdb_id, seq, chain, uni in groups[["sequence", "chain", "uniprot_id"]].itertuples():
        gene = gene_name(pdb_id)
        key = seq_key(seq)
        sequences[gene] = {"gene": gene, "protein_reference_id": gene, "sequence": seq, "key": key,
                           "length": len(seq), "accession": uni, "pdb_id": pdb_id, "chain": chain,
                           "source": "FireProtDB homologue-free benchmark; EVAL ONLY, never train"}
        rows = d[d.pdb_id == pdb_id]
        for (wt, pos, mut), part in rows.groupby(["wild_type", "position", "mutation"]):
            if seq[int(pos) - 1] != wt:
                raise ValueError(f"{gene} {wt}{pos}: WT/sequence mismatch")
            ddg = part.ddG.to_numpy(float)
            if len(part) > 1:
                n_averaged += 1
                max_spread = max(max_spread, float(ddg.max() - ddg.min()))
            corpus_rows.append({"gene": gene, "protein_reference_id": gene, "wt": wt, "ref_pos": int(pos),
                                "mut": mut, "score_value": float(ddg.mean()), "score_kind": "ddG",
                                "assay_id": "fireprot_ddg", "assay_type": "stability",
                                "supervision_tier": "benchmark", "pdb_id": pdb_id,
                                "pdb_position": int(part.pdb_position.iloc[0]),
                                "n_replicates": len(part), "ddg_spread": float(ddg.max() - ddg.min())})
        pdb_paths = [Path(args.pdbs) / f"{p}.pdb" for p in str(pdb_id).split("|")]
        missing = [str(p) for p in pdb_paths if not p.exists()]
        if missing:
            raise ValueError(f"{gene}: PDB files missing: {missing}")
        coords, conf, source, attempts = build_structure_arrays(pdb_paths, chain, seq)
        sites = sorted({int(p) for p in rows.position})
        if coords is None:
            status = {"gene": gene, "status": "sequence_only", "attempts": attempts,
                      "mutated_sites": len(sites)}
        else:
            np.savez_compressed(structures / f"{key}.npz", coords=coords, confidence=conf,
                                sequence=np.array(seq))
            uncovered = [p for p in sites if not conf[:, p - 1].any()]
            status = {"gene": gene, "status": "structure", "source_pdb": source,
                      "conformers": int(coords.shape[0]), "covered": int((conf[0] > 0).sum()),
                      "seq_len": len(seq), "mutated_sites": len(sites),
                      "uncovered_mutated_sites": uncovered, "attempts": attempts}
        struct_report[gene] = status
        print(f"{gene}: {len(rows)} rows, structure={status['status']}", flush=True)

    save_json(resources / "sequences.json", sequences)
    save_json(structures / "provenance.json", struct_report)
    corpus = pd.DataFrame(corpus_rows)
    corpus_path = out / "corpus_fireprot.csv"
    corpus.to_csv(corpus_path, index=False)
    checked = read_corpus(corpus_path)  # duplicate/variant-key validation must pass here
    design = ROOT / "external_data/TIER15_DESIGN.md"
    if design.exists():
        shutil.copy2(design, out / "TIER15_DESIGN.md")
    report = {"input_rows": n_in, "excluded_pdbs": excluded,
              "excluded_input_rows": int(n_in - len(d)),
              "eval_rows": len(checked), "genes": checked.gene.nunique(),
              "variants_averaged": n_averaged, "max_replicate_spread": max_spread,
              "with_structure": sum(1 for s in struct_report.values() if s["status"] == "structure"),
              "sequence_only": [g for g, s in struct_report.items() if s["status"] == "sequence_only"],
              "sites_lacking_coords": {g: s["uncovered_mutated_sites"] for g, s in struct_report.items()
                                       if s.get("uncovered_mutated_sites")}}
    save_json(out / "build_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
