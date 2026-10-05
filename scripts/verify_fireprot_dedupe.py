"""Verify FireProt benchmark sequences are deduped against training constructs (Tier 1.5a gate).

Compares every FireProt benchmark sequence against every v4/resources construct
with local alignment and reports the closest pair per benchmark sequence.
A threshold-crossing overlap (span >= 50, identity >= 0.40) is a FLAG only
when >= 1 benchmark test site falls inside the aligned span on the FireProt
side; overlaps with zero test sites inside are recorded as caveats. Short
k-mer collisions are expected by chance and are neither.

Writes fireprot_eval/dedupe_report.json. Read-only w.r.t. the frozen benchmark.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner

from mipo.common import save_json

ROOT = Path(__file__).resolve().parents[1]


def make_aligner():
    a = PairwiseAligner()
    a.mode = "local"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -2
    a.extend_gap_score = -0.5
    return a


def span_identity(aln, query, target):
    identical, span_q, span_t = 0, 0, 0
    qpos = set()
    for (q0, q1), (t0, t1) in zip(*aln.aligned):
        span_q += q1 - q0
        span_t += t1 - t0
        qpos.update(range(int(q0), int(q1)))
        identical += sum(query[i] == target[j]
                         for i, j in zip(range(q0, q1), range(t0, t1)))
    span = min(span_q, span_t)
    return int(span), float(identical / span if span else 0.), qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fireprot", default=str(ROOT / "fireprot_eval/resources_fireprot/sequences.json"))
    parser.add_argument("--corpus", default=str(ROOT / "v4/resources/sequences.json"))
    parser.add_argument("--out", default=str(ROOT / "fireprot_eval/dedupe_report.json"))
    parser.add_argument("--span", type=int, default=50)
    parser.add_argument("--identity", type=float, default=0.40)
    parser.add_argument("--benchmark-corpus", default=str(ROOT / "fireprot_eval/corpus_fireprot.csv"))
    args = parser.parse_args()
    fp = json.loads(Path(args.fireprot).read_text())
    corpus = json.loads(Path(args.corpus).read_text())
    sites = pd.read_csv(args.benchmark_corpus).groupby("gene").ref_pos.apply(set).to_dict()
    constructs = [(k, v["sequence"]) for k, v in corpus.items()]
    aligner = make_aligner()
    closest, flags, caveats = [], [], []
    for gene in sorted(fp):
        seq = fp[gene]["sequence"]
        best = (0, 0., None, set())
        for key, construct in constructs:
            aln = aligner.align(seq, construct)[0]
            span, ident, qpos = span_identity(aln, seq, construct)
            if (span, ident) > (best[0], best[1]):
                best = (span, ident, key, qpos)
        span, ident, key, qpos = best
        inside = sorted(int(p) for p in sites.get(gene, set()) if p - 1 in qpos)
        row = {"gene": gene, "len": len(seq), "closest_construct": key,
               "span": span, "identity": round(ident, 4),
               "test_sites": len(sites.get(gene, set())), "sites_inside_overlap": inside}
        closest.append(row)
        if span >= args.span and ident >= args.identity:
            (flags if inside else caveats).append(row)
        print(f"{gene}: span={span} id={ident:.3f} vs {key} "
              f"sites_inside={len(inside)}", flush=True)
    closest.sort(key=lambda r: (r["span"], r["identity"]), reverse=True)
    report = {"fireprot_sequences": len(fp), "corpus_constructs": len(constructs),
              "threshold": {"span": args.span, "identity": args.identity},
              "flags": flags, "caveats": caveats, "top10": closest[:10],
              "verdict": "FLAG" if flags else "PASS"}
    save_json(args.out, report)
    print(f"verdict: {report['verdict']} ({len(flags)} flags) -> {args.out}")


if __name__ == "__main__":
    main()
