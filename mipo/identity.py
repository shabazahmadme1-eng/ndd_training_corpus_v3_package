"""Report each held-out test gene's sequence identity to its nearest training gene.

Per-gene leave-one-out scores are only meaningful if the held-out gene is not a near
duplicate of a training gene: high train-test identity leaks the answer and inflates the
apparent score. This diagnostic reads the same sequences.json the features are built from,
computes each test gene's maximum identity to any train gene in its fold, and flags folds
above a threshold so a reviewer can discount them. It never looks at any target value.
"""
import json
from pathlib import Path

from .common import save_json
from .corpus import read_corpus


def _identity_biopython(a, b, aligner):
    aln = aligner.align(a, b)[0]
    identical = 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        identical += sum(a[i] == b[j] for i, j in zip(range(a0, a1), range(b0, b1)))
    return identical / min(len(a), len(b))


def _make_scorer():
    """Global-alignment identity via Biopython when available, else a difflib ratio."""
    try:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score, aligner.mismatch_score = 1, 0
        aligner.open_gap_score, aligner.extend_gap_score = -1, -0.5

        def score(a, b):
            # Guard the O(n*m) aligner against pathologically long pairs.
            if len(a) * len(b) > 6_000_000:
                from difflib import SequenceMatcher
                return SequenceMatcher(None, a, b, autojunk=False).ratio()
            return _identity_biopython(a, b, aligner)
        return score, "biopython_global_identity_over_min_length"
    except Exception:
        from difflib import SequenceMatcher

        def score(a, b):
            return SequenceMatcher(None, a, b, autojunk=False).ratio()
        return score, "difflib_ratio_fallback"


def gene_sequences(corpus, resources):
    """Map each corpus gene to the sequences of its protein reference ids."""
    d = read_corpus(corpus)
    records = json.loads((Path(resources) / "sequences.json").read_text())
    by_gene = {}
    for gene, ref in d[["gene", "protein_reference_id"]].drop_duplicates().itertuples(index=False):
        seq = records.get(ref, {}).get("sequence")
        if seq:
            by_gene.setdefault(gene, []).append(seq)
    return by_gene


def nearest_train_identity(corpus, resources, folds, flag_threshold=0.9):
    """For every test gene in every fold, its max identity to any train gene."""
    seqs = gene_sequences(corpus, resources)
    score, method = _make_scorer()
    report = {"method": method, "flag_threshold": flag_threshold, "missing_sequences": [], "folds": {}}
    for fold in folds:
        rows = []
        train = [(g, s) for g in fold["train"] for s in seqs.get(g, [])]
        for gene in fold["test"]:
            targets = seqs.get(gene)
            if not targets:
                report["missing_sequences"].append(gene)
                continue
            best_gene, best_id = None, 0.
            for train_gene, train_seq in train:
                identity = max(score(t, train_seq) for t in targets)
                if identity > best_id:
                    best_gene, best_id = train_gene, identity
            rows.append({"test_gene": gene, "nearest_train_gene": best_gene,
                         "max_identity": round(best_id, 4), "leakage_flag": best_id >= flag_threshold})
        report["folds"][str(fold["fold"])] = sorted(rows, key=lambda r: -r["max_identity"])
    report["flagged"] = sorted({r["test_gene"] for rows in report["folds"].values()
                                for r in rows if r["leakage_flag"]})
    return report


def write_identity_report(corpus, resources, splits_path, flag_threshold=0.9, test_genes=None):
    folds = json.loads(Path(splits_path).read_text())["folds"]
    if test_genes:
        folds = [f for f in folds if set(f['test']) & set(test_genes)]
    report = nearest_train_identity(corpus, resources, folds, flag_threshold)
    out = Path(splits_path).with_suffix(".identity.json")
    save_json(out, report)
    return out, report
