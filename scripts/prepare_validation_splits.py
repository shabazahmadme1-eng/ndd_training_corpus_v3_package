"""Expand shipped splits while preserving their test genes and cluster boundaries."""
import argparse
from pathlib import Path

from mipo.common import save_json
from mipo.corpus import read_corpus
from mipo.identity import write_identity_report
from mipo.splits import expand_holdouts, fold_task_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "source", "out"]:
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--validation-groups", type=int, default=5)
    parser.add_argument("--calibration-groups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-test-genes", help="Comma-separated test genes whose task coverage should be saved")
    parser.add_argument("--resources", help="Resources dir with sequences.json; if set, writes a train-test sequence-identity leakage report")
    parser.add_argument("--identity-threshold", type=float, default=0.9, help="Flag a test gene whose nearest train gene is at least this identical")
    args = vars(parser.parse_args())
    report_genes = args.pop("report_test_genes")
    resources = args.pop("resources")
    threshold = args.pop("identity_threshold")
    result = expand_holdouts(**args)
    if report_genes:
        targets = set(report_genes.split(","))
        data = read_corpus(args["corpus"])
        support = {str(fold["fold"]): fold_task_support(data, fold) for fold in result["folds"]
                   if targets.intersection(fold["test"])}
        save_json(Path(args["out"]).with_suffix(".support.json"), support)
    if resources:
        out, report = write_identity_report(args["corpus"], resources, args["out"], threshold,
                                            report_genes.split(',') if report_genes else None)
        print(f"Sequence-identity report ({report['method']}) -> {out}; flagged genes: {report['flagged']}")
    print(f"Prepared {len(result['folds'])} folds: {result['holdout_expansion']}")


if __name__ == "__main__":
    main()
