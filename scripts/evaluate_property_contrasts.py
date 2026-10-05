"""Descriptive held-out paired-assay rank contrasts, not causal mechanism labels.

Rank each assay over exactly the shared substitutions, then compare the difference
between its two observed rank vectors with the difference in predicted ranks.
Ranks use test observations only for reporting; they never enter model selection.
Score directions remain as deposited. This is not a function-loss classifier.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def contrasts(predictions, pairs):
    records = []
    for pair in pairs.itertuples(index=False):
        subset = predictions[predictions.gene == pair.gene]
        if subset.empty:
            continue
        left = subset[subset.assay_key == pair.assay_a][['variant_key', 'score_value', 'mu']]
        right = subset[subset.assay_key == pair.assay_b][['variant_key', 'score_value', 'mu']]
        joint = left.merge(right, on='variant_key', suffixes=('_a', '_b'), validate='one_to_one')
        record = dict(gene=pair.gene, assay_a=pair.assay_a, assay_b=pair.assay_b,
                      task_a=pair.task_a, task_b=pair.task_b,
                      abundance_anchored=pair.abundance_anchored, n_shared=len(joint),
                      expected_shared=int(pair.shared_variants), rank_contrast_spearman=np.nan,
                      rank_contrast_mae=np.nan, zero_contrast_mae=np.nan)
        if len(joint) != pair.shared_variants:
            record['status'] = 'incomplete_matched_predictions'
        elif len(joint) < 3 or not np.isfinite(joint.drop(columns='variant_key').to_numpy()).all():
            record['status'] = 'insufficient_or_nonfinite_predictions'
        else:
            ranks = joint.drop(columns='variant_key').rank(method='average', pct=True)
            observed = ranks.score_value_a - ranks.score_value_b
            predicted = ranks.mu_a - ranks.mu_b
            record['rank_contrast_mae'] = float((observed - predicted).abs().mean())
            record['zero_contrast_mae'] = float(observed.abs().mean())
            if observed.nunique() < 2 or predicted.nunique() < 2:
                record['status'] = 'undefined_constant_rank_contrast'
            else:
                record['rank_contrast_spearman'] = float(spearmanr(observed, predicted).statistic)
                record['status'] = 'ok'
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--pairs', default='v4/robust/mechanism_assay_pairs.csv')
    parser.add_argument('--panel', default='v4/robust/mechanism_panel.csv')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    pairs, panel = pd.read_csv(args.pairs), pd.read_csv(args.panel)
    tables = []
    for path in sorted(Path(args.runs).glob('*/seed_*/fold_*/test_predictions.csv')):
        table = contrasts(pd.read_csv(path), pairs)
        if table.empty:
            continue
        table['model'] = path.parents[2].name
        table['seed'] = int(path.parents[1].name.split('_')[1])
        table['fold'] = int(path.parent.name.split('_')[1])
        tables.append(table)
    if not tables:
        raise ValueError('No matched panel predictions found')
    result = pd.concat(tables, ignore_index=True).merge(panel[['gene', 'part', 'previously_examined']],
                                                       on='gene', validate='many_to_one')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print('Report each gene/pair and seed. Undefined contrasts remain visible. Abundance is a proxy;'
          ' non-NDD and previously examined genes are labelled separately. No causal folding claim.')


if __name__ == '__main__':
    main()
