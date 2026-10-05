"""Restore the fixed ten-gene panel using measured data and current split groups."""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from mipo.common import digest, save_json
from mipo.corpus import read_corpus
from mipo.splits import make_splits, expand_holdouts, verify_fold, fold_task_support

PANEL = ['PTEN', 'KRAS', 'SLC22A1', 'GCK', 'KCNJ2', 'CYP2C9', 'SRC', 'HLA-A', 'KCNE1', 'VKORC1']


def shared_pairs(data, genes):
    records = []
    for gene in genes:
        frame = data[(data.gene == gene) & (data.task != 'consensus')]
        assays = [(key, part) for key, part in frame.groupby('assay_key')]
        for (a, left), (b, right) in combinations(assays, 2):
            ta, tb = left.task.iloc[0], right.task.iloc[0]
            if ta == tb:
                continue
            count = len(set(left.variant_key) & set(right.variant_key))
            if count >= 3:
                records.append(dict(gene=gene, assay_a=a, assay_b=b, task_a=ta, task_b=tb,
                                    shared_variants=count, abundance_anchored='abundance' in [ta, tb]))
    return pd.DataFrame(records)


def main():
    root = Path('v4/robust')
    corpus = root/'curriculum.csv.gz'
    data = read_corpus(corpus)
    pairs = shared_pairs(data, PANEL)
    if set(pairs.gene) != set(PANEL):
        raise ValueError('A requested panel gene lacks matched multi-property observations')
    pairs.to_csv(root/'mechanism_assay_pairs.csv', index=False)
    verified = set(data.loc[pd.to_numeric(data.is_direct_ndd) == 1, 'gene'])
    panel = pd.DataFrame({'gene': PANEL})
    panel['part'] = panel.gene.map(lambda g: 'ndd_evaluation' if g in verified else 'mechanism_only')
    panel['previously_examined'] = panel.gene.eq('PTEN')
    panel['abundance_anchored'] = panel.gene.map(pairs.groupby('gene').abundance_anchored.any())
    panel.to_csv(root/'mechanism_panel.csv', index=False)
    # Draw validation/calibration from all eligible genes, not just ten test targets.
    # Restrict the outer fold list only after constructing valid corpus-wide groups.
    result = make_splits(corpus, root/'mechanism_splits.json',
                         clusters=root/'curriculum_groups.csv')
    result['folds'] = [f for f in result['folds'] if set(f['test']) & set(PANEL)]
    for i, fold in enumerate(result['folds']):
        fold['fold'] = i
    result['holdout_eligible_genes'] = sorted(data.gene.unique())
    result['evaluation_genes'] = sorted(PANEL)
    result['panel_policy'] = 'fixed_legacy_ten_genes_measured_rows_broad_holdout_eligibility'
    save_json(root/'mechanism_splits.json', result)
    expanded = expand_holdouts(corpus, root/'mechanism_splits.json',
                              root/'mechanism_expanded_splits.json', 5, 3)
    refs = json.loads(Path('v4/resources/sequences.json').read_text())
    sequences = {g: {refs[r]['sequence'] for r in rows.protein_reference_id.unique()}
                 for g, rows in data.groupby('gene')}
    for split in [result, expanded]:
        for fold in split['folds']:
            verify_fold(data[['gene']].drop_duplicates(), fold, split['gene_to_group'])
            roles = [set().union(*(sequences[g] for g in fold[role]))
                     for role in ['train', 'validation', 'calibration', 'test']]
            if any(roles[i] & roles[j] for i in range(4) for j in range(i)):
                raise ValueError('Identical reference sequence crosses split roles')
    report = {'passed': True, 'corpus': str(corpus), 'corpus_sha256': digest(corpus),
              'source_splits_sha256': digest(root/'mechanism_splits.json'),
              'panel_genes': PANEL, 'outer_folds': len(result['folds']),
              'matched_cross_task_assay_pairs': len(pairs),
              'all_gene_and_sequence_boundaries_passed': True,
              'holdout_eligibility': 'all measured corpus genes; fixed test panel',
              'task_support': {str(f['fold']): {'test': f['test'], 'tasks': fold_task_support(data, f)}
                               for f in expanded['folds']}}
    save_json(root/'mechanism_validation.json', report)
    print(panel.to_string(index=False))
    print(f'Validated {len(pairs)} paired assays and {len(result["folds"])} outer folds.')


if __name__ == '__main__':
    main()
