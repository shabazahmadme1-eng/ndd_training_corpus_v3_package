"""List the genes that can test whether predictions separate one mechanism from another.

A gene can only answer that question if the same substitutions were measured under two
different properties, so that a folding-driven loss and a function-only loss look
different in the data. Ranking one assay cannot distinguish the two.

The panel is written in two parts, because they support different claims:

  ndd_evaluation   PTEN and KRAS. NDD-verified under this release's evidence rule, so a
                   result here speaks to the release's own held-out NDD claim.
  mechanism_only   Genes with two measured properties that the evidence rule does not
                   verify as NDD. Holding one out tests mechanism separation and nothing
                   about neurodevelopmental phenotypes; it also removes that gene from
                   training, where it is currently one of the few contrast pairs.

Feed the chosen part to make_splits --test-genes. Read-only with respect to the corpus.
"""
import argparse
from pathlib import Path

import pandas as pd

from mipo.corpus import read_corpus

INDEX = 'v4/data/multi_property_variant_index.csv.gz'
EVALUATION = 'v4/data/ndd_evaluation_genes.csv'


def panel(corpus, index=INDEX, evaluation=EVALUATION):
    d = read_corpus(corpus)
    shared = pd.read_csv(index)
    shared = shared[shared.gene.isin(d.gene)]
    verified = set(pd.read_csv(evaluation).gene)
    assays = d.groupby(['gene', 'task']).assay_key.nunique().unstack(fill_value=0)
    rows = []
    for gene, part in shared.groupby('gene'):
        tasks = sorted({t for entry in part.tasks for t in entry.split('|')})
        rows.append({'gene': gene, 'tasks': '|'.join(tasks), 'shared_variants': part.variant_key.nunique(),
                     'corpus_tasks': '|'.join(sorted(assays.columns[assays.loc[gene] > 0])),
                     'part': 'ndd_evaluation' if gene in verified else 'mechanism_only',
                     'ndd_verified': gene in verified})
    return pd.DataFrame(rows).sort_values(['part', 'shared_variants'], ascending=[True, False])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', default='v4/data/ndd_training_corpus_v4.csv.gz')
    parser.add_argument('--out', default='v4/splits/mechanism_panel.csv')
    args = parser.parse_args()
    result = panel(args.corpus)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    counts = result.part.value_counts()
    print(f"\n{counts.get('ndd_evaluation', 0)} NDD-verified and {counts.get('mechanism_only', 0)} "
          f"mechanism-only genes measure two properties over shared variants.")
    print(f'Written to {args.out}. A mechanism-only gene held out is not an NDD test result.')
