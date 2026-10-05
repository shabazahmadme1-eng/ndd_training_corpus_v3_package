"""Convert MMseqs reference clusters to whole-gene connected components.

First run: mmseqs easy-cluster proteins.fasta cluster_result tmp_mmseqs
             --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1
Coverage/identity cutoffs are experimental choices, not a guarantee of no homology.
"""
import argparse
import pandas as pd

def gene_components(members, references=None):
    if members.gene.duplicated().any():
        raise ValueError('Each sequence member must have exactly one cluster')
    if references is None:
        if members.gene.str.contains('__').any():
            raise ValueError('V4 reference identifiers require --references')
        mapping = dict(zip(members.gene, members.gene))
    else:
        if references.protein_reference_id.duplicated().any():
            raise ValueError('Duplicate protein reference mapping')
        mapping = references.set_index('protein_reference_id').gene.to_dict()
        if set(members.gene) != set(mapping):
            raise ValueError('Cluster members must cover exactly all supplied references')
    parent = {g:g for g in mapping.values()}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for _, group in members.groupby('cluster'):
        genes = sorted({mapping[r] for r in group.gene})
        for g in genes[1:]:
            a,b = find(genes[0]),find(g)
            parent[max(a,b)] = min(a,b)
    return pd.DataFrame({'gene':sorted(parent), 'cluster':[find(g) for g in sorted(parent)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsv', required=True)
    parser.add_argument('--references', help='V4 protein_references.csv; unions all constructs of each gene')
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    members = pd.read_csv(a.tsv, sep='\t', names=['cluster','gene'], dtype=str)
    refs = pd.read_csv(a.references, dtype=str) if a.references else None
    gene_components(members, refs).to_csv(a.out,index=False)
