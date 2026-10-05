"""Finalize staging: curated TASK_LABELS.json, frozen target proteins, UniProt snapshot rows.

Labels start from staged heuristic tasks; overrides record the 2026-10-03 sign-off.
UniProt rows come from fetched FASTAs + API entry names (grounded, not invented).
"""
import glob
import json
import urllib.request
from pathlib import Path

import pandas as pd

H = Path('mavedb_expansion')
STAGED = H / 'staged'

# (gene, urn) -> curated task. Everything else keeps its staged heuristic task.
OVERRIDES = {
    ('G6PD', 'urn:mavedb:00001266-c-1'): 'fitness',   # yeast growth, not enzymatic activity
    ('MPL', 'urn:mavedb:00001214-i-1'): 'activity',   # growth-based activation readout, not binding
}

CODON = dict(zip((a + b + c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
                 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def protein_of(seq, seqtype):
    seq = seq.upper()
    if seqtype == 'dna':
        return ''.join(CODON.get(seq[i:i + 3], 'X') for i in range(0, len(seq) - 2, 3)).split('*')[0]
    return seq.rstrip('*')


v = pd.read_pickle(H / 'catalogue_v4.pkl')
labels = {}
for f in sorted(glob.glob(str(STAGED / '*/mapping.json'))):
    m = json.load(open(f, encoding='utf-8'))
    gene = m['gene']
    labels[gene] = {}
    for s in m['sets']:
        task = OVERRIDES.get((gene, s['urn']), s['heuristic_task'])
        labels[gene][s['urn']] = task
    # one target file per gene (both sets share the construct; assert it)
    seqs = {}
    for s in m['sets']:
        r = v[v.urn == s['urn']].iloc[0]
        seqs[s['urn']] = protein_of(r.seq, r.seqtype)
    uniq = set(seqs.values())
    print(f"{gene}: {len(uniq)} distinct target(s), lens {[len(x) for x in uniq]}")
    (STAGED / gene / 'target_protein.json').write_text(
        json.dumps({'sequence': max(uniq, key=len), 'per_urn': {u: len(x) for u, x in seqs.items()},
                    'identical': len(uniq) == 1}, indent=1), encoding='utf-8')
(STAGED / 'TASK_LABELS.json').write_text(json.dumps(labels, indent=1), encoding='utf-8')
print('wrote TASK_LABELS.json for', len(labels), 'genes')

# UniProt snapshot rows for accessions missing from v4/sources/uniprot_sequences.tsv
tsv = Path('v4/sources/uniprot_sequences.tsv')
have = set(pd.read_csv(tsv, sep='\t', keep_default_na=False).Entry)
need = {json.load(open(f, encoding='utf-8'))['uniprot'] for f in glob.glob(str(STAGED / '*/mapping.json'))} - have
print('missing accessions:', sorted(need))
rows = []
for acc in sorted(need):
    seq = ''.join((H / 'uniprot' / f'{acc}.fasta').read_text().splitlines()[1:])
    meta = json.loads(urllib.request.urlopen(
        urllib.request.Request(f'https://rest.uniprot.org/uniprotkb/{acc}.json',
                               headers={'Accept': 'application/json'}), timeout=60).read())
    gene = meta.get('genes', [{}])[0].get('geneName', {}).get('value', '')
    rows.append({'Entry': acc, 'Entry Name': meta['uniProtkbId'],
                 'Gene Names (primary)': gene, 'Organism': 'Homo sapiens (Human)', 'Sequence': seq})
    print(f"  {acc} {meta['uniProtkbId']} {gene} len={len(seq)}")
with open(tsv, 'a', encoding='utf-8') as fh:
    for r in rows:
        fh.write('\t'.join([r['Entry'], r['Entry Name'], r['Gene Names (primary)'], r['Organism'], r['Sequence']]) + '\n')
print('appended', len(rows), 'rows to uniprot_sequences.tsv')
