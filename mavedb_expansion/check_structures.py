"""Check staged target sequences against AlphaFold chains: exact match or subsequence crop."""
import glob
import json
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

H = Path('mavedb_expansion')
v = pd.read_pickle(H / 'catalogue_v4.pkl')
CODON = dict(zip((a + b + c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
                 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def protein_of(seq, seqtype):
    seq = seq.upper()
    if seqtype == 'dna':
        return ''.join(CODON.get(seq[i:i + 3], 'X') for i in range(0, len(seq) - 2, 3)).split('*')[0]
    return seq


parser = PDBParser(QUIET=True)
for f in sorted(glob.glob(str(H / 'staged/*/mapping.json'))):
    m = json.load(open(f, encoding='utf-8'))
    acc = m['uniprot']
    chain = parser.get_structure(acc, str(H / 'staged_structures' / f'{acc}.pdb'))
    residues = [r for r in next(chain.get_chains()) if r.id[0] == ' ' and 'CA' in r]
    afseq = ''.join(seq1(r.resname) for r in residues)
    verdicts = []
    for s in m['sets']:
        r = v[v.urn == s['urn']].iloc[0]
        tgt = protein_of(r.seq, r.seqtype)
        if tgt == afseq:
            verdicts.append('EXACT')
        elif afseq.count(tgt) == 1:
            verdicts.append(f'CROP@{afseq.index(tgt)}')
        else:
            verdicts.append(f'MISMATCH(tgt{len(tgt)}/af{len(afseq)}/count{afseq.count(tgt)})')
    print(f"{m['gene']:6s} af={len(afseq)} {verdicts} {m['sets'][0]['urn']} / {m['sets'][1]['urn'].split(':')[-1]}")
