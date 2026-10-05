"""List new references and assayed-position counts for the Colab feature pack."""
import json

import pandas as pd

s = json.load(open('v4/resources/sequences.json'))
new_refs = {rid: r for rid, r in s.items() if r.get('source', '').startswith('MaveDB staged')}
print('new refs:', len(new_refs))
d = pd.read_csv('v4/data/ndd_training_corpus_v4.csv.gz',
                usecols=['gene', 'protein_reference_id', 'ref_pos', 'source'], low_memory=False)
st = d[d.source == 'MaveDB_staged_contrast']
total = 0
for rid, r in sorted(new_refs.items(), key=lambda x: x[1]['gene']):
    npos = int(st[st.protein_reference_id == rid].ref_pos.nunique())
    total += npos
    print(r['gene'], rid, 'len=' + str(r['length']), 'assayed_positions=' + str(npos))
print('reused refs, new-assay positions:')
for g in ['CCR5', 'NUDT15', 'PRKN']:
    sub = st[st.gene == g]
    npos = int(sub.ref_pos.nunique())
    total += npos
    print(g, sub.protein_reference_id.iloc[0], 'new_positions=' + str(npos))
print('total LLR positions (upper bound):', total)
