"""Validate staged ingestion: new-corpus rows vs staged counts and vs pre-build backup."""
import glob
import json

import pandas as pd

new = pd.read_csv('v4/data/ndd_training_corpus_v4.csv.gz', low_memory=False)
old = pd.read_csv('v4_pre_staged_backup/data/ndd_training_corpus_v4.csv.gz', low_memory=False)
print(f'rows: {len(old)} -> {len(new)} (+{len(new) - len(old)})')
print(f'genes: {old.gene.nunique()} -> {new.gene.nunique()} | assays: {old.assay_key.nunique()} -> {new.assay_key.nunique()}')

# No old assay may lose rows or change.
a_old = old.groupby('assay_key').size()
a_new = new.groupby('assay_key').size()
missing = set(a_old.index) - set(a_new.index)
changed = {k: (a_old[k], a_new[k]) for k in a_old.index if k in a_new.index and a_old[k] != a_new[k]}
print('old assays missing:', missing or 'NONE')
print('old assays changed:', changed or 'NONE')

st = new[new.source == 'MaveDB_staged_contrast']
print(f'\nstaged rows: {len(st)} | genes: {sorted(st.gene.unique())}')
print('tiers:', st.supervision_tier.value_counts().to_dict())
print('tasks:', st.task.value_counts().to_dict())

ok = True
for f in sorted(glob.glob('mavedb_expansion/staged/*/mapping.json')):
    m = json.load(open(f, encoding='utf-8'))
    g = st[st.gene == m['gene']]
    if m['gene'] in ('GPR68', 'CD86', 'TP53'):
        status = 'HELD/EXCLUDED' if len(g) == 0 else 'LEAKED!'
        ok &= len(g) == 0
        print(f"  {m['gene']:6s} {status}")
        continue
    for s in m['sets']:
        n = len(g[g.assay_id == s['urn']])
        exp = s['mapped_missense']
        # parsed rows exclude WT-mismatches already counted in mapped; allow exact match only
        flag = 'OK' if n == exp else f'MISMATCH(staged mapped={exp})'
        ok &= n == exp
        print(f"  {m['gene']:6s} {s['urn'].split(':')[-1]}: built={n} {flag}")
    # shared recount on built rows
    a, b = (set(zip(g[g.assay_id == s['urn']].wt, g[g.assay_id == s['urn']].ref_pos,
                    g[g.assay_id == s['urn']].mut)) for s in m['sets'])
    sh = len(a & b)
    flag = 'OK' if sh == m['shared_missense'] else f"MISMATCH(staged={m['shared_missense']})"
    ok &= sh == m['shared_missense']
    print(f"  {m['gene']:6s} shared={sh} {flag}")
print('\nRECOUNT:', 'PASS' if ok else 'FAIL')
