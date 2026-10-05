"""Rank MaveDB genes that have >=2 score sets of different heuristic task types, and mark what the corpus already holds.
Task labels here are keyword heuristics for DISCOVERY ONLY; the corpus labels come from curated metadata and must be reviewed."""
import json, re
from pathlib import Path
import pandas as pd
H = Path(__file__).parent
sets = json.loads((H/'mavedb_all_scoresets.json').read_text())
corpus = pd.read_csv(H.parent/'v4/robust/curriculum.csv.gz', usecols=['gene', 'assay_key', 'task']).drop_duplicates('assay_key')
have_urn = set()
for k in corpus.assay_key:
    m = re.search(r'urn:mavedb:\d+-\w-\d+', k)
    if m: have_urn.add(m.group(0))
have_gene = set(corpus.gene)

RULES = [  # first match wins; order matters
    ('stability', r'trypsin|chymotrypsin|proteolysis|protease|cdna display|ddg|folding free|thermodynamic stability'),
    ('abundance', r'vamp-?seq|abundance|expression level|protein level|surface expression|steady[- ]state|abundancepca|stability'),
    ('binding', r'binding|interaction|two-hybrid|y2h|affinity|bindingpca|pull-?down|ligand|complex formation'),
    ('activity', r'activity|enzym|catalytic|phosphatase|kinase|electrophys|current|glycosylation|transport|uptake|function'),
    ('fitness', r'fitness|growth|survival|complementation|viability|resistance|sge|proliferation|selection|toxicity|drug'),
]
def task(text):
    t = text.lower()
    for name, pat in RULES:
        if re.search(pat, t):
            return name
    return 'unclassified'

rows = []
for s in sets:
    if not s.get('targetGenes') or s.get('private'):
        continue
    tg = s['targetGenes'][0]
    gene = (tg.get('name') or '').split('_')[0].upper()
    uni = tg.get('uniprotIdFromMappedMetadata')
    ext = [e['identifier']['identifier'] for e in tg.get('externalIdentifiers', []) if e.get('identifier', {}).get('dbName') == 'UniProt']
    uni = uni or (ext[0] if ext else None)
    text = ' '.join(filter(None, [s['title'], s.get('shortDescription'), (s.get('experiment') or {}).get('title')]))
    seq = (tg.get('targetSequence') or {}).get('sequence') or ''
    rows.append(dict(urn=s['urn'], gene=gene, uniprot=uni, title=s['title'], n=s['numVariants'], license=(s.get('license') or {}).get('shortName'),
                     task=task(text), seqlen=len(seq), seqhash=hash(seq) if seq else None, in_corpus=s['urn'] in have_urn,
                     gene_in_corpus=gene in have_gene, taxon=(tg.get('targetSequence') or {}).get('taxonomy', {}).get('commonName')))
df = pd.DataFrame(rows)
df.to_csv(H/'mavedb_scoresets_table.csv', index=False)
print(f'{len(df)} score sets, {df.gene.nunique()} gene labels, {df.uniprot.notna().sum()} with a UniProt id')
print('heuristic task counts:', df.task.value_counts().to_dict())

key = df.uniprot.fillna(df.gene)
df['key'] = key
g = df[df.task != 'unclassified'].groupby('key').agg(gene=('gene', 'first'), tasks=('task', lambda x: sorted(set(x))),
        n_tasks=('task', 'nunique'), n_sets=('urn', 'size'), total_variants=('n', 'sum'),
        in_corpus_sets=('in_corpus', 'sum'), gene_in_corpus=('gene_in_corpus', 'max'))
multi = g[g.n_tasks >= 2].sort_values(['n_tasks', 'total_variants'], ascending=False)
print(f'\ngenes with >=2 heuristic task types: {len(multi)}  (already in corpus: {int(multi.gene_in_corpus.sum())})')
multi.to_csv(H/'multitask_gene_candidates.csv')
print(multi.head(40).to_string())
