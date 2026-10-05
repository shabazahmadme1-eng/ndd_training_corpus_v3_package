"""Better task labels from title + description + abstract + methods. Discovery heuristic: every kept pair is reviewed by hand before ingestion."""
import json, re
from pathlib import Path
import pandas as pd
H = Path(__file__).parent
sets = json.loads((H/'mavedb_all_scoresets.json').read_text())

# (task, strong patterns). Evidence is counted per task over the score set's own text, experiment text second.
RULES = {
 'stability': [r'proteolysis', r'cdna display', r'trypsin', r'chymotrypsin', r'thermolysin', r'megascale', r'folding free energ', r'\bddg\b', r'thermodynamic stability'],
 'abundance': [r'vamp-?seq', r'abundance', r'abundancepca', r'surface (expression|presentation|level)', r'protein (level|expression|stability)', r'cell[- ]surface',
               r'steady[- ]state', r'expression level', r'dhfr-?pca', r'fluorescen\w+ (protein )?(tag|reporter)', r'gfp[- ]fusion', r'degradation', r'protein stability'],
 'binding':   [r'binding', r'bindingpca', r'two-hybrid', r'\by2h\b', r'affinity', r'interaction', r'pull-?down', r'complex formation', r'\bppi\b', r'ligand', r'antibody', r'nanobody', r'peptide'],
 'activity':  [r'enzym\w+ activity', r'catalytic', r'phosphatase', r'kinase activity', r'electrophysiolog', r'current density', r'glycosylation', r'transport', r'uptake', r'\bactivity\b',
               r'activation', r'signal(l)?ing', r'reporter assay', r'luciferase', r'ubiquitin', r'substrate', r'functional (assay|activity)', r'channel'],
 'fitness':   [r'fitness', r'growth', r'survival', r'complementation', r'viability', r'resistance', r'saturation genome editing', r'\bsge\b', r'proliferation', r'toxicity',
               r'drug', r'selection', r'haploid', r'hap1', r'competition', r'lethal'],
}
def counts(text):
    t = text.lower()
    return {k: sum(len(re.findall(p, t)) for p in ps) for k, ps in RULES.items()}

def label(s):
    primary = ' '.join(filter(None, [s['title'], s.get('shortDescription'), (s.get('experiment') or {}).get('title')]))
    longer = ' '.join(filter(None, [s.get('abstractText'), s.get('methodText'), (s.get('experiment') or {}).get('abstractText')]))
    c1, c2 = counts(primary), counts(longer[:6000])
    score = {k: 3*c1[k] + c2[k] for k in RULES}
    best = max(score, key=score.get)
    if score[best] == 0:
        return 'unclassified', 0
    # stability protease assays are decisive when present
    if c1['stability'] or c2['stability'] >= 2:
        return 'stability', score['stability']
    return best, score[best]

rows = []
for s in sets:
    if not s.get('targetGenes') or s.get('private'):
        continue
    tg = s['targetGenes'][0]
    ts = tg.get('targetSequence') or {}
    ext = [e['identifier']['identifier'] for e in tg.get('externalIdentifiers', []) if e.get('identifier', {}).get('dbName') == 'UniProt']
    uni = tg.get('uniprotIdFromMappedMetadata') or (ext[0] if ext else None)
    task, ev = label(s)
    rows.append(dict(urn=s['urn'], target=tg.get('name'), uniprot=uni, title=s['title'], n=s['numVariants'], license=(s.get('license') or {}).get('shortName'),
                     task=task, evidence=ev, seq=ts.get('sequence'), seqtype=ts.get('sequenceType'), taxon=(ts.get('taxonomy') or {}).get('commonName'),
                     published=s.get('publishedDate'), pubs=';'.join(p.get('identifier', '') for p in (s.get('primaryPublicationIdentifiers') or []))))
df = pd.DataFrame(rows)
df['n'] = pd.to_numeric(df.n, errors='coerce')
df.to_pickle(H/'catalogue_v2.pkl')
print(len(df), 'public score sets;', df.uniprot.notna().sum(), 'with UniProt;', df.seq.notna().sum(), 'with sequence')
print(df.task.value_counts().to_dict())
print('protein sequences:', (df.seqtype == 'protein').sum(), '| DNA targets:', (df.seqtype == 'dna').sum())
g = df[df.uniprot.notna() & (df.task != 'unclassified')].groupby('uniprot').agg(tasks=('task', lambda x: sorted(set(x))), sets=('urn', 'size'), nvar=('n', 'sum'))
g['k'] = g.tasks.apply(len)
print('UniProt-keyed genes with >=2 task types:', (g.k >= 2).sum(), '| >=3:', (g.k >= 3).sum())
print('sets belonging to those genes:', int(g[g.k >= 2].sets.sum()))
