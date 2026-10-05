"""For each candidate gene, download its score sets and count missense substitutions shared between sets of different heuristic task."""
import io, itertools, re, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
H = Path(__file__).parent
(H/'scores').mkdir(exist_ok=True)
df = pd.read_csv(H/'mavedb_scoresets_table.csv')
df['key'] = df.uniprot.fillna(df.gene)
df['n'] = pd.to_numeric(df.n, errors='coerce')
cand = pd.read_csv(H/'phase2_candidates.csv').key.tolist()
use = df[df.key.isin(cand) & (df.task != 'unclassified') & (df.task != 'stability') & (df.n >= 200)]
print(len(use), 'score sets for', use.key.nunique(), 'genes', flush=True)
MIS = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')

def fetch(urn, tries=3):
    path = H/'scores'/(urn.replace(':', '_')+'.csv')
    if path.exists():
        return pd.read_csv(path)
    for k in range(tries):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(f'https://api.mavedb.org/api/v1/score-sets/{urn}/scores',
                                         headers={'Accept': 'text/csv'}), timeout=100).read()
            path.write_bytes(raw)
            return pd.read_csv(io.BytesIO(raw))
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(5*(k+1))
    return None

def missense(d):
    col = 'hgvs_pro' if 'hgvs_pro' in d else None
    if col is None or 'score' not in d:
        return {}
    out = {}
    for h, s in zip(d[col], d['score']):
        if isinstance(h, str) and pd.notna(s) and MIS.match(h):
            out[h] = float(s)         # one value per substitution (multi-codon duplicates collapse)
    return out

rows = []
for key, grp in use.groupby('key'):
    sets = {}
    for r in grp.itertuples():
        d = fetch(r.urn)
        sets[r.urn] = (r, missense(d) if d is not None else {})
        print(f'  {r.gene[:28]:28s} {r.urn:26s} {r.task:9s} listed {r.n:>6.0f} missense {len(sets[r.urn][1]):>6}', flush=True)
    for (ua, (ra, a)), (ub, (rb, b)) in itertools.combinations(sets.items(), 2):
        if ra.task == rb.task:
            continue
        shared = set(a) & set(b)
        rows.append(dict(key=key, gene=ra.gene, urn_a=ua, task_a=ra.task, n_a=len(a), urn_b=ub, task_b=rb.task, n_b=len(b), shared=len(shared),
                         same_seqlen=ra.seqlen == rb.seqlen, seqlen_a=ra.seqlen, seqlen_b=rb.seqlen, license_a=ra.license, license_b=rb.license,
                         title_a=ra.title[:60], title_b=rb.title[:60]))
out = pd.DataFrame(rows).sort_values('shared', ascending=False)
out.to_csv(H/'phase2_pairs.csv', index=False)
best = out.groupby('gene').agg(best_shared=('shared', 'max'), pairs=('shared', 'size'), same_seqlen=('same_seqlen', 'max')).sort_values('best_shared', ascending=False)
print('\nBest shared missense per gene:')
print(best.to_string())
