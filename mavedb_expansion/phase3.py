"""Map every candidate score set onto UniProt coordinates and count shared missense substitutions between sets of different task."""
import io, itertools, json, re, time, urllib.request, urllib.error
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
H = Path(__file__).parent
(H/'scores').mkdir(exist_ok=True); (H/'uniprot').mkdir(exist_ok=True)
v = pd.read_pickle(H/'catalogue_v4.pkl'); cand = pd.read_pickle(H/'cand_genes.pkl')
use = v[v.up2.isin(cand.index) & (v.task_h != 'unclassified') & (v.n >= 100)].copy()
print(len(use), 'sets,', use.up2.nunique(), 'genes', flush=True)

A3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
MIS = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
CODON = dict(zip((a+b+c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
                 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))

def get(url, accept, timeout, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': accept}), timeout=timeout).read()
        except (urllib.error.URLError, TimeoutError):
            time.sleep(4*(k+1))
    return None

def scores(urn):
    p = H/'scores'/(urn.replace(':', '_')+'.csv')
    if not p.exists():
        raw = get(f'https://api.mavedb.org/api/v1/score-sets/{urn}/scores', 'text/csv', 120)
        if raw is None:
            return None
        p.write_bytes(raw)
    return pd.read_csv(p, usecols=lambda c: c in ('hgvs_pro', 'score'))

def uniprot_seq(acc):
    p = H/'uniprot'/f'{acc}.fasta'
    if not p.exists():
        raw = get(f'https://rest.uniprot.org/uniprotkb/{acc}.fasta', 'text/plain', 60)
        if raw is None:
            return None
        p.write_bytes(raw)
    return ''.join(p.read_text().splitlines()[1:])

def protein_of(seq, seqtype):
    if not isinstance(seq, str):
        return None
    seq = seq.upper()
    if seqtype == 'dna':
        return ''.join(CODON.get(seq[i:i+3], 'X') for i in range(0, len(seq)-2, 3)).split('*')[0]
    return seq

def offset(target, ref):
    """UniProt position = target position + offset, from the longest common block; None if the match is weak."""
    m = SequenceMatcher(None, ref, target, autojunk=False).find_longest_match(0, len(ref), 0, len(target))
    return (m.a - m.b, m.size) if m.size >= min(25, 0.4*len(target)) else (None, m.size)

rows, report = [], []
for acc, grp in use.groupby('up2'):
    ref = uniprot_seq(acc)
    if ref is None:
        print('  no UniProt sequence for', acc, flush=True); continue
    mapped = {}
    for r in grp.itertuples():
        d = scores(r.urn)
        if d is None or 'hgvs_pro' not in d:
            continue
        tgt = protein_of(r.seq, r.seqtype)
        off, block = offset(tgt, ref) if tgt else (None, 0)
        subs, ok, bad = {}, 0, 0
        if off is not None:
            for h, s in zip(d.hgvs_pro, d.score):
                m = MIS.match(h) if isinstance(h, str) else None
                if m is None or pd.isna(s) or m.group(1) not in A3 or m.group(3) not in A3:
                    continue
                pos = int(m.group(2)) + off
                if 1 <= pos <= len(ref) and ref[pos-1] == A3[m.group(1)]:
                    subs[(pos, A3[m.group(3)])] = float(s); ok += 1
                else:
                    bad += 1
        mapped[r.urn] = (r, subs)
        report.append(dict(uniprot=acc, urn=r.urn, task=r.task_h, title=r.title[:60], listed=r.n, mapped=len(subs), wt_mismatch=bad,
                           offset=off, block=block, span=(min((p for p, _ in subs), default=None), max((p for p, _ in subs), default=None))))
    print(f'{acc:8s} {grp.target.iloc[0][:30]:30s} sets={len(mapped)} mapped_variants={sum(len(s) for _, s in mapped.values())}', flush=True)
    for (ua, (ra, a)), (ub, (rb, b)) in itertools.combinations(mapped.items(), 2):
        if ra.task_h == rb.task_h or not a or not b:
            continue
        sh = set(a) & set(b)
        if len(sh) >= 8:
            rows.append(dict(uniprot=acc, gene=ra.target, task_a=ra.task_h, task_b=rb.task_h, shared=len(sh), n_a=len(a), n_b=len(b),
                             urn_a=ua, urn_b=ub, title_a=ra.title[:70], title_b=rb.title[:70], license_a=ra.license, license_b=rb.license,
                             in_corpus_a=ra.in_corpus if hasattr(ra, 'in_corpus') else None))
pairs = pd.DataFrame(rows).sort_values('shared', ascending=False)
pairs.to_csv(H/'phase3_pairs.csv', index=False)
pd.DataFrame(report).to_csv(H/'phase3_mapping_report.csv', index=False)
best = pairs.groupby('uniprot').agg(gene=('gene', 'first'), best_shared=('shared', 'max'), pairs=('shared', 'size'),
                                    tasks=('task_a', lambda x: '/'.join(sorted(set(x)))))
print('\nDONE phase3: genes with >=1 pair sharing >=8 substitutions:', len(best), flush=True)
print(best.sort_values('best_shared', ascending=False).to_string())
