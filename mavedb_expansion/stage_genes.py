"""Stage priority contrast genes: freeze score CSVs, UniProt mapping, and proposed labels.

Reads phase3_pairs.csv (best pair per gene), phase3_mapping_report.csv (offsets),
and mavedb_all_scoresets.json (licenses, experiment methods). Writes
mavedb_expansion/staged/<GENE>/{scores...,mapping.json,PROPOSED_LABELS.md}.
Corpus untouched.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

H = Path(__file__).parent
PRIORITY = {  # uniprot -> (symbol, role)
    'P29597': ('TYK2', 'contrast'), 'P11413': ('G6PD', 'contrast'),
    'P01130': ('LDLR', 'contrast'), 'P06213': ('INSR', 'contrast'),
    'Q15743': ('GPR68', 'contrast'), 'O60260': ('PRKN', 'contrast'),
    'P00441': ('SOD1', 'contrast'), 'P61073': ('CXCR4', 'contrast'),
    'P51681': ('CCR5', 'contrast'), 'Q9NV35': ('NUDT15', 'contrast'),
    'P40238': ('MPL', 'contrast'), 'P42081': ('CD86', 'contrast'),
    'P04637': ('TP53', 'eval-only'),
}

pairs = pd.read_csv(H / 'phase3_pairs.csv')
rep = pd.read_csv(H / 'phase3_mapping_report.csv')
sets = {s['urn']: s for s in json.loads((H / 'mavedb_all_scoresets.json').read_text())}


def fname(urn):
    return urn.replace(':', '_') + '.csv'


for acc, (sym, role) in PRIORITY.items():
    g = pairs[pairs.uniprot == acc].sort_values('shared', ascending=False)
    if g.empty:
        print(f'{sym}: NO PAIRS - skipped')
        continue
    best = g.iloc[0]
    d = H / 'staged' / sym
    d.mkdir(parents=True, exist_ok=True)
    urns = [best.urn_a, best.urn_b]
    info = []
    for u in urns:
        shutil.copyfile(H / 'scores' / fname(u), d / fname(u))
        r = rep[(rep.uniprot == acc) & (rep.urn == u)]
        r = r.iloc[0].to_dict() if not r.empty else {}
        s = sets[u]
        ex = s.get('experiment') or {}
        info.append({
            'urn': u, 'title': s['title'], 'license': (s.get('license') or {}).get('shortName'),
            'n_variants': int(s['numVariants']),
            'heuristic_task': best.task_a if u == best.urn_a else best.task_b,
            'mapped_missense': int(r.get('mapped', -1)),
            'wt_mismatch': int(r.get('wt_mismatch', -1)),
            'uniprot_offset': (None if pd.isna(r.get('offset')) else int(r['offset'])),
            'target_len': len(((s['targetGenes'][0].get('targetSequence')) or {}).get('sequence') or ''),
            'experiment_urn': ex.get('urn'), 'experiment_title': ex.get('title'),
            'experiment_method': ((ex.get('methodText') or '')[:800] + ' ' + (ex.get('abstractText') or '')[:400]).strip(),
            'published': s.get('publishedDate'),
        })
    (d / 'mapping.json').write_text(json.dumps({
        'gene': sym, 'uniprot': acc, 'role': role, 'shared_missense': int(best.shared),
        'sets': info,
    }, indent=1), encoding='utf-8')
    lines = [f'# {sym} ({acc}) - role: {role}', '',
               f'Best pair shares **{int(best.shared)}** WT-verified missense substitutions on UniProt coordinates.', '']
    for i, inf in enumerate(info, 1):
        lines += [f'## Assay {i}: proposed task `{inf["heuristic_task"]}` (REVIEW REQUIRED)',
                  f'- URN: {inf["urn"]} | license: {inf["license"]} | published: {inf["published"]}',
                  f'- Title: {inf["title"]}',
                  f'- Experiment: {inf["experiment_title"]} ({inf["experiment_urn"]})',
                  f'- Variants: {inf["n_variants"]} listed, {inf["mapped_missense"]} mapped missense, '
                  f'{inf["wt_mismatch"]} WT mismatches, offset {inf["uniprot_offset"]}, target len {inf["target_len"]}',
                  f'- Method: {inf["experiment_method"] or "(none in catalogue)"}', '']
    (d / 'PROPOSED_LABELS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{sym}: staged shared={int(best.shared)}')
