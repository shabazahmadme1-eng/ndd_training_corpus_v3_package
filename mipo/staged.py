"""Helpers for ingesting curated staged MaveDB contrast pairs.

Staged layout (built by mavedb_expansion/stage_genes.py + finalize_staging.py):
  staged/<GENE>/{mapping.json, target_protein.json, <urn>.csv} + staged/TASK_LABELS.json
Only role=='contrast' genes outside STAGED_HOLDS are ingested; labels come from
the curated TASK_LABELS.json, never from discovery heuristics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1

from .common import AA

STAGED_HOLDS = {
    'GPR68': 'CC BY-NC-SA 4.0 license: needs redistribution clearance before ingestion',
    'CD86': '27-aa TM-domain fragment: valid pair but no structure learning; revisit later',
}


def select_staged_genes(staged_dir, holds=None):
    """Return (ingest, skipped) gene records from a staged directory."""
    holds = STAGED_HOLDS if holds is None else holds
    root = Path(staged_dir)
    labels = json.loads((root / 'TASK_LABELS.json').read_text(encoding='utf-8'))
    ingest, skipped = [], []
    for path in sorted(root.glob('*/mapping.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        gene = m['gene']
        if gene in holds:
            skipped.append({'gene': gene, 'reason': holds[gene]})
            continue
        if m.get('role') != 'contrast':
            skipped.append({'gene': gene, 'reason': f"role={m.get('role')}"})
            continue
        sets = []
        for s in m['sets']:
            try:
                task = labels[gene][s['urn']]
            except KeyError:
                raise KeyError(f'{gene} {s["urn"]}: no curated task in TASK_LABELS.json')
            sets.append({'urn': s['urn'], 'task': task,
                         'csv': str(path.parent / (s['urn'].replace(':', '_') + '.csv')),
                         'license': s.get('license', ''), 'title': s.get('title', ''),
                         'published': s.get('published', ''), 'offset': s.get('uniprot_offset')})
        target = json.loads((path.parent / 'target_protein.json').read_text(encoding='utf-8'))
        ingest.append({'gene': gene, 'uniprot': m['uniprot'], 'shared_missense': m['shared_missense'],
                       'sequence': target['sequence'], 'sets': sets})
    return ingest, skipped


def _validates(sequence, frames):
    return all(1 <= p <= len(sequence) and sequence[p - 1] == wt
               for f in frames for p, wt in zip(f.ref_pos, f.wt))


def choose_reference(staged_sequence, canonical_sequence, frames):
    """Reference = the sequence the hgvs numbering validates against.

    Prefer the staged experimental target; fall back to UniProt canonical only
    when the target fails (e.g. Met-truncated DNA translations numbered on the
    full protein). Raise unless exactly the chosen sequence fully validates.
    """
    if _validates(staged_sequence, frames):
        return staged_sequence, 'staged_target'
    if canonical_sequence and _validates(canonical_sequence, frames):
        return canonical_sequence, 'uniprot_canonical_fallback'
    raise ValueError('staged rows validate against neither target nor canonical sequence')


def parse_staged_scores(csv_path):
    """Parse one staged MaveDB scores CSV into missense rows with finite scores."""
    raw = pd.read_csv(csv_path, float_precision='round_trip')
    parts = raw.hgvs_pro.fillna('').str.extract(r'^p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2})$')
    wt = parts[0].fillna('').map(lambda x: seq1(x) if x else '')
    mut = parts[2].fillna('').map(lambda x: seq1(x) if x else '')
    valid = wt.isin(list(AA)) & mut.isin(list(AA)) & (wt != mut)
    finite = pd.to_numeric(raw.score, errors='coerce').map(np.isfinite)
    good = valid & finite
    sigma = pd.to_numeric(raw['se'], errors='coerce') if 'se' in raw else pd.Series(pd.NA, index=raw.index)
    out = pd.DataFrame({'wt': wt[good], 'ref_pos': parts.loc[good, 1].astype(int), 'mut': mut[good],
                        'score_value': raw.loc[good, 'score'], 'source_variant': raw.loc[good, 'hgvs_pro'],
                        'measurement_sigma': sigma.loc[good], '_source_index': raw.index[good]})
    return out.reset_index(drop=True)
