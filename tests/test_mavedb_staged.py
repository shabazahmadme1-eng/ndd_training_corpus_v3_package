"""Focused tests for staged MaveDB ingestion helpers (mipo.staged)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from mipo.staged import choose_reference, parse_staged_scores, select_staged_genes


def _mapping(gene, role, sets):
    return {'gene': gene, 'uniprot': 'P00001', 'role': role, 'shared_missense': 1,
            'sets': [{'urn': u, 'heuristic_task': t} for u, t in sets]}


def _staged(tmp_path):
    root = tmp_path / 'staged'
    (root / 'G1').mkdir(parents=True)
    (root / 'G1' / 'mapping.json').write_text(json.dumps(
        _mapping('G1', 'contrast', [('urn:A', 'abundance'), ('urn:B', 'activity')])))
    (root / 'G1' / 'target_protein.json').write_text(json.dumps({'sequence': 'ACDEFG'}))
    (root / 'G2').mkdir(parents=True)
    (root / 'G2' / 'mapping.json').write_text(json.dumps(
        _mapping('G2', 'contrast', [('urn:C', 'abundance')])))
    (root / 'G3').mkdir(parents=True)
    (root / 'G3' / 'mapping.json').write_text(json.dumps(
        _mapping('G3', 'eval-only', [('urn:D', 'fitness')])))
    (root / 'TASK_LABELS.json').write_text(json.dumps(
        {'G1': {'urn:A': 'abundance', 'urn:B': 'fitness'}}))
    return root


def test_select_ingests_only_contrast_non_held(tmp_path):
    root = _staged(tmp_path)
    ingest, skipped = select_staged_genes(root, holds={'G2': 'test hold'})
    assert [g['gene'] for g in ingest] == ['G1']
    assert [(s['urn'], s['task']) for s in ingest[0]['sets']] == [('urn:A', 'abundance'), ('urn:B', 'fitness')]
    assert all(Path(s['csv']).parent.name == 'G1' for s in ingest[0]['sets'])
    assert {s['gene']: s['reason'] for s in skipped} == {'G2': 'test hold', 'G3': 'role=eval-only'}


def test_select_requires_curated_labels(tmp_path):
    root = _staged(tmp_path)
    (root / 'TASK_LABELS.json').write_text(json.dumps({'G1': {'urn:A': 'abundance'}}))
    with pytest.raises(KeyError):
        select_staged_genes(root, holds={})


def test_parse_keeps_only_finite_missense(tmp_path):
    csv = tmp_path / 'scores.csv'
    pd.DataFrame([
        {'hgvs_pro': 'p.Ala2Gly', 'score': 0.5, 'se': 0.1},     # keep
        {'hgvs_pro': 'p.Ala2Ter', 'score': 0.5, 'se': 0.1},     # nonsense -> drop
        {'hgvs_pro': 'p.Gly3Gly', 'score': 0.5, 'se': 0.1},     # synonymous -> drop
        {'hgvs_pro': 'p.Cys4Asp', 'score': float('nan'), 'se': 0.1},  # nonfinite -> drop
        {'hgvs_pro': '', 'score': 0.5, 'se': 0.1},              # non-protein -> drop
    ]).to_csv(csv, index=False)
    f = parse_staged_scores(csv)
    assert len(f) == 1
    row = f.iloc[0]
    assert (row.wt, row.ref_pos, row.mut, row.score_value, row.measurement_sigma) == ('A', 2, 'G', 0.5, 0.1)
    assert row.source_variant == 'p.Ala2Gly'


def test_choose_reference_prefers_staged_then_canonical():
    rows = pd.DataFrame({'wt': ['A', 'C'], 'ref_pos': [1, 2]})
    assert choose_reference('ACDE', 'XACDE', [rows]) == ('ACDE', 'staged_target')
    assert choose_reference('CDE', 'ACDE', [rows]) == ('ACDE', 'uniprot_canonical_fallback')
    with pytest.raises(ValueError):
        choose_reference('XXXX', 'YYYY', [rows])


def test_parse_sigma_optional(tmp_path):
    csv = tmp_path / 'scores.csv'
    pd.DataFrame([{'hgvs_pro': 'p.Ala2Gly', 'score': 0.5}]).to_csv(csv, index=False)
    f = parse_staged_scores(csv)
    assert len(f) == 1 and pd.isna(f.measurement_sigma.iloc[0])
