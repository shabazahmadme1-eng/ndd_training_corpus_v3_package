import json
import numpy as np
import pandas as pd
import pytest

from mipo.smoke import fixture
from mipo.splits import expand_holdouts
from scripts.build_robust_mechanism_panel import shared_pairs
from scripts.evaluate_property_contrasts import contrasts


def example():
    frame = pd.DataFrame({'gene': ['G'] * 8, 'assay_key': ['a'] * 4 + ['b'] * 4,
                          'task': ['abundance'] * 4 + ['activity'] * 4,
                          'variant_key': ['ref:A1C', 'ref:A2C', 'ref:A3C', 'ref:A4C'] * 2,
                          'score_value': [1, 2, 3, 4, 4, 2, 3, 1]})
    frame['mu'] = frame.score_value
    return frame


def test_shared_property_pairs_require_identical_reference_variants():
    data = example()
    assert shared_pairs(data, ['G']).iloc[0].shared_variants == 4
    data.loc[data.assay_key == 'b', 'variant_key'] = ['other:A1C', 'other:A2C', 'other:A3C', 'other:A4C']
    assert shared_pairs(data, ['G']).empty


def test_property_contrast_distinguishes_perfect_and_assay_invariant_predictions():
    data = example()
    pairs = shared_pairs(data, ['G'])
    perfect = contrasts(data, pairs).iloc[0]
    assert perfect.status == 'ok'
    assert perfect.rank_contrast_spearman == pytest.approx(1)
    assert perfect.rank_contrast_mae == 0
    data['mu'] = [1, 2, 3, 4] * 2
    collapsed = contrasts(data, pairs).iloc[0]
    assert collapsed.status == 'undefined_constant_rank_contrast'
    assert np.isnan(collapsed.rank_contrast_spearman)
    assert collapsed.rank_contrast_mae == collapsed.zero_contrast_mae
    assert contrasts(data.iloc[:-1], pairs).iloc[0].status == 'incomplete_matched_predictions'


def test_panel_holdouts_can_use_broad_population_without_changing_outer_test(tmp_path):
    paths = fixture(tmp_path)
    split = json.loads(paths[3].read_text())
    split['folds'] = split['folds'][:1]
    split['evaluation_genes'] = split['folds'][0]['test']
    paths[3].write_text(json.dumps(split))
    with pytest.raises(ValueError, match='insufficient eligible'):
        expand_holdouts(paths[0], paths[3], tmp_path/'no.json', 2, 1)
    split['holdout_eligible_genes'] = sorted(split['gene_to_group'])
    paths[3].write_text(json.dumps(split))
    expanded = expand_holdouts(paths[0], paths[3], tmp_path/'yes.json', 2, 1)
    assert expanded['folds'][0]['test'] == split['folds'][0]['test']
    assert len(expanded['folds'][0]['validation']) == 2
