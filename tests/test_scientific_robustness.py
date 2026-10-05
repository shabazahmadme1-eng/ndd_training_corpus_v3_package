import json
import numpy as np
import pandas as pd
import pytest
import torch
from mipo.common import META_FIELDS
from mipo.corpus import read_corpus
from mipo.data import FeatureStore, Metadata, TargetTransform, VariantDataset, collate
from mipo.metrics import metrics, calibration_quantiles
from mipo.model import MIPO
from mipo.smoke import fixture
from mipo.train import selection_score
from scripts.mmseqs_clusters import gene_components
from scripts.build_robust_release_data import parse_scores


def test_unreliable_geometry_cannot_change_selection_topology_or_prediction(tmp_path):
    torch.set_num_threads(1)
    paths=fixture(tmp_path)
    d=read_corpus(paths[0]);cfg=json.loads(paths[-1].read_text())
    store=FeatureStore(paths[1],paths[2],allow_synthetic=True)
    protein=store.protein(d.iloc[0].protein_reference_id)
    protein['confidence'][:]=.1
    ds=VariantDataset(d,store,TargetTransform().fit(d),Metadata().fit(d),cfg)
    model=MIPO(32,1,cfg).eval()
    first=collate([ds[0]])
    protein['coords']=np.random.default_rng(21).normal(size=protein['coords'].shape).astype(np.float32)*100
    ds.graph.cache_clear();second=collate([ds[0]])
    for key in ['ids','edge','edge_features','edge_weight','unit']:
        assert torch.equal(first[key],second[key]),key
    with torch.no_grad():
        assert torch.equal(model(first)['mu'],model(second)['mu'])


def test_condition_encoder_separates_cellular_backgrounds(tmp_path):
    rows=pd.DataFrame({'assay_key':['a','b']})
    rows['genetic_background']=['wild_type','null']
    for f in META_FIELDS:
        if f not in rows:rows[f]='unknown'
    path=tmp_path/'metadata.csv';rows.to_csv(path,index=False)
    m=Metadata(path).fit(rows)
    assert not np.array_equal(m.encode('a'),m.encode('b'))
    assert not m.encode('never_seen').any()


def test_undefined_assay_cannot_hide_inside_defined_gene():
    p=pd.DataFrame({'gene':['a']*6,'assay_key':['x']*3+['y']*3,'task':['activity']*6,
                    'y':[0,1,2]*2,'mu':[0,1,2]+[0,0,0],'sigma':[1.]*6,'task_seen_in_training':[True]*6})
    _,summary=metrics(p)
    assert summary['macro_gene_spearman'] is None
    assert summary['undefined_assays']==['y']
    with pytest.raises(ValueError):selection_score(summary,1)


def test_calibration_needs_independent_genes_not_just_many_variants():
    p=pd.DataFrame({'gene':['a']*99,'task':['activity']*99,'y':np.arange(99),'mu':np.arange(99),
                    'sigma':[1.]*99,'task_seen_in_training':[True]*99})
    assert not calibration_quantiles(p)['tasks']
    p['gene']=['a','b','c']*33
    assert calibration_quantiles(p)['tasks']['activity']['genes']==3


def test_reference_clusters_union_all_constructs_of_a_gene():
    refs=pd.DataFrame({'protein_reference_id':['A__1','A__2','B__1','C__1','D__1'],
                       'gene':['A','A','B','C','D']})
    members=pd.DataFrame({'gene':refs.protein_reference_id,'cluster':['x','y','x','y','z']})
    groups=gene_components(members,refs).set_index('gene').cluster
    assert groups.A==groups.B==groups.C
    assert groups.A!=groups.D
    with pytest.raises(ValueError,match='require --references'):gene_components(members)


def test_source_parser_alignment_exclusions_and_codon_aggregation():
    raw=pd.DataFrame({'hgvs_pro':['p.Ala1Gly','p.Ala1Gly','p.Cys2Asp','p.Cys2Ter','p.Ala1Val'],
                      'score':[1.,3.,4.,9.,np.nan]})
    result,accounting=parse_scores(raw,'AC')
    assert result.score_value.tolist()==[2.,4.]
    assert result.source_encoding_count.tolist()==[2,1]
    assert accounting['excluded_nonmissense_or_nonfinite']==2
    with pytest.raises(ValueError,match='WT'):parse_scores(raw,'CC')
