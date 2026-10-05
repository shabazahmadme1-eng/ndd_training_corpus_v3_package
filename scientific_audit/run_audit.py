"""Read-only scientific checks of the extracted release and supplementary results."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive'
sys.path.insert(0, str(ARCHIVE))
from mipo.common import META_FIELDS
from mipo.corpus import read_corpus
from mipo.data import FeatureStore, Metadata, TargetTransform, VariantDataset, collate
from mipo.model import MIPO
from mipo.smoke import fixture
import torch

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

report = {'archive_sha256': sha(ROOT.parent/'dist/mipo_ndd_v4_colab_upload.zip')}
cols = ['gene','protein_reference_id','variant_key','measurement_id','assay_key','task',
        'score_value','measurement_sigma','is_direct_ndd','source','ref_pos','wt',
        'possible_consensus_constituent_overlap','default_sample_weight']
d = pd.read_csv(ARCHIVE/'v4/data/ndd_training_corpus_v4.csv.gz', usecols=cols, low_memory=False)
refs = json.loads((ARCHIVE/'v4/resources/sequences.json').read_text())
ndd = d[d.is_direct_ndd == 1]
report['data'] = {'rows':len(d),'genes':d.gene.nunique(),'references':d.protein_reference_id.nunique(),
 'direct_rows':len(ndd),'direct_genes':ndd.gene.nunique(),
 'duplicate_measurements':int(d.measurement_id.duplicated().sum()),
 'duplicate_assay_variants':int(d.duplicated(['assay_key','variant_key']).sum()),
 'nonfinite_scores':int((~np.isfinite(d.score_value)).sum()),
 'measurement_sigma_nonmissing':int(d.measurement_sigma.notna().sum()),
 'consensus_overlap_rows':int(d.possible_consensus_constituent_overlap.sum()),
 'direct_task_counts':ndd.groupby('task').agg(rows=('gene','size'),genes=('gene','nunique')).to_dict('index'),
 'direct_nonconsensus_multi_task_genes':int((ndd[ndd.task!='consensus'].groupby('gene').task.nunique()>1).sum()),
 'direct_stability_only_genes':int(ndd.groupby('gene').task.agg(lambda s:set(s)=={'stability'}).sum())}
mismatches = 0
for ref, g in d.groupby('protein_reference_id'):
    seq = refs[ref]['sequence']
    mismatches += sum(not (0 < p <= len(seq) and seq[p-1]==w) for p,w in g[['ref_pos','wt']].drop_duplicates().itertuples(index=False,name=None))
report['data']['wt_reference_mismatches'] = mismatches
report['splits'] = {}
for name, frame in [('direct',ndd),('curriculum',d)]:
    s=json.loads((ARCHIVE/f'v4/splits/{name}.json').read_text())
    genes=set(frame.gene)
    seqs={g:{refs[r]['sequence'] for r in f.protein_reference_id.unique()} for g,f in frame.groupby('gene')}
    failures=[]
    for fold in s['folds']:
        roles=[set(fold[r]) for r in ['train','validation','calibration','test']]
        rs=[set().union(*(seqs[g] for g in role)) for role in roles]
        if set.union(*roles)!=genes or any(roles[i]&roles[j] or rs[i]&rs[j] for i in range(4) for j in range(i)):
            failures.append(fold['fold'])
    report['splits'][name]={'folds':len(s['folds']),'failed_isolation_folds':failures}

meta=pd.read_csv(ARCHIVE/'v4/data/assay_metadata.csv',keep_default_na=False)
keys=['protein_reference_id','task']+list(META_FIELDS)
collisions=[]
for _,g in meta.groupby(keys):
    if len(g)<2: continue
    a=d[d.assay_key.isin(g.assay_key)]
    piv=a.pivot(index='variant_key',columns='assay_key',values='score_value')
    assays=list(piv)
    for i in range(len(assays)):
        for j in range(i):
            pair=piv[[assays[i],assays[j]]].dropna()
            if len(pair): collisions.append({'assays':[assays[i],assays[j]],'shared_variants':len(pair),
                'spearman':float(spearmanr(pair.iloc[:,0],pair.iloc[:,1]).statistic),
                'mean_absolute_score_difference':float((pair.iloc[:,0]-pair.iloc[:,1]).abs().mean())})
report['indistinguishable_assay_pairs']=collisions

conformers=[]
for p in (ARCHIVE/'v4/resources/structures').glob('*.npz'):
    with np.load(p) as f: conformers.append(f['coords'].shape[0])
report['structure_conformer_counts']=pd.Series(conformers).value_counts().to_dict()

torch.set_num_threads(1)
torch.manual_seed(42)
with tempfile.TemporaryDirectory() as tmp:
    paths=fixture(tmp); data=read_corpus(paths[0]); cfg=json.loads(paths[-1].read_text())
    store=FeatureStore(paths[1],paths[2],allow_synthetic=True)
    gene=data.iloc[0].protein_reference_id
    p=store.protein(gene); orig=p['coords'].copy(); p['confidence'][:]=0.1
    ds=VariantDataset(data,store,TargetTransform().fit(data),Metadata().fit(data),cfg)
    model=MIPO(32,1,cfg).eval()
    with torch.no_grad():
        b=collate([ds[0]]); y1=model(b); p['coords']=orig*10; ds.graph.cache_clear()
        b2=collate([ds[0]]); y2=model(b2)
    report['low_confidence_geometry_probe']={'all_vector_weights_zero':bool((b['edge_weight']==0).all() and (b2['edge_weight']==0).all()),
        'absolute_mu_change':float((y1['mu']-y2['mu']).abs().max()),
        'max_field_change':float((y1['field']-y2['field']).abs().max())}
    cfg={**cfg,'structure':False}
    ds=VariantDataset(data,store,TargetTransform().fit(data),Metadata().fit(data),cfg)
    model=MIPO(32,1,cfg).eval()
    with torch.no_grad():
        b=collate([ds[0]]); y1=model(b); p['coords']=orig;ds.graph.cache_clear();y2=model(collate([ds[0]]))
    report['no_structure_probe']={'absolute_mu_change':float((y1['mu']-y2['mu']).abs().max())}
    ds.graph.cache_clear()
    for name in ['esm','llr']:
        if hasattr(p[name], '_mmap'): p[name]._mmap.close()
    store.protein.cache_clear()

report['results']={}
coverage_checks=[]
for root in sorted((ROOT/'results').iterdir()):
    scope='curriculum' if 'curriculum' in root.name else 'direct'
    runs=[]; tables=[]; checks=[]
    for p in root.glob('runs/*/seed_*/fold_*/provenance.json'):
        prov=json.loads(p.read_text()); run=p.parent
        status=json.loads((run/'training_status.json').read_text()) if (run/'training_status.json').exists() else {}
        history=json.loads((run/'history.json').read_text()) if (run/'history.json').exists() else []
        best=[h['stage'] for h in history if h['epoch']==status.get('best_epoch')]
        runs.append({'model':p.parents[2].name,'seed':p.parent.parent.name,'test_genes':prov['fold']['test'],
          'complete':(run/'test_metrics.csv').exists(),'best_epoch':status.get('best_epoch'),'best_stage':best,
          'epochs_run':status.get('epochs_run'),'code_mismatches':[n for n,h in prov['code_sha256'].items() if not (ARCHIVE/'mipo'/n).exists() or sha(ARCHIVE/'mipo'/n)!=h],
          'corpus_hash_matches':prov['corpus_sha256']==sha(ARCHIVE/'v4/data'/('ndd_direct_v4.csv.gz' if scope=='direct' else 'ndd_training_corpus_v4.csv.gz'))})
        if not (run/'test_metrics.csv').exists():continue
        t=pd.read_csv(run/'test_metrics.csv');t['model']=p.parents[2].name;t['seed']=p.parent.parent.name;tables.append(t)
        pred=pd.read_csv(run/'test_predictions.csv')
        for (gene,assay,task),a in pred.groupby(['gene','assay_key','task']):
            saved=t[t.assay_key==assay].iloc[0]
            rho=spearmanr(a.y,a.mu).statistic
            checks.append(abs(rho-saved.spearman))
            raw=np.mean(np.abs(a.y-a.mu)<=1.644854*a.sigma)
            coverage_checks.append(abs(raw-saved.coverage_90_gaussian))
            available=a.interval_low.notna() & a.interval_high.notna()
            if available.any():
                selected=a[available]
                cov=np.mean((selected.y>=selected.interval_low)&(selected.y<=selected.interval_high))
                coverage_checks.append(abs(cov-saved.coverage_calibrated))
    t=pd.concat(tables); t.to_csv(ROOT/f'{scope}_assay_metrics.csv',index=False)
    g=t.groupby(['model','seed']).spearman.mean().groupby('model').agg(['mean','std','count'])
    g.to_csv(ROOT/f'{scope}_model_summary.csv')
    independent=t[t.task!='consensus'].groupby(['model','seed']).spearman.mean().groupby('model').agg(['mean','std','count'])
    report['results'][scope]={'runs':runs,'mean_seed_spearman':g.to_dict('index'),
      'nonconsensus_mean_seed_spearman':independent.to_dict('index'),
      'max_recomputed_spearman_difference':max(checks),
      'task_metrics':t.groupby(['model','task'])[['spearman','coverage_calibrated','coverage_90_gaussian']].mean().reset_index().replace({np.nan:None}).to_dict('records')}

report['max_recomputed_coverage_difference']=max(coverage_checks)

def native(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
(ROOT/'audit_evidence.json').write_text(json.dumps(report,indent=2,default=native),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2,default=native))
print('Full results saved to audit_evidence.json')
