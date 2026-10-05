"""Validate actual robust data, all outer folds, context identity and pilot splits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from mipo.common import META_FIELDS, digest, save_json
from mipo.corpus import read_corpus
from mipo.splits import expand_holdouts, fold_task_support, verify_fold


def main():
    root=Path('v4/robust')
    refs=json.loads(Path('v4/resources/sequences.json').read_text())
    build=json.loads((root/'build_report.json').read_text())
    assert build['base_sha256']==digest('v4/data/ndd_training_corpus_v4.csv.gz')
    meta=pd.read_csv(root/'assay_metadata.csv',keep_default_na=False)
    assert not meta.assay_key.duplicated().any()
    assert not meta.duplicated(['protein_reference_id','task']+META_FIELDS).any()
    results={}
    for scope in ['direct','curriculum']:
        path=root/(scope+'.csv.gz'); d=read_corpus(path)
        assert not (d.task=='consensus').any()
        assert not d.measurement_id.duplicated().any()
        assert set(d.assay_key)<=set(meta.assay_key)
        assert set(d.protein_reference_id)<=set(refs)
        assert d.default_sample_weight.gt(0).all()
        for ref,part in d.groupby('protein_reference_id'):
            seq=refs[ref]['sequence']
            assert all(0<p<=len(seq) and seq[p-1]==w for p,w in part[['ref_pos','wt']].drop_duplicates().itertuples(index=False,name=None))
        split=json.loads((root/(scope+'_splits.json')).read_text())
        by_gene={g:{refs[r]['sequence'] for r in rows.protein_reference_id.unique()} for g,rows in d.groupby('gene')}
        for f in split['folds']:
            verify_fold(d[['gene']].drop_duplicates(),f,split['gene_to_group'])
            roles=[set().union(*(by_gene[g] for g in f[role])) for role in ['train','validation','calibration','test']]
            assert all(not roles[i]&roles[j] for i in range(4) for j in range(i))
        expanded=expand_holdouts(path,root/(scope+'_splits.json'),root/(scope+'_expanded_splits.json'),5,3)
        pilot={str(f['fold']):{'test':f['test'],'tasks':fold_task_support(d,f)} for f in expanded['folds'] if set(f['test'])&{'ASPA','KCNQ2','PAX6'}}
        results[scope]={'rows':len(d),'genes':int(d.gene.nunique()),'outer_folds':len(split['folds']),
            'corpus_sha256':digest(path),'all_gene_and_sequence_boundaries_passed':True,'pilot_task_support':pilot}
    results['passed']=True
    save_json(root/'validation.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
