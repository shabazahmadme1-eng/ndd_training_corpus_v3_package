"""Reproducible measured-assay views; preserve the original master corpus.

Only explicitly reviewed MaveDB sources below are imported. Source files, licenses,
reference sequences, exclusions and aggregation counts are retained for audit.
No clinical labels or inferred enzyme-activity predictions enter model tensors.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from mipo.common import AA, META_FIELDS, digest, save_json
from mipo.corpus import read_corpus
from mipo.splits import make_splits

SOURCE = Path('v4/sources/functional_expansion')
OUT = Path('v4/robust')
THREE = dict(zip(['Ala','Cys','Asp','Glu','Phe','Gly','His','Ile','Lys','Leu','Met','Asn','Pro','Gln','Arg','Ser','Thr','Val','Trp','Tyr'], AA))

# task, host, system, selection, treatment, cell line, background, partner, quantity, expression
SPECS = {
 '00000657-a-1': ('ASPA','abundance','human_cell_line','VAMP_seq','abundance','not_applicable','HEK293T','landing_pad','not_applicable','cellular_abundance','unknown'),
 '00000657-b-1': ('ASPA','fitness','human_cell_line','competitive_growth','growth','not_applicable','HEK293T','landing_pad','not_applicable','cellular_toxicity','unknown'),
 '00000664-a-1': ('KCNQ2','activity','hamster_cell_line','whole_cell_patch_clamp','electrophysiology','voltage_plus40mV','CHO','mutant_only','not_applicable','current_density_plus40mV','unknown'),
 '00000664-a-4': ('KCNQ2','activity','hamster_cell_line','whole_cell_patch_clamp','electrophysiology','voltage_plus40mV','CHO','wild_type_coexpression','not_applicable','current_density_plus40mV','unknown'),
 '00000665-a-1': ('PAX6','binding','yeast','yeast_one_hybrid','growth','geneticin','not_applicable','unknown','DNA_LE9','DNA_binding_growth_proxy','unknown'),
 '00000665-b-1': ('PAX6','binding','yeast','yeast_one_hybrid','growth','geneticin','not_applicable','unknown','DNA_BLX','DNA_binding_growth_proxy','unknown'),
 '00001197-a-5': ('FKRP','activity','human_cell_line','SMuRF','glycosylation','not_applicable','HAP1','unknown','not_applicable','alpha_dystroglycan_glycosylation','unknown'),
 '00001254-a-1': ('LARGE1','activity','human_cell_line','SMuRF','glycosylation','not_applicable','HAP1','unknown','not_applicable','alpha_dystroglycan_glycosylation','unknown'),
 '00001257-a-1': ('ADSL','fitness','yeast','yeast_functional_complementation','growth','doxycycline','not_applicable','ADE13_repressed','not_applicable','growth_complementation','high'),
 '00001257-a-3': ('ADSL','fitness','yeast','yeast_functional_complementation','growth','doxycycline','not_applicable','ADE13_repressed','not_applicable','growth_complementation','low'),
 '00001258-a-1': ('SPOP','fitness','yeast','yeast_survival','growth','unknown','not_applicable','unknown','not_applicable','loss_of_function_survival','unknown'),
}


def protein_change(value):
    m = re.fullmatch(r'p\.([A-Z][a-z]{2})([1-9]\d*)([A-Z][a-z]{2})', str(value))
    if not m or m[1] not in THREE or m[3] not in THREE or m[1] == m[3]:
        return None
    return THREE[m[1]], int(m[2]), THREE[m[3]]


def parse_scores(raw, sequence):
    changes = raw.hgvs_pro.map(protein_change)
    scores = pd.to_numeric(raw.score, errors='coerce')
    selected = changes.notna() & np.isfinite(scores)
    parsed = pd.DataFrame(changes[selected].tolist(), columns=['wt','ref_pos','mut'])
    parsed['score_value'] = scores[selected].to_numpy()
    parsed['source_row_id'] = np.flatnonzero(selected)
    if any(p > len(sequence) or sequence[p-1] != w for w,p in parsed[['wt','ref_pos']].itertuples(index=False,name=None)):
        raise ValueError('Source WT does not match the exact experimental reference')
    # These reviewed cDNA experiments have no intronic/splicing measurements.
    # Aggregate SNV encodings of the same protein substitution equally; record counts.
    grouped = parsed.groupby(['wt','ref_pos','mut'],as_index=False).agg(
        score_value=('score_value','mean'), source_encoding_count=('score_value','size'),
        source_row_id=('source_row_id', lambda x:'|'.join(map(str,x))))
    return grouped, {'downloaded_rows':len(raw), 'excluded_nonmissense_or_nonfinite':int((~selected).sum()),
                     'retained_encoding_rows':int(selected.sum()), 'unique_protein_substitutions':len(grouped),
                     'aggregation':'arithmetic mean of deposited protein-equivalent cDNA encodings; not independent replicate count'}


def write_source_credits(report):
    lines=['**Functional expansion source credits**', '',
           'Downloaded 26 September 2026 from the public MaveDB API. Original metadata and scores are bundled in v4/sources/functional_expansion. Changes: unsupported/nonfinite observations are excluded; protein-equivalent cDNA encodings are averaged with counts recorded. These counts are not independent biological replicates. No clinical annotation columns enter the model.', '']
    for record in report['imported']:
        urn=record['urn']
        detail=json.loads((SOURCE/(urn.replace('urn:mavedb:','')+'.json')).read_text(encoding='utf-8'))
        lines.append(f"- **{record['gene']} — {detail['title']}**: [{urn}](https://www.mavedb.org/score-sets/{urn}), [{detail['license']['shortName']}]({detail['license']['link']}); {record['unique_protein_substitutions']:,} retained substitutions.")
        publications=detail.get('primaryPublicationIdentifiers') or []
        lines.append('  Publication identifiers: '+('; '.join(str(p.get('identifier',p)) for p in publications) or 'See source record and linked experiment.'))
        lines.append('')
    (OUT/'SOURCE_CREDITS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    base_path = Path('v4/data/ndd_training_corpus_v4.csv.gz')
    base = read_corpus(base_path)
    meta = pd.read_csv('v4/data/assay_metadata.csv',keep_default_na=False)
    meta = meta[meta.assay_key.isin(base.assay_key)].copy()
    refs = json.loads(Path('v4/resources/sequences.json').read_text())
    for field in META_FIELDS:
        if field not in meta: meta[field] = 'unknown'
        meta[field] = meta[field].replace('', 'unknown')
    # Explicit corrections from the existing curated record and original study contexts.
    mask = meta.assay_key.str.contains('Giacomelli_2018')
    meta.loc[mask & meta.assay_key.str.contains('_Null_'), 'genetic_background'] = 'TP53_null'
    meta.loc[mask & meta.assay_key.str.contains('_WT_'), 'genetic_background'] = 'TP53_wild_type'
    for cell in ['HEK293T','RPE1']:
        meta.loc[meta.assay_key.str.contains('NPC1_HUMAN_Erwood_2022_'+cell),'cell_line'] = cell
    for index,row in meta[meta.task=='binding'].iterrows():
        if 'binding-' in row.assay_key:
            meta.loc[index,'interaction_partner'] = row.assay_key.split('binding-',1)[1]
    additions, records, imported = [], [], []
    for urn,spec in SPECS.items():
        gene,task,host,system,selection,treatment,cell,background,partner,quantity,expression = spec
        detail_path,score_path = SOURCE/(urn+'.json'),SOURCE/(urn+'.csv')
        detail=json.loads(detail_path.read_text(encoding='utf-8'))
        if len(detail['targetGenes']) != 1: raise ValueError('Expected one target')
        target=detail['targetGenes'][0]; target_seq=target['targetSequence']
        if target_seq['taxonomy']['code'] != 9606 or target['name'].upper()!=gene: raise ValueError('Unverified human gene')
        sequence=target_seq['sequence'] if target_seq['sequenceType']=='protein' else str(Seq(target_seq['sequence']).translate()).rstrip('*')
        matches=[key for key,value in refs.items() if value['gene']==gene and value['sequence']==sequence]
        if len(matches)!=1: raise ValueError(f'{urn}: need one exact reference match, got {matches}')
        ref=matches[0]
        raw=pd.read_csv(score_path)
        rows, accounting=parse_scores(raw,sequence)
        key=gene+'::urn:mavedb:'+urn
        if key in set(base.assay_key): raise ValueError('Assay already in base')
        template=base[base.gene==gene].iloc[0]
        assert int(template.is_direct_ndd)==1
        f=rows.copy()
        f['gene']=gene;f['protein_reference_id']=ref;f['assay_id']='urn:mavedb:'+urn;f['assay_key']=key
        f['variant_key']=ref+':'+f.wt+f.ref_pos.astype(str)+f.mut
        f['measurement_id']=f.assay_id+':'+f.variant_key
        f['task']=task;f['assay_type']='organismal_fitness' if task=='fitness' else task
        f['score_kind']='source_experimental_score';f['score_orientation']='higher_'+quantity+'_not_health'
        f['measurement_type']=quantity;f['position_system']='experimental_reference_1_based'
        f['supervision_tier']='B_DIRECT_NDD_FUNCTIONAL';f['training_role']='primary_ndd_functional'
        for column in ['is_direct_ndd','ndd_evidence_class','taxon_id','species','protein_group']:
            f[column]=template[column]
        f['default_sample_weight']=1.;f['possible_consensus_constituent_overlap']=0
        f['source']='MaveDB_reviewed_functional_expansion';f['assay_level_available']=1
        f['provenance_note']='Deposited experimental score. Source encoding counts are not independent replicates. No clinical annotations are features.'
        f['source_study']='urn:mavedb:'+urn.rsplit('-',2)[0]
        additions.append(f)
        m={c:'' for c in meta.columns}
        for c in ['gene','protein_reference_id','assay_id','assay_key','task','assay_type','score_kind','supervision_tier','taxon_id','is_direct_ndd','ndd_evidence_class','protein_group','source','measurement_type']:
            m[c]=f[c].iloc[0]
        m.update(dict(host=host,system=system,selection=selection,treatment=treatment,cell_line=cell,
                      genetic_background=background,interaction_partner=partner,assay_quantity=quantity,
                      expression_level=expression,region='full_experimental_reference',loaded_rows=len(f),
                      source_url='https://api.mavedb.org/api/v1/score-sets/urn:mavedb:'+urn,
                      source_sha256=digest(score_path),license=detail['license']['shortName'],
                      publication_title=detail['title'],source_file=str(score_path),
                      score_direction=f.score_orientation.iloc[0],orientation='as_deposited',
                      context_evidence=str(detail_path),context_status='reviewed_source_methods',
                      context_limitations='Task is experimental phenotype, not a clinical mechanism label; measurement uncertainty is not modeled.',
                      curation_review_date='2026-09-26'))
        # Use the established, source-curated host and cell line for SMuRF.
        if gene in ['FKRP','LARGE1']:
            old=meta[(meta.gene==gene)&(meta.task=='consensus')].iloc[0]
            for field in ['host','cell_line','system','selection','treatment','region']:
                m[field]=old[field]
        records.append(m)
        imported.append({'gene':gene,'task':task,'urn':'urn:mavedb:'+urn,'license':detail['license']['shortName'],
                         'reference':ref,'source_sha256':digest(score_path),'metadata_sha256':digest(detail_path),**accounting})
    # Preserve originals and raw imports; the primary analysis view excludes consensus
    # and one documented exact duplicate deposit rather than pretending they are independent.
    full=pd.concat([base.drop(columns='row_id'),*additions],ignore_index=True)
    full.to_csv(OUT/'master_with_provenance.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    duplicate='ITSN1::urn:mavedb:00000861-a-1'
    original='ITSN1::urn:mavedb:00000859-a-1'
    a=base[base.assay_key==duplicate].set_index('variant_key').score_value.sort_index()
    b=base[base.assay_key==original].set_index('variant_key').score_value.sort_index()
    assert a.equals(b), 'The documented duplicate changed; review before excluding'
    primary=full[(full.task!='consensus')&(full.assay_key!=duplicate)].copy()
    primary['row_id']=np.arange(len(primary))
    metadata=pd.concat([meta,pd.DataFrame(records)],ignore_index=True)
    metadata=metadata[metadata.assay_key.isin(primary.assay_key)].copy()
    metadata.to_csv(OUT/'assay_metadata.csv',index=False)
    views={'curriculum':primary,'direct':primary[pd.to_numeric(primary.is_direct_ndd)==1]}
    split_groups=pd.read_csv('v4/data/split_groups.csv')
    report={'base_sha256':digest(base_path),'imported':imported,'excluded_primary_assays':{
        'all_consensus':int((full.task=='consensus').sum()),duplicate:int((full.assay_key==duplicate).sum())},
        'excluded_candidates':{'00001257-0-1':'model-inferred enzyme activity, not independent measurement',
        '00000091-a-1':'BRAF V600E oncogenic background/fragment requires separate construct review',
        '00001197-a-4':'superseded FKRP deposit; use a-5 once'},'views':{}}
    for scope,frame in views.items():
        path=OUT/(scope+'.csv.gz');frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        # Re-read through the actual training contract to check every imported row.
        checked=read_corpus(path)
        evaluation=sorted(set(checked.loc[pd.to_numeric(checked.is_direct_ndd)==1,'gene']))
        pd.DataFrame({'gene':evaluation}).to_csv(OUT/(scope+'_evaluation_genes.csv'),index=False)
        groups=split_groups[split_groups.gene.isin(checked.gene)]
        groups.to_csv(OUT/(scope+'_groups.csv'),index=False)
        make_splits(path,OUT/(scope+'_splits.json'),clusters=OUT/(scope+'_groups.csv'),test_genes=OUT/(scope+'_evaluation_genes.csv'))
        counts=checked.groupby('task').agg(rows=('gene','size'),genes=('gene','nunique'),assays=('assay_key','nunique'))
        counts.to_csv(OUT/(scope+'_coverage.csv'))
        report['views'][scope]={'rows':len(frame),'genes':int(frame.gene.nunique()),
            'multi_task_genes':int((frame.groupby('gene').task.nunique()>1).sum()),
            'tasks':counts.to_dict('index'),'corpus_sha256':digest(path)}
    # Dataset-level context collisions are now visible rather than hidden in tensors.
    collision=metadata[metadata.duplicated(['protein_reference_id','task']+META_FIELDS,keep=False)]
    collision.to_csv(OUT/'context_collisions.csv',index=False)
    report['context_collision_assays']=len(collision)
    report['limitations']=['Additional conditions and recovered sources are not independent new genes.',
      'No guaranteed unseen-gene interval coverage; insufficient calibration tasks are suppressed.',
      'Error columns remain in original source tables; heterogeneous error definitions are not treated as common standard errors.',
      'Splits isolate genes and identical references, not all homologous families.']
    save_json(OUT/'build_report.json',report)
    write_source_credits(report)
    print(json.dumps(report['views'],indent=2))


if __name__=='__main__': main()
