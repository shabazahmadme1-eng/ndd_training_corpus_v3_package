"""Search MaveDB for abundance, activity and binding score sets of every corpus gene.

Read-only with respect to the corpus. Each search response is cached, so reruns are offline.
Keyword classification is a triage aid: every candidate still needs variant, reference and
orientation validation before import (see v4/REBUILD.md).
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import time

import pandas as pd
import requests

API = 'https://api.mavedb.org/api/v1/score-sets/search'
ROOT = Path('v4/sources/function_search')
CORPUS = 'v4/data/ndd_training_corpus_v4.csv.gz'
DIRECT = 'v4/data/ndd_direct_v4.csv.gz'
# First match wins: a folding ddG inferred from abundancePCA is stability, "binding fitness" and a
# one-hybrid antibiotic-resistance readout are binding, and "toxicity measured by VAMP-seq" is a
# phenotype. "Model combined scores" are Tsuboyama 2023 proteolysis stability sets.
CATEGORIES = [
    ('stability', r'domainome|trypsin|chymotrypsin|proteolysis|cdna display|folding free energ|thermo|\bstability\b'
                  r'|model combined scores'),
    ('binding', r'binding|interaction|affinity|two.hybrid|one.hybrid|\by2h\b|phage display'),
    ('fitness', r'toxicity|survival|prolifer|growth|resistance|phenotyp|saturation genome|\bsge\b|lof mutants'),
    ('abundance', r'vamp-?seq|abundance|expression level|surface express|protein level|steady.state|degron|degradation'),
    ('localization', r'imaging|locali[sz]ation|morpholog'),
    ('activity', r'activit|enzym|catalyt|complement|fitness|current|electrophys|reporter|transcripti|phosphatase'
                 r'|kinase|glycosylation|function'),
]
FUNCTION = ['abundance', 'activity', 'fitness', 'binding']


def search(gene):
    path = ROOT/'responses'/f'{gene}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    records, offset = [], 0
    while True:
        for attempt in range(4):
            r = requests.post(API, json={'text': gene, 'limit': 100, 'offset': offset}, timeout=(15, 120))
            if r.status_code not in [429, 500, 502, 503, 504]:
                break
            time.sleep(2**attempt)
        r.raise_for_status()
        page = r.json()
        records += page['scoreSets']
        offset += 100
        if offset >= page['numScoreSets']:
            break
    path.write_text(json.dumps(records, ensure_ascii=True), encoding='utf-8')
    return records


def targets(record, gene):
    """True when a target gene, not just free text, names this gene."""
    for t in record.get('targetGenes', []):
        names = [t.get('name') or '', t.get('mappedHgncName') or '', (t.get('targetSequence') or {}).get('label') or '']
        if any(gene in re.split(r'[^A-Z0-9]+', n.upper()) for n in names):
            return True
    return False


def classify(record):
    """Score-set text first; the experiment text only when the score set says nothing usable."""
    experiment = record.get('experiment') or {}
    for text in [f"{record.get('title') or ''} {record.get('shortDescription') or ''}",
                 f"{experiment.get('title') or ''} {experiment.get('shortDescription') or ''}"]:
        found = next((name for name, pattern in CATEGORIES if re.search(pattern, text.lower())), None)
        if found:
            return found
    return 'unclassified'


def likely_duplicate(gene, category, title, variants, assays):
    """A corpus assay of the same gene and kind with a similar variant count, e.g. a ProteinGym copy.
    Binding sets must also name the same partner: KRAS has six partners with similar counts."""
    same = assays[(assays.gene == gene) & (assays.task != 'consensus')
                  & ((assays.task == category) | (category in ['activity', 'fitness']) & assays.task.isin(['activity', 'fitness']))]
    if category == 'binding':
        squash = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
        partner = same.assay_key.str.extract(r'binding-(.+)$')[0].fillna('')
        same = same[partner.map(lambda p: bool(p) and squash(p) in squash(title)).to_numpy(bool)]
    close = same[(same.rows-variants).abs() <= .2*same.rows.clip(lower=variants)]
    return ';'.join(close.assay_key)


def species(record):
    taxa = {((t.get('targetSequence') or {}).get('taxonomy') or {}).get('code') for t in record.get('targetGenes', [])}
    return 'human' if taxa == {9606} else ('unknown' if taxa <= {None} else 'non_human_or_mixed')


if __name__ == '__main__':
    (ROOT/'responses').mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CORPUS, usecols=['gene', 'assay_id', 'assay_key', 'task'], low_memory=False)
    direct_genes = set(pd.read_csv(DIRECT, usecols=['gene'], low_memory=False).gene)
    in_corpus_ids = set(corpus.assay_id.astype(str))
    tasks = corpus.groupby('gene').task.agg(lambda s: '|'.join(sorted(set(s))))
    assays = corpus.groupby(['gene', 'assay_key', 'task']).size().reset_index(name='rows')
    # Sources that recover_consensus_provenance.py proved are the collapsed consensus labels.
    recovered = json.loads(Path('v4/sources/consensus_candidates/matched_details.json').read_text(encoding='utf-8'))
    consensus_source = {m['urn']: f'{gene}::gate_consensus' for gene, m in recovered.items()}
    consensus_study = {m['urn'].rsplit('-', 2)[0]: f'{gene}::gate_consensus' for gene, m in recovered.items()}
    genes = sorted(direct_genes)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(zip(genes, pool.map(search, genes)))
    rows = []
    for gene, records in results.items():
        for record in records:
            if not targets(record, gene):
                continue
            publication = (record.get('primaryPublicationIdentifiers') or [{}])[0]
            category = classify(record)
            rows.append({'gene': gene, 'corpus_tasks': tasks.get(gene, ''), 'urn': record['urn'],
                         'category': category, 'title': record.get('title'),
                         'num_variants': record.get('numVariants'), 'species': species(record),
                         'license': (record.get('license') or {}).get('shortName'),
                         'published': record.get('publishedDate'), 'doi': publication.get('identifier'),
                         'already_in_corpus': record['urn'] in in_corpus_ids,
                         'likely_duplicate_of': consensus_source.get(record['urn']) or likely_duplicate(
                             gene, category, record.get('title') or '', record.get('numVariants') or 0, assays),
                         'same_study_as': consensus_study.get(record['urn'].rsplit('-', 2)[0], '')})
    found = pd.DataFrame(rows)
    found.to_csv(ROOT/'candidates.csv', index=False)
    new = found[~found.already_in_corpus & (found.species != 'non_human_or_mixed') & (found.likely_duplicate_of == '')]
    wanted = new[new.category.isin(FUNCTION)]
    summary = wanted.pivot_table(index='gene', columns='category', values='urn', aggfunc='count', fill_value=0)
    summary['corpus_tasks'] = tasks.reindex(summary.index)
    summary.to_csv(ROOT/'summary_by_gene.csv')
    stability_only = sorted(g for g in genes if tasks.get(g) == 'stability')
    print(f'{len(genes)} direct-NDD genes searched; {len(stability_only)} have only stability assays')
    print(f'{len(found)} score sets target these genes; {len(wanted)} are new abundance/activity/binding candidates')
    print(f'stability-only genes with any new candidate: {sorted(set(wanted.gene) & set(stability_only))}')
    print(summary.to_string())
