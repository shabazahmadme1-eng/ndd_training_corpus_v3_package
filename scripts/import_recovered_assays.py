"""Load the real scores behind consensus assays whose source was identified at rho = +/-1.

The collapsed-consensus targets are rank transforms: every one of these assays is uniform
on [0, 1] with mean 0.500 and SD 0.2887, so the measured values are gone and only the order
survives. Relabelling those ranks as an activity or abundance task would teach a response
shape no assay has. `recover_consensus_provenance.py` already identified which deposited
score set each one is, so this loads that score set instead and adds it as its own assay.

Taking source values also removes the orientation trap. SPOP and MAPK1 are recorded as
`inverted_relative_to_source`, which matters only when relabelling corpus ranks; a source
score carries its own direction, so imported rows are oriented like every other assay.

Original rows are never edited except for the overlap flag and weight that the dataset card
already prescribes for a consensus row whose variant also has an assay-level measurement.
Writes new corpus files and a report; the inputs are left alone.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mipo.common import AA

SOURCES = Path('v4/sources/consensus_candidates')
THREE = {'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G',
         'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
         'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'}
# The measurement type each recovered source reports, from its MaveDB record and publication.
# Only genes this release verifies as NDD are listed; a gene whose consensus merely duplicates
# an assay the corpus already holds (MAPK1, OTC, PSAT1) is deliberately absent.
TASKS = {
    'ASPA': ('abundance', 'abundance', 'VAMP-seq cellular abundance, HEK293T landing pad'),
    'FKRP': ('activity', 'activity', 'SMuRF alpha-dystroglycan glycosylation'),
    'LARGE1': ('activity', 'activity', 'SMuRF alpha-dystroglycan glycosylation'),
    'SPOP': ('fitness', 'organismal_fitness', 'yeast survival selection'),
}


def protein_change(value):
    m = re.fullmatch(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(value).strip())
    if not m or m.group(1) not in THREE or m.group(3) not in THREE:
        return None  # Synonymous, nonsense, frameshift and multi-residue changes are out of scope.
    return THREE[m.group(1)], int(m.group(2)), THREE[m.group(3)]


def source_scores(urn):
    """One row per protein substitution; replicate nucleotide variants are averaged."""
    path = SOURCES/f"urn_mavedb_{urn.replace('urn:mavedb:', '')}.csv"
    if not path.exists():
        return None, f'no cached score table at {path}'
    d = pd.read_csv(path)
    parsed = d.hgvs_pro.map(protein_change)
    d = d[parsed.notna() & d.score.notna()].copy()
    if d.empty:
        return None, 'no single-substitution scores in the source table'
    changes = parsed[parsed.notna()]
    d['wt'] = [c[0] for c in changes]
    d['ref_pos'] = [c[1] for c in changes]
    d['mut'] = [c[2] for c in changes]
    grouped = d.groupby(['wt', 'ref_pos', 'mut'], as_index=False).agg(
        score_value=('score', 'mean'), measurement_sigma=('score', 'std'), n_snv=('score', 'size'))
    return grouped[grouped.wt != grouped.mut], None


def verify(source, consensus, sequences, reference, expected_rho, tolerance):
    """Reject anything that does not reproduce the recorded identification exactly."""
    sequence = sequences[reference]['sequence']
    off = np.fromiter((p > len(sequence) or sequence[p-1] != w
                       for p, w in zip(source.ref_pos, source.wt)), bool, len(source))
    if off.any():
        return f'{off.sum()} source rows disagree with the reference sequence at their position'
    merged = consensus.merge(source, on=['wt', 'ref_pos', 'mut'])
    if len(merged) != len(consensus):
        return f'source covers {len(merged)} of {len(consensus)} consensus rows, not all of them'
    rho = spearmanr(merged.score_value_x, merged.score_value_y).statistic
    if not np.isfinite(rho) or abs(abs(rho)-1) > tolerance or np.sign(rho) != np.sign(expected_rho):
        return f'rank match is {rho:+.4f}, not the recorded {expected_rho:+.4f}'
    return None


def build(corpus_path, metadata_path, tolerance=.01, genes=None):
    from mipo.corpus import read_corpus
    corpus = read_corpus(corpus_path)
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    sequences = json.loads(Path('v4/resources/sequences.json').read_text())
    wanted = {g: v for g, v in TASKS.items() if not genes or g in genes}
    added, report, flagged = [], [], []
    for gene, (task, assay_type, description) in sorted(wanted.items()):
        record = metadata[(metadata.gene == gene) & (metadata.assay_id == 'gate_consensus')]
        note = {'gene': gene, 'task': task}
        if record.empty or record.iloc[0].recovered_match_status != 'rank_identical_source':
            report.append({**note, 'status': 'skipped', 'detail': 'not recorded as a rank-identical source'})
            continue
        record = record.iloc[0]
        urn = record.recovered_source_urn
        source, problem = source_scores(urn)
        if problem:
            report.append({**note, 'status': 'skipped', 'urn': urn, 'detail': problem})
            continue
        consensus = corpus[(corpus.gene == gene) & (corpus.assay_id == 'gate_consensus')]
        reference = consensus.protein_reference_id.iloc[0]
        problem = verify(source, consensus, sequences, reference,
                         float(record.recovered_rank_correlation), tolerance)
        if problem:
            report.append({**note, 'status': 'rejected', 'urn': urn, 'detail': problem})
            continue
        rows = source.copy()
        rows['gene'], rows['protein_reference_id'] = gene, reference
        rows['assay_id'], rows['assay_type'], rows['task'] = urn, assay_type, task
        rows['score_kind'] = 'mavedb_recovered_source_score'
        rows['score_orientation'] = 'higher_source_assay_phenotype_not_health'
        rows['supervision_tier'], rows['training_role'] = 'B_DIRECT_NDD_FUNCTIONAL', 'primary_ndd_supervision'
        rows['is_direct_ndd'], rows['assay_level_available'] = 1, 1
        rows['ndd_evidence_class'] = record.ndd_evidence_class
        rows['species'], rows['taxon_id'] = 'Human', 9606
        rows['source'], rows['source_row_id'] = 'mavedb_recovered_source', urn
        rows['default_sample_weight'] = 1.
        rows['possible_consensus_constituent_overlap'] = 1
        rows['raw_dms_score'] = rows.score_value
        rows['measurement_id'] = [f'{urn}:{w}{p}{m}' for w, p, m in zip(rows.wt, rows.ref_pos, rows.mut)]
        rows['provenance_note'] = (f'{description}. Source score set {urn} identified as the collapsed '
                                   f'consensus for {gene} at rank correlation '
                                   f'{float(record.recovered_rank_correlation):+.4f}; this row is the source '
                                   'measurement, not the rank-transformed consensus value.')
        known = set(corpus.columns)
        added.append(rows[[c for c in rows.columns if c in known or c == 'n_snv']])
        shared = set(zip(rows.wt, rows.ref_pos, rows.mut))
        flagged.append((gene, shared))
        report.append({**note, 'status': 'imported', 'urn': urn, 'rows': len(rows),
                       'consensus_rows': len(consensus),
                       'new_variants': len(rows)-len(consensus),
                       'recorded_rho': float(record.recovered_rank_correlation),
                       'detail': description})
    return added, flagged, pd.DataFrame(report)


def apply_to(path, added, flagged, out):
    """Append the imported rows and down-weight the consensus rows they duplicate."""
    original = pd.read_csv(path, low_memory=False)
    updated = original.copy()
    for gene, shared in flagged:
        same_variant = np.fromiter(((w, p, m) in shared for w, p, m in
                                    zip(updated.wt, updated.ref_pos, updated.mut)), bool, len(updated))
        overlap = (updated.gene == gene) & (updated.assay_id == 'gate_consensus') & same_variant
        updated.loc[overlap, 'possible_consensus_constituent_overlap'] = 1
        updated.loc[overlap, 'default_sample_weight'] = .25
    genes = {gene for gene, _ in flagged}
    keep = [part for part in added if part.gene.iloc[0] in set(original.gene) & genes]
    if not keep:
        return None, 0
    combined = pd.concat([updated]+[part.drop(columns=[c for c in ['n_snv'] if c in part]) for part in keep],
                         ignore_index=True)
    combined.to_csv(out, index=False)
    return out, sum(len(part) for part in keep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full', default='v4/data/ndd_training_corpus_v4.csv.gz')
    parser.add_argument('--direct', default='v4/data/ndd_direct_v4.csv.gz')
    parser.add_argument('--metadata', default='v4/data/assay_metadata.csv')
    parser.add_argument('--suffix', default='_recovered')
    parser.add_argument('--genes', help='Comma-separated subset of the recovered genes')
    parser.add_argument('--tolerance', type=float, default=.01, help='Allowed departure from |rho| = 1')
    parser.add_argument('--report', default='v4/curation/recovered_assay_import.csv')
    args = parser.parse_args()
    added, flagged, report = build(args.full, args.metadata, args.tolerance,
                                   set(args.genes.split(',')) if args.genes else None)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    print(report.to_string(index=False))
    if not added:
        raise SystemExit('Nothing imported; see the report above.')
    for path in [args.full, args.direct]:
        name = Path(path).name.replace('.csv.gz', f'{args.suffix}.csv.gz')
        written, count = apply_to(path, added, flagged, Path(path).with_name(name))
        print(f'{written}: +{count} rows' if written else f'{path}: no listed gene present, unchanged')
    print(f'\nReport: {args.report}. Corpus hash changes, so regenerate splits before training.')
