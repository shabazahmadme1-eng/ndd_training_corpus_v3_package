"""Package the validated V4 data, source snapshots and Colab training code."""
from pathlib import Path
import json
import zipfile
import argparse
from mipo.common import digest, save_json

root = Path(__file__).resolve().parents[1]
files = []
for directory in ['mipo', 'tests', 'scripts', 'configs', 'notebooks', 'prepared', 'v4', 'fireprot_eval']:
    for path in (root/directory).rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.name == 'proteingym_v1.3.zip' or ('alphafold' in path.parts and path.suffix == '.pdb'):
            continue
        if path.suffix == '.html':
            continue
        # Cached full-text papers are read locally for curation and are not redistributed;
        # their licences are not uniformly clear. The curation evidence that *is* shipped is
        # the MaveDB record cache, which is public API metadata.
        if 'literature' in path.parts:
            continue
        # Raw MaveDB score downloads used to rank-match the consensus targets: ~73 MB of
        # evidence that training never reads. The conclusions live in
        # v4/audit/consensus_provenance_map.csv and the override document.
        if 'consensus_candidates' in path.parts:
            continue
        files.append(path)
files.extend(p for p in root.iterdir() if p.is_file() and (
    p.suffix in ['.md', '.toml'] or p.name in ['sequence_overrides.json',
    'README_ndd_training_corpus_v3.txt', 'ndd_training_corpus_v3.csv', 'ndd_training_corpus_v3_manifest.csv']))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', default=str(root/'dist/mipo_ndd_v4_complete.zip'))
parser.add_argument('--allow-stale-mechanism-panel', action='store_true')
parser.add_argument('--stale-panel-reason', default='')
args = parser.parse_args()
archive = Path(args.out).resolve()
archive.parent.mkdir(exist_ok=True)
validation = json.loads((root/'v4/robust/validation.json').read_text())
for scope in ['direct', 'curriculum']:
    if not validation[scope]['all_gene_and_sequence_boundaries_passed']:
        raise ValueError(f'{scope}: split validation failed')
    if digest(root/f'v4/robust/{scope}.csv.gz') != validation[scope]['corpus_sha256']:
        raise ValueError(f'{scope}: corpus changed since validation')
mechanism = json.loads((root/'v4/robust/mechanism_validation.json').read_text())
mechanism_ok = (mechanism['passed'] and
        mechanism['corpus_sha256'] == digest(root/'v4/robust/curriculum.csv.gz') and
        mechanism['source_splits_sha256'] == digest(root/'v4/robust/mechanism_splits.json'))
waiver = None
if not mechanism_ok:
    if not (args.allow_stale_mechanism_panel and args.stale_panel_reason):
        raise ValueError('Mechanism panel validation is stale or failed')
    waiver = args.stale_panel_reason
    print('WARNING: shipping with stale mechanism panel:', waiver, flush=True)
temporary = archive.with_suffix(archive.suffix + '.tmp')
manifest = {}
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
    for path in sorted(set(files)):
        compression = zipfile.ZIP_STORED if path.suffix in ['.gz', '.npz', '.parquet'] else zipfile.ZIP_DEFLATED
        name = path.relative_to(root).as_posix()
        manifest[name] = digest(path)
        z.write(path, name, compress_type=compression)
    z.writestr('RELEASE_MANIFEST.json', json.dumps({'sha256': manifest, 'mechanism_panel_waiver': waiver}, indent=2))
with zipfile.ZipFile(temporary) as z:
    if z.testzip() is not None:
        raise ValueError('ZIP integrity check failed')
    count = len(z.namelist())
temporary.replace(archive)
report = {'archive': archive.name, 'sha256': digest(archive), 'bytes': archive.stat().st_size,
          'files': count, 'zip_integrity_passed': True}
report_path = archive.with_suffix('.json') if '--out' in __import__('sys').argv else root/'dist/v4_release.json'
save_json(report_path, report)
print(json.dumps(report, indent=2))
