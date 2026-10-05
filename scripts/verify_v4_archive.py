"""Verify shipped bytes, notebook syntax and primary-data release fingerprints."""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import PurePosixPath


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError(f'Unsafe member: {name}')
        manifest = json.loads(archive.read('RELEASE_MANIFEST.json'))['sha256']
        if set(names) != set(manifest) | {'RELEASE_MANIFEST.json'}:
            raise ValueError('Manifest/member mismatch')
        for name, expected in manifest.items():
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f'Hash mismatch: {name}')
        validation = json.loads(archive.read('v4/robust/validation.json'))
        for scope in ['direct', 'curriculum']:
            if manifest[f'v4/robust/{scope}.csv.gz'] != validation[scope]['corpus_sha256']:
                raise ValueError(f'Stale validation: {scope}')
            for name in [f'configs/v4_robust_{scope}.json', f'v4/robust/{scope}_splits.json']:
                json.loads(archive.read(name))
        mechanism = json.loads(archive.read('v4/robust/mechanism_validation.json'))
        if (not mechanism['passed'] or
                mechanism['corpus_sha256'] != manifest['v4/robust/curriculum.csv.gz'] or
                mechanism['source_splits_sha256'] != manifest['v4/robust/mechanism_splits.json']):
            raise ValueError('Stale mechanism panel validation')
        source = archive.read('scripts/colab_v4_validation.py').decode('utf-8').replace('\r\n', '\n')
        ast.parse(source)
        for label in ['Upload_', 'Validation_', '']:
            name = f'notebooks/MIPO_NDD_V4_{label}Colab.ipynb'
            notebook = json.loads(archive.read(name))
            cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
            if cells[0] != source:
                raise ValueError(f'Notebook/script mismatch: {name}')
            for cell in cells:
                ast.parse(cell)
    return {'verified_files': len(manifest), 'all_member_hashes_passed': True,
            'notebooks_match_entrypoint': True, 'primary_data_fingerprints_passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    print(json.dumps(verify(parser.parse_args().archive), indent=2))
