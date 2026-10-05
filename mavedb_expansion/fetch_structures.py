"""Pre-fetch AlphaFold PDBs for staged genes (offline-safe for later import)."""
import json
import urllib.request
from pathlib import Path

H = Path(__file__).parent
OUT = H / 'staged_structures'
OUT.mkdir(exist_ok=True)

accs = set()
for f in (H / 'staged').glob('*/mapping.json'):
    accs.add(json.loads(f.read_text(encoding='utf-8'))['uniprot'])


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout).read()


for acc in sorted(accs):
    dest = OUT / f'{acc}.pdb'
    if dest.exists():
        print(f'{acc}: cached')
        continue
    try:
        meta = json.loads(get(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}'))
        if not meta:
            print(f'{acc}: NO AlphaFold record')
            continue
        pdb = urllib.request.urlopen(meta[0]['pdbUrl'], timeout=300).read()
        dest.write_bytes(pdb)
        print(f'{acc}: fetched {len(pdb)//1024} KB')
    except Exception as e:
        print(f'{acc}: FAIL {type(e).__name__} {e}')
