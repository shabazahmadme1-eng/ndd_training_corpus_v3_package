"""Import staged AlphaFold structures for new references (exact or unique-crop)."""
import json
from pathlib import Path

from mipo.resources import import_structure

RES = Path('v4/resources')
STAGED_PDB = Path('mavedb_expansion/staged_structures')
seqs = json.loads((RES / 'sequences.json').read_text(encoding='utf-8'))
report = json.loads((RES / 'structures/report.json').read_text(encoding='utf-8'))

n = 0
for ref_id, r in seqs.items():
    if not r.get('source', '').startswith('MaveDB staged'):
        continue
    pdb = STAGED_PDB / f"{r['accession']}.pdb"
    import_structure(str(RES), ref_id, str(pdb), confidence='plddt', allow_subsequence=True)
    report[ref_id] = 'imported_staged_alphafold_exact_or_unique_crop'
    n += 1
    print(f'{r["gene"]}: imported ({r["length"]} aa)')
(RES / 'structures/report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('imported', n, 'structures')
