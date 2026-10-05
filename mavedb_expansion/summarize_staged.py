"""Print staged genes with proposed labels for sign-off."""
import glob
import json

for f in sorted(glob.glob('mavedb_expansion/staged/*/mapping.json')):
    m = json.load(open(f, encoding='utf-8'))
    print(f"{m['gene']} [{m['role']}] shared={m['shared_missense']}")
    for s in m['sets']:
        print(f"  [{s['heuristic_task']}] {s['urn']} lic={s['license']} "
              f"map={s['mapped_missense']}/{s['n_variants']} mm={s['wt_mismatch']} off={s['uniprot_offset']}")
        print(f"    {s['title'][:95]}")
