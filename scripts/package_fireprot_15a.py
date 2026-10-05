"""Package the Tier 1.5a FireProt bundle as a Drive-ready zip.

Asserts the decided post-exclusion corpus shape (2427 rows / 80 genes) while
building, so a stale tree can never ship. Writes dist/fireprot_15a_bundle.zip
with top-level folder fireprot_15a/.
"""
import hashlib
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS, EXPECTED_GENES = 2427, 80


def main():
    corpus = pd.read_csv(ROOT / "fireprot_eval/corpus_fireprot.csv")
    shape = (len(corpus), corpus.gene.nunique())
    assert shape == (EXPECTED_ROWS, EXPECTED_GENES), f"corpus is {shape}, expected {(EXPECTED_ROWS, EXPECTED_GENES)}"
    import datetime
    import json

    prov = {"bundle": "fireprot_15a post-exclusion-2427x80",
            "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {}}
    for rel in ["fireprot_eval/corpus_fireprot.csv",
                "fireprot_eval/resources_fireprot/sequences.json",
                "mipo/fireprot.py", "scripts/evaluate_fireprot.py"]:
        prov["inputs"][rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:12]
    (ROOT / "fireprot_eval/resources_fireprot/provenance.json").write_text(json.dumps(prov, indent=2))
    files = ["mipo/fireprot.py", "scripts/evaluate_fireprot.py",
             "fireprot_eval/corpus_fireprot.csv",
             "fireprot_eval/resources_fireprot/sequences.json",
             "fireprot_eval/resources_fireprot/provenance.json",
             "fireprot_eval/TIER15_DESIGN.md",
             "fireprot_eval/build_report.json",
             "fireprot_eval/dedupe_report.json",
             "fireprot_eval/exclusions.json"]
    npz = sorted((ROOT / "fireprot_eval/resources_fireprot/structures").glob("*.npz"))
    assert len(npz) == 74, f"expected 74 structure npz, found {len(npz)}"
    out = ROOT / "dist/fireprot_15a_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            arc = "fireprot_15a/" + Path(rel).name
            if rel == "mipo/fireprot.py":
                arc = "fireprot_15a/mipo_fireprot.py"
            elif rel == "scripts/evaluate_fireprot.py":
                arc = "fireprot_15a/evaluate_fireprot.py"
            elif rel.startswith("fireprot_eval/resources_fireprot/"):
                arc = "fireprot_15a/resources_fireprot/" + Path(rel).name
            z.write(ROOT / rel, arc)
        for f in npz:
            z.write(f, f"fireprot_15a/resources_fireprot/structures/{f.name}")
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None, "zip integrity check failed"
        names = z.namelist()
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{len(names)} files, {out.stat().st_size / 1e6:.1f} MB -> {out}")
    print("SHA256:", sha)


if __name__ == "__main__":
    main()
