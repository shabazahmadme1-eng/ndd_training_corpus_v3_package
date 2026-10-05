# Tier 1.5a FireProt benchmark: paste this entire file into ONE Colab GPU cell.
# Eval-only, GPU-light (80 proteins, 2427 rows). Re-runnable after recycles.
#
# PREREQUISITE (once): upload dist/fireprot_15a_bundle.zip to Drive at
#   MyDrive/MIPO_NDD/fireprot_15a_bundle.zip
# (built by scripts/package_fireprot_15a.py, which asserts the decided
# post-exclusion 2427x80 corpus shape; the BUNDLE assertion below re-checks it.)
BUNDLE = "post-exclusion-2427x80"  # decided 2026-10-05: 1QLP/1B5M/1IET out (domain homology)
BUNDLE_ZIP = "fireprot_15a_bundle.zip"
PILOT = "contrast_pilot_v1"       # v1 champion (branch_weight=0)

from google.colab import drive

drive.mount("/content/drive")
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

NDD = Path("/content/drive/MyDrive/MIPO_NDD")
ARCHIVE = NDD / "mipo_ndd_v4_colab_upload.zip"
assert ARCHIVE.exists(), f"missing {ARCHIVE}"
PROJECT = Path("/content") / f"mipo_ndd_v4_validation_{hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()[:12]}"
if not (PROJECT / "mipo/model.py").exists():
    PROJECT.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(PROJECT)
assert "branch_probes" in (PROJECT / "mipo/model.py").read_text(), "NOT v3 code"
print("PROJECT:", PROJECT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[test]"],
               check=True, cwd=PROJECT)

BUNDLE_PATH = NDD / BUNDLE_ZIP
assert BUNDLE_PATH.exists(), f"upload {BUNDLE_ZIP} to {NDD} first"
BUNDLE_DIR = Path("/content/fireprot_15a_bundle/fireprot_15a")
if BUNDLE_DIR.exists():
    shutil.rmtree(BUNDLE_DIR.parent)
BUNDLE_DIR.parent.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE_PATH) as z:
    bad = z.testzip()
    assert bad is None, f"bundle zip corrupt at {bad}"
    z.extractall(BUNDLE_DIR.parent)
for name in ["mipo_fireprot.py", "evaluate_fireprot.py", "corpus_fireprot.csv",
             "resources_fireprot/sequences.json"]:
    assert (BUNDLE_DIR / name).exists(), f"bundle zip missing {name}; rebuild it"
shutil.copy2(BUNDLE_DIR / "mipo_fireprot.py", PROJECT / "mipo/fireprot.py")
shutil.copy2(BUNDLE_DIR / "evaluate_fireprot.py", PROJECT / "scripts/evaluate_fireprot.py")
FPEVAL = PROJECT / "fireprot_eval"
if FPEVAL.exists():
    shutil.rmtree(FPEVAL)
FPEVAL.mkdir()
shutil.copy2(BUNDLE_DIR / "corpus_fireprot.csv", FPEVAL / "corpus_fireprot.csv")
shutil.copytree(BUNDLE_DIR / "resources_fireprot", FPEVAL / "resources_fireprot")
for f in [PROJECT / "mipo/fireprot.py", PROJECT / "scripts/evaluate_fireprot.py",
          FPEVAL / "corpus_fireprot.csv"]:
    print(f.name, hashlib.sha256(f.read_bytes()).hexdigest()[:12])

corpus = pd.read_csv(FPEVAL / "corpus_fireprot.csv")
prefix = BUNDLE.rsplit("-", 1)[0]
shape = f"{prefix}-{len(corpus)}x{corpus.gene.nunique()}"
print("benchmark shape:", shape)
assert shape == BUNDLE, f"corpus {shape} != decided {BUNDLE}; re-upload bundle after exclusion rebuild"

def result_tag(ckpt, pilot_dir):
    return "_".join(ckpt.relative_to(pilot_dir).parent.parts)


ckpts = sorted((NDD / PILOT).rglob("best.pt"))
print("checkpoints:", [str(c.relative_to(NDD)) for c in ckpts])
assert ckpts, f"no best.pt under {NDD / PILOT}"

CACHE = PROJECT / "fireprot_cache"
subprocess.run([sys.executable, "-m", "mipo", "features", "--corpus",
                str(FPEVAL / "corpus_fireprot.csv"), "--resources",
                str(FPEVAL / "resources_fireprot"), "--cache", str(CACHE)], check=True)
META = PROJECT / "v4/robust/assay_metadata.csv"
assert META.exists()
seen_tags = set()
for ckpt in ckpts:
    tag = result_tag(ckpt, NDD / PILOT)
    assert tag and tag not in seen_tags, f"result tag collision on {tag}"
    seen_tags.add(tag)
    out = NDD / "fireprot_15a_results" / tag
    subprocess.run([sys.executable, str(PROJECT / "scripts/evaluate_fireprot.py"),
                    "--checkpoint", str(ckpt), "--corpus",
                    str(FPEVAL / "corpus_fireprot.csv"), "--resources",
                    str(FPEVAL / "resources_fireprot"), "--cache", str(CACHE),
                    "--metadata", str(META), "--allow-probeless",
                    "--out", str(out)], check=True)
    print("wrote", out)
print("DONE: paste each fireprot_metrics.json table back for the 1.5a verdict")
