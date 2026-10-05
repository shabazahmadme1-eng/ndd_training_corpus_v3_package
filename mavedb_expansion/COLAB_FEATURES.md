# ESM + LLR feature pack for the 10 staged genes (Colab A100)

Local torch is CPU-only and ~5,018 masked-LLR positions need a GPU, so this
step runs in the existing Colab runtime, not here. `mipo/features.py:extract`
is resumable and skips finished genes, so re-running it on the existing cache
fills in only the new work.

## What is already done locally

- Corpus rows built and recounted (178,793 staged rows, PASS).
- Structures for all 10 genes verified (`v4/resources/structures/`).
- New references registered in `v4/resources/sequences.json` (7 new; CCR5,
  NUDT15, PRKN reuse existing ProteinGym references).

## New references needing ESM + LLR

| Gene | protein_reference_id | Len | Assayed positions |
|---|---|---|---|
| CXCR4 | CXCR4__9606__745dbb36c096 | 352 | 351 |
| G6PD | G6PD__9606__0725c9221fe1 | 515 | 514 |
| INSR | INSR__9606__0e665f54d93e | 955 | 928 |
| LDLR | LDLR__9606__36a68c551082 | 860 | 858 |
| MPL | MPL__9606__bbf7d8b4523c | 49 | 49 |
| SOD1 | SOD1__9606__c56903dca972 | 154 | 152 |
| TYK2 | TYK2__9606__3f6b27ef15f1 | 1187 | 1187 |

Reused references may also gain LLR positions from the new assays (CCR5 351,
NUDT15 163, PRKN 465, upper bound — already-covered positions are skipped).

## Colab steps

1. On Drive (`MyDrive/MIPO_NDD/`): rename the existing
   `mipo_ndd_v4_colab_upload.zip` to `mipo_ndd_v4_colab_upload_DGX_release.zip`
   (keep it). Upload `dist/mipo_ndd_v4_colab_upload_staged.zip` (481,285,066
   bytes, sha256 `50f147eb...ffd4a0`) and rename it to
   `mipo_ndd_v4_colab_upload.zip` — cell 2 reads that fixed name. Do NOT delete
   the `.restored_*` marker. Cell 1 must print "Already restored"; if it tries
   to restore instead, stop.
2. Features-only pass in cell 2: `SCOPE='curriculum'`,
   `TEST_GENES=['TYK2','G6PD','LDLR']`, `DO_FEATURES=True`, `DO_TRAIN=False`,
   `DO_ABLATIONS=False`, `DO_BASELINE=False`, `EXPECTED_EXPERIMENT=None`
   (required: new corpus means a new protocol hash; anything else raises before
   features run). A new experiment folder is created and left empty — harmless.
2. Run against the EXISTING feature cache directory (same model/window config,
   or `extract` raises a cache-mismatch error):

```python
from mipo.features import extract
extract('v4/data/ndd_training_corpus_v4.csv.gz', 'v4/resources', '<existing-cache-dir>')
```

3. Expect roughly 2–4 hours on A100 for ~5k masked positions (checkpoints every
   50 positions; safe to interrupt and resume). ESM embeddings for the 7 new
   references take minutes.
4. Verify: every new `protein_reference_id` has `<cache>/<key>/esm.npy` and
   `llr.npy` with no NaN at assayed positions.

Do NOT run notebook cell 2 (retrains the failed runs). This pack only fills
feature caches; it trains nothing.
