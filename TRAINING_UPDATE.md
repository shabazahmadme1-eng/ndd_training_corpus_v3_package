# Multi-gene validation update

**Current primary workflow:** [ROBUST_RELEASE.md](ROBUST_RELEASE.md), 26 September 2026.
Earlier experimental contrast/residual/meta-learning options below are disabled in the
robust profiles pending matched evaluation. The original and recovered corpora remain historical views.

## Function-prediction changes, 2026-09-24

The first three-seed PTEN run showed MIPO matching its own `esm_mlp` ablation
(0.504 vs 0.518 mean test Spearman) and losing to the cached LLR on the activity
assay in all twelve runs. Diagnosis and the resulting changes:

- **Targets are scaled per assay, not per task.** Pooling a task put a generic
  binding assay reporting raw intensities in the millions on the same scale as
  everything else, reaching 1.2e8 standard deviations and a curriculum stage-1
  NLL near 1e12. Worst case is now 49.7 robust SD; `max_abs_target` (default 20)
  clips training targets only and reports the count. Held-out assays fall back to
  the median scale of their task's training assays, which Spearman does not depend
  on. **Old checkpoints and `target_transform.json` files are not compatible.**
- **Checkpoint selection and patience start at the final curriculum stage.**
  Previously `best` and `bad_epochs` carried across stages, so patience ran out
  during stage 1 and every curriculum checkpoint came from the generic-data stage
  with the broken scale, leaving the verified-NDD stage 1-2 of its 8 epochs.
- **`task_balanced_sampling` draws a task first, then a gene.** Under gene-uniform
  draws, 111 of 126 direct-scope training genes hold only stability, leaving
  activity at 0.8% and abundance at 0.3% of batches; each task now gets about 14%.
- **`contrast_fraction`/`contrast_weight` add a mechanism-separation term.** Some
  batches pair two assays of one gene over shared variants, and the loss ranks the
  gap between them. Pairs exclude collapsed consensus and any two assays whose
  shared training targets are a monotone relabel of each other. Direct scope has
  one usable pair (KRAS); the curriculum scope has nine.
- **`llr_residual` predicts a correction on top of a per-task affine function of
  the cached LLR**, so training cannot drive the zero-shot signal out. This assumes
  a task's training assays share its score direction.
- `scripts/discover_function_assays.py` searches MaveDB for function assays of every
  corpus gene; `scripts/plan_mechanism_panel.py` lists genes that can test mechanism
  separation. Findings are in `v4/sources/function_search/`.

Ablations `gene_sampling`, `no_contrast` and `no_llr_residual` isolate one change
each. None of these has been trained on a GPU here; no performance change is claimed.

Open `notebooks/MIPO_NDD_V4_Upload_Colab.ipynb` in a GPU Colab runtime. Use its
Files sidebar to upload `dist/mipo_ndd_v4_colab_upload.zip` to `/content/`, then run
the main cell. No Google Drive mount is required. Alternatively, paste all of
`scripts/colab_v4_validation.py` into one cell.

The release ZIP contains code, data and reference resources, not the precomputed
ESM embeddings/LLR cache from your previous Drive run. Optionally upload that
cache as `v4_650M_masked.zip` too (manifest.json at the ZIP root or inside
v4_650M_masked/). It is extracted and reused. Without it, the cell automatically
builds features before training. Set `DO_FEATURES=True` to resume an incomplete
cache; `False` requires an existing complete cache. Do not change model size or
disable LLR merely to bypass this preparation if comparing against the previous run.

Outputs are in `/content/v4_validation_...`. Run the notebook's final download
cell to save results and checkpoints to your computer. Runtime storage is temporary.
To save newly generated features for a future session, separately ZIP and download
`/content/v4_650M_masked/`. For optional Drive storage, set `STORAGE='drive'` and
place the same upload ZIP and feature ZIP under `MyDrive/MIPO_NDD/`.

Defaults preserve the requested training budget (3000 batches of 8 per epoch,
20 maximum direct-training epochs, patience 6) and architecture. Main MIPO and
three ablations run once each. Increase `SEEDS` and `TEST_GENES` only as part of a
predefined evaluation plan; runtime scales with models, seeds and folds.

## Split and checkpoint rules

- Five validation groups and three calibration groups by default. Existing
  validation/calibration groups stay in their roles; outer tests stay unchanged.
- Expansion uses task presence, favors scarce tasks when coverage counts tie,
  then uses seeded tie-breaking and the original eligible
  evaluation gene set. It preserves gene/cluster boundaries and retains at least
  one training group for each currently supported training task. It fails if
  the requested expansion cannot meet those constraints. It does not use target
  values or test performance. Task coverage is preferred, not guaranteed.
- The selection score is the assay-average Spearman within each validation gene,
  then the equal average across genes. Every validation gene must have a defined
  score. Calibration and test scores never select the checkpoint.
- Expanded holdouts reduce and change training data. Compare ablations within
  the new protocol; changes from the old PTEN score cannot be attributed solely
  to better checkpoint selection. The shipped clusters are not a complete
  homology-family holdout benchmark.
- Runs already early-stopped stay stopped on resume. Code, splits, configuration,
  metadata and feature provenance protect against mixing incompatible runs.
  Old checkpoints need the old release; use a new output folder for this update.

## New diagnostics

- `split_composition.csv`: role/gene/assay/task row counts.
- `task_support.json`: number of genes per task in each role. The Colab cell
  displays this before training. For the default direct PTEN fold, the only
  remaining abundance gene stays in training, so abundance has no separate
  validation or calibration support. Its calibrated intervals remain absent.
- `target_transform.json`: training-only centers and scales. Held-out target
  statistics never fit the transform.
- `history.json`: aggregate loss and its NLL, Huber, ranking, field and auxiliary
  components; per-gene/per-task validation scores; sample draws; stopping state.
- `validation_epochs/epoch_*.csv`: per-assay validation diagnostics each epoch.
  `validation_metrics.csv` and `validation_summary.json` describe the selected checkpoint.
- `training_status.json`: actual best epoch and early-stopping/epoch-cap reason.
- Test/calibration metrics include bias, target/prediction dispersion, predicted
  sigma and Gaussian 90% coverage. Test metrics additionally report calibrated
  interval coverage, width and supported row counts; absent calibration is missing,
  never substituted with Gaussian intervals.
- `matched_llr_metrics.csv`: trained and signed cached-LLR correlations on the
  exact same variants and targets. Partial legacy LLR columns cannot reduce the
  headline comparison to consensus. Undefined correlations suppress the headline.

Absolute metrics use training-task-transformed scores. Predictions also include
raw-score means, sigma and calibrated bounds. This adds visibility into scale
transfer; it does not establish or repair calibration on unseen genes. Empirical
calibration intervals have no assured coverage under gene distribution shift.

The entry point supports multiple seeds/folds, reports each seed separately, and
does not label a plateau as proof of memorization or automatically demand longer
training. Repeated seeds are not independent new test genes.

The companion notebook is generated from the Python cell with
`python scripts/build_validation_notebook.py`. Package the updated release with
`python scripts/package_v4_release.py --out dist/mipo_ndd_v4_colab_upload.zip`.
