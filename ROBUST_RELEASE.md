**MIPO-NDD robust release — 26 September 2026**

Use `notebooks/MIPO_NDD_V4_Upload_Colab.ipynb` with the rebuilt
`dist/mipo_ndd_v4_colab_upload.zip`. The release implements the scientific audit's
correctness fixes and adds reviewed experimental data. It does not claim a measured
accuracy improvement: the revised model has not been trained on a GPU here.

**What changed**

- Curriculum checkpoint selection and patience operate in the final stage. Saved
  status now identifies the selected stage. A regression test covers stronger early
  pretraining scores without allowing them to consume specialization patience.
- Robust training uses training-assay median/IQR scaling, task-balanced batches,
  and training-only outlier clipping. Held-out assay scales remain unknown; the
  fallback uses training-task statistics and is an assumption, not calibrated truth.
- The metadata encoder additionally accepts cell line, cellular genetic background,
  interaction partner, experimental quantity and expression level. TP53 background
  and NPC1 cell-line collisions are resolved. New KCNQ2, PAX6 and ADSL conditions
  remain distinct. Unknown/unseen training vocabulary still maps to unknown.
- Low-confidence coordinates cannot select spatial nodes/neighbors or contribute
  scalar distances or vector messages. Tests randomize unreliable coordinates and
  require unchanged graphs and predictions.
- An undefined assay correlation can no longer silently disappear from checkpoint
  selection or aggregate gene scores. All intended endpoints remain accountable.
- Empirical intervals require at least three calibration genes per task by default.
  Unsupported tasks retain missing intervals and an explicit reason. Three genes is
  a minimal support guard, not a theoretical or empirical coverage guarantee.
- The MMseqs converter maps reference IDs to biological genes and transitively
  unions all clusters linked by constructs of the same gene. Existing gene splits
  are not advertised as homology-family holdouts.
- Default ablations are ESM MLP, no structure and no global propagation, alongside
  full MIPO. Experimental contrast, LLR residual, head shrinkage and meta-learning
  options from the workspace remain available but are disabled in robust profiles.
- Reused frozen feature caches are resumed to fill newly observed mutation sites;
  a manifest alone is no longer taken as evidence of complete LLR coverage.

**Functional diversity: actual improvements and remaining limits**

The expansion adds 48,020 protein-substitution observations in 11 MaveDB score sets
across seven already evidence-qualified NDD genes. Deposited human sequences match
existing experimental references exactly. Each retained WT residue is validated;
nonmissense/nonfinite observations are excluded and counted. Protein-equivalent
cDNA encodings are averaged with their counts recorded; these counts are not
treated as independent biological replicates. No imputed labels, clinical annotation
features, or artificial assays are introduced.

| Measured task | NDD genes before | NDD genes now |
|---|---:|---:|
| Abundance | 2 | 3 |
| Activity | 4 | 7 |
| Binding / binding proxy | 2 | 3 |
| Fitness | 9 | 12 |
| Multiple non-consensus tasks | 5 | 7 |

Additions include ASPA abundance and toxicity, KCNQ2 current density with mutant-only
and wild-type coexpression, PAX6 DNA-binding growth proxies for two target sequences,
FKRP/LARGE1 glycosylation, ADSL yeast complementation at two expression levels, and
SPOP survival selection. Some recover measurements underlying old consensus labels;
they are not all new independent experiments. Multiple conditions of one gene do not
increase the independent gene count. Study/source lineage is retained.

The primary NDD view contains **280,969 measurements across 134 genes**. The broad
curriculum view contains **904,938 measurements across 490 genes**. Counts are lower
than the old master because all consensus observations and one exact ITSN1 duplicate
deposit are excluded from the primary analysis. TSC2 has only consensus in the base
release and is therefore absent from the primary view. Its original observations,
all other legacy data, and the added sources remain in the provenance master.

Three abundance or binding genes still cannot populate independent training,
validation, calibration and test groups with broad support. Task balancing increases
exposure to scarce tasks; it does not create independent evidence. The broad view
provides 17 abundance, 23 activity, 9 binding and 29 fitness genes, including generic
and teacher supervision, which must not be called NDD-only evidence.

The excluded ADSL “Estimated Enzyme Activities” table is model-inferred rather than
an independent activity experiment. BRAF V600E resistance needs separate construct
and biological-background review. These candidates are documented, not silently
promoted to direct functional supervision. Measurement-error columns remain in the
original source files; heterogeneous SD/SEM/model errors are not equated or used as
common precision weights.

**Data files and reproducibility**

- `v4/data/`: original corpus remains available and unchanged by the robust builder.
- `v4/robust/master_with_provenance.csv.gz`: original master plus reviewed imports.
- `v4/robust/direct.csv.gz`, `curriculum.csv.gz`: primary measured-assay views.
- `v4/robust/assay_metadata.csv`: matching expanded context schema.
- `v4/robust/*_splits.json`: matching fingerprints and gene/reference isolation.
- `v4/robust/build_report.json`: sources, hashes, licenses, row accounting and coverage.
- `v4/robust/validation.json`: real-data and split checks, including pilot task support.
- `v4/sources/functional_expansion/`: original public source snapshots and scores.
- `v4/robust/SOURCE_CREDITS.md`: attribution, source links and transformation policy.

Rebuild from bundled source snapshots with:

```text
python -m scripts.build_robust_release_data
python -m scripts.validate_robust_release
python -m scripts.build_robust_mechanism_panel
python -m pytest -q
python scripts/build_validation_notebook.py
python -m scripts.package_v4_release --out dist/mipo_ndd_v4_colab_upload.zip
```

For reference-level MMseqs output, the repaired converter requires:

```text
python scripts/mmseqs_clusters.py --tsv cluster_result_cluster.tsv --references v4/data/protein_references.csv --out clusters.csv
```

Use the resulting biological-gene mapping with the split command. Domain-aware
coverage and clustering thresholds require a predefined protocol; gene grouping
alone is not proof of unseen-family transfer.

**Running and interpreting the new release**

The default is now `SCOPE = 'mechanism'` with **10 genes and three seeds**
`[42, 123, 2026]`: PTEN, KRAS, SLC22A1, GCK, KCNJ2, CYP2C9, SRC, HLA-A,
KCNE1 and VKORC1. Four matched models imply **120 training runs**, plus frozen
ESM650/masked LLR extraction. This restores the earlier panel as an executable
option; the original panel data and planning script were never removed.

The mechanism option uses the robust broad measured corpus and joint training
with the robust direct model profile (no NDD-only final specialization). Its
new splits match that corpus and preserve gene/reference group boundaries.
Validation/calibration can draw from all 490 corpus genes, rather than removing
eight of the nine remaining paired-property panel genes in every fold. The panel
and exact-reference shared substitutions are checked in `mechanism_validation.json`
and `mechanism_assay_pairs.csv`. No old corpus fingerprints are reused.

Only PTEN and KRAS in this historical ten-gene panel are NDD-verified under the
dataset evidence rule; the other eight are reported as mechanism-only evidence.
PTEN remains included and marked as previously examined, so it is a development
case rather than an untouched confirmation. The historical panel is not a claim
that these are the only useful paired-property genes in the expanded dataset.
Three seeds estimate training variability, not independent biological replication.

`paired_property_contrasts.csv` compares observed versus predicted differences
in within-assay percentile ranks, computed on the exact shared substitutions for
each cross-task assay pair. It also reports rank-contrast error and a zero-contrast
reference. Missing/constant contrasts remain explicit. These are descriptive
test diagnostics, never a checkpoint-selection criterion. Deposited directions
are retained; this is not a function-loss classifier. Abundance is an indirect
folding proxy, and SRC's activity/fitness pairing is not abundance-anchored.
Neither successful contrasts nor learned field names establish causal mechanisms.
Related assay pairs and variants must not be counted as independent gene replicates.
Supporting experimental methods: [PTEN abundance](https://www.nature.com/articles/s41588-018-0122-z),
[PTEN function](https://pmc.ncbi.nlm.nih.gov/articles/PMC5986715/), and
[KRAS abundance and binding](https://www.nature.com/articles/s41586-023-06954-0).

Choose `SCOPE = 'direct'` for the smaller ASPA/KCNQ2/PAX6 pilot, or `'curriculum'`
for that panel with staged broad training. `TEST_GENES = None` selects the default
panel for each scope; an explicit list selects a subset of the available folds.
For confirmatory generalization predefine an untouched gene/family panel.
Robust profiles are
`configs/v4_robust_direct.json` and `configs/v4_robust_curriculum.json`; older configs
remain historical/experimental options. Do not mix their results. Old training
checkpoints are incompatible with the new metadata/model code; start a new run.
Sequence-compatible frozen ESM caches can be reused and extended.

Report task/source-specific ranking, all failures, and paired gene/family-level
uncertainty. Do not tune on the pilot and then call it confirmation. Signed LLR is
reported as signed LLR, not an automatically test-sign-flipped clinical predictor;
some source phenotypes such as toxicity have the opposite biological interpretation.
Any alternative baseline orientation must be specified from source semantics before
test evaluation.

New-gene interval coverage, family transfer, independent field localization, true
ensemble effects and clinical/mechanistic validity still require experiments. All
bundled structures remain single conformers. The latent fields and physical-proxy
outputs are not validated physical mechanisms.
