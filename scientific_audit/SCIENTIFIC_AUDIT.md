**Scientific audit of MIPO-NDD V4 — 26 September 2026**

**Verdict:** This is a credible research prototype with unusually explicit provenance and claim limitations. The supplied evidence does **not** establish that the response-field architecture improves unseen-gene prediction, that its fields identify biological mechanisms, or that its uncertainty transfers reliably to new assays. Several concrete implementation and experimental-design problems should be resolved before a larger benchmark or publication claim.

The most consequential findings are curriculum checkpoint selection, missing assay context, very sparse independent functional-task coverage, inadequate uncertainty transfer, and the absence of a multi-gene/family-held-out performance benchmark. More architectural complexity is not the next priority.

**Scope and evidence**

The primary target was `dist/mipo_ndd_v4_colab_upload.zip`, SHA-256 `0fd8ea7aba72aefaf39d41417466c5c3727a0f76b581fde96edd6772b0817cbd`. Its contents were extracted to [archive](archive) for inspection. The loose workspace contains newer code and is not interchangeable with this release.

I additionally inspected the existing `mipo_ndd_v4_results.zip` as supplementary evidence. Its corpus hashes match the archived datasets, but its recorded `train.py`, `features.py`, and `cli.py` hashes differ from this ZIP. Consequently, results below describe the supplied result files, not an independently reproduced training run of the exact upload ZIP. The workspace's current `train.py` also differs from the recorded training version.

Checks performed: all 994,673 dataset rows for measurement identity, finite targets and WT/reference consistency; all 270 folds in the two supplied split manifests for gene/exact-sequence isolation; task coverage; metadata collisions; structure ensemble counts; code inspection; synthetic geometry probes; saved-result correlation and coverage recomputation; and primary-literature checks. The archived suite passed **39 tests in 60.25 seconds**. No GPU retraining, independent wet-lab replication, exhaustive novelty search, or source-by-source re-derivation of every measurement was performed.

Reproducible computations are in [run_audit.py](run_audit.py), with full outputs in [audit_evidence.json](audit_evidence.json). Run `python scientific_audit/run_audit.py` from the workspace root after extraction. Supplementary per-assay tables are [direct](direct_assay_metrics.csv) and [curriculum](curriculum_assay_metrics.csv).

**What is already scientifically sound**

- The primary question is explicitly framed as prediction of experimental mutation effects on proteins excluded from supervised training. The documentation appropriately avoids equating this with pathogenicity, causal mechanisms, or physical dynamics.
- Exact experimental constructs have separate references and coordinates. I found zero WT/reference mismatches, nonfinite scores, duplicate measurement IDs, or duplicate assay/variant keys. These checks do not prove independent provenance across different assay IDs.
- All 135 direct and 135 curriculum folds passed gene and exact-sequence isolation. The intended supervised curriculum uses the same outer exclusions at every stage.
- Score transformations and categorical vocabularies are fitted on training groups; ranking pairs stay within assays; calibration and checkpoint-selection groups are distinct.
- Matched LLR comparison uses the same test variants and does not optimize the LLR sign on test outcomes. The documentation explicitly acknowledges unknown assay scales, consensus provenance, and foundation-model exposure.

**1. High priority: the saved “curriculum” model is a pretraining checkpoint**

In [archived train.py](archive/mipo/train.py:233), `best` and `bad_epochs` update during every stage, while stopping is activated in the final stage. There is no reset or final-stage-only checkpoint selection. A good pretraining epoch can therefore exhaust patience before NDD specialization starts and remain the exported best model.

This is not merely hypothetical in the supplementary results: all three MIPO curriculum runs execute nine epochs of a planned sixteen, and their best checkpoints are zero-based epochs **2, 2, and 1** for seeds 42, 123, and 2026. All three are in `generic_and_legacy_auxiliary`. The final NDD stage starts at epoch 8 and receives only one epoch before stopping. Their test predictions therefore do not measure the proposed completed three-stage curriculum.

**Required action:** select and count patience within the final stage, with explicit stage-specific checkpoint records; optionally retain a separately named best-over-all-stages comparator. Define whether transitions initialize from the previous stage's final or best checkpoint. Add a regression test where pretraining validation exceeds final-stage validation. The loose workspace already contains a final-stage-only selection change; rebuild and verify the ZIP before rerunning. Also compare freezing against continued backbone adaptation rather than assuming freezing helps.

**2. High priority: current results do not demonstrate an architecture advantage**

All completed supplementary runs hold out **PTEN only**. Direct runs provide three seeds for each of four models; curriculum results contain three completed MIPO seeds but only two completed seeds for each comparator. Recomputed test Spearman correlations agree with saved metrics to numerical precision.

Direct-training PTEN results, averaging the three assays within each seed:

| Model | Mean Spearman | SD across three seeds |
|---|---:|---:|
| No structure | 0.5274 | 0.0134 |
| ESM MLP baseline | 0.5181 | 0.0022 |
| Signed masked LLR | 0.5074 | Not applicable |
| Full MIPO | 0.5037 | 0.0231 |
| Fixed heads | 0.4604 | 0.0544 |

The ESM MLP runs also enable LLR, so their name must not be interpreted as an embedding-only baseline. Removing the unresolved consensus endpoint leaves mean abundance/activity Spearman of **0.4747 for MIPO**, **0.4942 for ESM MLP**, and **0.5077 for no structure**. MIPO is below the signed LLR on PTEN activity in every direct seed.

These are descriptive comparisons on one gene, not significance tests or proof that structure is generally harmful. Seeds measure training variability; they do not provide independent evidence across proteins. PTEN has already been examined and should now be treated as a development case for changes inspired by these findings.

**Required action:** freeze the revised protocol and assess untouched genes, then unseen homology families. Use paired model differences at the gene/family level, averaging seeds within gene before uncertainty estimation. Report task/source strata and every predefined endpoint, including failures. Add no-global-propagation, no-LLR, chemistry, and field-removal controls, along with a strong simple regularized predictor and an applicable external sequence/structure predictor under compatible supervision. ProteinGym provides useful benchmark infrastructure, but its published scores cannot replace matched evaluation on this corpus. [Official ProteinGym](https://github.com/OATML-Markslab/ProteinGym).

**3. High priority: independent functional coverage is much smaller than the corpus headline**

The direct dataset has 318,274 measurements across 135 genes, with the following coverage. Gene counts overlap across tasks.

| Task | Measurements | Genes |
|---|---:|---:|
| Stability / proxy | 159,715 | 116 |
| Consensus | 84,304 | 15 |
| Fitness | 49,996 | 9 |
| Activity | 12,664 | 4 |
| Abundance | 8,149 | 2 |
| Binding | 3,446 | 2 |

**112 of 135 genes have only stability-task observations**, and just **5 direct NDD genes have multiple non-consensus task types**. Millions of within-protein substitutions cannot replace independent genes, assay contexts, or mechanisms. Uniform-gene sampling also means stability-only genes account for a large share of training draws; equal-gene sampling does not balance tasks.

With only two abundance genes, disjoint training/validation/calibration/test genes cannot all contain abundance. In the direct PTEN fold, abundance has **1 training gene, 0 validation genes, and 0 calibration genes**. The run may predict abundance, but it has no abundance-based checkpoint selection or calibrated abundance intervals.

**Required action:** define a separate estimand for each task, report independent support explicitly, and collect additional matched abundance/activity/binding assays across NDD genes. Compare gene-balanced and task-balanced objectives transparently. Where support is insufficient, label results exploratory rather than manufacturing split coverage by sharing genes across roles.

Domainome measures abundance of isolated domains in a cellular selection assay. It supports a stability proxy, but does not make 116 genes into full-length neuronal-function maps. The original paper also discusses domain isolation and assay-selection limitations. [Beltran et al., Nature 2025](https://www.nature.com/articles/s41586-024-08370-4).

**4. High priority: biologically different assays can have identical model inputs**

The encoder uses only `host`, `selection`, `system`, `treatment`, and `region`, plus task. See [common.py](archive/mipo/common.py:16) and [model.py](archive/mipo/model.py:151). Richer curated fields exist in the table but do not enter the model.

Confirmed collisions for the same exact protein reference and task:

- **TP53 WT_Nutlin versus Null_Nutlin:** all five encoded fields match. Across 7,467 shared variants, their targets have Spearman **0.5884** and mean absolute raw-score difference **0.6991**. The deterministic model must return the same mean and dispersion for each shared variant in both conditions. It cannot express background-dependent effects from these inputs.
- **NPC1 HEK293T versus RPE1:** cell line is omitted; 57 shared variants have different targets despite identical encoded context.

The TP53 study deliberately used WT and null cellular backgrounds to distinguish responses, so this is meaningful biological context. [Giacomelli et al., Nature Genetics 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6168352/).

**Required action:** introduce controlled fields for cellular genetic background, cell line, interaction partner, dose/time and expression context where supported. Fit vocabularies on training only and flag unseen context. Do not solve transfer merely by adding assay IDs. Add a dataset check for distinct conditions collapsing to identical inputs. Include both metadata ablation and context-held-out evaluation. A larger hypernetwork cannot recover missing information.

**5. High priority: ranking transfers better than absolute scores or uncertainty**

[TargetTransform](archive/mipo/data.py:16) pools raw assay scores by broad task before median/IQR scaling. This avoids test-statistic leakage, but it does not reconcile different experimental units, WT baselines, normalization conventions or dynamic ranges. A task label and coarse context cannot determine an unseen assay's offset and scale. Gaussian NLL can consequently be dominated by incompatible score systems; predicted sigma can absorb scale mismatch rather than represent biological uncertainty.

Observed direct MIPO coverage, averaged across seeds:

| PTEN task | Nominal 90% Gaussian coverage | Separately calibrated coverage |
|---|---:|---:|
| Abundance | 6.83% | Unavailable |
| Activity | 4.48% | 22.45% |
| Consensus | 91.80% | 92.65% |

The apparent success on consensus must not conceal failure on functional endpoints. Curriculum activity coverage is also below nominal, and those runs have the checkpoint-selection limitation above.

**Required action:** preserve Spearman as the main uncalibrated cross-assay endpoint. Model the assay observation process explicitly, using published control-based units where justified. Separate an inductive zero-label protocol from any few-shot protocol that uses target-assay control/support labels. Evaluate calibration curves and interval width per gene and task. More rows from one calibration gene do not substitute for more calibration genes. Hierarchical calibration requires explicit group-level assumptions; ordinary variant residual quantiles do not guarantee new-gene coverage. [Lee et al., Distribution-free inference with hierarchical data](https://arxiv.org/abs/2306.06342).

**6. High priority for family-transfer claims: no demonstrated homology-held-out benchmark**

The verified split checks establish gene and exact-sequence isolation, not separation of homologous domains. The supplementary PTEN homology report is a diagnostic for one fold and is not a family-level holdout experiment. Foundation-model exposure is a separate issue from supervised-label leakage.

There is also a concrete V4 workflow gap: FASTA headers are `protein_reference_id`, but [mmseqs_clusters.py](archive/scripts/mmseqs_clusters.py:1) merely renames sequence members to `gene`. [make_splits](archive/mipo/splits.py:13) expects biological gene keys. Applying the documented converter directly to the V4 FASTA does not create the required gene mapping. It also does not union multiple construct clusters belonging to one gene.

**Required action:** map reference clusters to genes, union every component linked by any gene/ortholog/reference, and assign whole components to roles. Audit local/domain homology as well as full-length identity; short constructs and differing lengths can evade global coverage thresholds. Report threshold sensitivity and nearest-training similarity. Separately document the training-data exposure of external predictors. Protein/variant overlap is a known source of misleading variant-predictor evaluation. [Grimm et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25684150/).

**7. Medium–high priority: assay IDs do not guarantee independent observations**

I confirmed a cross-ID duplicate: ITSN1 score sets `urn:mavedb:00000859-a-1` and `urn:mavedb:00000861-a-1` share the same reference and coordinates **Q15811:1003–1059** and contain **1,021 identical variant scores**. Their metadata assign different Pfam accessions. This is consistent with the same observations represented through two domain annotations; independent replication is not established. They pass the current duplicate-key checks because their assay IDs differ.

The corpus additionally flags 60,535 consensus observations as potentially overlapping underlying assays. The 0.25 loss weight reduces their influence but cannot create independence, correct source dependence, or justify counting consensus and constituent assays as independent test endpoints.

The recovery report overstates evidence when it treats perfect ranks as proof of a unique historical source, or correlations below one as proof of aggregation. Rank agreement strongly supports a shared measurement lineage but does not identify a unique normalization pipeline. Imperfect agreement can also result from filtering, noise, reference changes, or transformations other than an aggregate.

**Required action:** retain all source records for provenance, but define independent observation groups across deposits/assays. Resolve ITSN1 before counting it twice in scientific summaries. Make non-consensus assays the primary evaluation and report consensus separately. Run exclusion sensitivities for overlapping rows and the entire consensus tier. Describe rank-recovered provenance as a supported candidate lineage with evidence strength, not proof beyond what the data establish.

**8. Medium priority: low-confidence structure still enters scalar computations**

In [data.py](archive/mipo/data.py:145), coordinates can select neighbors whenever confidence is greater than zero. The >0.5 threshold masks vector weights, while scalar distance/contact features remain populated. [model.py](archive/mipo/model.py:47) consumes those features when structure is enabled.

A synthetic probe with every confidence set to 0.1 had zero vector weights in both conditions, yet changing coordinates changed the latent field by approximately **4.46e-4** and the score by **1.04e-7** in an untrained model. This establishes an information path, not a measured performance problem of that magnitude in the trained network.

The distinct `structure=False` probe returned identical scores after changing coordinates. Thus the original suspicion that the no-structure ablation directly uses distance features was **not confirmed**; the concern is confidence handling in the structure-enabled model.

**Required action:** either mask unreliable scalar geometry and spatial-neighbor selection as well, or explicitly document the softer policy and test it. Add coordinate-randomization controls stratified by structural confidence and missing-structure status.

**9. Medium priority: measurement error is preserved but unused**

There are **513,496 rows with nonmissing `measurement_sigma`**, but the dataset/loss path does not supply it to the likelihood. All measured scores within a weighted group are treated as equally precise observations. Predicted dispersion therefore conflates experimental noise, assay mismatch and residual model error.

**Required action:** first audit the meaning and units of each source's uncertainty estimate. For compatible standard errors, test a likelihood incorporating transformed measurement variance, for example total variance `predicted_variance + known_measurement_variance`. Compare against the existing loss; do not blindly inverse-weight heterogeneous error columns. Report replicate reliability/noise ceilings where source data permit, and sensitivity to low-quality measurements. Distinguish measurement noise, predictive residual dispersion and between-model disagreement.

**10. Medium priority: fields, mechanisms and ensembles remain unvalidated hypotheses**

All **562 bundled structure files contain one conformer**. Setting `max_conformers=4` does not create ensemble evidence; a single-conformer ablation on these files provides no contrast. No physical-teacher directory is configured in the examined default profiles. The response field is inferred through scalar assay supervision and can be useful predictively without corresponding uniquely to a physical perturbation.

The documentation already acknowledges this correctly. The scientific work still missing is independent field evaluation: mutation-specific structural/biophysical observations, held-out functional-site tests, distance/confidence-matched nulls, seed stability, and comparison with simple conservation or distance-from-site maps. Explicitly test that removing the field degrades transfer after controlling for local sequence processing and LLR. A graph/no-graph comparison alone bundles multiple architectural changes.

Related work already covers sequence/structure graph models, multi-property learning, and mechanistic genotype-to-phenotype models. A defensible contribution needs the specific field/observation-model hypothesis to survive these tests, not a claim that the ingredients are new. [PreMode](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/), [SynFit official implementation](https://github.com/luo-group/SynFit), [MoCHI](https://link.springer.com/article/10.1186/s13059-024-03444-y).

**11. Medium priority: undefined assays can silently disappear during checkpoint selection**

[metrics.py](archive/mipo/metrics.py:35) uses pandas means, which omit NaNs. [selection_score](archive/mipo/train.py:27) verifies the number of genes with defined correlations, not the number of assays. If a gene has one defined assay and one constant-prediction assay, the undefined assay can disappear while the gene still passes. The matched-LLR summary has a stricter all-assays check, so selection and comparison do not share the same policy.

**Required action:** fix the assay eligibility/undefined-prediction policy before training. Track every intended assay, distinguish constant targets from constant predictions, and fail or penalize according to a predefined rule. Do not let model-specific omissions improve a headline average. This is a confirmed code path risk; I did not find it responsible for the supplied PTEN metrics.

**12. Reproducibility and claim boundaries need a final release pass**

The release documentation still contains V3-specific design text and statements that V4 training has not occurred; the supplementary results are newer and use different code. The current workspace includes further changes, including curriculum selection and other training/model changes, that are absent from the ZIP. None should be silently attributed to the archived methodology.

**Required action:** publish one immutable code/data/config/environment manifest per experiment, label partial run matrices, record all exclusions and stage-selected checkpoints, and archive the exact code that generated results. Rebuild after fixes and rerun the affected experiments. Preserve this audit against the original ZIP hash.

For a biological claim, distinguish experimental score prediction from NDD disease mechanism and clinical pathogenicity. The present corpus does not test dosage, neuronal context, dominant-negative effects, gain of function, splicing or clinical outcomes comprehensively. Gene–disease inclusion evidence is not a variant label. Clinical or mechanistic claims would require an independent, appropriately designed validation dataset; adding a clinical classifier to these labels would not establish that evidence.

**Recommended experiment order**

1. **Repair the release and data identity checks:** final-stage selection; assay-context collisions; cross-deposit duplicates; gene/reference homology mapping; undefined-assay accounting. Rebuild an immutable ZIP. Record which fixes already exist in the loose workspace versus which remain unimplemented.
2. **Choose the main scientific question:** initially, rank non-consensus experimental effects on unseen genes. Keep absolute-score transfer, uncertainty, unseen-family transfer and physical fields as separate claims with separate success criteria.
3. **Freeze a development/confirmation boundary:** keep PTEN as development. Predefine an untouched, task-stratified gene panel and family-grouped splits. Fix models, budgets, metrics and minimum useful improvement before seeing those outcomes. Use at least the already planned three seeds, while recognizing that adding independent genes matters more than multiplying seeds on PTEN.
4. **Run the minimum informative comparison:** signed LLR; ESM/LLR simple supervised baseline; no structure; full model; then the component ablations needed to explain any gain. Evaluate direct, pooled broad-data and curriculum training with matched outer holdouts and declared compute budgets. Do not compare incomplete seed sets as if balanced.
5. **Resolve task support and calibration:** add independent functional assays and controls, or narrow the supported tasks. Report non-consensus task-specific performance, paired gene/family uncertainty, interval coverage and width. Use independently held-out study/source tests where feasible.
6. **Only then pursue field/mechanism claims:** acquire real ensemble or mutation-response supervision and independent tests. A full model should earn its additional complexity through transfer improvements or validated explanatory information.

**Decision standard:** It is reasonable to continue this project. It is not yet reasonable to claim improved generalization, calibrated NDD predictions, or identified perturbation mechanisms. A useful outcome could be a strong simpler predictor, a well-supported negative result about the field architecture, or a revised model whose advantage survives the above tests.
