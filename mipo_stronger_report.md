# Strengthen-MIPO Plan — Synthesized ONLY from Compact Evidence

Prior results used: `prior result 1` (structure-over-PLM), `prior result 2` (revive-structure-path),
`prior result 3` (multitask-contrast-recipes), `prior result 4` (robustness-eval),
`prior result 5` (data-side-strength), `prior result 7` (gate-init numerics).
`prior result 6` carried no inspectable values in this compact bundle.
Note from bundle: completeness critic unavailable.

Diagnosed pathologies assumed from the bundle: dead 0.7% field/structure path under late fusion;
modality competition / gradient starvation; stuck gates with wrong init; modality-dropout bias;
per-assay offset memorization; NLL/ranking mismatch and collapse; gene memorization and
pretraining leakage; structure-signal erasure under supervised fine-tune.

Rule: every item below is tied to at least one bundled evidence value. No outside claims added.

---

## 1. DO FIRST — pilot-scale, one pathology each, measurable effect

### F1. Prove and unstick modality competition on the dead 0.7% path

- **Pathology:** late-fusion joint training provably learns only a modality subset
  (modality competition); explains a dead 0.7% field path.
  <https://arxiv.org/abs/2203.12221>
- **Also:** cross-entropy learns only a feature subset (gradient starvation); fix needs
  regularization decoupling per-feature learning dynamics.
  <https://arxiv.org/abs/2011.09468>
- **Pilot (no scale rerun):**
  1. Compute conditional utilization rate: gain from adding each branch, and conditional
     learning speeds per branch. <https://arxiv.org/abs/2202.05306>
  2. If structure-branch utilization ~0, apply OGM-GE for one pilot run: scale down the
     dominant branch gradient by its contribution gap, restore noise with dynamic
     Gaussian jitter. <https://arxiv.org/abs/2203.15332>
  3. Add noisy top-k gating plus auxiliary load-balancing loss (MoE Sec.4 Balancing
     Expert Utilization) to unstick gates. <https://arxiv.org/html/1701.06538v1>
- **Expected measurable effect:** structure-branch conditional utilization rises from ~0;
  dominant-branch gradient norm drops when its contribution gap is large; gate load
  spreads off a single expert.
- **Measure:** per-branch conditional utilization; per-branch gradient norms by step;
  gate assignment histogram; held-out-gene Spearman before/after (must not regress).

### F2. Fix small-fusion gate init to verified Switch/MoE numerics

- **Pathology:** stuck / high-variance gates from ad-hoc init.
- **Pilot — use ONLY verified numerics:**
  - Switch Transformers: `TruncatedNormal(mu=0, sigma=sqrt(s/n))`, `s=1.0` reduced 10x
    to `s=0.1`, values `>2sigma` resampled; float32 inside router. Table 3: 0.1x-init
    neg-log-perp `-2.72 std 0.01` vs 1.0x-init `-3.60 std 0.68`.
    (prior result 7; Switch arXiv HTML v3 inspected)
  - mesh_tensorflow `moe.py` defaults: `loss_coef=1e-2`, `switch_jitter=1e-2`,
    `switch_temperature=1.0`; gate logits via `mtf.layers.dense` with no explicit gate
    initializer (inherits dense default). (prior result 7; raw inspected)
  - TF official `moe.py` Router defaults: `kernel_initializer=TruncatedNormal(stddev=2e-2)`,
    `bias_initializer=Zeros()`, `jitter_noise=0.0`, `router_z_loss_weight=0.0`.
    (prior result 7; raw inspected)
- **Not verified — do not cite as primary:** Shazeer noisy-gating formula
  `H(x)_i=(x.Wgate)_i+eps.Softplus((x.Wnoise)_i)` was seen only in secondary sources,
  not verified in the primary paper body this pass. (prior result 7)
- **Expected measurable effect:** run-to-run variance collapses (analogue: std 0.68 → 0.01);
  gates move off init within warm-up.
- **Measure:** gate-logit std and gate entropy over first N steps; 3-seed std of val
  Spearman/NLL; keep float32 router check.

### F3. Cap modality dropout to short warm-up-only

- **Pathology:** excess structure/field dropout biases the net to the kept branch,
  hurting full-input scores. <https://arxiv.org/abs/2403.04245>
- **Pilot:** keep any modality dropout short and warm-up-only; run one ablation with
  dropout-off vs warm-up-only vs full-training dropout.
- **Expected measurable effect:** full-input (both modalities present) score recovers
  relative to full-training dropout; no loss of branch utilization from F1.
- **Measure:** full-input Spearman + per-branch-missing Spearman; conditional utilization
  under each schedule. Optimal PLM+graph-hybrid dropout rate/schedule stays unresolved —
  report the tried rates, do not claim an optimum.

### F4. Add per-branch teeth: uni-modal teacher + per-branch losses

- **Pathology:** joint loss alone starves the weak branch (F1).
- **Pilot:**
  1. Keep fusion loss but add L2 distillation from pretrained uni-modal features
     (Uni-Modal Teacher; +3% on VGGSound precedent).
     <https://arxiv.org/abs/2106.11059>
  2. Add per-branch losses with fixed weights fit from separate uni-modal runs
     (Gradient-Blending; modalities overfit at different rates).
     <https://arxiv.org/abs/1905.12681>
- **Expected measurable effect:** weak-branch-only eval stops degrading during joint
  training; joint held-out-gene Spearman lifts (VGGSound +3% is the analogue, not a promise).
- **Measure:** joint Spearman + each branch-only Spearman across training; uni-modal-run
  weights recorded; L2-distillation loss curve.

### F5. Stop tuning to the wrong objective: Spearman early-stop + collapse tripwires

- **Pathology:** NLL/MSE improvements that do not transfer to ranking; constant-low
  collapse scoring well on skewed sets.
- **Pilot:**
  - Early-stop on Spearman, not NLL; leakage study benchmarks SRCC vs train-mean
    baselines and cites ranking losses beating MSE.
    <https://www.biorxiv.org/content/10.1101/2024.07.23.604678v1.full>
  - Detect collapse vs a train-mean predictor; FLIP GB1 is 96% low-fitness so
    constant-low outputs score well — fix by downsampling.
    <https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full>
  - Add a gene-label-only baseline: it hits AUROC 0.74 (CAGI6), rivaling phyloP;
    supervised predictors degrade on gene-balanced sets.
    <https://www.biorxiv.org/content/biorxiv/early/2024/06/08/2024.06.06.597828.full.pdf>
- **Expected measurable effect:** selected checkpoint changes vs NLL-stopped; collapse
  caught before reporting; gene-memorization exposed if MIPO ≤ gene-label-only on
  gene-balanced sets.
- **Measure:** NLL-best vs Spearman-best checkpoint SRCC gap; train-mean-baseline delta;
  gene-label-only AUROC/Spearman vs MIPO on the same split.

### F6. Install honest held-out-gene eval before claiming any gain

- **Pathology:** random splits and in-gene splits overstate transfer; frozen-ESM
  leakage inflates scores.
- **Pilot (eval-only, no retrain needed):**
  - Use MMseqs2 20%-identity whole-cluster train/test splits (FLIP Meltome) for
    family-aware held-out-gene eval.
    <https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full>
  - Expect a drop: random splits mislead (FLIP Table 7: all models score far higher
    on sampled splits); engineer val splits to mimic transfer. (same URL)
  - Add ProteinGym supervised Random/Contiguous/Modulo 5-fold CV; it warns
    same-position similar-AA train/test overlap inflates scores.
    <https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full-text>
  - Run one ≤30% no-homology retest (VariPred precedent: MCC 0.75 → 0.65) plus a
    balanced-label retest (MCC 0.623 vs 0.567) to expose gene memorization.
    <https://discovery.ucl.ac.uk/id/eprint/10190972/1/s41598-024-51489-7%20%281%29.pdf>
  - Run one pretraining-aware EPA-style split if ESM-pretrained (precedent: Spearman
    0.548 vs 0.603 on FLIP split, p=0.007 Wilcoxon).
    <https://www.biorxiv.org/content/10.1101/2024.07.23.604678v1.full>
- **Expected measurable effect:** honest transfer number below in-gene/sampled number;
  magnitude of the gap quantifies leakage/memorization (analogues: −0.055 Spearman,
  −0.10 MCC).
- **Measure:** sampled-split vs cluster-split vs no-homology-split Spearman/MCC;
  inter-assay ceiling context: ESM-1v zero-shot ranked #1 of 55 VEPs on 26 human DMS
  with median inter-assay r=0.54 as a transfer ceiling.
  <https://www.pure.ed.ac.uk/ws/portalfiles/portal/356490544/msb.202211474_Livesey_and_Marsh.pdf>
  Also log per-gene AUROC spread: VEP AUROC varies hugely over 963 genes and disorder
  inflates AUROC via benign enrichment.
  <https://www.biorxiv.org/content/10.1101/2024.06.12.598724v1.full-text>

### F7. Joint multi-assay fitting without hand weights: per-experiment affine head + uncertainty/GradNorm + L2-SP

- **Pathology:** per-assay scale/offset memorization; hand-tuned task weights; loss of
  pretrained knowledge on small databases.
- **Pilot:**
  - MoCHI infers per-experiment affine scale+shift, fitting many DMS sets jointly with
    no explicit normalization.
    <https://www.biorxiv.org/content/10.1101/2024.01.21.575681v1.full-text>
  - MoCHI 3-state model precedent: held-out double-mutant binding R²=0.94; inferred
    folding ddG matches in vitro; effects additive at energy level. (same URL)
  - ddPCA precedent: pair BindingPCA+AbundancePCA; MoCHI fits both phenotypes jointly
    via extra design rows to split folding vs binding ddG. (same URL)
  - Kendall-Gal per-task log-variance `s`, loss `exp(-s)L+s`; 3-task Cityscapes 63.4%
    beats single-task and grid weights. <https://arxiv.org/html/1705.07115v3>
    Duplicate pointer for assay reweighting: <https://arxiv.org/abs/1705.07115>
  - Or GradNorm: balances gradient norms via learned `w_i`, one alpha; ~5% overhead;
    matches/exceeds grid search; beats uncertainty weighting.
    <https://ar5iv.labs.arxiv.org/html/1711.02257>
  - Preserve pretraining with L2-SP: penalize `||w−w0||` to init instead of origin;
    critical regularizer on small databases.
    <https://ar5iv.labs.arxiv.org/html/1802.01483>
- **Expected measurable effect:** joint fit converges without manual assay normalization;
  uncertainty/GradNorm matches or beats grid weights at ~5% overhead (GradNorm analogue).
- **Measure:** per-assay scale/shift values; task-weight trajectories; joint vs
  single-task Spearman; `||w−w0||` trace.

### F8. Add a ranking loss next to regression (one-list pilot)

- **Pathology:** ranking ≠ regression; MSE/NLL tuning leaves rank gains on the table.
- **Pilot:** add one listwise or pairwise loss from TF-Ranking (pairwise logistic,
  ListNet, ListMLE, softmax-CE); listwise wins on Gmail/Drive; ranking ≠ regression.
  <https://ar5iv.labs.arxiv.org/html/1812.00073>
  Context: RankNet → LambdaRank → LambdaMART progression; LambdaMART ensemble won
  Yahoo LTR Challenge Track 1.
  <https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/>
- **Expected measurable effect:** Spearman/NDCG-style rank metric rises with flat or
  worse MSE — that divergence is the win, consistent with F5.
- **Measure:** Spearman + MSE jointly; report both; keep F5 Spearman early-stop so the
  rank gain selects the checkpoint.

### F9. Pilot serial fusion (PLM embeddings → geometric net) before fixing late fusion

- **Pathology:** late fusion is the failing topology (F1); serial injection has the
  supervised precedent.
- **Pilot (frozen, cheap):** feed frozen PLM embeddings into the geometric net.
  Precedents: feeding PLM embeddings into geometric nets gives +20% over
  structure-only on interface/MQA/docking/affinity tasks.
  <https://arxiv.org/abs/2212.03447>
  TransFun (ESM + 3D-equivariant GNN) beats prior SOTA on CAFA3/new sets; equivariant
  path wins when supervised.
  <https://www.biorxiv.org/content/10.1101/2023.01.17.524477v1>
  Serial fusion (ESM-2 embeddings into structure encoder) plus structure pretraining
  sets SOTA function annotation. <https://arxiv.org/abs/2303.06275>
- **Expected measurable effect:** supervised ranking lift over structure-only and over
  current late fusion at frozen-embedding cost.
- **Measure:** structure-only vs PLM-only vs serial vs late-fusion Spearman on the same
  F6 cluster split; EGNN-seq corr 0.212 precedent says ensemble if serial underwhelms
  (see N2). <https://ar5iv.labs.arxiv.org/html/2306.12231v2>

---

## 2. DO NEXT — needs reruns, scale, or new data plumbing

### N1. Pretrain the structure encoder self-supervised (multiview contrast) before joint tuning

- Pretrain the structure encoder self-supervised (multiview contrast best on 7/8 sets)
  before joint tuning; rivals SOTA PLMs. <https://arxiv.org/abs/2203.06125>
- GearNet Multiview Contrast: EC 0.874, GO-BP 0.490 vs ESM-1b 0.864/0.452, using 805K
  structures vs 24M seqs. <https://ar5iv.labs.arxiv.org/html/2203.06125>
- Needs a structure-pretraining rerun; do after F9 proves serial topology on frozen
  embeddings.

### N2. Decide joint-vs-ensemble on stability-concentrated sets (do not assume joint wins)

- SaProt-650M beats ESM-2 zero-shot: Mega-scale 0.574v0.478, ProteinGym 0.457v0.414,
  ClinVar 0.909v0.862; 35M beats ESM2-15B.
  <https://www.biorxiv.org/content/10.1101/2024.05.24.595648v2.full>
- SaProt ranked 1st on blind ProteinGym leaderboard over 60+ methods; rivals
  ProteinMPNN design at 16x speed. (same URL)
- ProSST tops ProteinGym zero-shot (217 assays) incl best on 66-assay stability split;
  gains concentrate in stability.
  <https://www.biorxiv.org/content/10.1101/2024.04.15.589672v3.full>
- GEM25: structure models best on stability; simple StructSeq ensemble beats joint
  ProtSSN/SSEmb/SaProt/TranceptEVE. <https://ar5iv.labs.arxiv.org/html/2504.16886>
- GVP/EQGAT match Tranception on better-than-WT ranking on 181x fewer molecules;
  EGNN-seq corr 0.212 → ensemble. <https://ar5iv.labs.arxiv.org/html/2306.12231v2>
- Supervised fine-tune erases gap: SaProt equals ESM-1b/2 on Fluorescence, Stability,
  lactamase, AAV sans structure.
  <https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5.full>
- Action: rerun MIPO vs a simple StructSeq-style ensemble on a stability-heavy,
  held-out-gene split. If the ensemble wins, ship the ensemble; do not force joint fusion.
  Blocked claim: no published ablation of supervised structure+PLM fusion gain on
  strictly held-out-gene splits — MIPO must produce its own.

### N3. Scale stability supervision the proven ways (teachers + Megascale-class data)

- ThermoMPNN: PCC 0.754 vs 0.43 ProteinMPNN-zero-shot; Ssym 0.72/0.60, S669 0.43;
  full fine-tune overfits. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10861915/>
- Transfer + 776k Megascale points: SCC 0.725/0.657, 56% stabilizing PPV;
  Fireprot-only falls to 0.49/0.35. (same URL)
- ESM-IF summed likelihoods predict absolute dG: r=0.69, RMSE 1.4 kcal/mol (n=265);
  ESM-1b 0.25, FoldX 0.04. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11645669/>
- ESM-IF1: inverse folding on 12M AlphaFold2 structures; 51% recovery, 72% buried,
  ~10pp gain; zero-shot ddG teacher. <https://proceedings.mlr.press/v162/hsu22a.html>
- RaSP distills Rosetta ddG (Pearson 0.82, MAE 0.73) at 480–1036x speed, S669 parity;
  230M human ddGs split benign/pathogenic.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10266766/>
- Stability Oracle: Thermodynamic Permutations grow 120K to 2M ddGs over all 380 types;
  48% recall / 74% precision stabilizing.
  <https://www.biorxiv.org/content/10.1101/2023.05.15.540857v1>
- ESMtherm warning: ESM on 528k seqs/461 domains generalizes to distal seqs but fails
  larger scaffolds; needs length-diverse data.
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012248>
- Action: add one distilled teacher (RaSP/ESM-IF-style) + Megascale-class points +
  Thermodynamic-Permutation augmentation; freeze or L2-SP the transfer backbone
  (full fine-tune overfits); include length-diverse scaffolds.

### N4. Wire paired folding-vs-function supervision (ddPCA/MoCHI-style)

- PTEN+NUDT15: 6,749 paired activity+abundance effects; 1/3 LoF, ~1/2 of LoF also
  low-abundance; stability predicts sites.
  <https://www.biorxiv.org/content/10.1101/2020.09.28.317040v1>
- VAMP-seq: abundance for 7,595 PTEN+TPMT variants; 1,079 PTEN and 805 TPMT
  low-abundance, clinically actionable. <https://www.biorxiv.org/content/10.1101/211011v1>
- GCK: abundance for 95% of missense+nonsense variants; 43% of hypoactive variants
  show decreased abundance.
  <https://www.biorxiv.org/content/10.1101/2023.05.24.542036v1>
- VKOR: 2,695 abundance + 697 activity variants in human cells; 25% of human variants
  reduced. <https://www.biorxiv.org/content/10.1101/2020.05.10.087312v1>
- RBD DMS maps every mutation for ACE2 binding and expression-as-stability; ready
  paired mechanism set.
  <https://github.com/jbloomlab/sars-cov-2-rbd_dms/blob/HEAD/results/dms_view/description_spike.md>
- ProteinGym: 250+ standardized DMS assays, millions of sequences, 70+ models,
  zero-shot+supervised ranks; mine for paired genes.
  <https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1>
- Action: fit BindingPCA+AbundancePCA-style pairs jointly via MoCHI extra design rows
  to split folding vs binding ddG (F7), starting from the RBD + PTEN/NUDT15 + GCK +
  VKOR paired sets. Needs data plumbing + joint reruns.

### N5. Gradient-surgery and zero-shot-task upgrades (after F7)

- PCGrad projects task grads onto normal plane when cosine<0; gains on CelebA-40,
  CIFAR100-20, NYUv2, MT10/MT50 success+efficiency.
  <https://ar5iv.labs.arxiv.org/html/2001.06782>
- Hyper-X: one hypernet maps task+language embeddings to adapter weights; zero-shot to
  unseen pairs; beats mBERT/PSF on mixed sources.
  <https://ar5iv.labs.arxiv.org/html/2205.12148>
- SupCon: all same-class samples as positives; ResNet-200 ImageNet 81.4% (+0.8 SOTA);
  tau=0.1, smaller better; robust to corruptions.
  <https://ar5iv.labs.arxiv.org/html/2004.11362>
- SupCon subsumes triplet/max-margin/N-pairs (N-pairs 57.4% vs SupCon 78.7% R50);
  multi-view batch + many positives key. (same URL)
- Action: try PCGrad when F7 task grads conflict (cosine<0); try Hyper-X-style adapters
  for unseen assay/gene pairs; try SupCon tau=0.1 multi-view batch only if a
  contrastive head is added. All need reruns.

### N6. Structure-source and disorder hygiene reruns

- AF2 structures beat PDB for ESM-IF1 zero-shot; IDRs in 63/217 assays degrade
  sequence+structure models alike. <https://ar5iv.labs.arxiv.org/html/2504.16886>
- Action: rerun structure branch on AF2 vs PDB inputs; report IDR-stratified metrics;
  do not pool IDR-heavy assays with folded ones. Needs structure rebuild + reruns.

### N7. Per-new-gene recalibration (classification-grade first)

- Recalibrate per new-gene prior: CAGI6 applies Pejaver probabilistic calibration
  mapping scores to ACMG strengths given prior.
  <https://www.biorxiv.org/content/biorxiv/early/2024/06/08/2024.06.06.597828.full.pdf>
- PreMode reports only within-gene 80/20 transfer results (8 DMS + 9 G/LoF genes);
  zero-shot cross-gene is the open gap.
  <https://www.biorxiv.org/content/10.1101/2024.02.20.581321v1.full-text>
- Action: add Pejaver-style calibration for pathogenicity-style outputs on new genes;
  keep zero-shot cross-gene as the headline gap. Regression (mu,sigma) recalibration
  numbers are unresolved — do not invent them.

---

## 3. EXPLICITLY REJECTED (with bundled reasons)

1. **Long / full-training modality dropout.** Rejected: excess dropout biases the net to
   the kept branch, hurting full-input scores. <https://arxiv.org/abs/2403.04245>
2. **Pure joint late-fusion with no per-branch teeth.** Rejected: provably learns a
   modality subset <https://arxiv.org/abs/2203.12221> and cross-entropy starves
   features <https://arxiv.org/abs/2011.09468>; F1/F4 auxiliaries are mandatory.
3. **ESM-IF-style inverse-fold head as the sole zero-shot ranker.** Rejected: ESM-IF
   worst on both zero-shot tasks (inverse-fold objective).
   <https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5.full>
   Allowed: ESM-IF summed likelihoods for absolute dG (r=0.69) or as a ddG teacher.
4. **MSA retrieval as the fusion-gap fix.** Rejected as equalizer: MSA retrieval lifts
   all PLMs, SaProt stays top. (same URL)
5. **Claiming joint SOTA from supervised fine-tune without a held-out-gene ablation.**
   Rejected: supervised fine-tune erases the SaProt vs ESM-1b/2 gap sans structure
   (same URL), and no published ablation of supervised structure+PLM fusion gain on
   strictly held-out-gene splits exists (unresolved). Produce the ablation or drop the claim.
6. **Full fine-tune of a stability-transfer backbone.** Rejected: ThermoMPNN full
   fine-tune overfits; transfer + frozen/L2-SP is the verified recipe.
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC10861915/>
7. **Sampled/random splits, in-gene 80/20 only, or gene-unbalanced reporting as transfer
   evidence.** Rejected: FLIP Table 7 random-split inflation
   <https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full>,
   PreMode within-gene-only gap
   <https://www.biorxiv.org/content/10.1101/2024.02.20.581321v1.full-text>,
   CAGI6 gene-label-only AUROC 0.74.
   <https://www.biorxiv.org/content/biorxiv/early/2024/06/08/2024.06.06.597828.full.pdf>
8. **Early-stopping on NLL.** Rejected: stop on Spearman; ranking losses beat MSE in the
   leakage benchmark.
   <https://www.biorxiv.org/content/10.1101/2024.07.23.604678v1.full>
9. **Quoting AlphaMissense ablation magnitudes as verified.** Rejected this pass:
   primary ablation magnitudes unverified (Science p.403); only the secondary report
   (ClinVar auROC 0.940 vs EVE 0.911; AF pretrain+finetune both essential) was fetched.
   <https://cbirt.net/googles-deepmind-unveils-alphamissense-a-breakthrough-ai-tool-for-decoding-genetic-enigmas/>
10. **Quoting FoldX accuracy numbers.** Rejected: EuropePMC fetch returned shell only,
    PNAS page 403 (unresolved).
11. **Picking E(3)-equivariant vs invariant GNN for supervised DMS ranking.** Rejected:
    no head-to-head on supervised DMS ranking over PLM embeddings; only zero-shot
    better-than-WT exists (unresolved).
12. **Novel DMS per-assay offset-memorization penalty.** Rejected: no fetched source;
    L2-SP is the closest verified analogue. <https://ar5iv.labs.arxiv.org/html/1802.01483>
13. **DWA (Dynamic Weight Averaging) adoption.** Rejected until fetched; covered by
    UW/GradNorm/PCGrad instead (unresolved).
14. **Single numeric scheme for sigmoid-style (non-MoE) fusion gates.** Rejected: no
    single standard; value depends on start-open vs start-closed intent (unresolved).
    Use only the verified MoE/Switch numerics in F2.
15. **Citing PTEN phosphatase-MAVE counts, LoMuS +10% Spearman, or per-gene regression
    (mu,sigma) recalibration numbers.** Rejected: snippet-located or unfetched; Pejaver
    is classification-threshold based, not regression (unresolved).
16. **Claiming an early-stopping-rule head-to-head winner.** Rejected: no study compares
    in-gene-val vs held-out-gene-val stopping for transfer; FLIP/ProteinGym prescribe
    schemes without comparing them (unresolved).

---

## 4. WHAT TO MEASURE — decision table

| Intervention | Primary metric (must move) | Guardrail (must not regress) | Bundled analogue / threshold |
|---|---|---|---|
| F1 OGM-GE + noisy top-k + balance loss | structure-branch conditional utilization ↑ from ~0 | joint held-out-gene Spearman | utilization gain per 2202.05306 |
| F2 0.1x gate init, float32 router | 3-seed std ↓; gate entropy moves off init | val Spearman | std 0.68 → 0.01 analogue |
| F3 warm-up-only dropout | full-input Spearman recovers vs full-training dropout | branch-missing Spearman | bias warning 2403.04245 |
| F4 uni-modal L2 + per-branch losses | weak-branch-only Spearman stops decaying; joint ↑ | uni-modal-only baselines | +3% VGGSound analogue |
| F5 Spearman stop + collapse + gene-only baselines | Spearman-best vs NLL-best gap; train-mean delta | — (diagnostic) | GB1 96% skew; gene-only AUROC 0.74 |
| F6 cluster / no-homology / EPA splits | cluster-split Spearman (honest number) | sampled-split gap reported | −0.055 Spearman; MCC 0.75→0.65; ceiling r=0.54 |
| F7 affine head + UW/GradNorm + L2-SP | joint multi-assay Spearman ≥ best single-task | per-assay calibration; `||w−w0||` | Cityscapes 63.4%; GradNorm ~5% overhead; R²=0.94 double-mutant analogue |
| F8 ranking loss | Spearman ↑ at flat/worse MSE | MSE reported, not hidden | listwise wins; LambdaMART precedent |
| F9 serial PLM→geometric | serial vs late-fusion Spearman on F6 split | structure-only baseline | +20% over structure-only analogue |
| N1 multiview-contrast pretrain | EC/GO-style probe + downstream Spearman | — | EC 0.874/GO-BP 0.490; best 7/8 sets |
| N2 ensemble-vs-joint | stability-split Spearman; StructSeq-vs-joint delta | held-out-gene constraint | SaProt/ProSST/GEM25 numbers above |
| N3 teachers + Megascale + permutations | SCC/PCC; stabilizing PPV/recall/precision | full-fine-tune overfit check; length stratification | SCC 0.725/0.657; 56% PPV; 48%/74%; r=0.69 dG |
| N4 paired folding/function | folding-vs-binding ddG split quality; held-out double-mutant R² | per-phenotype calibration | R²=0.94; 6,749/7,595/95%/2,695+697/RBD counts |
| N5 PCGrad / Hyper-X / SupCon | conflict-cosine-gated gain; unseen-pair zero-shot; SupCon acc | overhead | cosine<0 rule; tau=0.1; 81.4% / 78.7-vs-57.4 analogues |
| N6 AF2-vs-PDB + IDR split | IDR-stratified Spearman; AF2-vs-PDB delta | — | 63/217 IDR-degraded |
| N7 Pejaver calibration | calibrated ACMG-strength accuracy on new genes | prior sensitivity | classification-only; no regression mu,sigma |

---

## 5. UNRESOLVED (carried from bundle — do not claim)

- Optimal structure-dropout rate/schedule values specific to PLM+graph hybrids.
- No head-to-head of E(3)-equivariant vs invariant GNN on supervised DMS ranking over
  PLM embeddings; only zero-shot better-than-WT exists.
- No published ablation of supervised structure+PLM fusion gain on strictly
  held-out-gene splits.
- AlphaMissense primary ablation magnitudes unverified (Science p.403).
- No fetched source for DMS-specific per-assay offset-memorization penalties; L2-SP is
  the closest verified analogue.
- DWA balancing recipe not fetched; covered by UW/GradNorm/PCGrad instead.
- FoldX accuracy numbers (original vs independent r): EuropePMC fetch shell only, PNAS 403.
- PTEN phosphatase-MAVE counts (Mighell et al.): snippet-located, page not fetched.
- LoMuS +10% Spearman claim: snippet-located, page not fetched.
- No fetched DMS study gives per-gene recalibration numbers for regression (mu,sigma)
  on unseen genes; Pejaver method is classification-threshold based.
- No head-to-head study comparing early-stopping rules (in-gene val vs held-out-gene
  val) for transfer prediction.
- Canonical numeric scheme for sigmoid-style (non-MoE) fusion gates: no single standard;
  depends on start-open vs start-closed intent.

---

## 6. Suggested pilot order (fewest reruns first)

1. F6 (eval-only) + F5 (stop/baselines) — establishes the honest baseline.
2. F2 (gate init) + F1 (OGM-GE + balance loss) + F3 (dropout cap) — unstick the dead path.
3. F4 (teacher + per-branch losses) + F7 (affine + UW/GradNorm + L2-SP) — joint teeth.
4. F8 (ranking loss) + F9 (serial pilot) — rank objective + topology check.
5. N1–N7 only after F-pilots pick a topology and an honest split.
