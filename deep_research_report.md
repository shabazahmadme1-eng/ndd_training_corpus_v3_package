# Stability-vs-Function Mechanism Prediction: Novelty, Data Constraint, and Scaled Design

Synthesized from compact evidence bundle (refs: prior results 1–7). All claims below trace to cited evidence values; URLs are the fetched sources.

> CORRECTION (2026-10-03, verified against primary source): the "NOVEL" verdict in §1 is wrong. PreMode (Nature Communications 2025, DOI 10.1038/s41467-025-62318-4, PMC12325985) already predicts per-variant mode-of-action (stability vs function) with an SE(3)-equivariant GNN over protein 3D structure, trained on 41,081 DMS variants in 8 genes (PTEN, SNCA, CCR5, CXCR4, NUDT15, CYP2C9, GCK, ASPA). The remaining gap is narrower: PreMode needs protein-specific transfer learning and does not demonstrate zero-shot generalization to fully held-out genes — that is the only novelty left for a whole-gene-holdout design. Its 8-gene ceiling independently corroborates the local ~11-gene paired-assay limit.

## 1. Novelty verdict + strongest competing work

### Verdict: NOVEL — no published supervised DMS/MAVE model classifies destabilizing vs function-disrupting-while-folded

The supervised prior-art sweep finds a uniform single-score pattern:

- Envision: gradient boosting on 27 evol/struct/physchem features, fit to 21,026 DMS scores; single effect score, no mechanism split. https://pmc.ncbi.nlm.nih.gov/articles/PMC5799033/
- DeMaSk: OLS linear model on 3 features (entropy, variant log-freq, DMS matrix), fit to ~109k scores; single impact score. https://pmc.ncbi.nlm.nih.gov/articles/PMC8016454/
- ProteinNPT: NPT over MSA-Transformer embeddings, supervised per-assay fitness regression; single fitness output, no mechanism split. https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1.full
- ProteinNPT multi-property mode jointly predicts several assay labels; still per-assay fitness scores, not a mechanism classifier. https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1.full
- ProteinGym supervised DMS: OHE models (+zero-shot augmentation, Hsu protocol) trained per assay with CV; one fitness target each. https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/
- Hsu-style head: ridge on one-hot sequence + unsupervised scores (DeepSequence/MSA-T), fit per assay; one fitness target. https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1.full
- Schulze et al.: ESM-IF + abundance net + per-experiment mapping, trained on 6 VAMP-seq sets (LOPO); abundance only. https://www.biorxiv.org/content/10.1101/2025.04.02.646878v1.full
- MAVE-NN: per-dataset G-P map (sequence to latent phenotype) + measurement model; one phenotype each, no cross-protein transfer. https://www.biorxiv.org/content/10.1101/2020.07.14.201475v2.full
- AlphaMissense: single 0–1 pathogenicity score; explicitly no structure/stability prediction. https://www.insideprecisionmedicine.com/topics/precision-medicine/alphamissense-classifies-mutation-pathogenicity/
- VERDICT: no published supervised DMS/MAVE model classifies destabilizing vs function-disrupting-while-folded; all emit one score. https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/

No fetched source shows a scaled 3D GNN jointly predicting destabilizing-vs-functional mechanism labels (unresolved item, carried forward).

### Strongest competing / adjacent work (what is closest and why it is not the same)

Ranked by closeness to a stability-vs-function classifier:

1. **ddPCA + MoCHI — the mechanism-label gold standard, but per-protein fit, not a transferable predictor.** Same yeast variant library read by AbundancePCA + BindingPCA (PSD95-PDZ3/CRIPT all singles+doubles); MoCHI fits aPCA via TwoStateFractionFolded, bPCA via ThreeStateFractionBound and infers per-variant dG-fold + dG-bind; held-out R² 0.83/0.91 binding, 0.76/0.74 abundance; inferred energies match in vitro (r 0.7–0.98). First allosteric maps of two human proteins; Faure et al. Nature 2022 mapped energetic+allosteric landscapes of GRB2/PSD95-PDZ3. This is the cleanest supervision signal for the proposed task, but it does not transfer across proteins. https://www.biorxiv.org/content/10.1101/2024.01.21.575681v2.full https://github.com/lehner-lab/MoCHI https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:35388192&format=json https://www.genengnews.com/topics/drug-discovery/novel-technique-reveals-potential-pathways-that-control-protein-function/
2. **LoGoFunc — the closest supervised mechanism classifier template, but on the orthogonal GOF-vs-LOF axis.** 3-class GOF/LOF/neutral soft-voting ensemble of 27 LightGBM classifiers on 474 encoded/imputed/scaled features with 3 probability outputs; training 1492 GOF + 13524 LOF (HGMD+NLP, ~90% TPR) + 13361 gnomAD neutral; 90/10 gene-disjoint split; nested 5×5 CV + Optuna macro-F1; oversampled GOF/neutral; 474 features = AF2 structure (RSA/contacts/DSSP/ConCavity/DDGun/GraphBind/pLDDT/3D-proximal counts) + conservation/pathogenicity/splice + gene-level + STRING node2vec 64-dim PPI; held-out AP 0.52/0.93/0.96, F1 0.56/0.87/0.89, MCC 0.54/0.75/0.80 (GOF/LOF/neutral); GOF-vs-LOF AP 0.63; ClinVar AP 0.52/0.98/0.99. Proves a supervised mechanism-polarity classifier is trainable and evaluable; does not separate destabilizing from folded-but-dysfunctional. Source: PMC10688473 full text inspected (prior result 7). Springer page fetch blocked, snippet-only.
3. **MAVISp — the closest modular stability/binding/allostery split, but via physics tools + curation, not supervised DMS learning.** https://www.biorxiv.org/content/10.1101/2022.10.22.513328v3.full
4. **SSEmb / ThermoMPNN / RaSP — structure-aware stability predictors that prove the stability half, stability-only.** SSEmb (MSA-T+GVP-GNN) vs MSA Transformer on ProteinGym: 0.45 vs 0.42 overall, 0.45 vs 0.39 on low-MSA subset; matches GEMME on activity MAVEs but wins on abundance MAVEs; zero-shot mega-scale stability Spearman 0.61, near dedicated predictors. ThermoMPNN: frozen ProteinMPNN embeddings + light-attention head; SSYM PCC 0.72 direct/0.60 inverse, SOTA; on Megascale (<75aa, 273k muts) SCC 0.725/0.657 Mega/FireProt; RMSE<1.0 under 100aa, rises after; S669 PCC 0.43; code+weights open, follow-ups -D (doubles) and -I (indels). RaSP: self-supervised 3D-CNN, fine-tuned on Rosetta ddG; test Pearson 0.82, MAE 0.73 kcal/mol; scanned 1381 human structures (8.8M ddG); crystal vs AlphaFold2 agree. Stability predictors train stability-only (RaSP Table 2: ACDC-NN, DDGun3D, PremPS, Dynamut, DUET). Retrained RaSP/PROSTATA gain on Mega test, not Fireprot-HF; ±5 vs ±10 kcal/mol range caps extrapolation. https://www.biorxiv.org/content/10.1101/2023.12.14.571755v1.full-text https://www.biorxiv.org/content/10.1101/2023.07.27.550881v1.full https://github.com/Kuhlman-Lab/ThermoMPNN https://pmc.ncbi.nlm.nih.gov/articles/PMC10266766/
5. **ProteinNPT multi-property + Schulze per-experiment mapping — multi-assay precedent without mechanism heads.** Joint per-assay fitness prediction and per-experiment mapping trained LOPO on 6 VAMP-seq sets (abundance only) show how to share representation across assays; neither emits a mechanism label. https://www.biorxiv.org/content/10.1101/2023.12.06.570473v1.full https://www.biorxiv.org/content/10.1101/2025.04.02.646878v1.full

### Structure-model context (why structure is required, and its limits)

- SaProt-35M seq-only matches ESM-2 (ProteinGym 0.337 vs 0.339; ClinVar 0.738 vs 0.722); adding 3Di tokens jumps to 0.392/0.794. https://github.com/westlake-repl/SaProt
- SaProt-650M vs ESM-2-650M: ClinVar AUC 0.862→0.909, Thermostability 0.680→0.724, HumanPPI 76.7→86.4, ProteinGym 0.475→0.478. https://github.com/westlake-repl/SaProt
- ESM-GearNet beats ESM2 by 9.7% on HumanPPI; SaESM2 lifts Contact P@L/5 +59% on CASP16, wins supervised Stability, ~flat on 9 function tasks. https://arxiv.org/html/2505.16896v2
- ProSST: every K>0 structure model beats no-structure K=0 on all metrics; dropping R2S attention raises PPL 9.03→12.14. https://www.biorxiv.org/content/10.1101/2024.04.15.589672v3.full
- ProtSSN: frozen ESM2-t33 node features + 6-layer EGNN on kNN graph, pretrained on 30,948 CATH domains; equivariant EGNN beats GCN/GAT; 6 EGNN layers suffice (12/18 no gain); interaction/stability assays more structure-sensitive than catalysis/activity ones. https://www.biorxiv.org/content/10.1101/2023.12.01.569522v2.full-text
- LM-GVP (ProtBERT+GVP end-to-end) best on all GO splits, wins 82.7% of GO terms; gains ~nil on Fluorescence/Protease; GVP>GAT. https://www.biorxiv.org/content/10.1101/2021.09.21.460852v1.full
- PrimateAI-3D: 2Å-voxel 3D-CNN over AF structure+MSA on 4.5M benign variants; top of 16 on ClinVar/DMS/DDD/ASD/CHD/UKBB. https://www.biorxiv.org/content/10.1101/2023.05.01.538953v1.full-text

Takeaway: structure helps stability/interaction/PPI most, catalysis/activity least — consistent with needing an explicit stability head plus a separate function-given-folded head rather than one fitness score.

## 2. Binding data constraint

**Constraint: paired stability+function DMS over shared variants exists at scale for only a handful of proteins; everything else is stability-only, abundance-only, fragment-only, or unpaired.**

Verified paired cases (same or overlapping library, both readouts):

- PTEN: yeast lipid-phosphatase MAVE + VAMP-seq abundance; 2,819 variants scored by all 4 methods; ~60% pathogenic LoF via destabilization. https://www.biorxiv.org/content/10.1101/688234v2.full
- NUDT15: VAMP abundance (2,922 vars) + thiopurine cytotoxicity (2,935 vars), same HEK293T library; 17 residues alter activity without stability loss. https://pmc.ncbi.nlm.nih.gov/articles/PMC7071893/
- CYP2C9: click-seq activity (6,142 vars, yeast) + VAMP abundance (6,370 vars, human); abundance explains ~half of function variance. https://pmc.ncbi.nlm.nih.gov/articles/PMC8456167/
- VKOR: VAMP abundance (2,695 vars) + activity (697 vars) in human cells; activity/abundance ratio maps active site. https://pmc.ncbi.nlm.nih.gov/articles/PMC7462613/
- ddPCA: PSD95-PDZ3/CRIPT all singles+doubles with AbundancePCA + BindingPCA; GRB2/PSD95-PDZ3 energetic+allosteric landscapes. https://www.biorxiv.org/content/10.1101/2024.01.21.575681v2.full

Verified non-paired or single-axis resources (useful for pretraining, not mechanism labels):

- Schulze 6 VAMP-seq sets: abundance only, LOPO. https://www.biorxiv.org/content/10.1101/2025.04.02.646878v1.full
- CYP2C19: VAMP abundance of 7,660 vars; 4,670 jointly analyzed with CYP2C9 abundance; stability-specificity tradeoff (abundance-only). https://pmc.ncbi.nlm.nih.gov/articles/PMC11538415/
- cDNA display proteolysis: 776k high-quality dG over 331 natural+148 designed 40–72aa domains. https://econpapers.repec.org/article/natnature/v_3a620_3ay_3a2023_3ai_3a7973_3ad_3a10.1038_5fs41586-023-06328-6.htm
- Megascale limits: all proteins <75aa, 400+ measures each; dDG range −3..5 vs FireProt −9..12 kcal/mol. https://www.biorxiv.org/content/10.1101/2023.07.27.550881v1.full
- ProtaBank: one repo for activity/binding/stability/folding/solubility DMS+lit data, full variant sequences, assay-annotated; Gb1/ubiquitin/lactamase cases. https://pmc.ncbi.nlm.nih.gov/articles/PMC5980626/
- ProThermDB: 31,580 stability entries (dG, dTm, dH, dCp) with methods/conditions, UniProt+PDB linked, free download. https://pmc.ncbi.nlm.nih.gov/articles/PMC7778892/
- ThermoMutDB: >14,669 curated ddG/dTm points over 588 proteins, +83% unique mutations vs prior DBs, REST API. https://pmc.ncbi.nlm.nih.gov/articles/PMC7778973/
- FireProtDB: manually curated single-point thermostability (ddG/dTm), merges ProTherm+ProtaBank+lit, ML-ready tables. https://pmc.ncbi.nlm.nih.gov/articles/PMC7778887/

Verified gaps (do not plan around them as if they exist):

- Count of genes with paired stability+function DMS over shared variants not verified from fetched pages.
- AVE Alliance hosts no stability+function-paired dataset independent of MaveDB (it develops MaveDB); no separate AVE pairing found.
- TPMT: VAMP-seq abundance exists but no overlapping full-length activity DMS located.
- GALT: only low-throughput activity/abundance per-variant data found, no paired DMS.
- No MAVE consortium beyond AVE/ProteinGym aggregating paired stability+function assays found.

**Design implication:** the mechanism head must be trained on ~5–6 paired proteins/domains (PTEN, NUDT15, CYP2C9, VKOR, PSD95-PDZ3/CRIPT, GRB2) plus derived ratio/threshold labels, with stability and function backbones pretrained on the large unpaired resources. Any claim of scaled training on dozens of paired genes is currently unsupported. Whole-gene holdout is mandatory because so few genes carry the mechanism label.

## 3. Credible scaled design

### 3a. Data

- **Stability pretraining (unpaired, large):** cDNA display 776k dG (40–72aa) + Megascale 273k muts (<75aa) + ProThermDB 31,580 + ThermoMutDB 14,669/588 proteins + FireProtDB ML-ready tables. Treat Megascale/cDNA as fragment-scale; FireProt/ProTherm/ThermoMutDB supply full-length range (−9..12 kcal/mol) that Megascale lacks (−3..5).
- **Abundance pretraining:** Schulze 6 VAMP-seq sets + CYP2C19 7,660 + CYP2C9 6,370 + NUDT15 2,922 + VKOR 2,695 + PTEN VAMP fraction. Per-experiment mapping head required (Schulze precedent).
- **Paired mechanism gold (small, precious):** PTEN 2,819×4-method; NUDT15 VAMP+cytotoxicity same-library; CYP2C9 activity+abundance; VKOR 2,695/697 activity/abundance ratio; ddPCA PSD95-PDZ3/CRIPT + GRB2/PSD95-PDZ3 with MoCHI dG-fold/dG-bind. Derive three mechanism labels: destabilizing (low abundance / large +ddG), function-disrupting-while-folded (normal abundance, low activity/binding — e.g. NUDT15 17 residues, VKOR active-site ratio hits), neutral.
- **Auxiliary fitness (ProteinGym 250+ DMS assays, zero-shot+supervised tracks, plus clinical sets):** multi-task regularizer only; never as mechanism labels. https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html
- **Structures:** AlphaFold2 for all (RaSP crystal-vs-AF2 agreement enables proteome scale). https://pmc.ncbi.nlm.nih.gov/articles/PMC10266766/
- **Disclosure:** release must list all training variants; evals exclude them plus other variants at same positions; VEPs trained on MAVE data must exclude those datasets from MAVE benchmarks (MaveDB accession disclosure suffices); allele-frequency features trigger same circularity as supervised VEPs since benign labels use AF evidence. https://pmc.ncbi.nlm.nih.gov/articles/PMC11998465/

### 3b. Architecture (structure-aware, two-phenotype + mechanism head)

- **Backbone:** frozen ESM2-t33 (or SaProt 3Di-augmented) node features + 6-layer equivariant EGNN on kNN structure graph, pretrained on CATH domains (ProtSSN recipe: 30,948 domains; EGNN>GNN; 6 layers suffice). Alternative with equal evidence: MSA-Transformer + GVP-GNN (SSEmb; wins abundance, 0.45 vs 0.39 low-MSA) or ProteinMPNN embeddings + light-attention head (ThermoMPNN SSYM 0.72/0.60). Prefer EGNN/GVP over GCN/GAT (ProtSSN, LM-GVP). Add SaProt-style 3Di tokens if sequence+structure fusion is used (35M: 0.337→0.392 ProteinGym on adding 3Di; 650M ClinVar 0.862→0.909).
- **Heads:** (i) stability head: ddG/abundance regression (ThermoMPNN-style light attention); (ii) function-given-folded head: activity/binding regression conditioned on predicted stability; (iii) mechanism classifier: 3-way (destabilizing / function-disrupting-while-folded / neutral) with probability outputs, following LoGoFunc's 3-probability design but learned end-to-end rather than 474 hand features; (iv) per-experiment/assay mapping head per Schulze + MAVE-NN-style measurement model (sequence→latent phenotype + assay-specific readout) to absorb VAMP vs click-seq vs BindingPCA scale differences.
- **Why two phenotypes:** interaction/stability assays are more structure-sensitive than catalysis/activity (ProtSSN); SaESM2 wins Stability but is flat on 9 function tasks; SSEmb matches GEMME on activity but wins on abundance — one head cannot serve both.

### 3c. Mechanism supervision (how the 3-way label is created without dozens of paired genes)

- **Primary:** MoCHI TwoStateFractionFolded / ThreeStateFractionBound fits on ddPCA aPCA/bPCA → per-variant dG-fold + dG-bind as continuous mechanism targets (held-out R² 0.83/0.91 binding, 0.76/0.74 abundance; in vitro r 0.7–0.98). https://github.com/lehner-lab/MoCHI
- **Secondary ratio rules on the 4 human paired sets:** VKOR activity/abundance ratio for active-site (folded-but-dead) labels; CYP2C9 abundance-explains-half-function residual for function-specific signal; PTEN 60%-destabilized prior as class-balance anchor; NUDT15 17 activity-without-stability-loss residues as hard positives for the folded-but-dysfunctional class.
- **Weak supervision at scale:** RaSP/ThermoMPNN ddG + VAMP abundance thresholds generate destabilizing-vs-other pseudo-labels on unpaired abundance sets; function head trains on ProteinGym activity/binding assays with stability as covariate. No GOF/LOF label is used as a mechanism label (LoGoFunc axis is orthogonal).
- **Splits:** gene-disjoint always — 90/10 gene-disjoint + nested 5×5 CV + Optuna macro-F1 per LoGoFunc; LOPO per Schulze for abundance; whole-gene holdout for eval (type-1/2 circularity fix). No paralog identity cutoff can be cited — no fetched source states an explicit cutoff; only homolog-aware type-1 guidance verified.

### 3d. Eval (NDD-aware, circularity-clean, ACMG-compatible)

- **Circularity hygiene:** type-1 = reusing train variants/homologs in eval; type-2 = gene-level imbalance inflating cross-gene scores; fix = whole-gene holdout. DMS-based VEP ranking (84 VEPs, 36 human proteins) matches clinical-classification ranking, validating held-out functional assays as eval. https://www.biorxiv.org/content/10.1101/2024.05.12.593741v1.full
- **Benchmarks:** ProteinGym supervised + zero-shot tracks and clinical sets; held-out paired genes (leave-one-paired-gene-out) reporting mechanism AP/F1/MCC per class (LoGoFunc precedent: GOF AP 0.52 is the minority-class bar to beat) plus stability SCC/RMSE and function SCC.
- **ACMG calibration path:** Bayesian ACMG LR+ 2.08/4.33/18.72/350.4 = supporting/moderate/strong/very strong (1/2/4/8 pts), reciprocals for benign. https://pmc.ncbi.nlm.nih.gov/articles/PMC11503184/ PS3/BS3 need pathogenic+benign controls classified without functional data: 8+ = supporting, 12+ = moderate, OddsPath = any strength. Functional eval is 4-step: define gene-disease mechanism (LoF/GoF/DN) first, then assay class, instance, per-variant call. https://www.biorxiv.org/content/10.1101/709428v1.full-text Calibrated REVEL precedent: PP3 at 0.644/0.773/0.932, BP4 at 0.290/0.183/0.016, one predictor per variant. https://github.com/tamerh/biobtree/blob/main/docs/datasets/revel.md Gene-level MAVE calibration via local LRs + prior, demoed BRCA1/PTEN. https://pmc.ncbi.nlm.nih.gov/articles/PMC12248162/
- **NDD specifics:** NDD VCEP (MECP2/CDKL5/FOXG1/TCF4/UBE3A): missense PP3 = REVEL≥0.75, BP4 = REVEL≤0.15; PS3 only via accepted assays. https://pmc.ncbi.nlm.nih.gov/articles/PMC9135956/ No NDD-specific VEP benchmark or NDD-calibrated predictor thresholds found; Rett/Angelman VCEP rules are the closest verified standard. Report NDD-gene slice (mechanism confusion matrix on held-out NDD genes) separately; do not claim NDD calibration without NDD benign/pathogenic controls.

## Unresolved / not verified from fetched sources

- Any preprint after Apr 2025 adding explicit mechanism heads; search window may miss newest work.
- AlphaMissense ablation isolating structure atop PLM embeddings with numbers (Science paywalled, no text-fetchable source found).
- PrimateAI-3D sequence-only vs combined-model ablation deltas (supplement PDF, not text-fetched).
- ESM-GearNet/GearNet original ablation tables (only secondary citation on fetched page).
- Direct train-on-fragments vs train-on-full-length ablation for equivariant GNNs; no length-stratified train/test curves beyond ThermoMPNN RMSE-vs-length.
- ProSST Table 3 numeric values (tables render as images in fetched HTML).
- No fetched source shows a scaled 3D GNN jointly predicting destabilizing-vs-functional mechanism labels.
- Count of genes with paired stability+function DMS over shared variants not verified from fetched pages.
- AVE Alliance hosts no stability+function-paired dataset independent of MaveDB; no separate AVE pairing found.
- TPMT: VAMP-seq abundance exists but no overlapping full-length activity DMS located.
- GALT: only low-throughput activity/abundance per-variant data found, no paired DMS.
- No MAVE consortium beyond AVE/ProteinGym aggregating paired stability+function assays found.
- No fetched source states an explicit paralog sequence-identity cutoff for VEP train/test splits; only homolog-aware type-1 guidance verified.
- No NDD-specific VEP benchmark or NDD-calibrated predictor thresholds found; closest verified standard is Rett/Angelman VCEP gene rules.
- Completeness critic unavailable (per input notes).
