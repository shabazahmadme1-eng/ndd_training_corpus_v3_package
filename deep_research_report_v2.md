# Deep Research Report v2 — Zero-shot held-out-gene stability-vs-function prediction (NDD)

Date: 2026-10-02. Synthesis scope: ONLY the compact evidence supplied (refs: prior results 1–7). No new fetching.
Supersedes nothing: `deep_research_report.md` is preserved; this is the corrected v2.

## Executive summary / corrected novelty verdict

**Verdict: the claimed novelty STANDS, narrowed.** No verified prior art does **zero-shot held-out-gene stability-vs-function (mechanism) prediction** that generalizes to unseen NDD genes without per-gene labels.

- **PreMode** (primary source verified) is the closest system: SE(3)-equivariant graph attention NN on ESM2-650M + AlphaFold2 + MSA, trained on 41,081 DMS variants / 8 genes for stability+activity — but it **requires protein-specific transfer learning** (per-gene 80/20 splits) and has **no zero-shot held-out-gene mechanism eval** ([PreMode](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/)).
- The next-closest mechanism classifier, the **Cagiada 2023 Functional Model**, is a CatBoost 4-class predictor (WT-like / total-loss / SBI stable-but-inactive / low-abundance-high-activity) trained on only 9,945 variants / 923 positions from 3 paired-MAVE genes (NUDT15, PTEN, CYP2C9), CV acc 58% / MCC 0.57 — not a held-out-gene zero-shot NDD predictor (full text via [PMC10345196](https://pmc.ncbi.nlm.nih.gov/articles/PMC10325985/)).
- Everything else is off-axis: **FunC-ESMs** (untrained zero-shot thresholds), **LoGoFunc** (genetic GOF/LOF/neutral axis, not stability-vs-function), **MAVISp** (curator-run physics, no learned classifier), **MoCHI/ddPCA** + **DETANGO** + **ProteinNPT** (per-protein/per-assay fits), **Envision / DeMaSk / SSEmb / ProteinGym-supervised** (single fitness/VEP score, no mechanism output). Refs in §2.

**Precise remaining gap:** no verified joint multi-assay model that predicts the **destabilized-vs-functional (SBI) split zero-shot on held-out genes**, with NDD-gene generalization proven under whole-gene + paralog holdout and per-gene metrics. Beating PreMode = exactly that ([PreMode](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/)).

**Data verdicts (supervision):** TYK2 = YES new paired supervision; G6PD = YES new paired supervision; TP53 = NO (function-only; eval set only); BRCA1 = NO (no protein-stability DMS; viability+RNA / function-only); GPR68 = NO (zero DMS/MAVE hits; absence still needs direct MaveDB/IGVF portal confirmation). Details in §6.

---

## 1. PreMode verified from primary source

All bullets verified against the PreMode paper ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/)) and corroborating preprint ([bioRxiv 2024.02.20.581321](https://www.biorxiv.org/content/10.1101/2024.02.20.581321v1)).

### 1.1 Architecture and inputs

- SE(3)-equivariant graph attention transformer; ESM2-650M (d=1280) + AlphaFold2 structures + MSA; star-graph then KNN-graph construction.
- Pretraining corpus: 83,844 benign + 64,480 pathogenic variants; pathogenicity AUROC **0.932** as sanity check; ~50 h on 4×A40.
- Paralog-leak ablation: pretrain AUROC drops 0.932 → **0.873** when paralogs are removed — direct evidence that gene/paralog leakage inflates scores, so NDD eval must hold out whole genes plus Ensembl paralogs.

### 1.2 Molecular (DMS) track

- 41,081 DMS variants in **8 genes** (PTEN, SNCA, CCR5, CXCR4, NUDT15, CYP2C9, GCK, ASPA), stability+activity assays ex MaveDB.
- Eval is **within-gene random splits only**. The sole cross-gene test is a **stability-only** experiment over >30 genes (Supp Fig 8) — no joint stability-vs-function cross-gene test. (Numeric Spearman values for that experiment are in Supp Fig 8 only, not the main text — unresolved.)
- Per-gene variant counts within the 41,081 sit in Supp Data 2 URNs — not fetched (unresolved).
- Per-new-gene DMS cost: 80/20 split, 40 epochs, **4–6 h on 1×A40**; needs **~40% (~2,000) labeled variants to saturate at Spearman ~0.6**.

### 1.3 GOF/LOF track

- Corpus: 2,043 GoF + 7,889 LoF in ~1,300 genes; eval covers **only 9 genes with ≥15 labels** (ABCC8, BRAF, CACNA1A, FGFR2, KCNJ11, RET, SCN2A, SCN5A, TP53).
- Transfer protocol: 4-fold CV ensemble, 20 min–1 h on 1×A40. AUC 0.8–0.9 on RET/KCNJ11/CACNA1A/BRAF; 0.7–0.8 on four others; **FGFR2 <0.6 fails**.
- Per-gene G/LoF train/test counts are in Supp Data 1 — not fetched (unresolved); whether other NDD genes sit unevaluated in the 1,300-gene pool is unresolved.
- **NOT zero-shot:** cross-gene G/LoF model gives **random / nearly-reversed predictions on unseen genes** (Supp Fig 9a); per-gene fine-tuning is required.

### 1.4 NDD and mechanism-split coverage

- NDD: **only SCN2A** (AUC 0.7–0.8, within-gene split). GoF-epilepsy vs LoF-autism/ID framing is **motivation text** — no held-out NDD eval exists.
- Destabilized-vs-functional split shown **only in PTEN**: 3 dominant-negatives (C124S, G129E, R130G; G129E held out); needs 20% (398 points) of PTEN labels.
- Repro: code + weights at ShenLab/PreMode GitHub + Zenodo 15825903 (see [README](https://raw.githubusercontent.com/ShenLab/PreMode/main/README.md)); **49,218 HGMD pretrain variants withheld** from shared data. HuggingFace weights page located but not fetched (unresolved).

---

## 2. Prior-art landscape (why each is not the same claim)

| System | What it does | Why it does not close the gap | Source |
|---|---|---|---|
| PreMode | SE(3)-GNN, ESM2+AF2+MSA; stability+activity; 41k variants / 8 genes | Per-gene transfer only; cross-gene G/LoF random/reversed; no zero-shot mechanism eval | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/) |
| Cagiada 2023 Functional Model | CatBoost 4-class (WT-like / total-loss / SBI / low-abund-high-act); 8 features (§2.1); 9,945 var / 923 pos / 3 genes | 3-gene fit, CV acc 58% / MCC 0.57; not held-out-gene zero-shot NDD | [PMC10345196](https://pmc.ncbi.nlm.nih.gov/articles/PMC10325985/) |
| FunC-ESMs | Zero-shot ESM-1b + ESM-IF thresholds → WT-like / total-loss / SBI genome-wide | Untrained threshold heuristic, not a learned joint model | [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.21.595203v1.full-text) |
| LoGoFunc | 27 LightGBM + 474 features → GOF/LOF/neutral genome-wide | Genetic axis, not stability-vs-function | [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.06.08.495288v2.full) |
| MAVISp | Curator-run FoldX/Rosetta ddG + AlloSigma2 + PTM modules | Physics layers, no learned classifier | [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.22.513328v4.full) |
| MoCHI / ddPCA | Per-protein thermodynamic NNs on that protein's Binding+AbundancePCA; free energies + couplings | Held-out = variants, not genes; no zero-shot-on-unseen-proteins test in any fetched source | [ddPCA](https://pubmed.ncbi.nlm.nih.gov/35388192/), [MoCHI](https://pubmed.ncbi.nlm.nih.gov/39617885/), [repo](https://github.com/lehner-lab/MoCHI), [couplings](https://pubmed.ncbi.nlm.nih.gov/39322666/) |
| DETANGO (2026) | Per-protein training disentangling ESM-1v effects via FoldX/abundance input; SBI maps | Per-protein, not zero-shot; bioRxiv full text HTTP 403 twice — gene-split/eval details beyond author README unverified | [GitHub](https://github.com/luo-group/DETANGO) |
| ProteinNPT | Non-Parametric Transformer trained per DMS assay (random/contiguous/modulo folds) | Fitness only, per-assay | [GitHub](https://github.com/OATML-Markslab/ProteinNPT) |
| Envision | Stochastic gradient boosting, 27 features, 21k DMS scores; LOPO | One effect score, no mechanism | [PMC](http://pmc.ncbi.nlm.nih.gov/articles/PMC5799033/) |
| DeMaSk | DMS-derived substitution matrix + linear conservation model | Single fitness score, no mechanism | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8016454/) |
| ProteinGym | 250+ DMS assays (217 substitution / 186 proteins per proteingympy); 79 zero-shot + 12 supervised models | Supervised = per-assay single-fitness fits; no mechanism labels | [NeurIPS](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html), [repo](https://github.com/ccb-hms/proteingympy) |
| SSEmb | GVP-GNN structure + MSA Transformer, self-supervised | Single VEP score, zero-shot, no mechanism output | [bioRxiv](http://www.biorxiv.org//content/10.1101/2023.12.14.571755v1.full) |
| AlphaMissense | Single pathogenicity score (claimed) | Verdict from primary source NOT fetched — unresolved | — |

### 2.1 Cagiada 2023 detail (the closest mechanism classifier)

Resolved via open-access PMCID PMC10345196 (PMID 37443362; Nat Commun 14:4175, DOI 10.1038/s41467-023-39909-0, CC-BY): PMC HTML (249 KB) fetched and Europe PMC fullTextXML parsed section-by-section. Model = **CatBoost gradient-boosting classifier** (multinomial loss + L2, v0.26.1) predicting 4 variant classes (WT-like, total-loss, SBI stable-but-inactive, low-abundance/high-activity) from **8 features**: variant Rosetta Cartesian-ddG/2.9 clipped 0–5 + GEMME dE, residue-average and neighbor-average ddG/dE, target-AA hydrophobicity, WCN (r0=7 Å). Trained on 9,945 variants / 923 positions from paired abundance+function MAVEs on NUDT15, PTEN, CYP2C9; residue = functional if ≥50% variants SBI. Performance: CV acc 58% / MCC 0.57; 95% train variants (90% SBI); 116/127 residues; 72%/68% on single-MAVE sets; 10-enzyme TPR 0.71 vs 0.40/0.60 priors. RaSP-ddG retrained twin matches Rosetta model (62/87 M-CSA). Code/data: KULL-Centre/_2022_functional-sites-cagiada + Zenodo 8046585.

**Why it leaves the gap open:** 3 training genes, handcrafted 8-feature GBM at modest accuracy, no held-out-gene zero-shot protocol on NDD genes, no joint deep sequence+structure representation. It is the baseline to beat on the SBI axis, not the solution.

---

## 3. Precise remaining gap

1. **No zero-shot held-out-gene stability-vs-function predictor exists in the verified set.** PreMode's joint model needs ~2,000 per-gene labels; its cross-gene G/LoF transfer is random/reversed; its only cross-gene success is stability-only. Cagiada predicts SBI but from 3 genes with no zero-shot NDD protocol. FunC-ESMs is zero-shot but untrained thresholds. DETANGO/MoCHI are per-protein.
2. **No published head-to-head zero-shot mechanism numbers vs PreMode on held-out genes** — none found (unresolved, and the experiment to run).
3. **NDD coverage is effectively empty:** one within-gene SCN2A AUC 0.7–0.8; GoF-epilepsy vs LoF-autism/ID is motivation prose. Any NDD claim needs whole-gene + Ensembl-paralog holdout, per-gene metrics, rare-gnomAD negatives (§7).
4. **Beating PreMode = a joint multi-assay model generalizing to unseen (NDD) genes** with zero per-gene labels, evaluated on the destabilized-vs-functional split — not another within-gene Spearman or per-gene AUC.

---

## 4. Design sketch to beat PreMode at zero-shot held-out-gene prediction

Constraints from evidence: must work with **zero labels on the test gene**; must output **mechanism (destabilized vs functional/SBI)**, not one fitness score; must survive **whole-gene + paralog holdout** and **per-gene metrics**.

### 4.1 Task heads (multi-task, mechanism-first)

- **Head A — abundance/stability:** predict VAMP-seq abundance / ddG-derived destabilization (continuous). Supervision: paired MAVEs (§6) + megascale K50/dG tables + RaSP/ThermoMPNN ddG as features/auxiliary targets, not sole truth.
- **Head B — function/activity:** predict assay-specific function (signaling, growth, transactivation) **conditioned on abundance** (cf. MoCHI's TwoStateFractionFolded / ThreeStateFractionBound trait mapping, [repo](https://github.com/lehner-lab/MoCHI)).
- **Head C — mechanism class (primary):** 4-way Cagiada-style output (WT-like / total-loss / SBI / low-abund-high-act), or minimal 3-way (WT-like / destabilized-loss / functional-loss-SBI). This head is the PreMode-beating claim; Heads A/B are the decomposition that justifies it.
- Additive ddG + sparse pairwise couplings tied to structural contacts (per [couplings work](https://pubmed.ncbi.nlm.nih.gov/39322666/)) as an interpretable prior inside Head A; keep it gene-agnostic (contact-map conditioned, not gene-ID conditioned) so it transfers zero-shot.

### 4.2 Inputs (no gene-ID shortcuts)

- Sequence: ESM2-650M embeddings (PreMode parity) and/or ESM-1v + EVE conservation features (CPT/popEVE recipe: ESM-1v + EVE + AlphaFold + ProteinMPNN, with gene-level calibration so scores compare across genes — [DMS-VEP review](https://www.biorxiv.org/content/10.1101/2024.05.12.593741v1.full)).
- Structure: AlphaFold2 + MSA (PreMode parity); WCN (r0=7 Å), neighbor-averaged ddG/dE, hydrophobicity (Cagiada 8-feature set as engineered priors alongside learned embeddings).
- Stability priors: RaSP ddG (proteome-wide, <1 s/residue; [PMID](https://pubmed.ncbi.nlm.nih.gov/37184062/), [23k-protein release](https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)) and ThermoMPNN / ThermoMPNN-D ([PMID](https://pubmed.ncbi.nlm.nih.gov/38285937/), [repo](https://github.com/Kuhlman-Lab/ThermoMPNN-D)); DDGun3D-on-AF2 as triage prior ([PMID](https://pubmed.ncbi.nlm.nih.gov/37086329/)). Self-supervised 3D + supervised-on-Rosetta recipe per [RaSP docs](https://github.com/machinelearninglifescience/poli-docs/blob/main/docs/poli-docs/using_poli/objective_repository/RaSP.md). Single-mutant scope noted.
- Explicitly **exclude allele-frequency features** (they circularize benign classification) and any gene-label / gene one-hot inputs.

### 4.3 Training protocol (gene-holdout native)

- Leave-genes-out (LOGO) CV over all paired-supervision genes (§6): TYK2, G6PD, plus PreMode-8 paired genes available ex MaveDB (PTEN, NUDT15, CYP2C9 at minimum — the Cagiada triple) and any further MaveDB paired genes confirmed by direct portal query (API fetch 404 / JS-heavy site unresolved — must query live).
- Loss: joint multi-task (abundance MSE + function MSE + mechanism cross-entropy) with assay-type embeddings so heterogeneous function assays share the mechanism head.
- Gene-level calibration layer à la popEVE so mechanism scores compare across genes.
- Baselines to beat in the same LOGO loop: PreMode per-gene-transfer (upper-bound with labels) vs zero-shot; Cagiada CatBoost retrained LOGO; FunC-ESMs thresholds; RaSP/ThermoMPNN-ddG-only triage. Report per-gene mechanism MCC/AUC, never pooled-only.

### 4.4 What "beats PreMode" means (acceptance bar)

- On **held-out genes with zero labels**, mechanism-head MCC/AUC exceeds (a) Cagiada-LOGO, (b) ddG-only triage, and (c) PreMode-zero-shot (expected ~random per Supp Fig 9a) — and approaches PreMode-with-20%-labels (PTEN 398-point regime) without any labels.
- Stability head must at least match PreMode's cross-gene stability generalization (the one axis where PreMode already transfers); the win comes from the **joint** mechanism split, not stability alone.

---

## 5. NDD eval design (from evidence)

- **Circularity:** Type 1 = same variants in train/test; Type 2 = same/homologous genes even if variants differ; Type 2 inflates cross-gene benchmarks. Supervised VEPs must be tested only on different, non-homologous genes vs training; per-gene/per-protein metrics sidestep Type 2 ([review](https://www.biorxiv.org/content/10.1101/2024.05.12.593741v1.full)).
- **DMS advantage:** DMS needs no clinical labels (cuts Type 1); per-protein correlation cuts Type 2; DMS–VEP ranks mirror clinical ranks for population-free VEPs (same review).
- **Holdout rule:** whole genes **plus Ensembl paralogs**, motivated by PreMode's 0.932→0.873 paralog-ablation drop ([PreMode](https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/)).
- **Negatives:** rare gnomAD variants, not ClinVar benigns; no AF features ([review](https://www.biorxiv.org/content/10.1101/2024.05.12.593741v1.full)).
- **Clinical calibration (if claiming ACMG relevance):** Tavtigian Bayesian odds — supporting 2.08, moderate 4.3, strong 18.7, very strong 350; LP posterior >0.90, P >0.99 ([PMC6336098](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336098/)); Brnich/ClinGen SVI 4-step PS3/BS3 validation, ≥11 mixed P/B controls for moderate, OddsPath = [P2(1−P1)]/[(1−P2)P1], controls classified without functional data ([PMC6938631](https://pmc.ncbi.nlm.nih.gov/articles/PMC6938631/)); Pejaver/ClinGen single-tool local-LR calibration to PP3/BP4 bands + indeterminate zone, validated on a later ClinVar release ([PMC9748256](https://pmc.ncbi.nlm.nih.gov/articles/PMC9748256/)). No NDD-gene-specific PS3/BS3 OddsPath calibrations were fetched (unresolved) — do not claim PS3/BS3 strength without an NDD VCEP spec.
- **Fitness benchmark context:** ProteinGym (217 substitution assays / 186 proteins; 79 zero-shot + 12 supervised models; [repo](https://github.com/ccb-hms/proteingympy)) is the standard **fitness** benchmark — use for Head A/B sanity, not mechanism proof.

---

## 6. Data plan

### 6.1 New paired supervision (YES)

- **TYK2 — YES.** >23k substitutions × IFN-α signaling + protein abundance; absent from the PreMode 8-gene set, so genuinely new paired supervision ([bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.11.681520v2)).
- **G6PD — YES.** Paired MAVEs: yeast activity 8,228 missense + HEK293T VAMP-seq abundance 9,676 missense, MaveDB 00001266 ([PMC13455394](https://pmc.ncbi.nlm.nih.gov/articles/PMC13455394/)). Splits low-activity/normal-abundance (dimer interface, substrate sites) vs low/low (buried, structural NADP+) — exactly the target mechanism split. Abundance MAVE generated by the IGVF consortium (data.igvf.org/analysis-sets/IGVFDS1322SJZB); IGVF portal is an ongoing paired-assay pipeline. Scores browsable at mavedb.org/experiment-sets/urn:mavedb:00001266. MaveDB 2024 holds 7M+ variant effects (cited ref).

### 6.2 NOT new paired supervision (with reason)

- **TP53 — NO (eval set, not supervision).** Function-only: Giacomelli A549 DMS (complete 3,629 DBD variants) + Kato 8-promoter yeast transactivation, both via MaveDB ([summary](https://www.k-dense.ai/blog/tp53-structure-vs-function-3629-missense-mutations)). Type-I discordance set — 1,116 structure-benign/function-lost variants, 91/92 ClinVar pathogenic — is a ready-made **mechanism eval set**, not supervision (no paired abundance DMS).
- **BRCA1 — NO.** Findlay SGE: 3,893 SNVs, HAP1 viability + RNA scores (**mRNA/splicing, not protein stability**), RING+BRCT exons only ([PMC6181777](https://pmc.ncbi.nlm.nih.gov/articles/PMC6181777/)). Starita 2015 RING-domain multiplexed functional analysis is a second **function** assay. No BRCA1 protein-stability DMS found → cannot supervise the stability-vs-function split.
- **GPR68 — NO (no data).** Zero DMS/MAVE hits across all query angles; confirming true absence needs a direct MaveDB/IGVF portal query (unresolved). Not supervision either way today.
- **CYP2C19 — auxiliary abundance only.** VAMP-seq abundance 7,660 missense with no paired activity DMS ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.06.561250v1.full-text)). Use for Head A pretraining, not mechanism supervision.
- **CYP2C9/NUDT15 decoupling fractions:** G3 paper (jkag248) blocked by HTTP 403, contents unverified — re-fetch before using any claimed fractions.

### 6.3 Stability-scale auxiliaries

- **Megascale cDNA-display proteolysis:** 1.8M measurements → ~776k stabilities over 331 natural + 148 designed domains, but only 40–72 aa each ([PMID](https://pubmed.ncbi.nlm.nih.gov/37468638/)). Release shares processed K50/dG tables + single/double/triple DMS lists + pipeline notebooks ([Zenodo 7992926](https://zenodo.org/records/7992926)). Use for Head A scale; note domain-size caveat for full-length NDD genes.
- **RaSP proteome-wide ddG** (~230M variants; [PMID](https://pubmed.ncbi.nlm.nih.gov/37184062/)) + 23,391-human-protein AF-structure release ([repo](https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)) as features/priors.
- **ThermoMPNN** (megascale-trained, transfer-learned ProteinMPNN features; [PMID](https://pubmed.ncbi.nlm.nih.gov/38285937/)) and **ThermoMPNN-D** siamese single+double ddG ([repo](https://github.com/Kuhlman-Lab/ThermoMPNN-D)). Held-out-gene splits and per-gene scores not in abstract (unresolved).
- **ProtaBank:** spans activity/binding/stability/folding/solubility incl. DMS with cross-assay compare tools, but live site unreachable — holdings unverified ([preprint](https://www.biorxiv.org/content/10.1101/272211v1.full-text)). Do not depend on it without a live query.
- **AVE Alliance:** snippet-level only beyond MaveDB mapping role; no page fetched (unresolved).
- **TPMT:** multiplexed activity DMS to pair with known VAMP-seq abundance — no fetched source confirms it (unresolved).

---

## 7. Thermo-decomposition toolkit (supporting, per-protein)

- ddPCA quantifies binding+abundance phenotypes across backgrounds, fitting thermodynamic NNs to infer per-mutation folding/binding energies in SH3, PDZ ([PMID](https://pubmed.ncbi.nlm.nih.gov/35388192/)).
- MoCHI fits interpretable models to DMS data, inferring free-energy changes and couplings from multimodal phenotypes; shipped as PyTorch package ([PMID](https://pubmed.ncbi.nlm.nih.gov/39617885/)); maps Abundance → TwoStateFractionFolded, Binding → ThreeStateFractionBound, fit per DMS dataset with CV ([repo](https://github.com/lehner-lab/MoCHI)).
- Use MoCHI-style trait mapping as the Head A/B inductive bias, but train it **cross-gene** (LOGO) rather than per-protein — that cross-gene lift is the novel step. No fetched source tests MoCHI-fitted energies zero-shot on unseen proteins (unresolved).
- PROSTATA peer-reviewed status and blind-test numbers not verified (unresolved).

---

## 8. Unresolved evidence (carried forward, must not be silently dropped)

- AlphaMissense single-pathogenicity-score verdict from primary source (not fetched).
- DETANGO bioRxiv full text (HTTP 403 twice); gene-split/eval details beyond author README unverified.
- Per-gene variant counts within the 41,081 (Supp Data 2 URNs, not fetched).
- Numeric Spearman of >30-gene cross-gene stability experiment (Supp Fig 8 values, not in main text).
- Per-gene G/LoF train/test counts (Supp Data 1, not fetched); whether other NDD genes sit in 1300-gene pool unevaluated.
- No fetched source tests MoCHI-fitted energies zero-shot on unseen proteins.
- ThermoMPNN held-out-gene splits and per-gene scores not in abstract.
- PROSTATA peer-reviewed status and blind-test numbers not verified.
- GPR68: zero DMS/MAVE hits across all query angles; absence needs direct MaveDB/IGVF portal query to confirm.
- TPMT multiplexed activity DMS to pair with known VAMP-seq abundance (no fetched source confirms it).
- CYP2C9/NUDT15 paired decoupling fractions: G3 paper (jkag248) blocked by HTTP 403, contents unverified.
- Live MaveDB paired-gene inventory: API fetch returned 404, site JS-heavy.
- ProtaBank live holdings: protabank.org connection failed, only 2018 preprint scope verified.
- AVE Alliance data beyond MaveDB mapping role: snippet-level only, no page fetched.
- NDD-gene-specific PS3/BS3 OddsPath calibrations: no NDD VCEP specification fetched.
- Published head-to-head zero-shot mechanism numbers vs PreMode on held-out genes: none found.
- PreMode released weights/code for direct comparison: HuggingFace page located but not fetched.
- Completeness critic unavailable (per input notes).

---

## 9. References (URLs appearing in the compact evidence)

PreMode: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12325985/> · <https://www.biorxiv.org/content/10.1101/2024.02.20.581321v1> · <https://raw.githubusercontent.com/ShenLab/PreMode/main/README.md> · Cagiada: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10325985/> · FunC-ESMs: <https://www.biorxiv.org/content/10.1101/2024.05.21.595203v1.full-text> · LoGoFunc: <https://www.biorxiv.org/content/10.1101/2022.06.08.495288v2.full> · MAVISp: <https://www.biorxiv.org/content/10.1101/2022.10.22.513328v4.full> · MoCHI/ddPCA: <https://www.biorxiv.org/content/10.1101/2024.01.21.575681v2.full> · <https://pubmed.ncbi.nlm.nih.gov/35388192/> · <https://pubmed.ncbi.nlm.nih.gov/39617885/> · <https://github.com/lehner-lab/MoCHI> · <https://pubmed.ncbi.nlm.nih.gov/39322666/> · ProteinNPT: <https://github.com/OATML-Markslab/ProteinNPT> · Envision: <http://pmc.ncbi.nlm.nih.gov/articles/PMC5799033/> · DeMaSk: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8016454/> · ProteinGym: <https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html> · <https://github.com/ccb-hms/proteingympy> · DETANGO: <https://github.com/luo-group/DETANGO> · SSEmb: <http://www.biorxiv.org//content/10.1101/2023.12.14.571755v1.full> · Megascale: <https://pubmed.ncbi.nlm.nih.gov/37468638/> · <https://zenodo.org/records/7992926> · RaSP: <https://pubmed.ncbi.nlm.nih.gov/37184062/> · <https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg> · <https://github.com/machinelearninglifescience/poli-docs/blob/main/docs/poli-docs/using_poli/objective_repository/RaSP.md> · ThermoMPNN: <https://pubmed.ncbi.nlm.nih.gov/38285937/> · <https://github.com/Kuhlman-Lab/ThermoMPNN-D> · DDGun3D: <https://pubmed.ncbi.nlm.nih.gov/37086329/> · TYK2: <https://www.biorxiv.org/content/10.1101/2025.10.11.681520v2> · G6PD: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13455394/> · TP53: <https://www.k-dense.ai/blog/tp53-structure-vs-function-3629-missense-mutations> · BRCA1: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6181777/> · CYP2C19: <https://www.biorxiv.org/content/10.1101/2023.10.06.561250v1.full-text> · ProtaBank: <https://www.biorxiv.org/content/10.1101/272211v1.full-text> · Eval/calibration: <https://www.biorxiv.org/content/10.1101/2024.05.12.593741v1.full> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC6336098/> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC6938631/> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC9748256/>
