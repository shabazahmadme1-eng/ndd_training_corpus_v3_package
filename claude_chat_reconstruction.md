# Claude Chat Reconstruction — MIPO NDD Resume / Diagnosis / Expansion

Synthesized only from compact evidence (prior results 1, 2, 3, 4, 5, 7).
No new filesystem inspection. Log-line refs preserved as given.

## 1. What the Claude chat did, in order

### A. Resume-notebook triage (log-early)
1. Session opened by user prompt `look at mipo ndd resume notebook` in package root (log line 3).
2. Identified resume notebook: 5 cells; resumes DGX run `v4_validation_mechanism_650M_bafe6e1c78fd218e`, 4×3×10 = 120 runs (log line 35).
3. Reconstructed grid: scope `mechanism`, 650M, seeds 42/123/2026, 3000 steps/epoch, patience 6, ablations `esm_mlp` / `no_structure` / `no_global_kernel` (log line 91).
4. Flagged inconsistency: markdown says `PARALLEL=4` but cell 2 sets `PARALLEL=10`; `AMP=False` unconfirmed at first (log line 35). Later config check reports `amp=false lr=0.0003 OK` (log line 253).
5. Judged BioRxiv 2025.04.07.647656 not close: Borzoi DNA-indel/expression vs MIPO protein MAVE; gave table (log line 54).
6. Gave prior-work survey from memory only: ESM/Tranception, EVE, AlphaMissense, ProteinNPT/Envision, ProteinGym benchmark (log line 67).
7. Deferred novelty verdict: corpus 994673 meas / 495 genes / 135 filtered + PTEN/KRAS mechanism angle possibly novel, unproven (log line 74).

### B. Collapse triage and reseed (log-early → log-middle)
8. Colab log: 113/120 done; 7 collapsed to constant preds at epoch 0; all same-seed retries failed; runtime 36.4 min (log line 82).
9. Failed 7: `no_structure` s42/f4 plus `mipo` / `no_structure` / `no_global_kernel` on s123 f0 and f2; `esm_mlp` 0 failures (log line 180).
10. Root cause stated: `Spearman undefined` on constant-pred assay; retry reruns same seed so sampler/init/dropout collapse identically (log line 156).
11. Fix cell 5: diagnose via `validation_epochs` CSVs to `failure_diagnosis.csv`; reseed +1000/+2000/+3000 into `runs_reseeded/` (log line 156).
12. Revised finding: 6/7 whole-model collapse (all assays flat, e.g. stability −0.15437); 1 partial 4/10; n hundreds–1000s, no data fault (log line 196).
13. 113 finished runs unaffected (passed per-epoch check); stability gap 7/90 complex vs 0/30 `esm_mlp` ~8%, must disclose (log line 207).
14. Timing: ~50 min/run at 5 parallel; reseed ETA 06:15–07:00 UTC (11:45–12:30 IST) (log line 253).
15. 5/7 runs done (50, 63, 64, 78, 107m); fold2 seed1123 collapsed all 3 models, 2123 survived; shared-batch hypothesis unproven (log L269).
16. Fold2 slowest, 20-epoch cap; `no_struct` 13ep/8.6m, `no_global` 11ep/9.4m; ETA 13:30–14:25 IST, `no_global` last (log L324).
17. 6/7 done; `no_global` f2 seed2123 provenance mismatch after runtime change; seed1123 fold2 recollapsed, deterministic (log L335).
18. 119/120: 113 orig + 6 replaced; cell6 bug set aside 3× wasting ~1h twice; cell7 resumes usable partial, kills orphans (log L353).
19. 120/120 done (113+7); last run epoch18 best val Spearman 0.570; decision: never rerun cell2, merge `runs` + `runs_reseeded` (log L372).

### C. LLR comparison and 1-hour diagnosis (log-middle)
20. LLR ok both folders, 120 slots / 7 replaced; NDD LLR 0.426 esm 0.470 mipo 0.467; mech LLR 0.472 esm 0.443 mipo 0.464 (log L412).
21. Collapse 7/90 vs 0/30 Fisher p~0.13 chance; 7 events = 3 (seed,fold) sharing batch sequence; report counts only (log L412).
22. Verdict: structure adds nothing; +0.04 over LLR is MAVE supervision; need more genes plus ESM-1v/AlphaMissense/ProteinNPT (log L423).
23. Decision: don't wait zip, don't start paper; 1h diagnosis first (contrasts, gates, per-task); retune val-only, fresh genes (log L435).
24. Cell9 diagnosis reads paired contrasts + seed42 checkpoints, no training; asks A separation B fusion use C per-task (log L477).
25. TableA: contrast ~0.04 mech ~0 NDD, `mae_gain<=0` all; mipo 0.044 vs esm 0.053; KRAS neg, PTEN weak one-gene (log L493).
26. TableB/C: field-pool share 0.007 vs 0.119 dead; LLR wins activity 0.490 binding 0.523; wiring problem not science (log L512).
27. Suspects: `model.py:40` step=−1 sig 0.27 ×3 layers; h zero-init; `losses.py:42-43` field L2 1e-5; check steps+history (log L537).
28. Gates stuck 0.274–0.293 vs init 0.269; field loss rises 0.00073→0.0047; nll 0.62→0.007 vs rank 0.65→0.53 overfit (log L550).

### D. Contrast pilot + corpus forensics + MaveDB expansion (log-expansion)
29. Pilot cell: one-factor contrast run, mipo vs `esm_mlp`, seed 42, 3 folds, frac 0.25 + weight 0.20, new folder (log L641).
30. Feasibility: every fold 10–11 contrast pairs from 10/487 train genes (~2%), median shared 3366–4630 (log L655).
31. Fraction cut 0.25→0.10: 10 genes would be 12× over-represented, memorization risk given overfit (log L658).
32. Corpus: 904938 rows, 490 genes, 610 assays; stability 537k (59%); 81 genes ≥2 assays, 18 ≥2 tasks (log L680).
33. 11 genes ≥8 shared variants: SLC22A1 9715 … VKORC1 692; 7 multi-task genes share 0 (log L680).
34. Zero-shared cause: non-overlapping residue ranges, e.g. UBE4B stab 1–71 vs activity 1072–1173 (log L692).
35. PAX6 has 75 shared subs missed: assays use different `protein_reference_id` so `variant_key` never matches (log L692).
36. Conclusion: 11 is ceiling, a data property; stability = domain-only ~1–70, function = other regions (log L696).
37. Candidates file: 28-gene Sept snapshot, much already in corpus; NEW entries mostly alternate readouts (log L740).
38. MaveDB API live: 2819 score sets; full metadata catalogue fetched to `mavedb_expansion/` (log L767).
39. `find_pairs.py` keyword heuristic: 83 genes ≥2 task types (38 in corpus); stability 1085, abundance 588 … (log L800).
40. Phase2 shortlist: 25 genes abundance+(activity|binding|fitness), 23 human: SOD1 LDLR GCK SRC CCR5 … (log L808).
41. `phase2_shared.py` counts missense shared across tasks; ~100 sets done, on CD86, no ranking yet (log L822).
42. Handoff saved to memory `mipo-ndd-diagnosis-findings.md`; pilot planned but not run yet (log L722).

### E. Phase2 → Phase3 + deep-research (log-late)
43. Phase2 done: LDLR 15391, INSR 14060 top shared-missense ranking; 13-gene all-CC0 shortlist, 11 contrast genes to ~21 (log L849).
44. `Go harder` push: hybrid classifier + UniProt-coord mapping; scope grew to 61 genes, 539 sets, 2.52M variants (log L888).
45. Phase3.py backgrounded as `brrahjrpp` with WT-residue check; ranked gene list promised on completion (log L917).
46. ProteinGym adds nothing: 11/186 proteins multi-assay, all already contrast genes (SLC22A1, GCK, PTEN …) (log L913).
47. Construct-size split: 522/598 constructs <100aa, median 61; all 536,243 stability rows <100aa, 0 function rows (log L975).
48. Dead-field link: field-pool share 0.004 on 300+aa vs 0.013; 35 genes `not_established` incl KRAS isoform P01116-2 (log L983).
49. User requested deep-research on scaled 3D-graph to find NDD irregularities, appended `/deep-research` (log L1003).
50. Scope locked via AskUserQuestion: prior-art + design, broad training with NDD eval, mechanism-assignment output (log L1009).
51. Workflow launched backgrounded: task `wl27m63ex`, run `wf_3c8f090e-727`, five search angles incl paired-data hunt (log L1013).
52. Completed 17:28 UTC but 0 confirmed / 0 refuted / 25 unverified: every verifier panel failed on session-limit errors (log L1030).
53. PreMode UNVERIFIED: Nat Commun Aug25 does exact stability-vs-function split; calls PTEN C124S/G129E/R130G (log L1030).
54. PreMode UNVERIFIED: ESM2+SE(3)-GNN/AlphaFold2+MSA stack; mechanism set only 8 genes / 41,081 variants (log L1030).
55. End state: both completions undelivered; assistant emitted only session-limit notices at 17:28 and 17:30 UTC (log L1036).
56. Phase3 finished 17:30 UTC exit 0 but ranked list never reported; `phase3_pairs.csv` on disk unreviewed (log L1039).

## 2. Established results with numbers

### 2.1 Run completion and quality (113 + 7 = 120)
- `MIPO_NDD_results_light.zip`: 113 `test_summary.json` in `runs/` (`esm_mlp` 30, `mipo` 28, `no_global_kernel` 28, `no_structure` 27) + 7 in `runs_reseeded/` = 120 slots (prior result 7).
- Test `macro_gene_spearman` over the 113: `esm_mlp` mean 0.4481 sd 0.14 min 0.0905 max 0.6863; `mipo` 0.4604/0.11/0.2659/0.6864; `no_global_kernel` 0.4591/0.113/0.2055/0.6688; `no_structure` 0.4589/0.116/0.1703/0.6718; `undefined_assays=0` in all 113 (prior result 7).
- Validation `macro_gene_spearman` over the 113: `esm_mlp` 0.5469 (0.4744–0.5965); `mipo` 0.5555 (0.4772–0.6072); `no_global_kernel` 0.5513 (0.4512–0.6001); `no_structure` 0.5528 (0.469–0.602); `undefined_assays=0` in all 113 (prior result 7).
- Low test runs are a fold effect, not collapse: all 8 lowest are fold_03 across every model/seed (0.09–0.27); top runs are fold_00 across models (~0.67–0.69) (prior result 7).
- `collapse_rate_by_model.csv` + `reseeded_runs.json`: 7 original slots collapsed (`mipo` 2, `no_global_kernel` 2, `no_structure` 3, `esm_mlp` 0), all 7 replacements finished with `lr_scale` 1.0 (prior result 7).
- Last run epoch 18 best val Spearman 0.570 (log L372).

### 2.2 LLR comparison (pending item now resolved)
- `macro_comparison_originals_only.csv`: `ndd_evaluation` model ~0.467–0.470 vs LLR 0.4256 (delta +0.041…+0.044 all models); `mechanism_only` model 0.4427–0.4578 vs LLR 0.4633–0.4719 (delta −0.005…−0.029) (prior result 7).
- Log-middle LLR shorthand consistent: NDD LLR 0.426 esm 0.470 mipo 0.467; mech LLR 0.472 esm 0.443 mipo 0.464 (log L412).

### 2.3 Collapse accounting
- 7/90 complex vs 0/30 `esm_mlp` ~8% stability gap, must disclose (log line 207).
- 6/7 whole-model collapse (all assays flat, e.g. stability −0.15437); 1 partial 4/10; n hundreds–1000s, no data fault (log line 196).
- Fold2 seed1123 collapsed all 3 models, 2123 survived; seed1123 fold2 recollapsed, deterministic (log L269, L335).
- Collapse 7/90 vs 0/30 Fisher p~0.13 chance; 7 events = 3 (seed,fold) sharing batch sequence (log L412). Shared-batch hypothesis unproven (log L269).

### 2.4 Diagnosis (wiring, not science)
- TableA: contrast ~0.04 mech ~0 NDD, `mae_gain<=0` all; mipo 0.044 vs esm 0.053; KRAS neg, PTEN weak one-gene (log L493).
- TableB/C: field-pool share 0.007 vs 0.119 dead; LLR wins activity 0.490 binding 0.523 (log L512).
- Gates stuck 0.274–0.293 vs init 0.269; field loss rises 0.00073→0.0047; nll 0.62→0.007 vs rank 0.65→0.53 overfit (log L550).
- Suspects: `model.py:40` step=−1 sig 0.27 ×3 layers; h zero-init; `losses.py:42-43` field L2 1e-5 (log L537).
- Dead-field link: field-pool share 0.004 on 300+aa vs 0.013; 35 genes `not_established` incl KRAS isoform P01116-2 (log L983).

### 2.5 Corpus / construct facts
- v4 master: 994673 rows, 495 genes, 598 refs, 625 assays; direct NDD 318274 rows / 135 genes (`v4/audit/summary.json`) (prior result 3).
- v4/data tier zips: E 33.6MB / 540234 rows, C 11.4MB / 159715, A 10.5MB / 136734; master csv.gz 62.9MB (`v4/data`) (prior result 3).
- v4/robust (2026-09-26): curriculum/direct/mechanism splits + `validation.json`; audit passed, 0 dupes, 562/598 structures (`v4/robust/validation.json`) (prior result 3).
- Expansion-time corpus snapshot: 904938 rows, 490 genes, 610 assays; stability 537k (59%); 81 genes ≥2 assays, 18 ≥2 tasks (log L680).
- Construct-size split: 522/598 constructs <100aa, median 61; all 536,243 stability rows <100aa, 0 function rows (log L975).
- 11 genes ≥8 shared variants: SLC22A1 9715 … VKORC1 692; 7 multi-task genes share 0 (log L680).
- Zero-shared cause: non-overlapping residue ranges, e.g. UBE4B stab 1–71 vs activity 1072–1173 (log L692).
- PAX6 has 75 shared subs missed: different `protein_reference_id` so `variant_key` never matches (log L692).

### 2.6 MaveDB expansion (workspace-grounded)
- Catalogue: `mavedb_all_scoresets.json` 15.9MB + `mavedb_scoresets_table.csv` 408KB with 2819 rows (`mavedb_expansion/mavedb_scoresets_table.csv`) (prior result 3).
- `fetch.log` ends `done 2819 unique 2818` after 50-step progress to 2819 (`mavedb_expansion/fetch.log`) (prior result 3).
- `mavedb_expansion/scores` holds 557 score CSVs; `uniprot` holds 61 FASTA files, all dated 2026-10-02 (prior result 3).
- Phase2: `phase2_candidates.csv` 25 genes; `phase2_pairs.csv` 322 rows; log opens `164 score sets for 25 genes` (`mavedb_expansion/phase2_pairs.csv`) (prior result 3).
- Phase3: `phase3_pairs.csv` 574 rows; `phase3_mapping_report.csv` 538 rows; log `539 sets, 61 genes` (`mavedb_expansion/phase3.log`) (prior result 3).
- Phase3 DONE: 52 genes with ≥1 pair sharing ≥8 subs; top TYK2 22513, LDLR 14617, INSR 14060 (`mavedb_expansion/phase3.log`) (prior result 3).
- `proteingym_reference.csv` 208KB, 217 rows, DMS_index to coarse_selection_type header (`mavedb_expansion/proteingym_reference.csv`) (prior result 3).
- `multitask_gene_candidates.csv` 5769B; e.g. KRAS 28 sets / 199665 vars, MPL 4 tasks / 16763 vars (`mavedb_expansion/multitask_gene_candidates.csv`) (prior result 3).

### 2.7 Deep-research outputs (workspace-grounded, post-log)
- `deep_research_report.md` 21049B 2026-10-02; correction: PreMode 41081 vars / 8 genes, only novelty is whole-gene holdout (`deep_research_report.md:5`) (prior result 3).
- Report s2: mechanism gold is ~5–6 paired proteins PTEN/NUDT15/CYP2C9/VKOR/ddPCA; dozens-of-genes unsupported (`deep_research_report.md:79`) (prior result 3).
- `old_deepresearch_result.txt` 22931B 2026-10-02: 0 findings, 0 refuted, 25 unverified; PreMode ×4, FunC ×3 (`mavedb_expansion/old_deepresearch_result.txt`) (prior result 3).

## 3. Decisions made

1. Never rerun cell2; merge `runs` + `runs_reseeded` (log L372).
2. Reseed failed slots +1000/+2000/+3000 into `runs_reseeded/` after `validation_epochs` → `failure_diagnosis.csv` triage (log line 156).
3. Disclose stability gap 7/90 complex vs 0/30 `esm_mlp` ~8% (log line 207); report counts only, not Fisher p~0.13 (log L412).
4. Don't wait zip, don't start paper; 1h diagnosis first (contrasts, gates, per-task) (log L435).
5. Retune val-only, fresh genes; need more genes plus ESM-1v / AlphaMissense / ProteinNPT comparison (log L423, L435).
6. Contrast pilot fraction cut 0.25 → 0.10 to avoid 12× over-representation / memorization (log L658).
7. `Go harder` scope expansion accepted: hybrid classifier + UniProt-coord mapping; 61 genes / 539 sets / 2.52M variants (log L888).
8. Deep-research scope locked: prior-art + design, broad training with NDD eval, mechanism-assignment output (log L1009).
9. Structure verdict recorded as wiring problem not science: field-pool dead, gates stuck, LLR wins activity/binding (log L512, L550).

## 4. Unfinished threads and what each needs

1. Final reseed outcome table and `reseeded_runs.json` — log-lines-1–260 view said pending (log line 196). Needs: paste/read final table; workspace evidence now says all 7 replacements finished (`lr_scale` 1.0) but the per-slot table was not in the compact evidence.
2. Image screenshots L272/L274/L284/L315/L317 — contents not decoded beyond assistant paraphrase. Needs: inspect image bodies.
3. Full pasted tables L327/L408/L483 — truncated by HTML; only assistant summaries grounded. Needs: re-extract full tables or source CSVs.
4. `contrast_weight` / `auxiliary_weight=0.0` fix — only proposed as question, no finding in range. Needs: targeted log/code read + experiment decision.
5. Phase3 UniProt-mapping design and results after L830 — needs: read `mavedb_expansion/phase3.log`, `phase3_pairs.csv` (574 rows), `phase3_mapping_report.csv` (538 rows); reconcile TYK2 22513 vs earlier LDLR-top ranking.
6. Final phase2 per-gene `best_shared` ranking table after L830 — needs: read `phase2_pairs.csv` (322 rows) and phase2 log fully; `phase2.log` middle per-gene yield rows not fully transcribed.
7. Sequence-reconstruction / construct-size distribution finding after L830 — needs: source behind 522/598 <100aa median-61 claim; check function-row coverage.
8. Truth of PreMode / FunC-ESMs / MoCHI prior-art claims: 0 of 25 verified in-session — needs: external-source verification; `deep_research_report.md` and `old_deepresearch_result.txt` exist but verifier panels failed in-log.
9. Phase3 ranked gene list contents: job finished but results never surfaced in log — needs: surface ranked list from `phase3_pairs.csv` on disk (log L1039 says unreviewed).
10. `deep_research_report.md` dated 23:42 IST postdates log end 17:30 UTC — later-session artifact. Needs: provenance check; treat as post-log, not in-chat evidence.
11. KRAS isoform correctness and new-gene ingestion: flagged/proposed, never executed in window — needs: resolve P01116-2 `not_established` mapping; execute ingestion.
12. Row counts of v4 recovered files vs master not compared — needs: compare `v4/data` tier zips + `v4/robust` splits against `v4/audit/summary.json` (994673 / 495 / 598 / 625).
13. Contrast pilot planned but not run (log L722) — needs: run or explicitly cancel pilot cell (mipo vs `esm_mlp`, seed 42, 3 folds, frac 0.10, weight 0.20).
14. Completeness critic unavailable (notes) — needs: independent completeness pass if required.

## 5. Log claims that no longer hold against workspace files

1. `11 is ceiling, a data property` (log L696) — superseded. Workspace: Phase3 DONE 52 genes with ≥1 pair sharing ≥8 subs (`mavedb_expansion/phase3.log`). The 11-gene ceiling was a `variant_key` / `protein_reference_id` artifact (cf. PAX6 75 missed, log L692), lifted by UniProt-coord mapping (61 genes / 539 sets scope, log L888).
2. `11 contrast genes to ~21` / `13-gene all-CC0 shortlist` with LDLR 15391 / INSR 14060 top (log L849) — superseded. Workspace: Phase3 top is TYK2 22513, LDLR 14617, INSR 14060 (`mavedb_expansion/phase3.log`); scope is 61 genes / 539 sets, not 13/21.
3. `Phase3 ranked list never reported; phase3_pairs.csv on disk unreviewed` (log L1039) — stale as a workspace statement. Files exist: `phase3_pairs.csv` 574 rows, `phase3_mapping_report.csv` 538 rows, log `539 sets, 61 genes` (prior result 3). Still true that the log never surfaced the ranking; false that results are unavailable.
4. `~100 sets done, on CD86, no ranking yet` (log L822) — stale. Workspace: 557 score CSVs + 61 FASTA, phase2 322 rows / phase3 574 rows complete (prior result 3).
5. `Spearman quality uninspected; cell 6 LLR comparison pending` (log line 207) — resolved. Now: full test/val `macro_gene_spearman` means/ranges over 113 + `macro_comparison_originals_only.csv` deltas (NDD +0.041…+0.044; mechanism −0.005…−0.029) (prior result 7).
6. `Final reseed outcome table pending` (log line 196) — largely resolved. Now: 120/120 done (113+7), last run epoch18 val 0.570; all 7 replacements finished `lr_scale` 1.0 (log L372; prior result 7). Per-slot pasted table still not in evidence.
7. `Deep-research 0 confirmed / 0 refuted / 25 unverified; both completions undelivered` (log L1030, L1036) — true in-log, stale on disk. Workspace has `deep_research_report.md` 21049B and `old_deepresearch_result.txt` 22931B, both 2026-10-02, with PreMode correction (41081 vars / 8 genes; only novelty whole-gene holdout) and mechanism-gold ~5–6 proteins (prior result 3). Provenance caveat: report postdates log end (23:42 IST vs 17:30 UTC).
8. `PreMode UNVERIFIED` mechanism-set descriptions (log L1030) — partially corrected on disk: PreMode 41081 vars / 8 genes confirmed as the report's correction (`deep_research_report.md:5`); Nat Commun / PTEN-call / stack details remain 0-of-25 verified in-session.
9. Corpus snapshot `904938 rows / 490 genes / 610 assays` (log L680) — stale vs v4 master `994673 rows / 495 genes / 598 refs / 625 assays; direct NDD 318274 / 135 genes` (`v4/audit/summary.json`). Use master, not expansion-time snapshot.
10. `AMP=False unconfirmed` (log line 35) — resolved: config `amp=false lr=0.0003 OK` (log line 253).
11. `PARALLEL=4 vs 10` inconsistency (log line 35) — still open; no workspace evidence resolves which value actually governed the run. Timing claims (`~50 min/run at 5 parallel`, ETAs 06:15–07:00 UTC / 13:30–14:25 IST) are obsolete either way.
12. `Dozens-of-genes mechanism gold` implication from early novelty discussion (log line 74) — contradicted by post-log report: mechanism gold is ~5–6 paired proteins (`deep_research_report.md:79`).

## Refs

- prior result 1 (log-middle), prior result 2 (log-early), prior result 3 (workspace-state), prior result 4 (log-expansion), prior result 5 (log-late), prior result 7 (Spearman/LLR quality). Prior result 6 carried no additional compact evidence. Completeness critic unavailable.
