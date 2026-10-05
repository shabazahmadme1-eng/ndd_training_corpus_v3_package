# Tier 1.5 design: close the in-gene 0.45 → cross-gene ~0.10 gap (frozen 2026-10-04)

## Problem
Paired-contrast rank differential is learnable in-gene (ridge CV 0.45 median)
but pilots transfer at ~0.03–0.18. Tier 1 (branch supervision) moved the field
without closing transfer. Tier 1.5 attacks supervision density, not architecture.

## Role assignments (frozen)
- FireProt homologue-free split (2,578 rows / 89 PDBs) = BENCHMARK ONLY.
  Void if it ever touches training. Dedupe FireProt-test PDB sequences against
  corpus constructs before first use (domain-level, same discipline as §ledger).
- S8754 + GeoStab ddG/dTm + megascale = TEACHER ONLY. Never benchmarked.

## 1.5a FireProt benchmark (eval-only, Colab GPU-light)
- Rows: homologue-free split, 2,511 rows / 83 PDBs after excluding 67 rows
  (exact corpus genes TP53/SERPINE1/BRCA1 + CRYGD/CPA2/MYB domain overlaps);
  WT verified 2578/2578 pre-exclusion. Package: `external_data/fireprot_benchmark/`.
- Transfer rule (frozen before numbers): MIPO's head is assay-conditioned, so new
  proteins are scored four ways — (a) cached LLR zero-shot, (b) model mu under the
  mean stability-assay embedding, (c) field-probe scalar (assay-free), (d) published
  FoldX/Rosetta/ThermoMPNN anchors. No assay-embedding fitting on benchmark rows.
- Metric: per-protein n too small for Spearman (test median 5.5, 7 singletons)
  → pooled Spearman on per-protein z-scored ddG + direction accuracy
  (sign of predicted Δstability vs sign of −ddG).
- Sign convention: ddG > 0 = destabilizing. Model stability scores run higher =
  more stable; flip one side and assert on a known stabilizing/destabilizing
  pair before trusting numbers (quantity shift ΔTm-style vs ΔΔG is the point:
  a legitimate transfer test).
- External anchor: published FoldX/Rosetta/ThermoMPNN ddG numbers on FireProt
  for the folding axis; our models must report the same table.

## 1.5b ddG auxiliary teachers (builder work, no architecture change)
- The `physical_proxies` head + `aux_mask`/`aux_target` plumbing exists and sits
  at zero (`auxiliary_weight` in `effective_loss_weights`, startup-logged).
- Map S8754 + FireProt-train ddG onto (protein, position, mut) rows as aux
  targets via the `prepare_field_teachers.py` path; WT-verify every row.
- Weight: start 0.1; the startup log must show it nonzero (no silent no-ops).

## 1.5c Megascale stability pretraining (GPU, gated on 1.5a)
- 2.6M proteolysis-derived ΔG rows, ~455 new proteins, fragment-scale.
- Pretrain the field encoder on ΔG ranking before DMS fine-tuning; success =
  FireProt benchmark lift, not training loss.

## Order
1.5a benchmark first (defines the gap) → 1.5b teachers → 1.5c pretraining.
Tier 2 (serial fusion + pretrained encoder) waits on 1.5a numbers.

## Amendments (append-only; frozen sections above unchanged)
- 2026-10-05, erratum: per-protein n is median 9 with 8 singletons (recomputed
  from corpus_fireprot.csv), not "median 5.5, 7 singletons". Pooled metric
  stands. Note pool dominance: FP_1STN_1EY0 contributes 559/2477 rows (23%).
- 2026-10-05, instrument: score_checkpoint gained require_probes (default True,
  original contract preserved). Surviving v1 champion (branch_weight=0) scores
  with --allow-probeless; field_probe is NaN-by-construction (missing, not
  zero). Post-hoc frozen-trunk probes remain the follow-up for arm (c).
- 2026-10-05, dedupe gate FIRED (scripts/verify_fireprot_dedupe.py →
  fireprot_eval/dedupe_report.json): 3/83 proteins show domain-level training
  overlap WITH test mutations inside the aligned span — FP_1QLP/serpin vs
  SERPINE1 (5 sites), FP_1B5M vs SUOX b5-domain (2 sites), FP_1IET vs SUOX
  b5-domain (1 site). Corpus DMS covers both overlap spans densely (72/73 and
  67 positions). 3 further overlaps carry zero test sites (1TEN/JAG1,
  1BNL/UBE4B, 1HME/PMS1): caveats, keep.
- 2026-10-05, dedupe DECIDED: the 3 proteins are EXCLUDED (50 rows out →
  2427 rows / 80 genes). Pre-exclusion tree preserved at
  fireprot_eval_pre_exclusion_2477x83/. Gate re-run with the site rule
  encoded (flag only if >=1 test site inside): PASS, 0 flags, 3 caveats.
  1.5a scoring is unblocked on the rebuilt corpus.
