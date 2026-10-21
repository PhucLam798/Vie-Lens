# E6 — frozen implementation specification

**Frozen: OPS-28 decided by the human on 2026-10-13.** This document records the E6 choices. It does not report E6 results.

## 1. Quantities and definitions

- **D_zh.** For each model and format, average over passing block layers with `0.25 ≤ ℓ/N ≤ 0.75` the difference between mean `LA_zh` for sino concepts and mean `LA_zh` for nonsino concepts. Failed and undefined layers are excluded under OPS-14-6 and OPS-14-16. The model-level value is the equal-weight mean of repetition and translation values; per-format estimates are descriptive. Ambiguous, other_loan, M1-extension and cloze records are excluded. The item counts are 481 sino and 706 nonsino.
- **§7.1 response.** For concept `c`, language `L`, layer `ℓ` and format `f`, `r(c,L,ℓ,f) = Δ(c,L,ℓ,f) − mean_{ℓ'∈ℓ_early} Δ(c,L,ℓ',f)`. `Δ` is the A1 matched-distractor contrast on metric `s`; zh uses the A3 no-space target. `ℓ_early` contains block layers with `ℓ/N ≤ 0.15`, excluding layer 0. Baselines are computed within format. The mean of `r` over concepts must equal stored `rise_L(ℓ)` on identical rows (absolute tolerance `1e-9`). Languages are vi, en, zh, fr and id; layers are 1…N; depth is `ℓ/N`. The A6 gate does not apply to this lens quantity.
- **§7.1 model.** Repetition and translation are pooled for each family. Fixed effects are `C(language, Treatment('en')) * cr(depth, df=5) + C(stratum, Treatment('nonsino')) * C(language, Treatment('en')) + C(task_format) + zipf_en + n_tokens + n_syllables + concreteness`. Include a concept random intercept, fit by REML with statsmodels 0.15 `MixedLM`. `task_format` is fixed from the start: its random-effect variance is not estimable with two levels, and a crossed random effect would require an infeasible single-group fit at this row count. `n_tokens` is the number of content tokens (`is_content`) in the scored correct P-L target for the family × format × language × concept. Use complete cases only: 469 sino and 688 nonsino concepts.
- **Model instrument for H3 and Holm family (b).** Report the coefficient `stratum[sino]:language[zh]`, equal to `(sino − nonsino difference in rise_zh) − (the same difference in rise_en)`. Report `β_sino + β_sino:zh`, the zh-specific stratum effect, descriptively.
- **H3 rule.** Qwen supports H3 when `D_zh > 0` and its stratified bootstrap 95% CI excludes zero. Gemma and Llama CIs include zero. All E6 results are labelled exploratory.
- **Inference.** Use 1,000 concept bootstrap draws, stratified by stratum, with 95% percentile CIs. For §7.1 each draw is shared across repetition and translation. A concept sampled k times is represented by k distinct groups. Warm-start each bootstrap MixedLM refit from the full-sample fit; count and exclude non-converged draws, and report valid draws. Report the MixedLM Wald CI of the interaction beside its bootstrap CI. The Holm-family (b) bootstrap p-value is two-sided: `p = min(1, 2·(1 + min(#{θ_b ≤ 0}, #{θ_b ≥ 0})) / (1 + B))`; apply Holm across six tests (3 models × 2 instruments: D_zh and the §7.1 interaction).
- **Matched subsample.** Before any outcome is read, form and commit one shared concept-ID set from the 1,157 complete cases. Covariates are `zipf_en`, `n_syllables`, `concreteness`, and the mean of `ntok_gemma_vi`, `ntok_qwen_vi`, `ntok_llama_vi`, standardized by pooled SD over complete-case sino+nonsino concepts. Traverse sino concepts in seeded random order using the `e6_match` stream and seed 20260925. Match each to its nearest unmatched nonsino concept by Euclidean distance in standardized covariates; accept only when each absolute standardized covariate difference is at most 0.2. No replacement. Report balance as SMDs only.
- **Matched estimates.** Recompute both instruments on matched concepts, resampling matched pairs with key `bootstrap|h3m|<model>|pooled` and B=1,000. Classify each estimate as `+` if its CI is above zero, `−` if its CI is below zero, and `0` otherwise. H3 is `unresolved` if either instrument's class differs between the full and matched sets for any model.
- **Sensitivities (descriptive; no Holm).** For §7.1 only, substitute `zipf_vi`, then `zipf_zh`, for `zipf_en`; replace nonsino with `native_strict` concepts whose stratum is not sino; analyze each v1.2 subset (`bt_pass_strict_v1`, `external_attested`, each of the zh/fr/id NLLB flags, all three NLLB flags, and `min_surface_dist ≥ 0.60`); and exclude the six R11 concept IDs. D_zh sensitivities receive bootstrap CIs; §7.1 term sensitivities receive Wald CIs only.
- **Bootstrap keys.** D_zh: `bootstrap|h3|<model>|pooled`; full §7.1: `bootstrap|h3cc|<model>|pooled`; matched: `bootstrap|h3m|<model>|pooled`. One concept-index draw per model is shared across its two formats. All streams use the registered seed 20260925 and `derived/seeds.json` stream scheme.

## 2. Inputs and output columns

| Quantity | Inputs and selected columns | Item set |
|---|---|---|
| D_zh | `la.parquet`: `concept_id, format, layer, target_lang, metric, value`; select `metric=LA`, `target_lang=zh`, main sino/nonsino, passing layers only | main (481 sino / 706 nonsino) |
| §7.1 | `rise.parquet`: `concept_id, format, layer, target_lang, metric, value`; use `rise_main` rows for consistency validation and the matched-distractor rows needed to build the per-concept response. Join concept covariates from the preparation `concepts.tsv` by `concept_id`; read target token counts from P-L lens rows using `is_content` for correct targets, grouped by family, format, language and concept | complete cases (469 sino / 688 nonsino) |
| Matched estimates | Same LA and rise/lens rows as the corresponding full instrument, filtered to committed pair IDs | matched complete-case pairs |
| Sensitivities | Same LA/rise inputs, joined to only the named sensitivity columns from `concepts.tsv` | named subset per sensitivity |

The long-form P-L analysis outputs are under `results/analysis/primary/family=<family>/`. They use the shared analysis columns `model, model_revision, pass, format, condition, fewshot_set, concept_id, target_lang, target_variant, role, ref_concept_id, tok_idx, token_id, is_content, layer, metric, value`. Input readers must project only required columns, in family-scoped chunks. The matched-set stage reads only concepts.tsv and asserts it does not open anything under `results/`.

Summary outputs contain counts, estimates, intervals, p-values, model terms, convergence/valid-draw counts, classifications, and covariate SMDs. They must not contain raw concreteness values or covariate means.

## 3. Former open points and the decision that settles each

| Point in the draft | Settled by |
|---|---|
| 1. Per-concept rise response | OPS-28-2: within-format concept Δ minus that concept's early-layer mean |
| 2. Layers and A6 gate for §7.1 | OPS-28-2: layers 1…N; A6 gate does not apply |
| 3. Format pooling and MixedLM random effects | OPS-28-3: pool repetition and translation; concept random intercept; task_format fixed from the start |
| 4. Stratum × language instrument | OPS-28-4: `stratum[sino]:language[zh]`; report zh-specific stratum effect descriptively |
| 5. D_zh sidedness and Holm p-value | OPS-28-6: two-sided bootstrap p-value with plus-one correction; Holm over 6 tests |
| 6. Matched estimates and disagreement | OPS-28-9: recompute both instruments on matched pairs; compare `+ / − / 0` classes |
| 7. D_zh window and failed layers | OPS-14-6/16: passing layers only in the inclusive 0.25–0.75 depth window |
| 8. Gate and format scope | OPS-28-1/5: main sino/nonsino; exclude ambiguous, other_loan, M1-extension and cloze; equal-weight pool format D_zh |
| 9. Matching token-count field | OPS-28-8: mean of the three family-specific `ntok_*_vi` columns |
| 10. Pooled and matched bootstrap key | OPS-28-6/9: `h3`, `h3cc`, and `h3m` pooled keys |
| 11. Statsmodels version | D11: statsmodels 0.15 MixedLM; installed inventory recorded 0.15.0 |

## 4. Compute plan and outputs

The §7.1 primary model has 1,157 complete-case concepts, five readout languages, N block layers and two formats. Arithmetic row counts are:

| Family | Arithmetic | Rows |
|---|---:|---:|
| Gemma (N=34) | 1,157 × 5 × 34 × 2 | 393,380 |
| Qwen (N=36) | 1,157 × 5 × 36 × 2 | 416,520 |
| Llama (N=28) | 1,157 × 5 × 28 × 2 | 323,960 |
| Total | — | 1,133,860 |

Run `scripts/analyze_e6.py` through `job_analyze_e6.py`, profile `normal`, with at most the slice CPU count in parallel workers and resume-safe bootstrap chunks. A synthetic-shape timing job precedes the primary run. If projected primary wall time exceeds 48 hours, stop and ask.

Outputs:

- `reports/E6.md`
- `reports/E6_dzh.csv`
- `reports/E6_model_terms.csv`
- `reports/E6_holm.csv`
- `reports/E6_matched.csv`
- `reports/E6_sensitivity.csv`
- `reports/E6_matched_pairs.csv`
- `reports/E6_balance.csv`

Primary analysis artifacts are written under `results/analysis/primary/e6/`. The matched pairs and balance table are created and committed before any E6 outcome value is read.
