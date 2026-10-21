# Pre-registration v1.4

> v1.4 (2026-09-30) keeps the frozen v1.1 text and the v1.2 amendments below unchanged, adds the v1.3 cloze rebuild by reference and the v1.4 analysis amendment in full, and ends with the **effective analysis specification**, which governs where texts differ. Tag: `prereg-v1.4`. No study-LLM run preceded this version.

## Frozen protocol text (v1.1)

## 2. Hypotheses `[PREREG]`

### H1 — High-resource pivot

Middle-layer representations of a Vietnamese prompt occupy a region aligned with a high-resource language (English by default, Chinese for Qwen), before moving into a Vietnamese-specific region in late layers.

**Predicts:**
- Normalized language-alignment to the pivot language rises well above null in middle layers.
- In the five-way readout, the pivot language's *relative rise* substantially exceeds that of the control languages.
- Steering along the Vietnamese direction has a sharply localized flip-rate peak.

### H2 — Language-neutral concept space

There is no English stage. Middle layers hold an abstract representation roughly orthogonal to all language-specific directions. The logit lens decodes to English merely because the unembedding is English-shaped.

**Predicts:**
- Alignment near null for *all* languages in middle layers, while semantic decodability is already high.
- In the five-way readout, **all** language versions of the correct concept rise together.
- Tuned lens substantially reduces the apparent English dominance.

### H3 — Etymology-conditional routing (Qwen)

Qwen routes Sino-Vietnamese vocabulary through Chinese and native Vietnamese vocabulary through English. Gemma and Llama show no such split.

**Predicts:**
- Qwen: Chinese alignment significantly higher for the Sino stratum than the native stratum, after covariate adjustment.
- Gemma and Llama: no significant stratum difference.

### 2.1 Decision rules `[PREREG]`

Let `LA_L(ℓ)` be the normalized language-alignment score (§5.3), scaled so 0 = random direction, 1 = a native prompt in language `L`.

Let `rise_L(ℓ)` be the within-language relative rise in readout score (§5.2), measured in that language's own z-units from its early-layer baseline. **Relative rise, not absolute score, is the comparison quantity** — see §5.2 for why.

| Outcome | Rule |
|---|---|
| Support H1 | `max_ℓ LA_en(ℓ) ≥ 0.50` on Vietnamese prompts, **and** `max_ℓ rise_en(ℓ) ≥ 2 × mean(max_ℓ rise_fr, max_ℓ rise_id)` |
| Support H2 | `max_ℓ LA_L(ℓ) < 0.20` for all L, **and** all five `max_ℓ rise_L` fall within a 1.5× band |
| Support H3 | Qwen: `LA_zh(Sino) − LA_zh(native) > 0`, bootstrap 95% CI excluding zero; Gemma and Llama CIs include zero |
| Inconclusive | Any pattern between the H1 and H2 bands — report as such, do not reframe post hoc |

All thresholds frozen at v1.1. Revision requires a version bump documented in §14.

---

### 5.2 M2 — Multi-token logit lens (the instrument)

For target string `t = t₁…t_k` in language `L`:

1. Teacher-force the full target after the prompt.
2. At each layer `ℓ`, apply the final norm and unembedding to the residual stream at each target position.
3. Compute `log P_ℓ(tᵢ | prompt, t₁…tᵢ₋₁)` for each `i`.
4. Score for language `L` at layer `ℓ` = mean over `i`.

One forward pass per (prompt, target-language) pair.

#### `[CORRECTION vs v1.0]` Cross-language scores are not directly comparable

**This was a real flaw in v1.0 and it would have broken E4.** Chinese targets are typically 1–2 tokens; French and Indonesian are multi-token; Vietnamese is multi-token. Mean log-probability per token is systematically affected by target length, tokenizer fertility, and unigram priors. Comparing raw M2 scores *across languages* therefore measures tokenization economics as much as internal representation.

**Fix — three layers of normalization, applied in this order:**

1. **Frequency normalization.** Divide out each language's marginal token probability under the unembedding.
2. **Within-language standardization.** Convert each language's layer curve to z-units using that language's own mean and SD across layers. This makes the curve *shape* comparable even when levels are not.
3. **Baseline-relative rise.** Define `rise_L(ℓ) = z_L(ℓ) − z_L(ℓ_early)`, where `ℓ_early` is a fixed early-layer reference (first 15% of depth). **This is the primary statistic for all cross-language comparisons**, including the H1/H2 decision rules in §2.1.

Report raw scores in the appendix for transparency, but never use them for cross-language claims. Within-language comparisons across layers, models or strata may use the frequency-normalized score directly.

### 5.6 Landmark definitions `[PREREG]`

`[CORRECTION vs v1.0]` v1.0 defined `L_concept` by a fixed absolute probability threshold. That is not comparable across models with different vocabulary sizes and calibration. Replaced with a scale-free relative definition.

| Landmark | Definition |
|---|---|
| `L_concept` | First layer where the correct concept in *any* readout language reaches **10% of its own final-layer probability** |
| `L_peak` | Layer of maximum `rise_L` among non-Vietnamese readout languages |
| `L_cross` | First layer after `L_peak` where `rise_vi` exceeds the pivot language's `rise` |
| Pivot width | `(L_cross − L_concept)` as a fraction of total depth |
| Pivot magnitude | `max_ℓ rise` for the leading non-Vietnamese language |

The 10% figure is frozen. Report sensitivity at 5% and 20% as a robustness check.

---

## 7. Analysis plan `[PREREG]`

### 7.1 Primary model

```
rise ~ language * depth_spline + stratum * language
       + log_freq + n_tokens + n_syllables + concreteness
       + (1 | concept) + (1 | task_format)
```

Mixed-effects, concept as random intercept. The same concepts appear across languages and conditions; treating observations as independent will inflate significance.

### 7.2 Uncertainty

Bootstrap over **concepts**, 1,000 resamples. 95% CIs on every landmark and every decision-rule quantity.

### 7.3 Multiple comparisons

Families: (a) H1/H2 tests across 3 models × 5 languages; (b) H3 tests across 3 models × 2 strata. Holm–Bonferroni within each family. Family definitions frozen here.

### 7.4 Power

With ~1,200 concepts and bootstrap over concepts, the H1/H2 contrasts are amply powered. **H3 is the binding constraint** — its power depends on the `sino` and `native` stratum sizes after filtering, which are unknown until §4 runs. After building the concept set and before running E6, compute and report the minimum detectable stratum difference. If it exceeds a plausible effect size, declare H3 exploratory rather than confirmatory.

### 7.5 Robustness checks

- E4 repeated with tuned lens on at least one model.
- All three task formats reported separately — a finding holding for only one format is a format artifact.
- Results with and without frequency normalization.
- `L_concept` at 5%, 10%, 20% thresholds.
- Prompt-matched vs FLORES-based language directions.

---

## Protocol amendments (v1.2)

Note: amendments override v1.1 where they conflict.

# Protocol amendments

v1.2 (2026-09-26; pre-data; no model outputs exist at the time of this amendment)

- §4.3 filter 1: attestation = ≥2 of {EN-entry translation table, VI-entry
  English gloss, MUSE, Wikidata}. Reason: MUSE's Vietnamese side has 0/73,874
  multi-syllable entries; the v1.1 rule would have limited the concept set to
  monosyllables, undermining C1. Robustness subset: external_attested
  (MUSE or Wikidata).
- §4.4: strata are sino vs nonsino (not native). native_strict (Proto-Vietic/
  Mon-Khmer/Austroasiatic evidence or an inh template) is reported as a
  robustness subset. Reason: only ~7% of pool entries carry native-origin evidence.
- §4.2: POS limited to noun/verb/adj; English lemmas ≤ 2 words; Chinese = Mandarin
  (cmn), Simplified.
- §4.3 filter 4: threshold set to 0.45 at an empty histogram bin (0.45–0.50),
  after inspecting items around 0.30 (predominantly loans). Added a hyphenated-
  transliteration filter. Pre-declared robustness check: primary results
  re-run excluding min_surface_dist < 0.60.
- §4.3 filter 2: pass = round-trip
  whole-syllable containment (≤ +4 syllables) OR forward en→vi hit. Reason: NLLB
  hallucinates on isolated words, and exact-match round-trip failed 71% of
  monosyllables vs 47% of disyllables. Pre-declared robustness subset:
  bt_pass_strict_v1. Directions pool exempt from filter 2.
- Vietnamese spelling variants (tone placement, final i/y) are treated as equivalent; the display form is the more frequent one. If back-translation validates only an alternative form, it becomes the canonical form and is re-filtered.
- Split components and all disjointness checks use the Vietnamese spelling-equivalence key; splits were regenerated (same seed) before any model outputs existed.
- 'Model outputs' in this document means outputs of the study LLMs (Gemma,
  Qwen, Llama); NLLB preprocessing outputs are not included.
- §4.4 H3 status rule (fixed before the corrected Signal B was run): H3 is
  confirmatory iff the lower 95% CI bound of Cohen's κ(A, B) ≥ 0.60 AND the
  Holm-level minimum detectable d (sino vs nonsino, test) ≤ 0.30; otherwise H3
  is reported as exploratory.
- §4.3 filter 3 unchanged (≤ median senses). The adequacy of the M1
  single-token subset is assessed at the E1 gate (minimum 150 items per model).
- zh/fr/id NLLB agreement flags are robustness variables, not filters.
  Pre-declared robustness subset: all three agree.
- Checkpoint C outcome: primary κ(A,B) = 0.594, 95% CI [0.557, 0.631]; Holm-level
  MDE (sino vs nonsino, test) = 0.206. Under the pre-committed rule, H3 is
  EXPLORATORY. Strata frozen with the primary Signal B (test: sino 481, nonsino 706,
  ambiguous 347 excluded from H3). 25.8% of verified Han strings used Wiktionary
  character entries for readings (Unihan kVietnamese coverage gaps); Signal B is
  therefore only partly independent of Signal A. No further changes to the
  etymology rules will be made.
- E1 tokenizer audit (no model weights loaded): single-token Vietnamese coverage
  is 7.0% (Gemma), 6.3% (Qwen), 6.3% (Llama); all single-token items are
  monosyllabic. M1 adequacy (≥150) is not met by the main set. Planned remedy (main
  dataset unchanged): an M1-only extension set of monosyllabic concepts with
  exactly 2 senses that pass all other filters, used solely for the M1-vs-M2
  comparison (C1).
- M1 set = single-token items of the main test set ∪ M1-only extension (monosyllabic, exactly 2 senses, all other filters passed). Sizes: Gemma 186, Qwen 166, Llama 162 (final counts in agreement.md). M1-vs-M2 is also reported on main-set items only.
- Russian forms are stored without stress marks; ё/е are treated as equivalent when matching. Few-shot eligibility: ru canonical with NLLB agreement and filter-2 pass (collapse status is irrelevant, because few-shot examples are never diacritic-stripped).
- Checkpoint D outcome: Russian is the translation-prompt source; templates are unquoted; cloze examples come from Vietnamese Wiktionary examples. Cloze coverage is 337 test concepts (primary; threshold 300). Few-shot selection uses 30 examples in six sets of five, with set 1 primary.
- E9 nodiac condition: all Vietnamese text in the prompt and the Vietnamese target are diacritic-stripped (simulating undiacritized input); items whose stripped test form collides with another word are excluded. The cloze primary/supplementary status is determined on the main test set only.

### Data build complete (2026-09-26)

- Frozen data sizes: main test = 1,535; M1-only extension = 123; directions = 132; few-shot = 30 (six sets of five). `concepts.tsv` has 1,837 rows including the extension.
- Checkpoint C: primary κ(A, B) = 0.594470, 95% CI [0.557133, 0.631475]; H3 is exploratory under the pre-committed rule. Test strata: sino = 481, nonsino = 706, ambiguous = 347, other_loan = 1.
- Cloze is supplementary: 265/1,535 main-test concepts (17.26%) have a qualifying cloze example.
- E1 single-token Vietnamese coverage on the 1,535-item main test set: Gemma = 108 (7.04%); Qwen = 96 (6.25%); Llama = 96 (6.25%). M1 eligible counts including the extension: Gemma = 186; Qwen = 166; Llama = 162.
- FLORES+ devtest fertility (tokens/character; Vietnamese tokens/syllable):

  | model | vi | en | zh | fr | id |
  |---|---:|---:|---:|---:|---:|
  | Gemma | 0.268329 / 1.210382 | 0.205462 | 0.676101 | 0.242200 | 0.219199 |
  | Qwen | 0.286035 / 1.290252 | 0.209304 | 0.640515 | 0.276869 | 0.298768 |
  | Llama | 0.272802 / 1.230561 | 0.205894 | 0.804846 | 0.275771 | 0.293554 |

- Concreteness matching across the release: exact = 1,653/1,837 (89.98%); head = 135/1,837 (7.35%); none = 49/1,837 (2.67%). Counts and within-group rates by split/scope and stratum:

  | split/scope | stratum | n | exact | head | none |
  |---|---|---:|---:|---:|---:|
  | fewshot_reservoir | ambiguous | 14 | 13 (92.86%) | 1 (7.14%) | 0 (0.00%) |
  | fewshot_reservoir | nonsino | 17 | 13 (76.47%) | 4 (23.53%) | 0 (0.00%) |
  | fewshot_reservoir | sino | 16 | 14 (87.50%) | 2 (12.50%) | 0 (0.00%) |
  | directions | ambiguous | 34 | 30 (88.24%) | 4 (11.76%) | 0 (0.00%) |
  | directions | nonsino | 57 | 48 (84.21%) | 6 (10.53%) | 3 (5.26%) |
  | directions | other_loan | 1 | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | directions | sino | 40 | 38 (95.00%) | 1 (2.50%) | 1 (2.50%) |
  | main_test | ambiguous | 347 | 323 (93.08%) | 13 (3.75%) | 11 (3.17%) |
  | main_test | nonsino | 706 | 594 (84.14%) | 94 (13.31%) | 18 (2.55%) |
  | main_test | other_loan | 1 | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | main_test | sino | 481 | 460 (95.63%) | 9 (1.87%) | 12 (2.49%) |
  | m1_extension | ambiguous | 27 | 24 (88.89%) | 1 (3.70%) | 2 (7.41%) |
  | m1_extension | nonsino | 68 | 67 (98.53%) | 0 (0.00%) | 1 (1.47%) |
  | m1_extension | other_loan | 2 | 2 (100.00%) | 0 (0.00%) | 0 (0.00%) |
  | m1_extension | sino | 26 | 25 (96.15%) | 0 (0.00%) | 1 (3.85%) |

## Frozen data pointer

- `data/concepts.tsv` SHA-256: `6ed1faf4210e43f7e37058bf975573d43ed76613897b8e6b5d609524546d4de8`
- Pipeline commit: `0629a4519e7cadfd61d370156ca2b76fd926cb8d`
- Data-build-complete amendment commit: `36d88c0ebdeb06161b4ebf1e99326a663b43cde9`
- concepts.tsv was generated at pipeline commit b6ecb6f (the header line) and reproduced byte-for-byte at 0629a45 after the Makefile order fix.

## Protocol amendments (continued)

### v1.2 corrections (2026-09-27, before any study-LLM run)

- Correction to the Checkpoint D line: the 337 cloze count included the M1
  extension. On the main test set, cloze coverage is 265/1,535; cloze is
  SUPPLEMENTARY (see step 12b).
- §4.3 filter 4 (recorded late; applied since step 06): surface distance is
  normalized Levenshtein after removing diacritics, spaces, hyphens and
  apostrophes, with Vietnamese 'ph'→'f'. An etymology-based loanword filter
  drops borrowings from European languages and Malay/Indonesian
  (bor/bor+/lbor/der/der+/obor templates); calques and Sinitic, Sino-Japanese
  and areal/proto-language derivations are retained.
- §7.1 log_freq = Vietnamese Zipf frequency in the primary H3 model; a
  pre-declared sensitivity run adds the readout language's own Zipf frequency
  (e.g. Chinese for LA_zh). Matched-subsample check (§4.5): nearest-neighbour
  matching without replacement, caliper 0.2 SD on the four §4.5 covariates
  (Vietnamese frequency, syllables, concreteness, mean Vietnamese token count).
  Reason: the pre-model data analysis found sino vs nonsino imbalance of
  d = +0.89 (Vietnamese frequency), +0.83 (Chinese frequency), −0.83
  (concreteness), −0.62 (Vietnamese tokens).

### v1.3 cloze rebuild (2026-09-27; before any study-LLM run)

Included by reference: see `PROTOCOL_AMENDMENTS.md` (section "v1.3 cloze rebuild") and data tag `data-v1.3.1`. Summary: cloze files rebuilt (diac 41 = 34 main + 7 extension; nodiac 29 main); cloze is supplementary and descriptive only, excluded from decision rules, Holm families, the §7.1 model and all stratum analyses; `cloze_available` is deprecated.

### v1.4 analysis amendment (2026-09-30; before any study-LLM run)

**Status when this amendment was written.**

- No study model (base or instruct Gemma 3 4B, Qwen3 4B, Llama 3.2 3B) has run a forward pass on any viLens prompt.
- The only model outputs seen are three exploratory single-prompt notebooks (`gemma-3-4b-pt`, Llama-3.1-8B, Qwen3.5-9B). They used unpinned libraries, hand-written prompts and no test item.
- Everything else was done on tokenizers and `config.json` files only: the X0 audits in the experiment repo `PhucLam798/Vie-Lens` (commit `59e69ca`).
- The decisions below were taken by the author on 2026-09-29 and are recorded in the experiment repo as `docs/decisions.md` D1–D12 and OPS-7.
- Where this amendment conflicts with v1.1–v1.3, v1.4 governs.

**Why v1.4 is needed.** Four problems would make the frozen headline tests uninterpretable:

1. The v1.1 `rise` normalization is invariant to positive affine transforms of a curve, so it discards amplitude. A synthetic French curve at 1/20 of the English amplitude gets the same `max rise`, and a flat noisy curve can exceed a clean pivot.
2. `L_concept` is anchored to each language's own suppressed final-layer probability.
3. With the released leading space, Chinese targets tokenize into a bare space token (Gemma 58.8% of items) or UTF-8 byte fragments (Qwen 80.1%, Llama 53.4%).
4. 747 of the 1,535 main concepts have an English–French surface distance below 0.45, so French and Indonesian are weak controls for English.

**Model set (D10; changes protocol §3).**

- **Primary study checkpoints are the same-size base models:**
  - `google/gemma-3-4b-pt` @ `cc012e0a6d0787b4adcc0fa2c4da74402494554d`;
  - `Qwen/Qwen3-4B-Base` @ `906bfd4b4dc7f14ee4320094d8b41684abff8539`;
  - `meta-llama/Llama-3.2-3B` @ `13afe5124825b4f3751f836b40dafda64c1ed062`.
- **Reasons:**
  - the prompts are raw few-shot completions, the setting of the reference study;
  - the base models have the same depth, width and vocabulary as the registered instruct models;
  - their tokenizers produce identical token ids on all 229,487 prompt-and-target encodings of the release (X0c-a, check C31). They differ only in the end-of-sequence token, which no stopping rule uses.
  - Every frozen token column (`ntok_*`, `single_token_vi_*`, `m1_eligible_*`) therefore stays valid.
- The registered instruct checkpoints (`gemma-3-4b-it` @ `093f9f3`, `Qwen3-4B` @ `1cfa9a7`, `Llama-3.2-3B-Instruct` @ `0cb88a4`) are an optional secondary replication, run after the primary analysis.

**A1 — Matched-distractor contrast replaces per-language z-scoring (D1; changes §5.2 normalization steps 1–3 and the definition of `rise` in §2.1).**

- `s(c, L, ℓ)` is the mean content-token lens log-probability (A4) of concept c's L form, teacher-forced after c's prompt.
- `d(c, L)` is one distractor, another main-test concept chosen by these rules:
  - it has the same number of content tokens for L in the model's tokenizer; if no such concept exists, the nearest count is used ("count fallback");
  - its first content token differs from c's;
  - its L string differs from c's;
  - among the remaining candidates, it has the nearest `zipf_en`;
  - exact ties are broken by a seeded draw.
- The distractor is teacher-forced after **c's** prompt.
- `Δ(c, L, ℓ) = s(c, L, ℓ) − s(d, L, ℓ | c's prompt)`.
- `rise_L(ℓ) = mean_c Δ(c, L, ℓ) − mean_c mean_{ℓ' ∈ ℓ_early} Δ(c, L, ℓ')`, where `ℓ_early` is the blocks with ℓ/N ≤ 0.15, excluding the embedding output.
- The §2.1 H1/H2 rules apply unchanged in form to `max_ℓ rise_L`, which is now in nats. The v1.1 z-statistic is reported in the appendix only.
- The distractor manifests are frozen in the experiment repo:
  - `derived/distractors.jsonl` (32,235 rows);
  - `derived/e0_distractors.jsonl`;
  - `provenance.derived_sha256 = e0ed01a7d7b9923d1fb17e05225d3821100c5164f910570715635ea66e539cf3`.
- Exact-count matches cover 1,531–1,535 of the 1,535 concepts for every released readout. For the `vi_nodiac` diagnostic, 108 (Qwen) and 119 (Llama) concepts use the count fallback.

**A2 — `L_concept` (D2; changes §5.6).**

- `L_concept` = the first ℓ at which, for any readout language L, `p_L(ℓ) ≥ 0.10 · p_vi(N)`.
- `p = exp(s)` on concept-mean curves.
- The anchor is the probability of the model's actual (Vietnamese) answer at the final block. The 5% and 20% sensitivities are kept.

**A3 — Chinese target form (D3).**

- The primary zh target is the released form **without** its leading space, teacher-forced directly after the final `:`.
- The with-space form is a sensitivity.
- Latin-script targets keep their space.
- Joint tokenization keeps the prompt a strict prefix for the no-space form in all six tokenizers (0 violations).

**A4 — Content-token scoring.**

- Whitespace-only target tokens are teacher-forced but excluded from the mean.
- Pre-declared decompositions: `s¹` (first content token), `s⁺` (continuation tokens) and the sequence sum (R8). If the verdict or `L_cross` differs between `s` and `s¹`, both are reported.

**A4b — Along-path readout (diagnostic, R10).**

- Teacher-force the Vietnamese answer.
- At each Vietnamese target position and layer, read the lens log-probability of the first content token of each L form and of its A1 distractor.
- Not a decision-rule input.

**A5 — Item set for the H1/H2 rise rules (D4).**

- The **cognate-free subset** is 729 main concepts. Conditions:
  - English–French and English–Indonesian filter-4 surface distance ≥ 0.45;
  - the first content tokens of the vi, en, zh, fr and id targets are pairwise distinct under **all three** tokenizers.
- sha256 of the sorted id list: `f1e7a6b5a0c197b1769215224503ee33a8f2e216943ae0d8088d55219042a58f`.
- All items are sensitivity R9. H3 uses the full strata.

**A6 — Language alignment (M3) specification.**

- **Activation:** the block output at the last prompt token (the final `:`), in fp32.
- **Centering:** subtract the mean of the five fit-split class means.
- **Validation and diagnostics:**
  - frozen gate: ≥ 95% of the 130 validate prompts per layer and format;
  - FLORES transfer, 199 × 5 validate sentences (layers below 95% are flagged "label-dominated");
  - cross-format transfer;
  - `c_random` is reported.
- **H2 "for all L"** means L ∈ {en, zh, fr, id}.
- **Supporting statistic:** `D_en = LA_en − mean(LA_fr, LA_id)`, with a bootstrap CI.
- **Format matching:** repetition with repetition, translation with translation. Cloze (descriptive) uses the repetition directions.

**A7 — Steering (D6; replaces the perplexity guard).**

- **Items:** translation-format set-1 main items. Up to 300 concepts per model whose unsteered greedy answer is scored VI (A10), selected by a frozen, stratum-proportional, seeded rule (`derived/e8_order.jsonl`, `derived/e8_rule.md`).
- **Direction:** `u_ℓ = μ_en^ℓ − μ_vi^ℓ` (translation directions, fit split), added as `k · u_ℓ` at block ℓ's output at the last prompt position and at every generated position. One layer at a time, k ∈ {0.5, 1, 2}. The control is a random direction of the same norm.
- **Generation:** greedy, stopping at `\n`, `" -"` or 8 tokens.
- **Outcome classes:** VI, EN (clean flip), OTHER-LANG and DAMAGE. The guard is the DAMAGE rate; GlotLID is descriptive only. A continuous companion is reported: `log P(en) − log P(vi)` at the output.
- **Coincidence with `L_cross`:** `|depth(flip peak) − depth(L_cross)| ≤ 0.10`, and the bootstrap CI of the difference includes 0. Flip curves are reported in blocks as well as depth.

**A8 — H3 frequency covariate (D5; replaces the v1.2 choice of `log_freq`).**

- `zipf_en` is the primary covariate; `zipf_vi` and `zipf_zh` are sensitivities.
- The matched-subsample check uses `zipf_en` in place of Vietnamese frequency.
- **Reason:** `zipf_vi` tracks syllable frequency, not word frequency. Its mean is flat across syllable counts, and it correlates with `zipf_en` at Spearman 0.30.

**A9 — Inference set-up.**

- Raw completion with the released prompt strings unchanged: no chat template, tokenizer-default special tokens, joint prompt-and-target tokenization with a prefix assertion. No prompt ends in whitespace.
- bf16 weights; the analysis lens (final norm + unembedding) runs in fp32.
- One code path for all models: Hugging Face forward hooks on block outputs. For Gemma 3, the hooks sit on the text tower, and the post-norm attention and MLP outputs are the residual contributions.
- Library versions are pinned (`torch 2.8.0`, `transformers 4.57.6`, `tokenizers 0.22.2`).
- **Lens-validity gate (E1, OPS-7):**
  - The model-dtype lens path (the model's own final norm and unembedding applied to the hooked block-N output) must reproduce the model logits: max |Δ log-prob| < 1e-3 (expected exact), with 100% top-1 agreement.
  - The residual decomposition must hold to a relative error below 1e-4.
  - The fp32 analysis lens's difference from the bf16 model logits is reported, not gated. It cannot meet 1e-3 by construction (about 2e-3 on tiny test models).

**A10 — Competence reporting and scoring.**

- A greedy answer is produced for every test prompt, with the same stopping rule as A7. Accuracy is reported by model and format.
- **An answer is scored VI** if, after NFC, trimming and casefolding, it equals or begins with (at a word boundary) the concept's vi form or any `vi_alts` form.
- **Shared Russian sources.** Some translation prompts share their Russian source across concepts: 27 groups (55 records) per diac set and 18 groups (37 records) per nodiac set; for example, палец → ngón tay / ngón chân. For these, a match to the vi form of **any** concept in the same source group also counts as VI. These items are flagged and counted in every competence table and in the E8 sample.
- **EN, OTHER-LANG and DAMAGE:**
  - **EN:** the same matching against the en form.
  - **OTHER-LANG:** the same matching against the zh, fr, id or ru form.
  - **DAMAGE:** anything else.
- The competent subset is sensitivity R7. E8 uses competent items only.

**A11 — Remaining pins.**

- **`D_zh`:** the mean over ℓ/N ∈ [0.25, 0.75] of `LA_zh(sino) − LA_zh(nonsino)`, with a stratified bootstrap.
- **Holm families (D9):**
  - (a) the H1/H2 tests, 3 models × 5 languages = 15, one-sided;
  - (b) the H3 tests, 3 models × 2 instruments (`D_zh`, and the stratum × language term for `rise_zh` in the §7.1 model) = 6.
- **§7.1 model:**
  - natural spline with 5 df for depth;
  - `n_tokens` = the readout target's content tokens;
  - complete cases only (no imputation);
  - if the `task_format` random effect is singular, it is refitted as fixed;
  - fitted with statsmodels 0.15 `MixedLM` (D11; R/lme4 is not available on the compute server).
- Cloze stays excluded from §7.1 (v1.3).
- Landmarks are computed on concept-mean curves with 1,000 bootstrap resamples over concepts, seed `20260925`.

**A12 — `vi_nodiac` diagnostic readout.**

- In the diac condition, the concept's diacritic-stripped form (case-preserving, identical to the nodiac-file targets) and its matched distractor's form are also scored.
- Descriptive only; it feeds the interpretation of E9.

**Directions/test prompt overlap (D12).**

- Five directions-split records (translation format, Vietnamese target) are byte-identical to `translation_diac_set1` test prompts because they share a Russian source:

  | Directions concept | Test concept |
  |---|---|
  | `9b523e0c2ba7` | `1516bff503d4` |
  | `c8cac6e6221a` | `24d4529a0170` |
  | `70d57c0663fa` | `84b2d9c93a58` |
  | `44b4b01dae07` | `cc2d2ff59954` |
  | `5fec3d7df089` | `fb2a9a229f1a` |

- They are sent to no model before this tag. The experiment repo's integrity lock enforces this.
- After the tag, the direction fit (P-D) uses the frozen fit split unchanged, and the overlap is reported.
- One repetition-format zh prompt is shared by a fit concept (`92286b25eb72`) and a validate concept (`d014c4a6e97c`). It is kept and footnoted in the A6 gate report.

**Tuned lens (R5, D8).** Trained for Llama only, on a balanced five-language sample. The corpus and training details are fixed in the R5 task before training, and no test item is used.

**Pre-declared robustness battery:**

| # | Check |
|---|---|
| R1 | Few-shot sets 2–6 (fallback: sets 2–3) on the cognate-free items |
| R2 | FLORES vs matched directions |
| R3 | `L_concept` at 5/10/20% |
| R4 | Raw `s` vs `Δ` |
| R5 | Tuned lens (Llama) |
| R6 | v1.2 subsets (`bt_pass_strict_v1`, `external_attested`, all-NLLB-agree, `min_surface_dist ≥ 0.60`) |
| R7 | Competent subset |
| R8 | `s¹` / `s⁺` / sum, and zh with its space |
| R9 | All-items five-way |
| R10 | Along-path readout |
| R11 | Exclusion of the six card-flagged questionable targets |

E9 (diacritic-stripped input) remains optional.

**Adjudication order.**

- The per-model H1/H2 verdicts are written (Checkpoint H) before the E6, E7 and E8 analyses run.
- They are compared with the three pre-written abstracts in `abstracts_prewritten.md`, which are committed with this amendment.
- Every departure from v1.2/v1.3 in the paper cites its v1.4 item. The paper states that v1.4 predates any study-model run.

---

## Effective analysis specification (v1.4)

This section restates §2, §5.2, §5.6 and §7 as they apply after the v1.2, v1.3 and v1.4 amendments. The frozen v1.1 text above is kept unchanged for audit. Where the two differ, this section governs.

### Models

| Role | Checkpoints |
|---|---|
| Primary | `google/gemma-3-4b-pt` @ `cc012e0`, `Qwen/Qwen3-4B-Base` @ `906bfd4`, `meta-llama/Llama-3.2-3B` @ `13afe51` (bf16 weights; fp32 analysis lens) |
| Optional secondary replication | The registered instruct checkpoints |

### 2. Hypotheses (unchanged)

H1 (high-resource pivot), H2 (language-neutral concept space) and H3 (etymology-conditional routing in Qwen) are as frozen above. H3 is **exploratory** (v1.2, Checkpoint C).

### 2.1 Decision rules

**Quantities:**

- `LA_L(ℓ)` is as in §5.3, with the A6 specification: last prompt token, centered class means, 95% validate gate.
- `rise_L(ℓ)` is the **matched-distractor rise in nats** (A1):
  - `Δ(c,L,ℓ) = s(c,L,ℓ) − s(d(c,L),L,ℓ | c's prompt)`;
  - `rise_L(ℓ) = mean_c Δ(c,L,ℓ) − mean_c mean_{ℓ'∈ℓ_early} Δ(c,L,ℓ')`;
  - `ℓ_early` = blocks with ℓ/N ≤ 0.15.
- **Items for the rise rules:** the cognate-free 729 main concepts (A5), per model and per format (repetition and translation reported separately). The zh readout uses the no-space target (A3).

**Rules:**

| Outcome | Rule |
|---|---|
| Support H1 | `max_ℓ LA_en(ℓ) ≥ 0.50` on Vietnamese prompts, **and** `max_ℓ rise_en(ℓ) ≥ 2 × mean(max_ℓ rise_fr, max_ℓ rise_id)` |
| Support H2 | `max_ℓ LA_L(ℓ) < 0.20` for all L ∈ {en, zh, fr, id}, **and** all five `max_ℓ rise_L` (vi, en, zh, fr, id) fall within a 1.5× band |
| Support H3 (exploratory) | Qwen: `D_zh > 0` with a stratified bootstrap 95% CI excluding 0; Gemma and Llama: CIs include 0 |
| Inconclusive | Any pattern between the H1 and H2 bands. Report as such; do not reframe post hoc |

**Reporting:** every rule quantity is reported with its point estimate and its 95% bootstrap CI over concepts (1,000 resamples). Where a CI straddles a threshold, this is stated.

### 5.2 M2 — Multi-token logit lens

1. Teacher-force the full target after the unchanged prompt, with joint tokenization.
2. At each block output ℓ = 1..N, apply the fp32 copy of the final norm and the unembedding (and the softcap, if the model has one).
3. `s(c,L,ℓ)` = the mean of `log P_ℓ(t_i | prompt, t_<i)` over **content** tokens (A4).
4. `s¹` and `s⁺` are pre-declared decompositions (R8).
5. Cross-language comparison uses **only** `Δ` / `rise` (A1). Raw `s` and the v1.1 z-statistic go to the appendix (R4).

### 5.6 Landmarks

| Landmark | Definition |
|---|---|
| `L_concept` | First ℓ where, for any readout language L, `p_L(ℓ) ≥ 0.10 · p_vi(N)` (A2; 5% and 20% sensitivities) |
| `L_peak` | Layer of maximum `rise_L` among non-Vietnamese readout languages |
| `L_cross` | First layer after `L_peak` where `rise_vi` exceeds the pivot language's `rise` |
| Pivot width | `(L_cross − L_concept) / N` |
| Pivot magnitude | `max_ℓ rise` (nats) of the leading non-Vietnamese language |

All landmarks are computed on concept-mean curves, with 1,000 bootstrap resamples over concepts.

### 7. Analysis plan

- **7.1 Primary model:**

  ```
  rise ~ language * ns(depth, df = 5) + stratum * language
         + zipf_en + n_tokens + n_syllables + concreteness
         + (1 | concept) + (1 | task_format)
  ```

  - `n_tokens` = the readout target's content tokens.
  - Complete cases only.
  - If the `task_format` random effect is singular, it is refitted as fixed.
  - Fitted with statsmodels 0.15 `MixedLM`.
  - Sensitivities: `zipf_vi`, `zipf_zh`.
  - Cloze is excluded.
- **7.2 Uncertainty:** bootstrap over concepts, 1,000 resamples, 95% percentile CIs, stratified by stratum for H3.
- **7.3 Multiple comparisons (Holm within family):**
  - (a) H1/H2, 3 models × 5 languages = 15 one-sided tests;
  - (b) H3, 3 models × 2 instruments (`D_zh`, and the stratum × language term for `rise_zh` in §7.1) = 6.
- **7.4 Power:** H3 is exploratory (κ lower bound 0.557; Holm MDE d = 0.206).
- **7.5 Robustness:**
  - R1–R11 as listed in v1.4;
  - E9 optional;
  - cloze is descriptive only (v1.3).
- **Adjudication:** the per-model verdicts are committed (Checkpoint H) before the E6–E8 analyses, and compared with `abstracts_prewritten.md`.
