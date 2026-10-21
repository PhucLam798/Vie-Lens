# E2–E5 analysis specification (T13a)

Frozen: OPS-14 decided by the human on 2026-10-02. The governing source remains `docs/prereg/PREREG.md` at `prereg-v1.4`.

## Table of contents

1. Analysis inputs, item sets, and layer convention
2. M2 sequence scores, matched-distractor contrasts, and rise
3. H1/H2 quantities, bootstrap, and Holm family (a)
4. E3 landmarks and C1
5. M3 directions, LA, and E5 validity diagnostics
6. E2 Wendler prefix-sum probability
7. A10 competence table
8. Checkpoint H outputs
9. OPS-14 decisions and resolved specification gaps

## 1. Analysis inputs, item sets, and layer convention

### Input tables

| Input | Table and columns used | Analysis role |
|---|---|---|
| P-L lens output | Long parquet columns `model, model_revision, pass, format, condition, fewshot_set, concept_id, target_lang, target_variant, role, ref_concept_id, tok_idx, token_id, is_content, layer, metric, value` | E2–E4 M1/M2 scores, contrasts, rises, landmarks, and the answer-position join for generations. |
| P-L residual index | JSONL columns `array_row, concept_id, format, lang, condition, fewshot_set, m1_extension` plus the aligned `.resid.npy` row | E2 answer-position residual selection. The residual array stores `h_ℓ` for ℓ=0…N at the last prompt token. |
| Generation table | P-L generations JSONL fields `concept_id, format, text`, joined to concept forms and Russian-source groups | A10 competence classes and shared-source flags. |
| P-D matched index and arrays | Parquet columns `row, concept_id, lang, format, split, n_tokens, n_special`; aligned `.resid.npy` is float32 `[row, ℓ=0…N, hidden]` | Fit/validate directions, gate, and cross-format transfer. |
| P-D FLORES index and arrays | Parquet columns `row, sentence_id, lang, format, split, n_tokens, n_special`; aligned `flores.last.npy` and `flores.mean.npy` are float32 `[row, ℓ=0…N, hidden]` | Held-out FLORES transfer, without sentence text. |
| Frozen target forms | `derived/target_forms.jsonl`: `concept_id, target_lang, target_variant, target` | E2 prefix token sets, joined by concept and readout. |
| Frozen A5 IDs | `derived/cognate_flags.jsonl`: `concept_id, cognate_free` | H1/H2 primary item subset. Hash sorted IDs using `id_list_sha256`; required result is `f1e7a6b5a0c197b1769215224503ee33a8f2e216943ae0d8088d55219042a58f`. |
| Concept table | `data/external/viLens-concepts/concepts.jsonl`: `concept_id, split, m1_extension, m1_eligible_<family>, stratum, vi, vi_alts, en, zh, fr, id, ru` | Main/extension filters, E2 eligibility, H3 strata, and A10 matching. |

### Item filters

- **Main:** `split == "test"` and `m1_extension == false` (1,535 concepts). Readouts are filtered to this set for E3/E4 unless a stated E2/E3 extension analysis applies. [AGENTS.md rule 31; PREREG A1/A5]
- **H1/H2 rise rules:** the sorted 729 IDs in `derived/cognate_flags.jsonl` with `cognate_free == true`; assert the stated SHA-256 before calculating. [PREREG A5; decisions D4]
- **Formats:** repetition and translation are calculated and reported separately. Never pool them. [PREREG Effective §2.1; docs/experiments.md §6]
- **Cloze:** descriptive curves/tables only; excluded from H1/H2 decisions, Holm family (a), §7.1, and stratum analyses. It uses repetition directions for LA descriptions. [PREREG A6; v1.3 amendment; AGENTS.md rule 31]
- **M1 extension:** included only in E2 and E3; never included in H1/H2 rise, A10 test competence, or E5 test-prompt LA. [AGENTS.md rule 31; decisions OPS-13]
- **E2/E3 extension boundary:** E2 includes every family-eligible main and extension prompt. P-L extension records have correct targets but no A1 distractors, so they never contribute to `Δ`, `rise`, H1/H2, or an A1-based landmark. C1 reports the instrument comparison on `m1_all` as its separate extension-inclusive descriptive line. [AGENTS.md rule 31; decisions OPS-13; OPS-14-13]
- **H3 strata:** use the full main set; compare `sino` with `nonsino`, excluding `ambiguous` and `other_loan`. [AGENTS.md rule 31; PREREG v1.2 and A11]

### Layers

All E2–E5 curves use decoder block outputs ℓ=1…N; embedding output ℓ=0 is stored but excluded. `N` is the model's frozen number of decoder blocks (Gemma 34, Qwen 36, Llama 28). For rise baselines, `ℓ_early = {ℓ: 1 ≤ ℓ ≤ N and ℓ/N ≤ 0.15}`, using `e1.early_depth_max` from `configs/experiments.yaml`. [PREREG Effective §5.2; A1; AGENTS.md rule 36]

## 2. M2 sequence scores, matched-distractor contrasts, and rise

For a target sequence with content-token indices `I(c,L,v)` in the P-L long table, define:

\[
s(c,L,v,ℓ)=\frac{1}{|I|}\sum_{i\in I}\texttt{value}_{c,L,v,\ell,i},\qquad
s^1(c,L,v,ℓ)=\texttt{value}_{i_1},\qquad
s^+(c,L,v,ℓ)=\frac{1}{|I|-1}\sum_{i\in I, i\ne i_1}\texttt{value}_{i}.
\]

Here `metric == lens_logprob`, `role == correct`, and `is_content == true`; `i₁` is the smallest `tok_idx` among content tokens. `s⁺` is NA for one-content-token targets. Also retain the R8 sequence sum `s_sum=Σᵢ valueᵢ`. Whitespace-only tokens are teacher-forced in P-L but excluded from each score. [PREREG A4; Effective §5.2; decisions D1/D3]

Apply the same functions to `role == distractor` rows. The A1 distractor was teacher-forced after the target concept's prompt, so pair by `(model, format, condition, fewshot_set, concept_id, target_lang, target_variant, layer)` and keep its `ref_concept_id` as `d(c,L)`. For each score component `q ∈ {s,s¹,s⁺,s_sum}`:

\[
\Delta_q(c,L,v,ℓ)=q_{correct}(c,L,v,ℓ)-q_{distractor}(d(c,L),L,v\mid c\text{'s prompt},ℓ).
\]

The primary contrast is `Δ = Δ_s`, in nats. Sensitivity contrasts use `Δ_s¹`, `Δ_s⁺`, and the sequence-sum contrast. [PREREG A1/A4; decisions D1]

For each concept, language/readout variant, model, and format:

\[
rise_{L,v}(c,ℓ)=\Delta(c,L,v,ℓ)-\frac{1}{|\ell_{early}|}\sum_{j\in\ell_{early}}\Delta(c,L,v,j),
\quad rise_{L,v}(ℓ)=\frac{1}{C}\sum_c rise_{L,v}(c,ℓ).
\]

The primary five readouts are `vi/released`, `en/released`, `zh/nospace`, `fr/released`, `id/released`. Sensitivities are `zh/released`, `vi/nodiac`, and `ru/released` on translation only. H1/H2 use the 729 cognate-free main items; cloze is descriptive. Inputs are the P-L long table columns listed in §1; outputs are rule-38 long tables `results/analysis/<source>/scores.parquet`, `deltas.parquet`, and `rise.parquet`. [PREREG A1/A3/A5/A12; Effective §2.1 and §5.2; docs/experiments.md §5 E4]

Per-concept scores, contrasts, and rises are not bootstrapped. In `rise.parquet`, `rise_main` is the primary `Δ_s` curve; `rise_first_token`, `rise_continuation`, and `rise_sequence_sum` preserve the A4/R8 decompositions; `rise_readout_sensitivity` stores the `zh/released`, `vi/nodiac`, and translation-only `ru/released` sensitivity curves. Only the registered primary readouts enter rule and landmark calculations. Concept bootstrap CIs are produced for the aggregate rule and landmark quantities in §§3–4. [PREREG A1/A4; decisions D1/D3; docs/experiments.md §5 E4]

## 3. H1/H2 quantities, bootstrap, and Holm family (a)

For each model × format, let `M_L = max_{ℓ=1…N} rise_L(ℓ)` for `L ∈ {vi, en, zh, fr, id}`. The H1 rise quantity is `Q_H1 = M_en − 2·mean(M_fr, M_id)`; it holds iff `Q_H1 ≥ 0`. Report `M_en / mean(M_fr, M_id)` when its denominator is positive. The H2 rise band holds iff `min_L M_L > 0` and `max_L M_L / min_L M_L ≤ 1.5`. Every maximum is over block outputs ℓ=1…N. (OPS-14-1–3)

For each block, compute primary LA for `L∈{en,zh,fr,id}` on all main Vietnamese-query prompts of the matching format. Use `h̃=h−μ̄`, `v_L=μ_L−μ̄`, `c_nat,L` as the mean validate cosine for native-language L, and one seeded `c_rand` per family × format × layer from 1,000 unit Gaussian directions. If any `c_nat,L−c_rand≤0`, that layer's LA is undefined, flagged and treated as failed. A layer passes only when at least 124/130 validate prompts are classified correctly. Failed and undefined layers are excluded from every primary maximum, `D_en` scalar, and window mean; calculate the maximum over all defined layers as a sensitivity and state whether including failed layers would change a rule outcome. If no layer passes for a family × format, LA is not evaluable and its H1/H2 outcome is inconclusive. Bootstrap query concepts only, holding fitted directions, native validation cosines, and `c_rand` fixed. Report each maximum, its 95% CI, and whether it straddles its threshold. (OPS-14-4–7, 11)

For exploratory H3, `D_zh = mean_{ℓ/N∈[0.25,0.75]}(mean_{c∈sino} LA_zh(c,ℓ) − mean_{c∈nonsino} LA_zh(c,ℓ))`. Use only gate-passing layers in the window, and bootstrap within each stratum over concepts for 1,000 draws. Inputs are the main Vietnamese-query LA values joined by `concept_id` to the concept table's `stratum`; exclude `ambiguous` and `other_loan`, and use the full main set, not the A5 cognate-free subset. The downstream E6 report owns this statistic (`reports/E6.md`); it is not calculated in the T13a smoke run or this Checkpoint-H E2–E5 collector. (OPS-14-16; PREREG Effective §2.1/§7.2/A11; AGENTS.md rule 31)

Bootstrap over concepts with `B=1,000`, 95% percentile intervals. For each item set × model × format, draw one concept-index matrix with key `bootstrap|<item_set>|<model>|<format>` and reuse it across readouts and quantities for that item set. The stream uses the `derived/seeds.json` CRC32 NumPy scheme and seed `20260925`. Item sets are `cf729` (729 repetition / 721 translation), `main` (1,535 / 1,524), `m1_main`, and descriptive `m1_all`. Query concepts alone are resampled for LA; fitted directions, `c_nat`, and `c_rand` stay fixed. Report valid-draw counts whenever a statistic can be undefined. (OPS-14-11; PREREG Effective §2.1/§7.2; decisions D9; derived/seeds.json)

For family (a), within each format test the 15 `model × L` hypotheses using `W_L`, the mean of `rise_L(ℓ)` over block outputs satisfying `0.25 ≤ ℓ/N ≤ 0.75`, with `H0: W_L ≤ 0`. The bootstrap p-value is `p=(1 + #{b: W_L^(b) ≤ 0})/(1+B)`. Holm-adjust the 15 raw p-values within each format; repetition and translation are separate families. `M_L` remains the decision-rule quantity and is reported with its CI. (OPS-14-10; PREREG Effective §7.3; decisions D9)

## 4. E3 landmarks and C1

For E3, calculate concept-mean curves from P-L sequence scores. For each model × format, readout `L`, and block, let `\bar{s}_L(ℓ)=mean_c s(c,L,ℓ)` and `p_L(ℓ)=exp(\bar{s}_L(ℓ))`. Let `p_vi(N)=exp(\bar{s}_vi(N))`. Then `L_concept(q)` is the first block where any readout language in `{vi,en,zh,fr,id}` has `p_L(ℓ) ≥ q·p_vi(N)`; report `q ∈ {0.05,0.10,0.20}`. Primary landmarks use `cf729`; `main` is the R9 sensitivity. `L_peak` is the earliest block attaining the largest non-Vietnamese `rise_L`; `L_cross` is the first later block where `rise_vi > rise_pivot`. If no crossing occurs, `L_cross=NA` and pivot width is NA. Pivot width is `(L_cross−L_concept(0.10))/N`; pivot magnitude is `max_ℓ rise_pivot(ℓ)`. Report the leading pivot language, and the A4 `s¹` rule and `L_cross` variants. (OPS-14-8, 11, 12, 17; PREREG Effective §5.6; decisions D2)

Bootstrap every landmark over concepts using the shared indices in §3; each draw recomputes all concept means and landmarks before its percentile interval. Inputs: P-L table `concept_id,target_lang,target_variant,role,ref_concept_id,tok_idx,is_content,layer,metric,value`; primary curves use main items, with extension records limited to the E2/E3 subset analysis. Outputs are `results/analysis/<source>/landmarks.parquet` plus `reports/analysis_core_<source>.md`. [PREREG Effective §5.6; A2; docs/experiments.md §5 E3]

**C1 instrument effect:** on `m1_main`, compare the M2 `L_concept` with the M1 `L_concept` computed from mean prefix-sum probability, for every configured threshold; report differences in blocks and normalized depth. **C1 selection effect:** compare M2 `L_concept(m1_main)` with `L_concept(main)`, paired on the `main` bootstrap draw after filtering that draw to its M1-eligible concepts. Report differences in blocks and normalized depth. The extension-inclusive instrument comparison on `m1_all` is a separate descriptive line. No rise-based landmark is computed on extension items. These are descriptive, with no decision rule or Holm test. (OPS-14-13; docs/experiments.md §5 E2–E3)

## 5. M3 directions, LA, and E5 validity diagnostics

### Verbatim v1.1 §5.3 source

The governing method is amended by A6 in v1.4. The frozen §5.3 language is quoted here verbatim from `data/external/viLens-concepts-preparation/docs/protocol_v1.1.md`, “5.3 M3 — Language directions (token-free)”:

> For each language `L`, compute `v_L^(ℓ)` per layer as the difference of means between activations on `L` text and activations on all other languages.
>
> **`[REFINEMENT vs v1.0]` Directions must be estimated on prompt-format-matched data.** v1.0 estimated directions on FLORES sentences and applied them to short word-completion prompts — a domain mismatch that could bias the alignment scores. Estimate directions on the same prompt templates used in §4.6, populated with held-out vocabulary. Retain the FLORES-based directions as a robustness check and report both.
>
> **Validation gate:** each `v_L` must classify held-out prompts by language at >95% accuracy before use. If not, the direction is not measuring language and the analysis stops.
>
> Normalized alignment:
>
> ```
> LA_L(ℓ) = ( cos(h_ℓ^vi-prompt, v_L) − c_random ) / ( cos(h_ℓ^L-prompt, v_L) − c_random )
> ```
>
> `c_random` = mean cosine to random unit directions of matched norm, per layer.
>
> This measurement adjudicates H1 vs H2. The logit lens cannot, because it reads through an English-shaped unembedding by construction.

### A6 fit, gate, and diagnostics

For each family, format, and block `ℓ`, read float32 `h_ℓ` from the P-D matched array at the last prompt token. On the 106 fit concepts, compute five class means `μ_L^ℓ`; center by `μ̄^ℓ = mean_L μ_L^ℓ`; set `v_L^ℓ=μ_L^ℓ−μ̄^ℓ`. Use only the matching format's fit prompts for primary directions. [PREREG A6; decisions OPS-12; docs/experiments.md §5 E5]

On the 26 validate concepts × 5 languages (130 prompts per format), center each activation by `μ̄`, classify it as `argmax_L cos(h−μ̄,v_L)`, and report correct/130 per layer. In general, transfer accuracy on rows `R` is `|R|⁻¹ Σ_{r∈R} 1[argmax_L cos(h̃_r,v_L)=lang_r]`. A layer passes at ≥124/130 (≥95%); only passing layers enter `max_ℓ LA`. [PREREG A6]

For FLORES transfer, apply each matched-format direction to the 199 validate IDs × 5 languages (995 rows). Report accuracy separately for `flores.last.npy` (primary) and `flores.mean.npy`; flag each pooling's layer when accuracy <95%. For cross-format transfer, apply repetition directions to the translation validate prompts and translation directions to repetition validate prompts (130 rows). (OPS-14-15 decided; PREREG A6; decisions OPS-12)

For `c_random` diagnostics in T13a, use the held-out matched Vietnamese validate activations for the fitted format/layer and 1,000 seeded standard-Gaussian vectors normalized to unit norm. For prompt `i`, `c_random(i,ℓ)=mean_{r=1…1000} cos(h_i^ℓ−μ̄^ℓ,u_r^ℓ)`; report the mean across the 26 held-out Vietnamese prompts. The report records the exact stream key `la_random|<family>|<format>|<layer>`. The null reference for exchangeable class offsets is −1/(K−1)=−0.25, while `c_random` is expected near zero. This diagnostic uses Monte Carlo averaging, not a concept bootstrap; gate and transfer accuracies are exact counts on the fixed validate sets. (OPS-14-4; docs/experiments.md §3 A6)

For the main LA query in T13b, `c_nat,L(ℓ)=mean_{i∈validate,label_i=L} cos(h_i^ℓ−μ̄^ℓ,v_L^ℓ)`. At each family × format × layer, `c_rand(ℓ)` is the mean cosine of the full set of centered Vietnamese query activations with 1,000 seeded unit Gaussian directions, independently drawn per layer with key `la_random|<family>|<format>|<layer>`. Use the same scalar `c_rand(ℓ)` in each query's ratio: `LA_L(c,ℓ)=(cos(h̃_c^ℓ,v_L^ℓ)−c_rand(ℓ))/(c_nat,L(ℓ)−c_rand(ℓ))`; `LA_L(ℓ)` is the concept mean. Bootstrap the query concepts only, holding fit means, validate baselines, and `c_rand` fixed. Inputs are P-D matched `row,lang,split` plus residual arrays and P-L residual-index `array_row,concept_id,format,lang,split,m1_extension` plus aligned residual arrays; use main Vietnamese test prompts for repetition (1,535), translation (1,524), and descriptive cloze (34) with repetition directions; layer is block output ℓ=1…N. Cloze is descriptive and excluded from verdicts. Outputs are rule-38 `results/analysis/<source>/la.parquet` and its summary. (OPS-14-4–6, 11; PREREG §5.3/A6/A11)

For every query prompt and block, `D_en(c,ℓ)=LA_en(c,ℓ)−(LA_fr(c,ℓ)+LA_id(c,ℓ))/2`; the curve is its concept mean with pointwise 95% concept-bootstrap CIs. The scalar is the mean over passing blocks with `0.25 ≤ ℓ/N ≤ 0.75`. (OPS-14-7; docs/experiments.md §3 A6)

Report D12's five translation Vietnamese overlap records without changing the fit split. Retain the repetition zh fit/validate shared-prompt pair as a footnote: `92286b25eb72` fit / `d014c4a6e97c` validate. [PREREG D12; decisions D12]

T13a computes only direction validity on P-D rows; it does not compute LA on test prompts, so no test result file is an input. E5 outputs are `reports/E5_gate.md` and `reports/E5_gate.csv`; the CSV has one row per family × fit-format × block.

## 6. E2 Wendler prefix-sum probability

Use M1-eligible concepts by `m1_eligible_<family> == true`, including main and extension; report cloze only descriptively. From the P-L long table/residual index select `model,format,concept_id,target_lang,target_variant,role,array_row,m1_extension` and correct target rows only. For each prompt, use its saved final-prompt-token residual `h_ℓ` at stored layers ℓ=0…N, apply the full-vocabulary fp32 final norm and unembedding, and softmax over the whole vocabulary. The E2 output retains layer 0 for the full per-prompt record; any layer maximum uses only block outputs ℓ=1…N (OPS-14-1). For target word `w`, construct the deduplicated token set `T(w)` from every non-empty character prefix, bare and with the tokenizer's word-start marker (`▁` for Gemma, `Ġ` for Llama and Qwen). For zh and ru, also add the token for the first UTF-8 byte of the first character, resolved through the tokenizer vocabulary: `<0xHH>` for Gemma's SentencePiece byte fallback or the byte-level BPE byte-to-Unicode symbol for Llama and Qwen. If that token is absent, add nothing and retain the item in the count. The E2 score is

\[
P_{M1}(w\mid prompt,ℓ)=\sum_{t\in T(w)} softmax(logits_ℓ(h_ℓ))_t.
\]

Use `zh/nospace` as the primary Chinese target; Latin targets retain the leading space in source form, with plain/marker variants covered by the token set. The P-L prompt residual index and `.resid.npy` are joined by `array_row`; `concept_id` then joins to concept eligibility and frozen target forms. Report the exact per-prompt/per-layer probability without a bootstrap or inferential CI; cloze rows are descriptive. Output rule-38 long tables by family × format at `results/analysis/m1/<source>/family=<family>__format=<format>__pass=m1__source=<source>.parquet`, with provenance sidecars. Full-vocabulary final norm/unembedding weights are loaded only by the GPU entrypoint `job_e2_m1.py`; it is not submitted in T13a. (OPS-14-14 decided; docs/experiments.md §2.5 and §5 E2; task source description)

The required Wendler comparator is `process_tokens` in the official [`Translation.ipynb`](https://github.com/epfl-dlab/llm-latent-language/blob/main/Translation.ipynb). It enumerates every non-empty Unicode-character prefix, checks both the bare spelling and the word-start-marked spelling against the vocabulary, deduplicates IDs, and adds the model-specific token for the first UTF-8 byte of zh/ru when present. Russian is supported only for translation descriptions; it is not part of the frozen primary M1 language family. (OPS-14-14; task Sources; docs/experiments.md §2.5)

## 7. A10 competence table

For each main test generation row (`model,format,concept_id,text` plus stop metadata), call `vilens.score.classify` after the frozen stop rule. A10 classifies VI first (NFC, trim, casefold, Vietnamese orthographic key; exact or word-boundary prefix), then EN, then OTHER-LANG (zh/fr/id/ru), then DAMAGE. For translation prompts whose normalized Russian source belongs to a shared-source group, any grouped concept's VI form/alternative can match; flag every such row and report its count. `competent` means class `VI` and defines the R7 subset used by E8. Report exact class counts and denominators by model × format; no bootstrap is specified for this descriptive table, and no generation is regenerated. Output `results/analysis/primary/competence.parquet` and `reports/analysis_core_primary_competence.csv`. [PREREG A10; decisions OPS-8/OPS-9; docs/experiments.md §3 A10]

## 8. Checkpoint H outputs

Produce: (1) a rule table per model × format with each point estimate, 95% CI, and threshold-straddling note; (2) one model verdict. A model supports H1 or H2 only if both repetition and translation meet that rule; otherwise report “inconclusive” with the per-format pattern. Select the matching pre-written abstract A/B/C from `docs/prereg/abstracts_prewritten.md`; fill only brackets, and keep unresolved bracket alternatives when the computed values do not decide them. Write only `reports/adjudication_draft.md` for the human to finalise and commit as `reports/adjudication.md`. No E6–E8 analysis begins until the human has finalised and committed the adjudication. (OPS-14-9, 18; PREREG Effective §2.1/§7; docs/experiments.md §5 Checkpoint H)

## 9. OPS-14 decisions and resolved specification gaps

The human resolved the former OPS-14-z gaps in `docs/decisions.md`, OPS-14:

| Resolved question | Decision |
|---|---|
| C1 scales, effects, and paired selection draw | OPS-14-13 |
| `L_peak` tie handling and missing `L_cross` | OPS-14-12 |
| Bootstrap item sets, keys, shared draws, and LA uncertainty scope | OPS-14-11 |
| LA random-direction PRNG key and per-layer independence | OPS-14-4 |
| Extension-inclusive C1 summary and exclusion from rise landmarks | OPS-14-13, with OPS-13 |
| H3 handling of failed layers in its depth window | OPS-14-16 |

The other operational choices are recorded and implemented at their corresponding OPS-14 items: block-layer and decision rules (1–3), LA baselines, query sets, gate and summaries (4–7), `L_concept` (8), verdict aggregation (9), Holm family (10), token prefixes (14), FLORES pooling (15), and the Checkpoint-H draft workflow (18). The exact human decision text is in `docs/decisions.md`; this specification follows those decisions without adding defaults.
