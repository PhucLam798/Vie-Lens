# viLens — Experiment plan (`experiments.md`, draft 2.1)

**Paper:** *The Latent Language of Vietnamese Prompts: Where and When Do LLMs Translate Back?*
**Date:** 2026-09-28 (draft 1: 2026-09-27; changes are listed in §11)

**Execution:** all work runs on the lab server, in `/datastore/luannt/Vie-Lens` (GitHub `PhucLam798/vie-lens`), with GPU steps submitted as Slurm jobs. Operational rules are in `AGENTS.md`, which takes precedence on how to run things.

**Basis:**
- Protocol v1.1 (`docs/protocol_v1.1.md`), as amended by v1.2 and by the **v1.3 cloze rebuild** (`PROTOCOL_AMENDMENTS.md`, preparation repo tag `data-v1.3.1`).
- The frozen hypotheses and analysis plan are in `PREREG.md` at tag `prereg-v1.2`.

**Status:**
- The data are frozen at HF `v1.3.1`.
- **No run on the study models has happened yet.** Three exploratory single-prompt notebooks exist (§2.9). They use different checkpoints and touch no test items.

**Naming:** the preparation repo already uses "v1.3" for the cloze rebuild. The analysis amendment proposed here is therefore **v1.4** (draft 1 called it v1.3).

**Provenance of the numbers:**
- All numbers come from the released files, the model tokenizers (no weights), and the saved outputs of the three notebooks.
- The mirror tokenizers reproduce every stored `ntok_*` value: 0 mismatches across 27,555 cells.

---

## 0. Summary

1. **The next step is still an amendment (now v1.4, §3), not a model run.** Four problems in the frozen analysis would make the headline tests uninterpretable:
   - the `rise` normalization discards amplitude;
   - `L_concept` is anchored to a suppressed quantity;
   - the released Chinese targets tokenize badly;
   - about half the French and Indonesian control readouts are English cognates.

   The notebooks add direct, if qualitative, support for two of the fixes:
   - Qwen3.5 decodes the space-free `学校`, not `' 学校'`;
   - early-layer lens distributions are model-specific junk, not a uniform floor.
2. **New decision D10: which checkpoints.** The notebooks use base and larger models (`gemma-3-4b-pt`, Llama-3.1-8B, Qwen3.5-9B), not the registered instruct models.
   - The base checkpoints of the registered sizes, and Llama-3.1-8B, share the frozen tokenizers exactly.
   - **Qwen3.5-9B does not.** Its forms disagree with the frozen counts for 644 of 1,837 vi forms and 1,440 zh forms. It is also a hybrid linear/full-attention model.

   Settle this before v1.4 is tagged (§3.12, §10).
3. **Cloze is now small and descriptive only.** v1.3.1 has 41 diac records (34 main + 7 extension) and 29 nodiac (all main). By data amendment v1.3, cloze is out of every decision rule, Holm family, the §7.1 model and all stratum analyses.
4. **Order of work:**
   1. X0 setup, then X1 (v1.4 + pre-written abstracts + tag).
   2. E0 pilot on the directions split.
   3. E1 model gate.
   4. One primary lens pass per model, which feeds E2–E5, then **Checkpoint H**.
   5. E8, E6 (exploratory), E7.
   6. Robustness, then E9 (optional).

---

## 1. Frozen inputs

| Artifact | Source and pin | Check before use |
|---|---|---|
| Concept table **with concreteness and `in_muse`** (71 cols) | GitHub `tarudesu/viLens-concepts-preparation`, `data/concepts.tsv`. Unchanged from `prereg-v1.2` through `data-v1.3.1` (annotated tag → commit `873def8`). | SHA-256 `6ed1faf4…d4de8`. Line 1 is a `# Pipeline commit` comment, so use `skiprows=1`. |
| Public concept table (69 cols) | HF `tarudesu/viLens-concepts` @ `v1.3.1` (= `main`, `8a2d99c`), `concepts.jsonl`. Byte-identical to yesterday's `main`. | SHA-256 `f3d009bb…7fc7`. **Every field is a string**, so cast explicitly. `cloze_available` is **deprecated**; do not use it. |
| Repetition, translation and directions prompts (25 files) | HF `v1.3.1`; unchanged since `v1.2` | SHA-256 table in the HF card (these match `BUILD_OUTPUTS.sha256`) |
| **Cloze v1.3.1** (2 files) | HF `v1.3.1`; GitHub `data/prompts/v1.3/` | `cloze_diac_set1` `8da79d5e…bca1`; `cloze_nodiac_set1` `322401ef…c4a8`. The v1.2 cloze files (HF tag `v1.2`) are superseded and not used. |
| `directions_flores.jsonl` (FLORES+ dev, 997 ids × 5 langs; 199 validate) | Not distributed (gated). Rebuild with `scripts/build_flores_directions.py` from FLORES+ @ `5fec6c1`. | SHA-256 `e768392c…ea25e` |
| Registered models | `google/gemma-3-4b-it` @ `093f9f3`, `Qwen/Qwen3-4B` @ `1cfa9a7`, `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4` | Commit hash only. If D10 changes the checkpoints, pin the new hashes in v1.4. |

**Rule:** the experiment repo never modifies these files. Anything derived is a new file in `viLens/derived/`, committed before it is used. Examples: distractor manifests, cognate flags, no-space Chinese targets, vi_nodiac readout targets.

---

## 2. What the data (and the pilots) imply

### 2.1 Item sets (main = main test set; ext = M1-only extension)

| File | Records | Main | Ext | Notes |
|---|---:|---:|---:|---|
| `repetition_diac_set{1..6}` | 1,658 each | 1,535 | 123 | Set 1 is primary |
| `translation_diac_set{1..6}` | 1,647 each | 1,524 | 123 | 11 main concepts have no Russian form |
| `cloze_diac_set1` (**v1.3.1**) | **41** | **34** | 7 | Main strata: sino 17, nonsino 10, ambiguous 7. **Descriptive only** (data amendment v1.3) |
| `repetition_nodiac_set{1..6}` | 1,236 each | 1,231 | 5 | 304 main concepts are collapsed and excluded |
| `translation_nodiac_set{1..6}` | 1,226 each | 1,221 | 5 | |
| `cloze_nodiac_set1` (**v1.3.1**) | **29** | 29 | 0 | Main strata: sino 15, nonsino 8, ambiguous 6 |
| `directions_matched` | 1,320 | — | — | 132 concepts × 5 langs × {repetition, translation}; fit 106 / validate 26. **No cloze-format directions.** |

- The extension is used **only** in E2/E3. Everything else filters to `m1_extension == False`.
- **Cloze v1.3.1 schema:**
  - It adds `demo_concept_ids`, and `fewshot_set` is `null`.
  - All 41 prompts use the same three demonstrations:
    - `bạo lực` (few-shot set 1);
    - `ướt` (few-shot set 5);
    - `năng lực` (`0a52c03dec8c`, **directions fit split**).
  - The card warns that some contexts admit more than one Vietnamese answer; only the target sense is verified.

### 2.2 Main test composition (n = 1,535)

- **POS:** noun 1,280 (83.4%), verb 142, adjective 113.
- **Syllables:** 1 → 207; 2 → 1,059; 3 → 190; 4 → 74; 5–6 → 5.
- **Strata:** sino 481, nonsino 706, ambiguous 347 (excluded from H3), other_loan 1.
- **Other flags:** `native_strict` 51 (48 nonsino); `vi_alts` non-empty 420.
- **Robustness subsets (v1.2):** `bt_pass_strict_v1` 1,111; `external_attested` 1,086; all three NLLB agree 964; `min_surface_dist ≥ 0.60` 1,485.
- **Known issues in the v1.3.1 card that affect analysis:**
  - Six main-test targets are flagged as questionable but kept as frozen data: `chết đuối`, `rưỡi`, `tủy xương`, `hội nghị thượng đỉnh`, `cảnh sát chính tả`, and `đừng` (whose fr target is the template "ne... verb ...pas"). Four of the six are in the cognate-free subset.
  - "Imprecise targets" are estimated at a few percent. Both issues feed sensitivity R11.
  - `vi_nodiac` is lowercased for 2 concepts; prompts preserve case. Answer matching casefolds, so this is harmless.
- **Discrepancy to fix in the card:** the card lists 3 collapsed nodiac demonstrations (thứ tư, tiệc, sắt). The concept table flags **6 of 30** as collapsed:

  | Set | Collapsed demonstrations |
  |---|---|
  | 1 | thứ tư |
  | 3 | nhóm |
  | 4 | khí đốt |
  | 5 | vàng |
  | 6 | tiệc, sắt |

  So the primary nodiac set 1 contains the ambiguous stripped demonstration `thu tu`.

### 2.3 Registered models, depth and prompt length

| | Gemma 3 4B-it | Qwen3-4B | Llama 3.2 3B-Inst |
|---|---|---|---|
| Blocks N / d_model / vocab | 34 / 2560 / 262,208 | 36 / 2560 / 151,936 | 28 / 3072 / 128,256 |
| BOS added by tokenizer | `<bos>` | none | `<|begin_of_text|>` |
| `ℓ_early` (ℓ/N ≤ 0.15) | 1–5 | 1–5 | 1–4 |
| Logit softcapping (config) | null | null | null |

The same-size base checkpoints have identical depth, width, vocabulary and tokenizer (§3.12).

- **Prompt length:** the longest prompt plus target is 121 tokens (Gemma). Gemma's 1,024-token sliding window never binds, so local layers see the whole prompt.
- **Prefix property:** prompt tokens are always a prefix of the prompt+target tokenization. There are 0 violations in 212,112 checks per tokenizer, plus 420 more for the v1.3.1 cloze files (Qwen3.5 included).

### 2.4 Target tokenization in context (main test, repetition set 1)

Cells show content tokens (not whitespace only) / continuation share.

| Model | vi | en | zh (released) | fr | id | vi nodiac |
|---|---|---|---|---|---|---|
| Gemma | 2.40 / 58% | 1.21 / 17% | 1.81 / 45% | 1.93 / 48% | 2.12 / 53% | 2.93 / 66% |
| Qwen3 | 2.58 / 61% | 1.33 / 25% | 3.27 / 69% | 2.48 / 60% | 2.96 / 66% | 3.45 / 71% |
| Llama | 2.51 / 60% | 1.33 / 25% | 2.81 / 64% | 2.51 / 60% | 2.91 / 66% | 3.19 / 69% |

**Chinese targets with the released leading space:**

| | Gemma | Qwen3 | Llama | *Qwen3.5 (notebook)* |
|---|---|---|---|---|
| First token is a bare space | **58.8%** | 12.5% | 5.4% | ***99.9%*** |
| Contains UTF-8 byte fragments, with space | 0.2% | **80.1%** | **53.4%** | *0.6%* |
| Same, without the space | 0.2% | 0.1% | 18.8% | *0.6%* |
| Mean tokens, with → without space | 2.40 → 1.67 | 3.40 → 1.46 | 2.87 → 2.23 | — |

Example (Qwen3): `' 千年'` → `[' �','�','年']`, but `'千年'` → `['千年']`.

### 2.5 M1 (single-token) sets

- **M1-eligible:** Gemma 186 (108 main + 78 ext), Qwen 166 (96 + 70), Llama 162 (96 + 66). Every single-token item is monosyllabic.
- **Usable first-token readouts** per language, after excluding whitespace, byte-fragment and cross-language-collision first tokens:

| Model | en | fr | id | zh released | zh no-space |
|---|---:|---:|---:|---:|---:|
| Gemma | 161 | 165 | 181 | **66** | 186 |
| Qwen3 | 141 | 144 | 160 | **15** | 166 |
| Llama | 139 | 141 | 158 | **61** | 136 |

### 2.6 Cognates between English and the control languages

Distances use the filter-4 normalization.

- **en–fr:** 746 of 1,535 main concepts are below 0.45; **240 are identical strings**.
- **en–id:** 416 below 0.45; 64 identical.
- **First content token identical to English:** fr 213 / 259 / 259 and id 78 / 109 / 109 (Gemma / Qwen3 / Llama).
- **Cognate-free five-way subset** (en–fr ≥ 0.45, en–id ≥ 0.45, five distinct first content tokens): **729 concepts, the same in every model.**
  - Translation 721.
  - Cloze v1.3.1: 14 of 34 (nodiac 11 of 29).
  - Strata: nonsino 454, sino 146, ambiguous 129.
  - Nouns 78.6%.

### 2.7 H3 confounds (sino vs nonsino, main test)

| | sino (481) | nonsino (706) |
|---|---:|---:|
| mean `zipf_vi` | 5.33 | 4.86 |
| mean `zipf_zh` | 4.42 | 3.78 |
| mean concreteness | 3.22 | 4.02 |
| mean syllables | 1.98 | 2.19 |
| 3+ syllables | 33 | 178 |
| concreteness NA | 12 | 18 |

- **Complete cases:** sino 469, nonsino 688.
- **Registered imbalance:** d = +0.89, +0.83, −0.83, −0.62.
- **H3 status:** exploratory (κ lower bound 0.557). Holm MDE d = 0.206.
- **`zipf_vi` fails a validity check:**
  - Mean by syllable count is 4.85 (1), 5.12 (2), 5.02 (3), 5.06 (4).
  - `số học` scores 6.17 and `đại từ` 6.10.
  - Spearman correlation with `zipf_en` is 0.30.
  - The v1.3.1 card now states that wordfreq "tracks syllable rather than word frequency" for multi-syllable forms.

### 2.8 Directions data and few-shot sets

- **Directions split** (132 concepts): nonsino 57, sino 40, ambiguous 34, other_loan 1.
  - Prompts carry explicit language labels, so the 95% gate is near-trivial (A6).
  - One fit concept (`năng lực`) is also a cloze v1.3.1 demonstration. This is harmless: cloze prompts are never used to fit directions, and no test concept is involved.
- **Few-shot set strata (sino / nonsino / ambiguous):**

| Set | sino / nonsino / ambiguous |
|---|---|
| 1 (primary) | 2 / 3 / 0 |
| 2 | 3 / 1 / 1 |
| 3 | 1 / 1 / 3 |
| 4 | 2 / 2 / 1 |
| 5 | 1 / 2 / 2 |
| 6 | 0 / 3 / 2 |

### 2.9 What the three pilot notebooks show (qualitative, one prompt each)

All three ran on a Colab GPU in bf16, with `pip install -U`, i.e. no pinned versions.

| Notebook | Checkpoint | Lens implementation | Prompt |
|---|---|---|---|
| `Gemma3_4b` | `google/gemma-3-4b-pt` (base) | TransformerLens `TransformerBridge`, `blocks.ℓ.hook_resid_post` → `ln_final` → `unembed` (+ soft-cap if set) | English→Vietnamese; quoted demos `book→sách`, `water→nước`; test `hospital` |
| `Llama3.1_8b` | `meta-llama/Llama-3.1-8B` (base; 32 blocks, d = 4096) | HF forward hooks on each decoder layer → `model.norm` → `lm_head` | Chinese→Vietnamese; `书→sách`, `水→nước`; test `学校` |
| `Qwen3.5_9b` | `Qwen/Qwen3.5-9B` (32 blocks, d = 4096, vocab 248,320) | Same HF hooks (`Qwen3_5ForCausalLM`) | Same as Llama |

**Findings and their consequences for the plan:**

1. **Final-block lens = model output, in both code paths.**
   - Gemma: identical top-1 probabilities at all 5 steps (e.g. 0.449531 / 0.449531).
   - Llama: 0.371382 vs 0.371383.
   - Qwen3.5: 0.123051 vs 0.123064.
   - Block-output hooks + final norm + unembedding are therefore implemented correctly. This is pilot support for E1 check 2, though not yet on the registered checkpoints.
2. **Trailing space plus a quote mismatch derails the first step.** Every notebook prompt ends in `Vietnamese: ` (trailing space) after quoted demonstrations, so the first generated token is not the answer:
   - Gemma emits an HTML tag token (P = 0.28, output `<i>bệnh viện</i>`);
   - Llama emits `'\xa0'` (0.37);
   - Qwen3.5 emits `' "'` (0.12).

   The released templates end at `:` and are unquoted, which avoids this. **X0 asserts that no prompt ends in whitespace** and that demonstrations and test line share one template.
3. **Source-language confound.** In the Gemma and Qwen3.5 runs, the language that dominates the middle layers is the prompt's *source* language:
   - Gemma: `' hospital'`, layers 25–32 (English source);
   - Qwen3.5: `学校`, layers 24–30 (Chinese source).

   These cannot be separated from copying. Llama is the exception: `' school'` is top-1 at layers 22–30 although the source is Chinese. This supports the protocol's design (vi-only repetition, ru-source translation) and the ru readout diagnostic. **None of this is evidence for H1.**
4. **The decision is late.** Vietnamese first becomes top-1 at:
   - Llama: layers 31–32 of 32;
   - Qwen3.5: layers 28–31 of 32;
   - Gemma: layers 27–30 of 34.

   That is normalized depth ≈ 0.8–0.97, where one block ≈ 3% of depth. Consequences:
   - `L_cross` is expected in the top ~20% of depth.
   - The ±0.10 coincidence window (A7) is about ±3 blocks.
   - The E7 band can be very narrow, so it gets a pre-declared minimum width (§5, E7).
5. **The latent concept reappears at continuation positions.** Llama's `' school'` is top-1 at layers 22–30 when predicting `tr` and again (mixed with `'uong'`) before `ường` and `' học'`. Gemma's `' hospital'` leads at layers 22–26 before `' viện'`. This motivates a new **along-path readout** (§3 A4b, R10): teacher-force the Vietnamese answer and read the other languages' first content tokens at every Vietnamese position. No foreign prefix is forced.
6. **An undiacritized intermediate appears.** Before emitting `ường`, Llama's top-1 is `'uong'` at most layers from 18 to 31, and Qwen3.5 shows `'uong'` at layers 30–31. This motivates a new `vi_nodiac` **diagnostic readout in the diac condition** (A12), which is cheap and relevant to E9.
7. **The early-layer lens is not a uniform floor.**
   - Gemma layers 1–4 put 0.94–0.99999 on one junk token (entropy 0.0–0.3 nats).
   - Llama layers 1–20 decode junk tokens (entropy ≈ 5–10 nats).
   - Qwen3.5 layers 1–18 are near-uniform (≈ 11.2 nats, against ln V ≈ 12.4).

   A raw `ℓ_early` baseline is therefore model-specific junk. This supports A1 (distractor contrast), and A2's wording is corrected.
8. **Qwen3.5 decodes the space-free `学校`**, and at layers 19–23 it decodes Chinese task words (`翻译` "translate", `是什么` "what is", `哪里` "where"). This supports A3 directly, and it supports treating Chinese as a candidate pivot for the Qwen family. It is n = 1, not evidence.
9. **Environment:** the notebooks used unpinned `pip install -U`, so versions must be pinned (A9). All study runs happen on the server (`AGENTS.md`).

---

## 3. Amendment v1.4 — resolve before any study-model run

Each item is marked **changes frozen text** (version bump) or **specification**. §10 lists the ones that need your decision. A1–A11 are carried over from draft 1 with corrections marked ✎. A4b, A12 and §3.12 are new.

### A1 — `rise` normalization discards amplitude `[changes frozen §5.2 step 2]` — critical

**Problem.** Per-language z-standardization across layers is invariant to any positive affine transform of the curve, so amplitude disappears. Synthetic check (34 layers, `ℓ_early` = 15%, frozen `rise`):

| Synthetic curve | max `rise` |
|---|---:|
| Strong English pivot (+8 nats) | 2.24 |
| French, same shape at 1/20 amplitude | 2.24 (ratio **1.00**) |
| French drifting 0.05 nats + concept-averaged noise (n = 1,535) | **2.68** |
| Per concept: pure noise vs clean step | 2.14 vs 2.03 |

**Consequence.** The H1 "≥ 2×" condition can fail under a strong pivot, and the H2 "1.5× band" is met by flat curves. Notebook finding 7 adds a second reason: early-layer lens values are model-specific junk.

**Proposal.** Use a **matched-distractor contrast** and drop step 2:

- `s(c,L,ℓ)`: mean content-token log-probability of c's L form, teacher-forced after c's prompt.
- `d(c,L)`: one distractor. It is another main-test concept with the same content-token count in L for this model, a different first content token, and the nearest `zipf_en`. It is seeded and written to `derived/` before any run, and teacher-forced after **c's** prompt.
- `Δ(c,L,ℓ) = s(c,L,ℓ) − s(d|c,L,ℓ)`.
- `rise_L(ℓ) = mean_c Δ(c,L,ℓ) − mean_c mean_{ℓ'∈ℓ_early} Δ(c,L,ℓ')`.
- The §2.1 rules apply unchanged to `max_ℓ rise_L`, now in nats.
- The v1.1 z-statistic is kept in the appendix.

### A2 — `L_concept` anchored to a suppressed quantity `[changes frozen §5.6]` — critical ✎

**Problem.** "10% of *its own* final-layer probability" is a trivially low bar for languages the model is not outputting. Their final-layer probability is suppressed, and early-layer lens values can be far below it:

- in Gemma, one junk token holds ~0.99 of the mass at layers 1–4, so every other token is far below 1/V;
- the early-layer regime differs by model (§2.9, finding 7).

`L_concept` would then track a model-specific floor, and pivot width would inflate.

**Proposal.** `L_concept` = the first ℓ where, for any readout language L, `p_L(ℓ) ≥ 0.10 · p_vi(N)`.

- `p = exp(s)`.
- The anchor is the model's actual answer; curves are concept means.
- The 5% and 20% sensitivities are kept.

### A3 — Chinese target form `[specification]` — critical

**Problem.** See §2.4 and §2.5. With the leading space:

- only 15 of 166 Qwen3 M1 items have a usable zh first token;
- 59% of Gemma zh targets start with a bare space, and 99.9% of Qwen3.5 targets do.

The lens itself decodes the space-free word (Qwen3.5 `学校`, §2.9 finding 8).

**Proposal.**

- The **primary zh target is the released form without the leading space**, teacher-forced directly after the final `:`. Wendler et al. scored Chinese after `中文: "`, also without a space.
- The with-space form becomes a sensitivity.
- Latin-script targets keep their space.

### A4 — Content-token scoring and decomposition `[changes frozen §5.2 "mean over i"]` ✎

- Whitespace-only target tokens are teacher-forced but excluded from the mean. After A3 this leaves only a handful: vi Gemma 2 / Qwen3 14 / Llama 5; fr 1 each in Qwen3 and Llama.
- Pre-declared decompositions:
  - `s¹`: first content token only;
  - `s⁺`: continuation tokens only;
  - the sequence sum.
- **Why:** continuations are 58–61% of scored vi tokens but only 17–25% for en. A continuation token of a *foreign* target is conditioned on a forced foreign prefix. If the verdict or `L_cross` differs between `s` and `s¹`, report both.

### A4b — Along-path readout (new) `[specification; diagnostic]`

- Teacher-force the **Vietnamese** answer.
- At each Vietnamese target position i and each layer ℓ, record log P_ℓ of the first content token of each L form and of its A1 distractor (L ∈ {en, zh-nospace, fr, id}).
- `Δ^path_L(ℓ)` = mean over positions and concepts of (correct − distractor).
- This reads the latent along the model's own output path and forces no foreign prefix (§2.9, finding 5).
- It needs no extra forward passes; the lens outputs from the vi sequence are enough.
- It is reported next to E4 as R10. It is **not** a decision-rule input.

### A5 — Item set for the five-way rules `[specification]`

- The H1/H2 `rise` rules use the **cognate-free 729** (§2.6), mirroring Wendler et al.'s prefix-sharing exclusion.
- All-items results are a sensitivity (R9).
- H3 uses the full strata.

### A6 — LA (M3) specification `[specification]`

- **Position and activation:** last prompt token (the final `:`); block output `h_ℓ`, fp32.
- **Centering:** subtract the mean of the five fit-split class means.
- **`c_random`:** ≈ 0 by construction; report it anyway. Also note that the exchangeable-geometry null for class offsets is −1/(K−1) = −0.25.
- **H2 "for all L"** means L ∈ {en, zh, fr, id}.
- **Supporting statistic:** `D_en = LA_en − mean(LA_fr, LA_id)`, with a bootstrap CI.
- **Format matching:** repetition ↔ repetition and translation ↔ translation. Cloze (descriptive) uses repetition directions.
- **Frozen gate:** at least 95% of the 130 validate prompts per layer and format (at most 6 errors).
- **Validity diagnostics:** FLORES transfer (199 × 5 validate sentences) and cross-format transfer. Layers below 95% on FLORES are flagged "label-dominated".

### A7 — Steering (E8) specification `[specification; replaces the perplexity guard]` ✎

- **Items:** main-test translation prompts, set 1; up to 300 concepts per model whose unsteered greedy answer is vi or a `vi_alts` form; seeded and stratum-proportional.
- **Vector:** `u_ℓ = μ_en^ℓ − μ_vi^ℓ` (translation-direction fit split). This equals the protocol's `α·(v_vi − v_en)` with α < 0.
- **Injection:** add `k·u_ℓ` at block ℓ's output, at the last prompt position and at generated positions. One layer at a time, k ∈ {0.5, **1**, 2}.
- **Control:** a random direction of the same norm.
- **Generation:** greedy, stopping at `\n`, `" -"` or 8 tokens.
- **Outcome classes:** VI / **EN (clean flip)** / OTHER-LANG / DAMAGE.
- **Guard:** DAMAGE rate is the guard. GlotLID is descriptive only.
- **Continuous companion:** `log P(en) − log P(vi)` at the output.
- **Coincidence:** `|depth(flip peak) − depth(L_cross)| ≤ 0.10` **and** the bootstrap CI of the difference includes 0. ✎ Given §2.9 finding 4, the flip peak is expected near the top of the network. Report flip curves in **blocks as well as depth** so a 1-block difference is visible.

### A8 — H3 frequency covariate `[changes the v1.2 correction]` — D5

- **Recommended:** `zipf_en` becomes the primary covariate, `zipf_vi` a sensitivity, and the `zipf_zh` sensitivity is kept. Matching uses the primary variable.
- **Alternative:** keep `zipf_vi` primary (H3 is exploratory anyway). The card now concedes `zipf_vi` tracks syllable frequency.

### A9 — Inference set-up `[specification]` ✎

- Raw completion, **no chat template**. `enable_thinking` does not apply.
- Tokenizer-default special tokens.
- Joint tokenization of prompt+target; X0 asserts no prompt ends in whitespace (§2.9, finding 2).
- bf16 weights; fp32 lens.
- **One code path for all models: HF forward hooks on block outputs**, as in the Llama and Qwen notebooks. TransformerLens is optional for Gemma. Either path must pass E1 checks 2 and 4. M4 needs the attention and MLP outputs actually added to the residual; for Gemma these are the post-norm outputs.
- Do not use the last element of `output_hidden_states`, which is typically post-norm.
- **Pin library versions** (`transformers`, `torch`, `transformer-lens` if used) in the lockfile. The notebooks used unpinned `pip install -U`.

### A10 — Competence reporting `[specification]`

- Greedy answer for every test prompt; accuracy by model × format.
- Competent-subset sensitivity (R7).
- E8 uses competent items only.
- Vietnamese is not an officially supported Llama 3.2 language.

### A11 — Remaining pins `[specification]`

- **`D_zh`:** mean over ℓ/N ∈ [0.25, 0.75] of `LA_zh`(sino) − `LA_zh`(nonsino), with stratified bootstrap.
- **Holm family (b)** = 3 models × 2 instruments = 6 (D9). **Family (a)** = 15 one-sided tests.
- **§7.1 model:** natural spline, 5 df; `n_tokens` = the readout target's content tokens; complete cases; if the `task_format` random effect is singular, refit it as fixed; lme4.
- **Cloze** is excluded from §7.1, as fixed by data amendment v1.3.
- **Landmarks:** concept-mean curves, 1,000 bootstrap resamples, seed `20260925`.

### A12 — `vi_nodiac` diagnostic readout (new) `[specification; diagnostic]`

- In the **diac** condition, also score the concept's `vi_nodiac` form (+ its distractor's nodiac form).
- **Question:** does the latent Vietnamese pass through an undiacritized form before diacritics are realised late (§2.9, finding 6)?
- Descriptive; it feeds the interpretation of E9.

### 3.12 — Model set (new) `[changes protocol §3]` — **D10**

| Checkpoint | Tokenizer = frozen? (ntok mismatches / 1,837 per language) | N / d / vocab | Notes |
|---|---|---|---|
| `gemma-3-4b-it` (registered) | ✓ | 34 / 2560 / 262,208 | Instruct |
| `gemma-3-4b-pt` (notebook) | **✓ 0 / 0 / 0 / 0 / 0** | 34 / 2560 / 262,208 | Base; same multimodal wrapper |
| `Qwen3-4B` (registered) | ✓ | 36 / 2560 / 151,936 | Hybrid-thinking post-trained |
| `Qwen3-4B-Base` | **✓ 0 / 0 / 0 / 0 / 0** | 36 / 2560 / 151,936 | Base |
| `Qwen3.5-9B` (notebook) | **✗ vi 644, en 12, zh 1,440, fr 667, id 1,109** | 32 / 4096 / 248,320 | Hybrid: **24 linear-attention + 8 full-attention** layers (full at 3, 7, …, 31); untied embeddings; multimodal wrapper. Main single-token vi would be 174 (frozen Qwen: 96). |
| `Llama-3.2-3B-Instruct` (registered) | ✓ | 28 / 3072 / 128,256 | Instruct |
| `Llama-3.2-3B` | **✓ 0 / 0 / 0 / 0 / 0** | 28 / 3072 / 128,256 | Base |
| `Llama-3.1-8B` (notebook) | **✓ 0 / 0 / 0 / 0 / 0** | 32 / 4096 / 128,256 | Base; untied; 8B bf16 ≈ 16 GB of weights |

**Implications.**

- **Base checkpoints of the registered sizes** change no frozen column: `ntok_*`, `single_token_*`, `m1_eligible_*`, the M1 sizes (186 / 166 / 162) and every §2 number stay valid. Base models also fit raw few-shot completion prompts, which is the reference paper's setting and the notebooks' own stated rationale.
- **Llama-3.1-8B** is tokenizer-compatible but breaks the 3–4B size match. It is better as an optional scale check; the protocol already allows one.
- **Qwen3.5-9B** would require all of the following:
  - re-deriving every Qwen tokenizer column as a documented derived file;
  - new M1 sets and a new E1 audit;
  - new distractor matching;
  - head-level M4 would be defined only on its 8 full-attention layers.

  That is a new study arm, not a drop-in.

**Recommendation:** use the base checkpoints of the registered sizes (`gemma-3-4b-pt`, `Qwen3-4B-Base`, `Llama-3.2-3B`) as the primary models. Keep the registered instruct models as a secondary replication if time allows. Keep Qwen3.5-9B out of the main study. Whatever you choose goes into v1.4 with pinned commit hashes before any run.

### How to commit v1.4 (task X1)

1. Keep one amendment log: append **v1.4 (analysis; before any study-model run)** to the preparation repo's `PROTOCOL_AMENDMENTS.md`, after the v1.3 cloze entry. Include the chosen A-items and D10, with one-line reasons.
2. Regenerate `PREREG.md` (§2, §5.2, §5.6, §7 + v1.2 + v1.3 + v1.4). Write `abstracts_prewritten.md` with H1-supported, H2-supported and inconclusive versions.
3. Commit `derived/` in the experiment repo:
   - distractors;
   - no-space zh targets;
   - `vi_nodiac` targets;
   - cognate flags;
   - the E8 sampling rule;
   - seeds.
4. Tag `prereg-v1.4` in the preparation repo, and record that tag in the experiment repo's config.
5. **No test prompt reaches a study model before that tag.**

---

## 4. Execution design

### 4.1 Repository and conventions

Carry over `AGENTS.md`:

- config-driven, deterministic, cached;
- one commit per step;
- REPORT BACK;
- stop on anything unexpected;
- GPU work runs as Slurm jobs on the server; the login node does only light CPU work (`AGENTS.md` §4).

```
viLens/
├── configs/experiments.yaml   # data tags, model revisions, seed, ℓ_early, L_concept fracs, k grid, bands
├── derived/                   # frozen pre-run manifests
├── src/  audit.py  lens.py  directions.py  attribute.py  steer.py  score.py
├── results/<pass>/<model>/…parquet (+ float16 residual caches)
├── analysis/                  # §7 notebooks / lme4 scripts
└── abstracts_prewritten.md
```

### 4.2 Three passes

| Pass | What runs | Stores | Feeds |
|---|---|---|---|
| **P-D** directions | 1,320 matched prompts; FLORES 997 × 5 | `h_ℓ` at the last token (mean-pooled too for FLORES) | E5, E7, E8 |
| **P-L** lens | Each test prompt × {vi, en, zh-nospace, zh-space, fr, id, vi-nodiac} + 6 distractors (vi, en, zh-nospace, fr, id, vi-nodiac), + ru on translation | Per-layer log-probs of every target token; along-path logits for A4b; top-20 tokens at the answer position; `h_ℓ` and attention/MLP outputs at the last prompt position; greedy answer | E2–E7, E9, A10, R10 |
| **P-S** steering | E8 grid | Answer, class, continuous Δ | E8 |

Counts for the registered or base-equivalent models (different if D10 picks other sizes):

| Run | Per model | Total (3 models) |
|---|---:|---:|
| P-L primary (diac, set 1: rep 1,658 × 13 + tr 1,647 × 14 + cz 41 × 13) | 45,145 sequences; 3,346 prompts; 3,346 greedy answers | 135,435 |
| P-L few-shot sensitivity R1 (sets 2–6; rep 729 + tr 721; 10 targets each) | 72,500 | 217,500 |
| P-L nodiac E9 (rep 1,236 × 10 + tr 1,226 × 11 + cz 29 × 10) | 26,136 | 78,408 |
| P-D | 1,320 + 4,985 | 18,915 |
| P-S (≤ 300 × N × 4 settings) | 40,800 / 43,200 / 33,600 generations | 117,600 generations (+ as many teacher-forced passes) |

Residual storage is about 0.6 GB per model at the last position, or about 1.8 GB with attention and MLP outputs.

**Throughput:** E0 measures sequences per second and peak GPU memory on the server GPU, and projects GPU-hours for the table above.

---

## 5. Experiments, in execution order

### X0 — Setup and data verification (no weights)

- **Pin and verify:**
  - HF `v1.3.1` hashes;
  - `data/concepts.tsv` hash;
  - rebuild and verify `directions_flores.jsonl`.
- **Load with explicit casting.** Ignore `cloze_available`.
- **Build `derived/`:** A1 distractors, A3 zh no-space, A12 nodiac targets, cognate flags / 729 subset, E8 sampling rule, seeds.
- **Assert:**
  - tokenizer counts reproduce `ntok_*` for the chosen checkpoints (§3.12);
  - 0 prefix violations;
  - **no prompt ends in whitespace**;
  - item counts per file (§2.1);
  - no few-shot or directions concept is a test item.
- **REPORT BACK:** hashes; regenerated §2.4–2.6 tables; distractor match quality; the collapsed-demonstration list (§2.2).

### X1 — v1.4, abstracts, tag

As in §3. **Gate:** no test prompt reaches a study model before `prereg-v1.4`.

### E0 — Pipeline pilot (directions split; discard results)

- **Items:** 50 seeded directions-split concepts; distractors from other directions concepts.
- **Run:**
  - on the smallest chosen model: P-D on all 132 directions concepts, then P-L on the 50;
  - a 10-concept smoke test on the other two models;
  - throughput and peak memory on the server GPU.
- **Checks:** schemas, hooks, memory, resumability, byte-identical reruns. No curve interpretation.
- **REPORT BACK:** throughput and peak GPU memory per model, rerun reproducibility (max |Δ log-prob|), and projected GPU-hours for §4.2.

### E1 — Model-side audit `[GATE]`

The tokenizer half is done for the frozen tokenizers:

- single-token vi coverage is 7.04% / 6.25% / 6.25%, so C1 stands;
- M1 adequacy is met via the extension.

If D10 picks a non-matching tokenizer, redo this half first.

**On directions-split prompts only:**

1. Load at the pinned revisions and confirm §2.3.
2. **Lens validity:** max |Δ log-prob| < 1e-3 at block N on 200 prompts, with 100% top-1 agreement. The notebooks passed the equivalent check (§2.9, finding 1).
3. Gemma text-tower hook path.
4. **Residual decomposition:** `‖h_ℓ − (h_{ℓ−1} + a_ℓ + m_ℓ)‖/‖h_ℓ‖ < 1e-4`.
5. bf16 vs fp32 on 50 prompts.
6. Competence preview (132 × {rep, tr}).

**Gate:** if check 2 fails, fix it (or use a tuned lens, or drop the model). If check 6 is below 20% on translation, stop for a checkpoint discussion.

### P-L primary → E2, E3, E4 (+ the A10 table)

#### E2 — M1 replication

- **Items:** M1-eligible items (§2.5, zh no-space).
- **Measure:** Wendler-style single-token probability. Before implementing, check whether `epfl-dlab/llm-latent-language` sums over prefix-token variants, and match it.
- **Cloze:** the M1 n is tiny (≤ 7 extension + a few main); report it descriptively only.

#### E3 — M2 landmarks and C1

- Landmarks per model × format (A1/A2), with bootstrap CIs.
- **C1 instrument effect:** M1 − M2 on the same items, main-only as primary (108 / 96 / 96).
- **C1 selection effect:** M2 on the M1 subset vs M2 on the full main set. Monosyllables are 13.5% of main.

#### E4 — Five-way readout `[CENTERPIECE]`

- **Primary:** `rise_L` (A1) on the cognate-free 729, per model × format (repetition, translation). Apply the H1/H2 `rise` rules.
- **Shown alongside:**
  - all items;
  - `s¹` / `s⁺`;
  - zh with space;
  - the v1.1 z-statistic;
  - the ru readout (translation);
  - **along-path readout (A4b)**;
  - the `vi_nodiac` readout (A12).
- **Cloze (34 main, 14 of them cognate-free):** per-model curves with bootstrap CIs, descriptive only.

### P-D + E5 — Direction analysis

- Fit `v_L` on 106 concepts per format. Run the gate and the A6 diagnostics.
- Compute `LA_L` for vi test prompts (L ∈ {en, zh, fr, id}) and `D_en`, and apply the LA conditions.
- E4 and E5 must agree; if they don't, use the frozen §10.1 framing.

#### ▶ Checkpoint H — adjudication

Write the per-model verdicts into `results/adjudication.md` **before** analysing E6–E8. Compare them with the pre-written abstracts.

### E8 — Steering sweep

As in A7. Report:

- clean flip / DAMAGE / OTHER-LANG vs block and depth, for each k, plus the random control;
- the continuous Δ;
- the coincidence with `L_cross` (translation);
- repetition as a secondary (continuous only).

If there is no `L_cross`, report the flip peak alone.

### E6 — Etymology (H3, exploratory)

- **Items:** sino 481 vs nonsino 706 (complete cases 469 / 688). Cloze is excluded (data amendment v1.3).
- **Instruments:**
  - `D_zh` (A11);
  - `rise_zh` / `rise_en` through §7.1 (A8 covariate);
  - a matched subsample (caliper 0.2 SD, four covariates).
- **Sensitivities:** `zipf_zh`; `native_strict` (51; Holm MDE d ≈ 0.51); v1.2 subsets; R11.
- Every result is labelled exploratory. If the adjusted and matched estimates disagree, H3 is "unresolved".

### E7 — Component attribution ✎

- **Band:** [`L_peak`, `L_cross`], with a **pre-declared minimum of 3 blocks ending at `L_cross`** (§2.9, finding 4). If there is no `L_cross`, use the top-quartile slope band of `Δ_vi`.
- **Per layer:** project attention and MLP updates onto `v_vi` and `(v_vi − v_en)`, and compute the attention/MLP share.
- **Heads:** head-level ranking, with Gemma local and global layers shown separately. If D10 ever admits a hybrid model, report heads only on its full-attention layers.
- **Optional:** mean-ablate the top-10 heads and re-measure on the E8 sample.

### R — Robustness battery (pre-declared)

| # | Check | Items |
|---|---|---|
| R1 | Few-shot sets 2–6 (fallback: sets 2–3) | Cognate-free 729 / 721 |
| R2 | FLORES vs matched directions | E5 |
| R3 | `L_concept` at 5 / 10 / 20% | E3 |
| R4 | Raw `s` vs `Δ` | E4 |
| R5 | Tuned lens on Llama with a language-balanced corpus (D8) | E4 |
| R6 | v1.2 subsets (`bt_pass_strict_v1`, `external_attested`, all-NLLB-agree, `min_surface_dist ≥ 0.60`) | E4, E6 |
| R7 | Competent subset | E3–E5 |
| R8 | `s¹` / `s⁺` / sum; zh with space | E3, E4 |
| R9 | All-items five-way | E4 |
| **R10** | **Along-path readout (A4b)** | E4 |
| **R11** | **Exclude the 6 card-flagged questionable targets** | E4, E6 |

### E9 — Diacritic-stripped input (optional)

- **Items:** non-collapsed main concepts: repetition 1,231, translation 1,221, cloze 29 (descriptive).
- **Comparison:** paired against diac on the same concepts. Distractors are re-matched, because nodiac vi targets are longer: Gemma 2.93 vs 2.40, Qwen3 3.45 vs 2.58, Llama 3.19 vs 2.51 content tokens.
- **Caveat:** in nodiac set 1 the demonstration `thu tu` is itself a collapsed form (§2.2). Report this, and run set 6 as a check only if R1 capacity allows. Set 6 contains two collapsed demonstrations, so it is not a clean contrast either.
- **Interpretation:** read E9 alongside A12. If the latent passes through `vi_nodiac` in the diac condition, stripping diacritics may change little internally.

### E10 — Dropped

---

## 6. Analysis rules

- **Bootstrap:** over concepts, 1,000 resamples, 95% percentile CIs, stratified for E6.
- **Holm families:** as in A11.
- **Formats:** reported separately.
- **Cloze:** descriptive only, per model with bootstrap CIs. It is excluded from decision rules, Holm families, §7.1 and stratum analyses (data amendment v1.3).
- **Raw scores:** appendix only.
- **Extension:** E2/E3 only.
- **Traceability:** every departure from v1.2/v1.3 cites its v1.4 item. The paper states that v1.4 predates any study-model run, and that the only model outputs seen beforehand were the three exploratory notebooks (different checkpoints, no test items).

## 7. Figures and tables

| # | Content | Status |
|---|---|---|
| Fig 1 | Fertility and single-token coverage | Can be made now |
| Tab A | Target tokenization, including the zh space problem (§2.4) | Can be made now |
| Tab B | Item sets, cognate-free subset, cloze v1.3.1 (§2.1, §2.6) | Can be made now |
| Tab C | Checkpoint / tokenizer compatibility (§3.12) | Can be made now; appendix |
| Fig 2 | M1 vs M2 (instrument and selection effects) | E3 |
| Fig 3 | Five-way `rise_L` per model (**the paper**) | E4 |
| Fig 4 | `LA_L(ℓ)` with null band and label-dominated flag | E5 |
| Fig 5 | Sino vs nonsino (exploratory) | E6 |
| Fig 6 | Steering flip rate vs block/depth with `L_cross` and DAMAGE | E8 |
| Fig 7 | Attention vs MLP | E7 |

## 8. Timeline (estimate)

| Week | Dates | Work |
|---|---|---|
| 1 | 28 Sep – 4 Oct | **D10 decision**, X0, X1 (v1.4 + abstracts + tag), E0, E1 |
| 2 | 5 – 11 Oct | P-D, P-L → E2–E5 → **Checkpoint H** |
| 3 | 12 – 18 Oct | E8; E6 |
| 4 | 19 – 25 Oct | E7; R1–R11 (R5 is the long pole) |
| 5 | 26 Oct – 1 Nov | E9 (optional); figures; writing |

## 9. Risks

| Risk | Likelihood | Response |
|---|---|---|
| v1.4 not adopted | — | Verdicts would be uninterpretable (A1) |
| Model set changes after the tag | Low if D10 is settled now | Settle D10 inside v1.4 |
| Qwen3.5 adopted anyway | — | New study arm: re-derive tokenizer columns, M1 and E1; heads only on full-attention layers |
| Weak Vietnamese competence (Llama) | Medium | A10; R7; E1 checkpoint |
| Cognate-free subset nonsino-heavy (146 / 729 sino) | Certain | Fine for H1/H2; H3 uses full strata |
| LA gate passes only on labels | High | A6 diagnostics |
| Steering damages rather than flips | Medium | DAMAGE + random control |
| Flip peak and `L_cross` both at the top, so the coincidence is uninformative | Medium (§2.9, finding 4) | Report in blocks; add the continuous Δ; state the resolution limit |
| H3 confounded | Certain | Exploratory; adjusted + matched; A8 |
| Cloze too small to say anything | Certain (34 main) | Descriptive only (v1.3) |
| One-job-at-a-time queue slows R1/P-S | Medium | Pre-declared R1 fallback (sets 2–3); batch several units into one job |
| E4/E5 disagree | Medium | Frozen §10.1 framing |

## 10. Decisions needed from you

| # | Decision | Recommendation |
|---|---|---|
| D1 | A1 matched-distractor Δ; drop per-language z | **Yes** |
| D2 | A2 `L_concept` anchored to the vi answer | **Yes** |
| D3 | A3 zh without the leading space as primary | **Yes** |
| D4 | A5 H1/H2 rules on the cognate-free 729 | **Yes** |
| D5 | A8 covariate: `zipf_en` primary vs `zipf_vi` | `zipf_en` |
| D6 | A7 steering design | As specified |
| D7 | ~~Compute platform~~ | **Resolved:** lab server, Slurm jobs (`AGENTS.md`); profiles named per task |
| D8 | Tuned-lens model and corpus (R5) | Llama; balanced 5-language sample |
| D9 | Holm family (b) = 3 models × 2 instruments | Confirm |
| **D10** | **Checkpoints:** registered instruct / same-size base / notebook models | **Same-size base** (`gemma-3-4b-pt`, `Qwen3-4B-Base`, `Llama-3.2-3B`); instruct as optional replication; not Qwen3.5-9B |

## 11. Changes from draft 1

- **Data:** re-pinned to HF `v1.3.1` / GitHub `data-v1.3.1`.
  - Cloze v1.3.1 is 41 / 29 records (34 / 29 main) and descriptive only; counts, the compute table and E-sections are updated.
  - `cloze_available` is deprecated.
  - The card's known issues were added (§2.2), including the collapsed-demonstration discrepancy (6 vs 3).
- **Draft 2.1 (same day):** execution moved to the lab server (`/datastore/luannt/Vie-Lens`, Slurm, shared model cache); D7 resolved; M4/Colab references removed; `AGENTS.md` added for the experiment repo.
- **Naming:** the analysis amendment is renamed **v1.4** because "v1.3" is the cloze rebuild; the tag is `prereg-v1.4`.
- **New:** §2.9 notebook findings; A4b along-path readout (R10); A12 `vi_nodiac` readout; §3.12 model set (D10); R11 questionable-target exclusion; E7 minimum band; X0 whitespace assertion; A9 single hook path + version pins.
- **Corrected:** A2's "≈ 1/V floor" claim (Gemma early layers put ~0.99 on one junk token); A7 now reports in blocks.