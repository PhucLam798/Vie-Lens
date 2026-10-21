# E7 component attribution — DRAFT for human decisions

Status: FROZEN by OPS-37 (decisions.md); the inventory below is unchanged

## E1. Governing text and status

`docs/experiments.md:595–600` states:

> ### E7 — Component attribution ✎
>
> - **Band:** [`L_peak`, `L_cross`], with a **pre-declared minimum of 3 blocks ending at `L_cross`** (§2.9, finding 4). If there is no `L_cross`, use the top-quartile slope band of `Δ_vi`.
> - **Per layer:** project attention and MLP updates onto `v_vi` and `(v_vi − v_en)`, and compute the attention/MLP share.
> - **Heads:** head-level ranking, with Gemma local and global layers shown separately. If D10 ever admits a hybrid model, report heads only on its full-attention layers.
> - **Optional:** mean-ablate the top-10 heads and re-measure on the E8 sample.

The supporting §2.9 finding 4 in `docs/experiments.md:235–243` states:

> 4. **The decision is late.** Vietnamese first becomes top-1 at:
>    - Llama: layers 31–32 of 32;
>    - Qwen3.5: layers 28–31 of 32;
>    - Gemma: layers 27–30 of 34.
>
>    That is normalized depth ≈ 0.8–0.97, where one block ≈ 3% of depth. Consequences:
>    - `L_cross` is expected in the top ~20% of depth.
>    - The ±0.10 coincidence window (A7) is about ±3 blocks.
>    - The E7 band can be very narrow, so it gets a pre-declared minimum width (§5, E7).

No raw notebook file was located in the repository; this finding is reproduced in `docs/experiments.md`.

`docs/prereg/PREREG.md:437` and `docs/prereg/v1.4_amendment.md:189` mention E7 only in the Checkpoint H ordering. `docs/decisions.md:128` also mentions E7 as a later analysis in OPS-26; no E7 computation or uncertainty rule is specified there.

## E2. P-L component outputs available

P-L outputs are under `results/pl/`, with basenames of the form `condition=diac__family={family}__fewshot_set=1__format={format}__pass=pl__shard={0..6}.{kind}`, where `kind` is `resid.npy`, `attn.npy`, `mlp.npy`, `resid_index.jsonl`, or the corresponding parquet unit. The arrays contain the registered prompts, including their P-L records; each listed row is an activation row, not a statistic.

| Family | Format | Prompt rows | Residual array shape and dtype | Attention / MLP array shape and dtype |
|---|---|---:|---|---|
| Gemma | repetition | 1,658 | (1,658, 35, 2,560), float32 | each (1,658, 34, 2,560), float32 |
| Gemma | translation | 1,647 | (1,647, 35, 2,560), float32 | each (1,647, 34, 2,560), float32 |
| Qwen | repetition | 1,658 | (1,658, 37, 2,560), float32 | each (1,658, 36, 2,560), float32 |
| Qwen | translation | 1,647 | (1,647, 37, 2,560), float32 | each (1,647, 36, 2,560), float32 |
| Llama | repetition | 1,658 | (1,658, 29, 3,072), float32 | each (1,658, 28, 3,072), float32 |
| Llama | translation | 1,647 | (1,647, 29, 3,072), float32 | each (1,647, 28, 3,072), float32 |

The `.resid.npy` layer axis includes embedding layer 0; `.attn.npy` and `.mlp.npy` have one entry per transformer block. `src/vilens/hooks.py:25–33, 84–95` captures selected decoder positions and attention/MLP output-module activations; `src/vilens/passes.py:1356–1364` selects the last prompt token and stores fp32 arrays. The saved attention and MLP outputs have a single hidden-width axis, not a head axis. No per-head attention output files were found. The P-L index fields are `array_row, concept_id, condition, fewshot_set, format, lang, m1_extension, shard, split`. The index and sidecars do not record Gemma local/global layer types.

## E3. Direction inputs

The P-D fit-split activation and index files are `results/pd/family={family}__format={repetition|translation}__pass=pd__source=matched.resid.npy` and the paired `.parquet` files. Per family and format, the residual arrays have 660 rows and shape `(660, N+1, d_model)`, float32; the index has 660 rows and columns `row, concept_id, lang, format, split, n_tokens, n_special`. The fit-split class means are not separately saved, but can be reconstructed by grouping these stored residuals on the index's language and split columns.

E8 saved six family-by-format vector arrays at `results/ps/family={family}__format={format}__vectors.npy`, each `(N+1, d_model)`, float32. `src/vilens/steer.py:15–52` defines these as fp32 `μ_en − μ_vi`. Negating the saved vector gives the sign of `v_vi − v_en` if E7 defines its class vectors as those same means; the saved difference alone cannot supply `v_vi`. The matched P-D activation/index files are the available source for separately reconstructing class means.

## E4. Band inputs and Δ_vi availability

`reports/E2-E5_landmarks.csv` contains the `L_peak` and `L_cross` metric rows keyed by model and format, with layer/depth/CI/valid-draw columns. The six decision-format cells are `google/gemma-3-4b-pt × {repetition, translation}`, `Qwen/Qwen3-4B-Base × {repetition, translation}`, and `meta-llama/Llama-3.2-3B × {repetition, translation}`; four rows for each metric are present per cell. Numeric values are omitted from this inventory. The file also contains cloze rows, which are not decision-format E7 cells.

`results/analysis/primary/family={family}/deltas.parquet` and `rise.parquet` are per-concept, per-language, per-layer tables. Their common key fields include `concept_id, format, target_lang, layer, metric`; `deltas.parquet` also has `value`. Key-column metadata confirms `target_lang=vi` rows for repetition and translation in each family, so per-layer `Δ_vi` data are available. The `deltas.parquet` row counts are 3,031,140 (Gemma), 3,204,348 (Qwen) and 2,511,516 (Llama); `rise.parquet` row counts are 2,806,564, 2,971,656 and 2,311,288 respectively. No E7 value was used to compute or report an E7 result.

## E5. Existing code and tests

No `src/vilens/attribute.py`, E7 analysis module, `scripts/analyze_e7.py`, `job_e7_ablate.py`, or E7-specific test file exists. `src/vilens/hooks.py` provides the shared forward-hook recorder, and `src/vilens/steer.py` provides E8 steering-vector code; neither implements E7 attribution or head ranking.

## E6. Decisions needed and compute planning

The following choices are not specified by the current E7 text. Options are listed without a recommendation.

1. **Band when `L_cross` is NA:** use the stated top-quartile slope fallback; omit E7 for that cell; or use another fixed band chosen by the human.
2. **Top-quartile slope band:** use the concept-mean `Δ_vi` curve or a per-concept slope distribution; use repetition, translation, or each format separately; resolve tied cut points by including all ties or by a deterministic layer rule.
3. **Minimum-width interaction with the fallback:** apply the three-block minimum ending at a defined endpoint; apply it only when `L_cross` exists; or choose a distinct fallback endpoint.
4. **Direction definitions:** use raw class means or centered/normalized directions; compute separately by format or pool formats; use fit-split concepts only or a different frozen source.
5. **Projection quantity:** report each layer's update projection or cumulative projections; report raw dot products or normalize by vector/update norms.
6. **Attention/MLP share:** define the share from absolute projection magnitudes, signed projections, or component norms; define the denominator and zero-denominator handling.
7. **Head ranking:** acquire per-head outputs with a new model forward pass; rank by a decomposition of stored layer outputs if one can be justified; or report layer-level attribution only.
8. **Gemma local/global layers:** use model-configuration layer types after recording them in E7 metadata; infer them from the pinned architecture; or report blocks without type labels.
9. **Optional top-10 head ablation:** use the 300 E8 primary items or the E8 directions-split items; rank globally or within each layer/type; derive mean-ablation activations from the P-L prompts or another human-specified source.
10. **Uncertainty and multiplicity:** use a concept bootstrap with a new E7-specific key or report descriptive point estimates; apply no Holm adjustment or define a family before analysis. No E7 bootstrap key or Holm family is pre-registered.
11. **Format reporting:** keep repetition and translation separate; pool them with a task-format term; or provide both format-specific and pooled summaries.

### Compute items and planning estimates

These are engineering estimates for future sizing, not measured resource use. E7 and independent R jobs can be scheduled as one CPU-oriented `small` job plus one model-running `normal` job, subject to the queue rule and separate human approvals.

| E7 compute item | Expected profile | Projected peak VRAM | Projected peak RAM | Basis |
|---|---|---:|---:|---|
| Resolve landmarks and choose the approved layer band | small | <1 GB | ≤4 GB | CSV/key processing only |
| Reconstruct P-D class means from saved activations and indices | small | <1 GB | ≤8 GB | memory-mapped arrays, one family × format at a time |
| Project stored attention/MLP updates and calculate layer shares | small | <1 GB | ≤8 GB | stored fp32 arrays, streamed by unit |
| Bootstrap or summarize component estimates, if decided | small | <1 GB | ≤8 GB | CPU-only resampling of summary rows |
| Capture/rank per-head outputs if required | normal | 20–35 GB | ≤32 GB | new model forward pass; conservative estimate for 4B-class weights and activations, to be checked on a trial |
| Mean-ablate ranked heads on E8 items, if required | normal | 20–35 GB | ≤32 GB | model inference and intervention; to be measured on a trial |
| Write `reports/E7.md` and summary tables | small | <1 GB | ≤4 GB | report/table writing |

Expected report and output names are `reports/E7.md` and summary CSVs for layer attribution, head ranking, and any human-approved ablation. Their schemas, item sets, uncertainty columns and provenance layout remain to be specified before implementation.

## E7. Decisions

See OPS-37 in docs/decisions.md
Check 10(b) amended and smoke check added by OPS-38 (docs/decisions.md).
Batch replay and check 10(c′) added by OPS-39 (docs/decisions.md).
