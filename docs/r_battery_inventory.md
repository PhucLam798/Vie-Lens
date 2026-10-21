# R1–R11 and E9 inventory

Metadata/code inventory only. No robustness statistic was calculated. Existing reports are identified by location and section; no result values are copied here. Input shapes and table row counts are storage metadata.

## Shared inputs and resource references

The registered analysis tables are present under `results/analysis/primary/family={gemma,qwen,llama}/`. Each of `la.parquet`, `deltas.parquet`, `rise.parquet`, and `scores.parquet` has the rule-38 17-column long schema. The per-family row counts, in Gemma / Qwen / Llama order, are:

| Table | Rows by family |
|---|---|
| `la.parquet` | 426,870 / 557,820 / 433,860 |
| `deltas.parquet` | 3,031,140 / 3,204,348 / 2,511,516 |
| `rise.parquet` | 2,806,564 / 2,971,656 / 2,311,288 |
| `scores.parquet` | 10,146,640 / 10,726,448 / 8,407,216 |

P-D matched residuals and indices are present at `results/pd/family={family}__format={format}__pass=pd__source=matched.{resid.npy,parquet}`. Per family × format the residual shape is `(660, N+1, d_model)`, fp32: Gemma `(660,35,2560)`, Qwen `(660,37,2560)`, Llama `(660,29,3072)`. P-D FLORES residual files `results/pd/family={family}__pass=pd__source=flores.{last,mean}.npy` are each `(4985,N+1,d_model)`, fp32; no sentence text is read or included.

Measured P-L/E8 GPU peaks used only as resource references: P-L job 1564 reported sweep peaks of 24.49 GB Gemma, 23.14 GB Qwen and 17.50 GB Llama; E8 primary job 3183 reported 11.66 / 10.26 / 8.00 GB respectively. These are memory measurements, not study outcomes. CPU-only estimates below use zero GPU VRAM. RAM and wall-time figures are planning estimates, not measured job results.

## Registered robustness battery

| Item | Governing text | Existing result location (section only) | Inputs present / missing | Re-analysis or new pass | Expected profile and planning resources | Code present / missing |
|---|---|---|---|---|---|---|
| **R1** Few-shot sets 2–6; fallback sets 2–3 | `docs/experiments.md:606`; `docs/prereg/PREREG.md:421` | No R1 result table identified. Existing Checkpoint H tables are set-1 analyses. | Released prompts and registered item manifests are present. No `results/pl/` units named for fewshot sets 2–6 were found. | New P-L GPU pass, then CPU summaries. | `normal`; planning VRAM ceiling from P-L job 1564: 24.49 / 23.14 / 17.50 GB by family; RAM up to 32 GB; about 10–20 h using the P-L throughput range and the planned sequence volume in `docs/experiments.md:474` (rough estimate). | P-L pass infrastructure exists; no R1 orchestration/report code identified. |
| **R2** FLORES vs matched directions | `docs/experiments.md:607`; `docs/prereg/PREREG.md:422` | Partly represented in `reports/E2-E5.md`, §E5 language alignment and `reports/E5_gate.csv`; no separate R2 report identified. | P-D matched and FLORES arrays/indices are present with the shapes above; `data/external/directions_flores.jsonl` is present. | CPU comparison of stored direction outputs; no new model pass indicated by the current stored inputs. | `small`; 0 GB VRAM, up to 16 GB RAM, roughly 30 min. | P-D and E5 comparison helpers exist; standalone R2 report writer not identified. |
| **R3** L_concept at 5 / 10 / 20% | `docs/experiments.md:608`; `docs/prereg/PREREG.md:423` | Present in `reports/E2-E5.md`, §E3 landmarks and C1; source table `reports/E2-E5_landmarks.csv`. | Stored landmark and per-concept LA tables are present. | CPU re-analysis/reporting from committed summaries. | `small`; 0 GB VRAM, up to 8 GB RAM, under 15 min. | E3 landmark computation and report outputs exist. |
| **R4** Raw s vs Δ | `docs/experiments.md:609`; `docs/prereg/PREREG.md:424` | Present in `reports/E2-E5.md`, §E4 rules and Holm family (a), and `reports/E2-E5_rules.csv`. | Stored `scores.parquet` and `deltas.parquet` are present. | CPU re-analysis of stored tables. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | Core score/delta and E4 rule code exists; a separate R4 wrapper was not identified. |
| **R5** Tuned lens on Llama with balanced corpus (D8) | `docs/experiments.md:610,692`; `docs/prereg/PREREG.md:415,425`; `docs/decisions.md:247–248` | Untuned E4 result exists in `reports/E2-E5.md`, §E4; no tuned-lens result identified. | Llama P-L inputs and the concept table are present. A dedicated language-balanced training corpus and trained tuned-lens checkpoint were not found in the current input inventory. | Requires the corpus/training specification in the future R5 task and a GPU training/inference pass. | `normal` planning lane; base Llama P-L reference peak 17.50 GB (job 1564), but tuned-lens peak is unmeasured; plan up to 32 GB RAM; wall time cannot be estimated until corpus and training details are fixed. | Core lens code exists; tuned-lens training, checkpoint and R5 orchestration are absent. |
| **R6** v1.2 subsets | `docs/experiments.md:611`; `docs/prereg/PREREG.md:426` | Present in `reports/E2-E5.md`, §E4, and `reports/E6.md`, §Sensitivities; source `reports/E6_sensitivity.csv`. | Core analysis tables and the frozen concept metadata are present. | CPU re-analysis of saved summaries, if a consolidated robustness table is needed. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | E4 and E6 sensitivity implementations/results exist. |
| **R7** Competent subset | `docs/experiments.md:612`; `docs/prereg/PREREG.md:427` | Competence outputs are in `reports/E2-E5.md`, §A10 competence and `reports/E2-E5_c1.csv`; competent-subset analyses are in the E3–E5 sections. | `c1.parquet`, core analysis tables and A10 summary files are present. | CPU subset analysis from stored tables. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | A10 competence and E3–E5 subset code exists; no separate R7 report identified. |
| **R8** s¹ / s⁺ / sequence sum; zh with space | `docs/experiments.md:613`; `docs/prereg/PREREG.md:314,428` | Present in `reports/E2-E5.md`, §§E3–E4, and the committed landmark/rule/curve CSVs. | `scores.parquet`, `deltas.parquet`, and `rise.parquet` contain the corresponding stored metrics. | CPU re-analysis/reporting from saved tables. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | The alternate-readout and zh-space core paths exist. |
| **R9** All-items five-way | `docs/experiments.md:614`; `docs/prereg/PREREG.md:328,429` | Present as sensitivity rows in `reports/E2-E5.md`, §§E3–E4, and `reports/E2-E5_rules.csv`. | Main-scope core tables are present. | CPU summary from stored analyses. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | Core main-item sensitivity path exists. |
| **R10** Along-path readout | `docs/experiments.md:322–329,615`; `docs/prereg/PREREG.md:316,430` | Present in `reports/E2-E5.md`, §E4 diagnostic material and the committed analysis tables. | P-L along-path score rows and core score/delta files are present. | CPU re-analysis from stored along-path rows. | `small`; 0 GB VRAM, up to 24 GB RAM, about 1 h for a full-family read. | P-L along-path capture and E4 analysis code exist. |
| **R11** Exclude six card-flagged targets | `docs/experiments.md:616`; `docs/prereg/PREREG.md:431`; `docs/decisions.md:157–158` | Present in `reports/E6.md`, §Sensitivities and `reports/E6_sensitivity.csv`; E4 exclusion is represented in the E2–E5 outputs. | Core tables and the six frozen IDs in `configs/experiments.yaml → e6.r11_concept_ids` are present. | CPU sensitivity re-analysis from stored tables. | `small`; 0 GB VRAM, up to 16 GB RAM, about 30 min. | E4/E6 sensitivity code and outputs exist. |

## E9 — diacritic-stripped input (optional)

**Governing text:** `docs/experiments.md:618–623`; `docs/prereg/PREREG.md:180,433`; `docs/prereg/v1.4_amendment.md:185`. The planned P-L scale is recorded at `docs/experiments.md:475`.

**Existing result:** no separate E9 result report was identified. `results/pl/` has no files named with `condition=nodiac`; the existing `vi_nodiac` target diagnostic in the diacritic condition is A12 and is not the E9 input condition.

**Inputs:** released prompt files and target forms are present; the nodiac-condition P-L residual/lens outputs are missing. The existing P-L code supports `nodiac` as a condition but no E9 output unit exists.

**Compute:** new P-L inference is required. Expected profile `normal`; use P-L job 1564's measured reference peaks (Gemma 24.49 GB, Qwen 23.14 GB, Llama 17.50 GB), up to 32 GB RAM, and a rough 2–4 h wall-time planning range. A separate CPU report stage can use `small` (0 GB VRAM, up to 16 GB RAM, about 30 min).

## Open decisions and compute pairing order

- R5's balanced-corpus identity and training details remain for the R5 task, as stated by D8; its runtime/resource estimate is therefore incomplete.
- E9 is optional in the preregistration; whether to run the optional pass remains a human scheduling decision.
- The R1 fallback from sets 2–6 to sets 2–3 is already predeclared; no new scientific choice is recorded here.
- E7's optional mean-ablation is deferred by OPS-37-8 and would require a later OPS entry.

Compute pairing order for queue planning: while the P-A E7 `normal` job runs, the `small` lane can run CPU-only R2, R3, R4, R6, R7, R8, R9, R10 and R11 analyses one at a time. R1 and E9 require `normal` GPU inference and would follow the P-A job; R5 waits for its corpus/training specification. This is a scheduling inventory only, not a recommendation about scientific choices.
