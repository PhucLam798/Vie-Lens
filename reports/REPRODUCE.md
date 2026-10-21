# Reproducibility reference

Sources: reports/audit/job_ledger.csv; reports/audit/results_inventory.csv; the inventoried provenance sidecars; configs/experiments.yaml; requirements.txt; reports/figures/README.md; reports/tables/README.md. Current verifier invocations are recorded in logs/t29/resume_verify.provenance.json.

The artifact index is [reproduce_map.csv](reproduce_map.csv). Each report CSV, figure PDF/PNG, table Markdown/TeX and data sidecar has one row; the map itself has a row. README files are documentation, not figure/table artifacts. Blank fields mean the permitted sources do not record that fact. Job associations and code commits are copied from the audit ledger/inventory. Source-CSV commits in READMEs are listed below; they are not substituted for generator code commits. Last-touch commits and filenames inside code-tree manifests are not treated as producing commits or entrypoints. No unrecorded arguments or governing decisions are inferred.

## Environment

Venv: /datastore/luannt/.cache/sb (requirements.txt).
Recorded Python version(s): 3.12.13 (inventoried provenance sidecars).

Exact package pins copied from requirements.txt:

```text
accelerate==1.15.0
datasets==5.0.1
hf-xet==1.6.0
huggingface-hub==0.36.2
matplotlib==3.11.2
numpy==1.26.4
pandas==3.0.6
protobuf==7.36.2
pyarrow==25.0.1
pytest==9.1.1
pyyaml==6.0.3
safetensors==0.8.0
scipy==1.17.1
sentencepiece==0.2.2
statsmodels==0.15.0
tokenizers==0.22.2
torch==2.8.0
tqdm==4.70.1
transformers==4.57.6
```

## Model and data pins

models.study_set: base.

| Set | Family | HF ID | Revision |
| --- | --- | --- | --- |
| base | gemma | google/gemma-3-4b-pt | cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| base | qwen | Qwen/Qwen3-4B-Base | 906bfd4b4dc7f14ee4320094d8b41684abff8539 |
| base | llama | meta-llama/Llama-3.2-3B | 13afe5124825b4f3751f836b40dafda64c1ed062 |
| registered | gemma | google/gemma-3-4b-it | 093f9f388b31de276ce2de164bdc2081324b9767 |
| registered | qwen | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| registered | llama | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b7a12671c53f0838cd831a0843b95 |

| Config field | Copied value |
| --- | --- |
| data.hf_repo | tarudesu/viLens-concepts |
| data.hf_revision | a8d9ca1a75cd9944da6a3364a679094913859210 |
| data.prep_repo | tarudesu/viLens-concepts-preparation |
| data.prep_tag | data-v1.3.1 |
| data.prep_commit | 873def8829f4024b72c0993356447d3cca31b549 |
| flores.repo | openlanguagedata/flores_plus |
| flores.revision | 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 |
| flores.split | dev |
| prereg.data_tag | data-v1.3.1 |
| prereg.analysis_tag | prereg-v1.4 |

The study used dataset revision 8a2d99c; its data files are byte-identical to the v1.0 release at the configured revision.

## Pass records

Rows below retain ledger order. E8 rows are recorded under that label in the ledger; the P-S unit provenance records pass=ps and job 3183. P-D/P-L/P-A and R1/E9 capture labels are copied from the ledger. Failed, cancelled and trial records retain their recorded state/status.

| Job | Ledger step | State | Status in reports | Code commit |
| --- | --- | --- | --- | --- |
| 1562 | P-D | COMPLETED | completed-and-used | a0077d3cd50f840abcfaee864b5a0367744b7520 |
| 1563 | P-L | CANCELLED by 659418558 | failed-and-kept |  |
| 1564 | P-L | FAILED | failed-and-kept | 1cad93f91a08745d5270999a0c281e4040f7c09b |
| 1572 | P-L | COMPLETED | smoke-or-trial | 860614f23fafd2bf4cf50fa27aed5c2da3629b91 |
| 1575 | P-L | FAILED | failed-and-kept |  |
| 1591 | P-L | FAILED | failed-and-kept | 2cb990e052d608c0943f71d90a53fe1bd7ad3532 |
| 1624 | P-L | COMPLETED | completed-and-used | e18569d258dabd075165ba38cbda8c8f147243bf |
| 3134 | E8 | COMPLETED | smoke-or-trial | d1b3d5e02196426abd2242c95059dbf423f341ed |
| 3183 | E8 | COMPLETED | completed-and-used | 566653a83674711cd7002cf00f953b6ac1eb5f67 |
| 3928 | P-A | FAILED | failed-and-kept | ef88a2fffa0f620138d99950b414ba658813cf66 |
| 3943 | P-A | FAILED | failed-and-kept | e0ad0d20756af69c1a65b18fd92c7c95adaad35c |
| 3963 | P-A | COMPLETED | smoke-or-trial | 8b47f2e55ca05d5481991414aaa07769827e9494 |
| 4062 | P-A | COMPLETED | smoke-or-trial | 953dfeea553c479d264c9188237ef664d4ad3c91 |
| 4068 | P-A | COMPLETED | completed-and-used | 97f3fd6581141974f9af07df386c0f917dc93dda |
| 4302 | R1 capture | COMPLETED | completed-and-used | 48df1e037dd05bd640569a67556afe5d4ce4a14b |
| 4630 | E9 capture | COMPLETED | completed-and-used | 2897c2d87c8ae934dde999c386891833f20765ee |

## Analysis and audit job records

Entrypoints are copied only where a provenance sidecar explicitly records one; missing arguments remain unrecorded.

| Job | Ledger step | State | Entrypoint | Report CSV outputs |
| --- | --- | --- | --- | --- |
| 1831 | E2 | FAILED |  |  |
| 2632 | E2 | FAILED |  |  |
| 2666 | E2 | FAILED |  |  |
| 2709 | E2 | FAILED |  |  |
| 2715 | E2 | FAILED |  |  |
| 2728 | E2 | COMPLETED |  | reports/E2-E5_e2.csv |
| 2779 | E2-E5 | FAILED |  |  |
| 2782 | E2-E5 | FAILED |  |  |
| 2783 | E2-E5 | FAILED |  |  |
| 2788 | E2-E5 | FAILED |  |  |
| 3017 | E2-E5 | FAILED |  |  |
| 3026 | E2-E5 | COMPLETED |  | reports/E2-E5_rules.csv;reports/checkpoint_h_rules.csv;reports/checkpoint_h_verdicts.csv |
| 3280 | E6 | COMPLETED |  |  |
| 3302 | E6 | COMPLETED |  |  |
| 3326 | E6 | FAILED |  |  |
| 3357 | E6 | COMPLETED |  |  |
| 3360 | E6 | FAILED |  |  |
| 3423 | E6 | FAILED |  |  |
| 3522 | E6 | CANCELLED by 659418558 |  |  |
| 3523 | E6 | CANCELLED by 659418558 |  |  |
| 3524 | E6 | CANCELLED by 659418558 |  |  |
| 3525 | E6 | CANCELLED by 659418558 |  |  |
| 3526 | E6 | CANCELLED by 659418558 |  |  |
| 3527 | E6 | COMPLETED |  | reports/E6_dzh.csv;reports/E6_holm.csv;reports/E6_matched.csv;reports/E6_model_terms.csv;reports/E6_sensitivity.csv |
| 3884 | E6 | OUT_OF_MEMORY |  |  |
| 3891 | E6 | COMPLETED |  | reports/E6_closeout.csv |
| 3905 | E6 | COMPLETED |  | reports/E6_moment_refit.csv |
| 4103 | E7 | FAILED | job_analyze_e7.py |  |
| 4301 | R2 | COMPLETED |  | reports/R2_alignment.csv;reports/R2_classification.csv;reports/R2_directions.csv |
| 4391 | R1 | COMPLETED |  | reports/R1_metrics.csv;reports/R1_set_outcomes.csv |
| 4422 | E7 | FAILED | job_analyze_e7.py |  |
| 4473 | E7 | COMPLETED | job_analyze_e7.py | reports/E7_band.csv;reports/E7_layers.csv |
| 4621 | E7 | COMPLETED | job_analyze_e7.py | reports/E7_heads.csv;reports/E7_heads_all.csv |
| 4629 | R-battery | COMPLETED |  | reports/R_battery_e4.csv |
| 4682 | E9 | COMPLETED |  | reports/E9.csv |
| 4683 | R10 | COMPLETED |  | reports/R10.csv |

## Regeneration and verification commands

Run from /datastore/luannt/Vie-Lens with the existing shared venv and datastore caches. These are the T29 operational commands; no historical unrecorded invocation is asserted. Run each command to completion before starting the next Python process. The map copies the verifier invocation from current verification provenance; its ignored wrapper expands to the environment below. Regeneration is documented here and was not executed during this resume.

```bash
cd /datastore/luannt/Vie-Lens
export HF_HOME=/datastore/luannt/.cache/huggingface
export HF_HUB_CACHE=/datastore/shared/models
export TORCH_HOME=/datastore/luannt/.cache/torch
export PIP_CACHE_DIR=/datastore/luannt/.cache/pip
export UV_CACHE_DIR=/datastore/luannt/.cache/uv
export XDG_CACHE_HOME=/datastore/luannt/.cache/xdg
export TRITON_CACHE_DIR=/datastore/luannt/.cache/triton
export MPLCONFIGDIR=/datastore/luannt/.cache/matplotlib
export CUDA_CACHE_PATH=/datastore/luannt/.cache/nv
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2 ARROW_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
timeout 1200s /datastore/luannt/.cache/sb/bin/python scripts/figures.py
timeout 1200s /datastore/luannt/.cache/sb/bin/python scripts/tables.py
timeout 1200s /datastore/luannt/.cache/sb/bin/python scripts/figures.py --verify
timeout 1200s /datastore/luannt/.cache/sb/bin/python scripts/tables.py --verify
```

## Allocated GPU-hours

| Recorded project total |
| --- |
| |

The permitted ledger records allocations per job and has no aggregate-total row or field. No total is recomputed. Per-job values below are copied exactly from gpu_hours; allocated_gpus, elapsed and profile are copied alongside them.

| Job | Profile | Elapsed | Allocated GPUs | Allocated GPU-hours |
| --- | --- | --- | --- | --- |
| 1484 | normal | 00:28:38 | 1 | 0.4772222222222222 |
| 1504 | normal | 00:05:40 | 1 | 0.09444444444444444 |
| 1512 | normal | 00:11:12 | 1 | 0.18666666666666668 |
| 1523 | normal | 00:10:40 | 1 | 0.17777777777777778 |
| 1562 | normal | 00:16:20 | 1 | 0.2722222222222222 |
| 1563 | large | 00:00:00 | 0 | 0.0 |
| 1564 | normal | 01:10:49 | 1 | 1.1802777777777778 |
| 1572 | normal | 00:46:48 | 1 | 0.78 |
| 1575 | normal | 00:00:11 | 1 | 0.0030555555555555557 |
| 1591 | normal | 09:01:33 | 1 | 9.025833333333333 |
| 1624 | normal | 00:52:01 | 1 | 0.8669444444444444 |
| 1831 | normal | 00:01:15 | 1 | 0.020833333333333332 |
| 2632 | normal | 00:00:26 | 1 | 0.007222222222222222 |
| 2666 | normal | 00:01:11 | 1 | 0.01972222222222222 |
| 2709 | normal | 00:01:38 | 1 | 0.02722222222222222 |
| 2715 | normal | 00:02:40 | 1 | 0.044444444444444446 |
| 2728 | normal | 00:06:19 | 1 | 0.10527777777777778 |
| 2779 | normal | 00:00:16 | 1 | 0.0044444444444444444 |
| 2782 | normal | 00:00:56 | 1 | 0.015555555555555555 |
| 2783 | normal | 00:03:55 | 1 | 0.06527777777777778 |
| 2788 | normal | 00:02:49 | 1 | 0.04694444444444444 |
| 3017 | normal | 00:00:23 | 1 | 0.006388888888888889 |
| 3026 | normal | 00:00:34 | 1 | 0.009444444444444445 |
| 3134 | normal | 00:18:00 | 1 | 0.3 |
| 3183 | normal | 03:26:39 | 1 | 3.444166666666667 |
| 3280 | normal | 00:33:17 | 1 | 0.5547222222222222 |
| 3302 | normal | 01:15:48 | 1 | 1.2633333333333334 |
| 3326 | normal | 00:01:16 | 1 | 0.021111111111111112 |
| 3357 | normal | 00:21:42 | 1 | 0.3616666666666667 |
| 3360 | normal | 00:00:55 | 1 | 0.015277777777777777 |
| 3423 | normal | 00:24:00 | 1 | 0.4 |
| 3522 | normal | 00:00:00 | 0 | 0.0 |
| 3523 | large | 00:00:00 | 0 | 0.0 |
| 3524 | full | 00:00:00 | 0 | 0.0 |
| 3525 | full | 00:00:00 | 0 | 0.0 |
| 3526 | small | 00:00:00 | 0 | 0.0 |
| 3527 | normal | 21:12:51 | 1 | 21.214166666666667 |
| 3884 | normal | 00:00:45 | 1 | 0.0125 |
| 3891 | normal | 00:17:27 | 1 | 0.29083333333333333 |
| 3905 | small | 00:01:48 | 1 | 0.03 |
| 3928 | small | 00:00:35 | 1 | 0.009722222222222222 |
| 3943 | small | 00:00:41 | 1 | 0.01138888888888889 |
| 3963 | normal | 00:01:22 | 1 | 0.02277777777777778 |
| 4062 | normal | 00:00:37 | 1 | 0.010277777777777778 |
| 4068 | normal | 00:14:54 | 1 | 0.24833333333333332 |
| 4103 | small | 00:00:15 | 1 | 0.004166666666666667 |
| 4301 | small | 00:03:35 | 1 | 0.059722222222222225 |
| 4302 | normal | 14:32:56 | 1 | 14.54888888888889 |
| 4391 | small | 00:05:28 | 1 | 0.09111111111111111 |
| 4422 | small | 00:00:18 | 1 | 0.005 |
| 4473 | small | 00:00:39 | 1 | 0.010833333333333334 |
| 4621 | small | 00:01:07 | 1 | 0.01861111111111111 |
| 4629 | small | 00:06:43 | 1 | 0.11194444444444444 |
| 4630 | normal | 04:58:10 | 1 | 4.969444444444444 |
| 4682 | small | 00:04:57 | 1 | 0.0825 |
| 4683 | small | 00:01:01 | 1 | 0.016944444444444446 |

## Source CSV revisions recorded in the READMEs

| Source CSV | Recorded source commit |
| --- | --- |
| reports/E2-E5_c1.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E2-E5_e2.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E2-E5_e5.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E2-E5_la_curves.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E2-E5_landmarks.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E2-E5_rules.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/E5_gate.csv | db5ee400f3780b51a637ccd0ed0e0142ab076fed |
| reports/E6_dzh.csv | 00cb2b58cd222ea54def21fd383a031828dea8b1 |
| reports/E6_holm.csv | 00cb2b58cd222ea54def21fd383a031828dea8b1 |
| reports/E6_matched.csv | 00cb2b58cd222ea54def21fd383a031828dea8b1 |
| reports/E6_model_terms.csv | 00cb2b58cd222ea54def21fd383a031828dea8b1 |
| reports/E6_sensitivity.csv | 00cb2b58cd222ea54def21fd383a031828dea8b1 |
| reports/E7_band.csv | 37a1ea5575f33d3586289280b12945a684c45b0d |
| reports/E7_heads.csv | 88845c4c7632824df31f407b24b9884fe657111b |
| reports/E7_heads_all.csv | 88845c4c7632824df31f407b24b9884fe657111b |
| reports/E7_layers.csv | 37a1ea5575f33d3586289280b12945a684c45b0d |
| reports/E8_delta.csv | 247bea75b0399bb7f4722a2bce8ccb209194331f |
| reports/E8_rates.csv | 247bea75b0399bb7f4722a2bce8ccb209194331f |
| reports/E9.csv | 97e96fa33d3cae3d18892bc3108e60f798b1b856 |
| reports/R10.csv | b003bbc69bd167138a5b2f40c01e0face51e7041 |
| reports/R1_metrics.csv | 2bee4da450c911de864acfeba91fc719a4de3b3f |
| reports/R1_set_outcomes.csv | 2bee4da450c911de864acfeba91fc719a4de3b3f |
| reports/R2_alignment.csv | 9af24ac4a07330bb545f650b8ae0cbd4ffd6af42 |
| reports/R2_classification.csv | 9af24ac4a07330bb545f650b8ae0cbd4ffd6af42 |
| reports/R2_directions.csv | 9af24ac4a07330bb545f650b8ae0cbd4ffd6af42 |
| reports/R_battery_e4.csv | 7dd53aa8ef87fd1e4b5bbaf6c0c431f006f4d856 |
| reports/checkpoint_h_verdicts.csv | 3f0bc6f86d5a147eb6830e03c4ec6a1cbe1f9b2c |
| reports/tables/tableA_tokenization.csv | 418096dda03ebe596c3f55b80efc8d2f8943c54a |

The audit inventory records its own snapshot. The map includes the current tracked artifact paths, including artifacts added after that snapshot; missing producer fields for those artifacts remain blank.
