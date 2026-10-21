# Vie-Lens: The Latent Language of Vietnamese Prompts

Code for a mechanistic-interpretability study of how multilingual LLMs internally process Vietnamese prompts: do they route through English, a language-neutral concept space, or (for Qwen) Chinese for Sino-Vietnamese words?

The standard logit lens only scores **single-token** targets. Most Vietnamese words are multi-syllabic and split into several tokens, so that restriction biases which words get measured. This repo implements a **multi-token logit lens** and uses it to test three preregistered hypotheses:

- **H1, English pivot:** Vietnamese is processed via English in middle layers.
- **H2, language-neutral concepts:** middle layers encode concepts without a preferred language.
- **H3, Chinese routing in Qwen:** Qwen routes Sino-Vietnamese vocabulary through Chinese.

The study covers five languages (Vietnamese, English, Chinese, French, Indonesian), prompt formats (repetition, translation, cloze), and a diacritics-stripped condition. It includes steering, etymology stratification, component attribution, and a robustness battery.

**Models (base checkpoints):** `google/gemma-3-4b-pt`, `Qwen/Qwen3-4B-Base`, `meta-llama/Llama-3.2-3B`. Exact revisions are pinned in `configs/experiments.yaml`.

## Repository layout

| Path | Contents |
|---|---|
| `src/vilens/` | Core package: lens, hooks, scoring, steering, attribution, analysis |
| `job_*.py` | Experiment entry points, run via `submit_job.sh` (Slurm) |
| `scripts/` | Input verification, tokenizer audit, analyses, figures and tables |
| `configs/experiments.yaml` | All settings, data and model pins, seeds |
| `derived/` | Frozen derived inputs: seeds, distractors, target forms, E8 order |
| `docs/` | Experiment plan, analysis spec, decisions, deviations; `docs/prereg/` holds the preregistration |
| `reports/REPRODUCE.md` | Step-by-step reproduction commands and output map |
| `tests/` | CPU unit tests |

## Reproduce

**1. Environment.** Use Python 3.12 and a GPU with 20–40 GB of VRAM per job (see the profiles in `submit_job.sh`).

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

The code was run on our Slurm cluster. Before running, set your own paths and cache locations in `configs/experiments.yaml` (`paths`, `flores`), `submit_job.sh`, and `src/vilens/env_guard.py`.

**2. Data**

```bash
# Concept dataset and prompts
hf download tarudesu/viLens-concepts --repo-type dataset --revision v1.0 \
  --local-dir data/external/viLens-concepts
# Dataset-construction pipeline (needed for concepts.tsv and the FLORES+ directions builder)
git clone --branch data-v1.3.1 --depth 1 \
  https://github.com/tarudesu/viLens-concepts-preparation.git data/external/viLens-concepts-preparation
# FLORES+ is gated: accept the terms at https://huggingface.co/datasets/openlanguagedata/flores_plus,
# set HF_TOKEN, then rebuild the FLORES inputs
python scripts/x0_flores.py
```

**3. Models.** Gemma and Llama are gated on Hugging Face. Request access, then download the pinned revisions into your HF cache.

**4. Verify inputs**

```bash
python scripts/x0_verify.py          # checks inputs against data/MANIFEST.sha256
python scripts/x0c_tokenizers.py     # tokenizer audit
python scripts/x0d_derived.py --check
```

**5. Run the experiments** in this order, each with `bash submit_job.sh <job>.py <profile>`:

| Stage | Entry points |
|---|---|
| Pilot and model audit gates | `job_e0_pilot.py`, `job_e1_audit.py` |
| Lens and direction passes | `job_pl_lens.py`, `job_pd_directions.py`, `job_e2_m1.py` |
| Primary analysis (E2–E5) | `job_analyze_core.py` |
| Steering (E8) | `job_ps_steer.py` |
| Etymology (E6), attribution (E7) | `job_analyze_e6.py`, `job_e7_attribute.py`, `job_analyze_e7.py` |
| Robustness and diacritics (E9) | `job_analyze_r1.py`, `job_r2_directions.py`, `job_analyze_r10.py`, `job_analyze_rbattery.py`, `job_analyze_e9.py` |
| Figures and tables | `python scripts/figures.py`, `python scripts/tables.py` |

Exact arguments, run order, and which output feeds each table or figure are in `reports/REPRODUCE.md`.

**Tests:** `pytest`

## Related resources

- Dataset: [tarudesu/viLens-concepts](https://huggingface.co/datasets/tarudesu/viLens-concepts)

## Citation

Coming soon.



