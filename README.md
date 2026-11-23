# PraxisTrace-LLM: Python methodological reproduction

This project implements an **executable, independent methodological reproduction** of the attached manuscript *A Unified Evidence-Grounded Temporal Graph Framework for Modeling Persistent Belief-to-Practice Transitions in Teacher Professional Development*. It **does not claim to reproduce the manuscript's published numerical results**. Every generated result is new and labeled as such.

## What is implemented

- Six waves × 360 synthetic teacher trajectories, 120 teachers per site A/B/C; 60:20:20 progression/regression/plateau mixture, shocks, and reported initial belief statistics.
- Four evidence sources with evidence spans and uncertainty; deterministic **offline proxy encoder** for immediate execution. Optional evidence-grounded hosted LLM calls via Anthropic API.
- Teacher-level causal temporal graph surrogate: learned embedding, GRU temporal edges, five mechanism prototype cross-attention vectors, joint state/evidence/prototype/temporal/alignment losses.
- Grouped 60/20/20-ish train/validation/test teacher splits (64/16/20 overall), validation early stopping, F1, Cohen kappa, AUROC, Brier score, confusion-independent probability outputs.
- Static logistic-regression comparison, four ablations, site transfer, missing evidence sensitivity, data exports, charts, model checkpoint.

## Quick start

Python 3.11 recommended. From the extracted project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py --mode generate
python run.py --mode main --epochs 50
python run.py --mode full --epochs 50
```

CPU works; CUDA is automatically used when available. For a ~1-minute functionality check on a laptop:

```bash
python run.py --mode main --n 60 --epochs 2 --hidden 32 --out quick_check
python -m pytest -q   # requires: pip install pytest
```

On Windows, the commands run from PowerShell or Anaconda Prompt. Results are placed in `outputs/`, including CSV, checkpoint, JSON scores and PNG figures. Change `--out` to select a directory. Run `python run.py --help` for options.

### Optional hosted LLM mode

The manuscript describes `claude-sonnet-4-6`; access/model availability and API terms must be verified with your Anthropic account. **API usage incurs costs**. Run explicitly:

```bash
pip install anthropic
# macOS/Linux:
export ANTHROPIC_API_KEY='your_key'
# Windows PowerShell: $env:ANTHROPIC_API_KEY='your_key'
python run.py --mode main --encoder anthropic --model-name claude-sonnet-4-6
```

The encoder produces construct values, exact evidence spans, and confidence, and rejects unsupported spans. There is no hardcoded API key. The optional LLM is an inference-only coding interface; no fine-tuning of a hosted model is claimed.

## Critical methodological gaps in the manuscript

1. The manuscript references an unreleased repository placeholder, so neither original source code, generation scripts, anchoring vignettes nor the original 360 trajectories are provided. The 48 manually annotated vignettes and their reported inter-rater kappa **cannot be reconstructed**. There is no reason to expect exact paper metrics from this independent simulator.
2. TALIS Starting Strong 2018 and NSECE 2019 public use survey files are **not supplied**. Reported belief mean 3.41, SD 0.62 and within-wave correlations 0.38 and 0.29 are taken from the manuscript without independently verifying their provenance. Other distributions, mechanism effects, site-shift sizes, professional-development participation and prompt wording are **explicit approximations**, not extracted original estimates. A real empirical calibration requires obtaining these microdata, understanding survey weights and mappings, and rerunning the parameterization.
3. Equation `A = cos(W_b b, W_p p)` cannot be computed exactly without the unknown learned projection matrices and vector state representations; this reproduction uses a transparent proxy `clip(1-|belief-practice|/2.5,0,1)`. Thus `gamma=0.6` is not equivalent to the manuscript's original cosine threshold.
4. The manuscript defines 5 total loss terms but not their operational formulations, or the exact graph architecture, prompt templates, evidence spans, learning-rate schedules, annotation data, parameter seeds or full model training configuration. This implementation chooses a causal GRU as a **temporal graph surrogate**, not a verified reimplementation of the authors' graph network. Mechanism classification and supervised contrastive loss are both used.
5. The generated vignettes literally include their numerical state scores, so offline evidence coding is intentionally an **oracle-like feature-extraction diagnostic** rather than a defensible semantic-coding benchmark. The generated mechanism values similarly encode synthetic mechanism signals. Do **not** interpret offline results as genuine LLM performance or empirical teacher outcomes.
6. The original paper provides five-split averages. The default `run.py` uses one group-preserving split; to emulate five independent splits, run five commands with `--seed 41` through `--seed 45` and aggregate the independent metrics. The manuscript's exact held-out splits are unknown.
7. Synthetic benchmark labels use future waves as ground truth. The training input for each position contains only evidence available **up to that wave**. Offline features expose current state values, not the future target label.
8. Missing evidence experiments mask belief/practice/context/reflection inputs, not entire synthetic free-form text records. All reported figures from these new experiments are simulation-only.

## Data source landing pages from the manuscript

- OECD TALIS Starting Strong: https://www.oecd.org/en/data/datasets/TALIS-Starting-Strong-Database.html
- NSECE 2019, ICPSR: https://doi.org/10.3886/ICPSR37941.v10

These links are cited by the manuscript; this package does not automatically download survey microdata or assert that a particular version or variable mapping was verified.

## Output contract

`synthetic_benchmark.csv` (synthetic states and text), `encoded_offline.csv` (encoder inputs), `split_ids.json` (teacher IDs), `training_history.csv`, `test_predictions.csv`, `metrics.json`, `model.pt`, and PNG plots. Full mode adds cross-site results, missingness and ablations. All holdout scores reflect **this newly simulated dataset and code**.

## Manuscript reference table (for comparison only, not produced by this code)

| Metric | Manuscript claim |
|---|---:|
| Macro-F1 | 0.872 |
| Cohen kappa | 0.838 |
| AUROC | 0.901 |
| Brier score | 0.118 |
| Cross-site A→C AUROC | 0.801 |
| 50% missing evidence AUROC | 0.807 |

Scientific use: validate variable definitions and scoring against independently verified survey/codebook data, obtain ethically collected longitudinal ground truth, and avoid presenting simulator accuracy as real-world performance.
