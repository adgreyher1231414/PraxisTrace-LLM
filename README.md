# PraxisTrace-LLM

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

The encoder produces construct values, exact evidence spans, and confidence, and rejects unsupported spans.


