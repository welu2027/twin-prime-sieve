# Twin Prime + AI Project

Uses a fast Rust segmented sieve (SSoZ) + Python ML to study twin prime gaps.

## Quick Start

```bash
cd project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Full pipeline (10^8 default, ~1 min)
python main.py

# Small test first
python main.py --limit 1_000_000

# Skip ML training, just generate data
python main.py --limit 100_000_000 --skip-train
```

## Pipeline Steps

| Step | File | What it does |
|------|------|--------------|
| 1 | `data_generation.py` | Python segmented sieve → `data/twin_primes.parquet` |
| 2 | `feature_engineering.py` | Adds log_p, gaps, residues, H-L density → `twin_primes_features.parquet` |
| 3 | `train_baselines.py` | XGBoost + Random Forest gap prediction |
| 4 | `train_lstm.py` | LSTM on gap sequences |
| 5 | `pysr_symbolic.py` | Symbolic regression — AI discovers formulas |
| 6 | `analyze_results.py` | Plots: density, gap distribution, model comparison |

## Note on the Rust Binary

`twinprimes_ssoz` is a counting sieve — it returns the total count and last twin
prime for a range, not individual values. Python generates the actual twin prime
list; the binary is used to verify the Python count matches.

## Output Files (`data/`)

- `twin_primes.parquet` — raw twin primes
- `twin_primes_features.parquet` — with engineered features
- `model_xgboost.pkl`, `model_randomforest.pkl`, `model_lstm.pt`
- `pysr_equations.csv` — discovered symbolic formulas
- `twin_prime_density.png`, `gap_distribution.png`, `model_comparison.png`
