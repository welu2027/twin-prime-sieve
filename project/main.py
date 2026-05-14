"""
Run the full twin prime + AI pipeline.

Usage:
    python main.py --limit 1_000_000          # quick test (~1M)
    python main.py --limit 100_000_000        # 10^8, good default
    python main.py --limit 1_000_000_000      # 10^9, takes a few minutes
    python main.py --skip-data                # re-use existing data
    python main.py --skip-train               # only generate data, no ML
"""
import argparse
import subprocess
import sys
from pathlib import Path

from utils import DATA_DIR

STEPS = {
    "data":     "data_generation.py",
    "features": "feature_engineering.py",
    "baseline": "train_baselines.py",
    "lstm":     "train_lstm.py",
    "analyze":  "analyze_results.py",
}


def run(script: str, extra_args: list[str] = ()):
    cmd = [sys.executable, str(Path(__file__).parent / script)] + list(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {script} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Twin Prime + AI pipeline.")
    parser.add_argument("--limit", type=int, default=10**8)
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation (reuse existing parquet)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training")
    parser.add_argument("--skip-symbolic", action="store_true",
                        help="Skip PySR symbolic regression (slow)")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training")
    args = parser.parse_args()

    if not args.skip_data:
        run(STEPS["data"], ["--limit", str(args.limit)])
        run(STEPS["features"])

    if not args.skip_train:
        run(STEPS["baseline"])
        if not args.skip_lstm:
            run(STEPS["lstm"])
        if not args.skip_symbolic:
            run(STEPS["analyze"])  # analyze first so plots exist even if pysr is skipped
            run("pysr_symbolic.py")
        else:
            run(STEPS["analyze"])
    else:
        run(STEPS["analyze"])

    print("\nPipeline complete. Check data/ for output files and plots.")


if __name__ == "__main__":
    main()
