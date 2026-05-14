"""
Run the full twin prime + AI pipeline. Saves all output to data/pipeline_results.txt.

Usage:
    python main.py                             # 10^8 default
    python main.py --limit 1_000_000           # quick test
    python main.py --limit 1_000_000_000       # 10^9 full run (~1-2 hrs)
    python main.py --skip-data                 # reuse existing parquet
    python main.py --skip-train                # data only, no ML
    python main.py --skip-lstm                 # skip LSTM (saves time)
    python main.py --skip-symbolic             # skip PySR (slow)

Analysis + visualization are separate:
    python analyze_results.py                  # saves analysis_results.txt + plots
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils import DATA_DIR

STEPS = {
    "data":     "data_generation.py",
    "features": "feature_engineering.py",
    "baseline": "train_baselines.py",
    "lstm":     "train_lstm.py",
    "symbolic": "pysr_symbolic.py",
}


def run(script: str, extra_args: list[str] = (), log=None):
    cmd = [sys.executable, str(Path(__file__).parent / script)] + list(extra_args)
    header = f"\n{'='*60}\nSTEP: {script}  {' '.join(extra_args)}\n{'='*60}"
    print(header)
    if log:
        log.write(header + "\n")
        log.flush()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log:
            log.write(line)
            log.flush()
    proc.wait()

    if proc.returncode != 0:
        msg = f"\nERROR: {script} failed (exit {proc.returncode}). Stopping.\n"
        print(msg)
        if log:
            log.write(msg)
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Twin Prime + AI pipeline.")
    parser.add_argument("--limit", type=int, default=10**8)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-symbolic", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    log_path = DATA_DIR / "pipeline_results.txt"

    with open(log_path, "w") as log:
        start = datetime.now()
        header = (
            f"Twin Prime + AI Pipeline\n"
            f"Started : {start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Limit   : {args.limit:,}\n"
            f"{'='*60}\n"
        )
        print(header)
        log.write(header)

        if not args.skip_data:
            run(STEPS["data"], ["--limit", str(args.limit)], log)
            run(STEPS["features"], [], log)

        if not args.skip_train:
            run(STEPS["baseline"], [], log)
            if not args.skip_lstm:
                run(STEPS["lstm"], [], log)
            if not args.skip_symbolic:
                run(STEPS["symbolic"], [], log)

        elapsed = datetime.now() - start
        footer = (
            f"\n{'='*60}\n"
            f"Pipeline complete.\n"
            f"Elapsed : {elapsed}\n"
            f"Results : {log_path}\n"
            f"Next    : python analyze_results.py\n"
        )
        print(footer)
        log.write(footer)

    print(f"Full log saved to {log_path}")


if __name__ == "__main__":
    main()
