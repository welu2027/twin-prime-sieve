"""
Stage 3: write the preregistered prediction table and stop.

Dumps the FULL predicted table (ledger/data/predicted.parquet) to
ledger/prereg/predictions_YYYYMMDD.csv with a `preregistered_prior` column
marking the 150 twin mean_norm_gap le_1e9 cells whose predictions were
already committed to the repo before the original mod-210/2310 sieving
(pred_rho_bar_mod210.csv / pred_rho_bar_mod2310.csv).  Everything else —
cousins, sexy, all windows, all non-mean statistics — is NEW and is locked
by committing this file.

Prints the file's SHA-256.  COMMIT THIS FILE TO GIT BEFORE RUNNING STAGE 4:
    git add ledger/prereg/predictions_YYYYMMDD.csv
    git commit -m "Preregister grand-ledger predictions"

Stage 4 refuses to run until the file is committed and clean.
"""

import csv
import hashlib
import os
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "20260827"
OUT = os.path.join(HERE, "prereg", f"predictions_{DATE}.csv")


def main():
    t = pq.read_table(os.path.join(HERE, "data", "predicted.parquet"))
    cols = t.column_names
    data = t.to_pydict()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["preregistered_prior"])
        for i in range(t.num_rows):
            row = [data[c][i] for c in cols]
            prior = (data["constellation"][i] == "twin"
                     and data["statistic"][i] == "mean_norm_gap"
                     and data["window"][i] == "le_1e9")
            g = data["g"][i]
            row[cols.index("g")] = "" if g is None else g
            row[cols.index("predicted")] = f"{data['predicted'][i]:.8g}"
            row[cols.index("predicted_wheel_only")] = f"{data['predicted_wheel_only'][i]:.8g}"
            w.writerow(row + [int(prior)])

    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print(f"wrote {OUT} ({t.num_rows:,} rows, {os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"SHA-256: {h}")
    print("\nNow commit it (stage 4 will refuse to run until then):")
    print(f"  git add {os.path.relpath(OUT, os.path.join(HERE, '..'))}")
    print(f"  git commit -m 'Preregister grand-ledger predictions ({DATE})'")


if __name__ == "__main__":
    sys.exit(main())
