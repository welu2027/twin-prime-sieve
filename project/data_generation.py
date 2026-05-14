"""
Generate twin prime dataset up to a given limit.

Uses a Python segmented sieve to enumerate individual twin primes (p, p+2),
then saves them to data/twin_primes.parquet.

The Rust binary is used at the end to verify the count matches.

Start small: limit=10**8 is fast (~seconds), 10**10 takes a few minutes and ~2GB RAM.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_DIR, run_sieve_count, segmented_sieve


def generate_twin_primes(limit: int, seg_size: int = 2**20) -> list[int]:
    """
    Return list of lower twin primes p such that p and p+2 are both prime, p+2 <= limit.
    Uses segmented sieve to keep memory bounded.
    """
    sqrt_limit = int(math.isqrt(limit)) + 1
    print(f"Building base primes up to sqrt({limit:,}) = {sqrt_limit:,} ...")
    base_primes = segmented_sieve(sqrt_limit)
    print(f"  Found {len(base_primes):,} base primes.")

    twins = []
    low = 3
    print(f"Sieving for twin primes up to {limit:,} ...")

    while low <= limit:
        high = min(low + seg_size - 1, limit)
        size = high - low + 1
        is_prime = bytearray([1]) * size

        # sieve out composites in [low, high]
        for p in base_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            is_prime[start - low :: p] = bytearray(len(is_prime[start - low :: p]))

        # mark even numbers composite
        first_even = low if low % 2 == 0 else low + 1
        is_prime[first_even - low :: 2] = bytearray(len(is_prime[first_even - low :: 2]))
        if low <= 2 <= high:
            is_prime[2 - low] = 1  # 2 is prime

        # collect twins: need p and p+2 both prime
        for i in range(size - 2):
            if is_prime[i] and is_prime[i + 2]:
                p = low + i
                if p >= 3:
                    twins.append(p)

        if low % (10 * seg_size) == 3 or low == 3:
            print(f"  Progress: {low:,} / {limit:,} — {len(twins):,} twins so far", end="\r")
        low = high + 1

    print(f"\nDone. Found {len(twins):,} twin primes (lower value) <= {limit - 2:,}.")
    return twins


def build_dataframe(twins: list[int]) -> pd.DataFrame:
    """Build a DataFrame from the list of lower twin primes."""
    p = np.array(twins, dtype=np.int64)
    q = p + 2
    gap_before = np.empty(len(p), dtype=np.int64)
    gap_before[0] = 0
    gap_before[1:] = p[1:] - p[:-1]

    df = pd.DataFrame({
        "p": p,         # lower twin prime
        "q": q,         # upper twin prime (p+2)
        "gap_before": gap_before,  # gap from previous lower twin prime
        "index": np.arange(len(p), dtype=np.int64),
    })
    return df


def verify_count(df: pd.DataFrame, limit: int):
    """Cross-check our Python count against the Rust binary."""
    python_count = len(df)
    print(f"Python sieve count: {python_count:,}")
    try:
        rust_result = run_sieve_count(limit)
        rust_count = rust_result.get("count", -1)
        print(f"Rust binary count: {rust_count:,}")
        if python_count == rust_count:
            print("Counts match.")
        else:
            print(f"WARNING: counts differ by {abs(python_count - rust_count)}")
    except Exception as e:
        print(f"Could not verify with Rust binary: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate twin prime dataset.")
    parser.add_argument("--limit", type=int, default=10**8,
                        help="Generate twin primes up to this value (default: 10^8)")
    parser.add_argument("--seg-size", type=int, default=2**20,
                        help="Segment size for sieve (default: 1M)")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "twin_primes.parquet"),
                        help="Output parquet file path")
    args = parser.parse_args()

    twins = generate_twin_primes(args.limit, args.seg_size)
    df = build_dataframe(twins)
    verify_count(df, args.limit)

    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
