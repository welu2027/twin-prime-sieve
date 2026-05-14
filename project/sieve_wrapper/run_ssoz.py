"""
Thin wrapper around the twinprimes_ssoz binary.
Use this when you only need the count and last twin for a range,
not the individual twin prime values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import run_sieve_count


def count_twins(n: int, start: int = None) -> dict:
    """Return {'count': int, 'last_twin': int} for twins <= n (or in [start, n])."""
    return run_sieve_count(n, start)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ssoz.py <N> [start]")
        sys.exit(1)
    n = int(sys.argv[1])
    start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    result = count_twins(n, start)
    print(f"Twin primes <= {n}: {result.get('count', 'N/A')}")
    print(f"Last twin prime: {result.get('last_twin', 'N/A')}")
