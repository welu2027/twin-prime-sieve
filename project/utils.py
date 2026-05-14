import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
BINARY = PROJECT_DIR / "twinprimes_ssoz"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def run_sieve_count(n: int, start: int = None) -> dict:
    """Run the Rust binary and return parsed output (count + last twin)."""
    if not BINARY.exists():
        raise FileNotFoundError(f"Binary not found: {BINARY}")
    cmd = [str(BINARY), str(n)] if start is None else [str(BINARY), str(start), str(n)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    info = {}
    for line in output.splitlines():
        if "total twins" in line:
            parts = line.split(";")
            info["count"] = int(parts[0].split("=")[1].strip())
            info["last_twin"] = int(parts[1].split("=")[1].split("|")[0].strip())
    return info


def segmented_sieve(limit: int) -> list[int]:
    """Return all primes up to limit using a segmented sieve."""
    import math
    seg_size = max(int(math.sqrt(limit)) + 1, 65536)
    small_limit = int(math.sqrt(limit)) + 1
    small_sieve = bytearray([1]) * (small_limit + 1)
    small_sieve[0] = small_sieve[1] = 0
    for i in range(2, int(math.sqrt(small_limit)) + 1):
        if small_sieve[i]:
            small_sieve[i * i :: i] = bytearray(len(small_sieve[i * i :: i]))
    small_primes = [i for i, v in enumerate(small_sieve) if v]

    primes = list(small_primes)
    low = small_limit + 1
    while low <= limit:
        high = min(low + seg_size - 1, limit)
        seg = bytearray([1]) * (high - low + 1)
        for p in small_primes:
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            seg[start - low :: p] = bytearray(len(seg[start - low :: p]))
        for i, v in enumerate(seg):
            if v:
                primes.append(low + i)
        low = high + 1
    return primes
