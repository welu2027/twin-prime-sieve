# Ledger build environment

- date: 2026-08-27
- python: 3.14.3 (macOS-26.5.2-arm64-arm-64bit-Mach-O)
- numpy: 2.5.2
- pyarrow: 25.0.1
- bootstrap: Poisson bootstrap, B = 1000, seed = (20260827, constellation_idx,
  modulus, window_idx, class); autocorr via pair bootstrap, same seed stream
- sieve: odd-only segmented numpy sieve (ledger/01_sieve.py), XMAX = 1e9
- hazard model: ledger/hazard.py; Euler products truncated at p < 2e6;
  gmax = 12x predicted mean gap; window nodes weighted e^u/u^2
- preregistration: ledger/prereg/predictions_20260827.csv (SHA-256 printed by
  03_commit_predictions.py; committed to git before stage 4 ran)
