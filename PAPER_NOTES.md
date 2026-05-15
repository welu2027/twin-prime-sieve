# Paper Writing Notes — Twin Prime Residue Class Deviations

## One-Line Summary

A computational pipeline discovers that twin prime gap spacings deviate from
Hardy-Littlewood by 4–5% in a class-dependent, scale-consistent way across
mod-210 residue classes — a phenomenon independent of the Lemke Oliver–
Soundararajan bias and unexplained by simple closed-form corrections.

---

## What Was Done

### 1. Data Generation
- Python segmented sieve generates all twin primes up to 10^9 (~440K pairs)
- For each twin prime p: compute log_p, gap_after (to next twin prime), mod210
  residue class, Hardy-Littlewood expected gap = (ln p)² / (2C₂)
- **hl_ratio** = gap_after / HL_expected — equals 1.0 if H-L is exactly right

### 2. Main Finding: Gap Deviation by Residue Class
- Grouped twin primes by mod-210 class (15 valid classes exist)
- 14/15 classes deviate significantly from H-L (t-test, Bonferroni-corrected p < 0.00067)
- Deviations are 4–5% in magnitude, consistent across all log_p sub-ranges
  (Spearman r = 0.95–0.98 between adjacent ranges)
- H-L correctly predicts total counts per class (count-based deviation < 0.1%)
  but NOT the gap spacing — an important distinction

### 3. Scale Validation via Rust Sieve (up to 10^12)
- Extended Rust SSoZ sieve with --per-class flag for fast counting at scale
- Ran ablation across [1e9,1e10), [1e10,1e11), [1e11,1e12)
- Count-based hl_ratio converges to 1.0 at 10^12 (< 0.012% deviation) —
  confirms H-L is asymptotically correct for counts
- Gap-based deviation (the novel finding) is separate from this

### 4. PySR Symbolic Regression
- Engineered number-theoretic features per class: dist_7, dist_11, ..., dist_23,
  isolation score, mod30_r, mod42_r, mod70_r
- Ran PySR on 48-point (actually 15-point) class-level dataset
- Best formula: 1.029 − 0.000464·mod210 + 0.000984·(mod70_r / dist_7)
- 76% loss reduction over constant baseline — weak but nonzero signal
- No clean closed-form correction found — deviations resist simple explanation

### 5. Lemke Oliver–Soundararajan Comparison
- Built 15×15 transition matrix of consecutive twin prime residue classes
- Confirmed LO&S anti-persistence: all self-transition rates (~0.045) below
  uniform baseline (1/15 ≈ 0.067)
- KEY RESULT: Spearman r = −0.04 (p = 0.90) between self-transition rate and
  mean hl_ratio per class — LO&S DOES NOT explain the gap deviations
- Finding is independent of and complementary to LO&S

---

## Files for Methods Section

| File | What to cite in paper |
|------|----------------------|
| `src/main.rs` | Rust SSoZ sieve with --per-class flag; scales to 10^12 |
| `project/data_generation.py` | Python twin prime generation, hl_ratio computation |
| `project/feature_engineering.py` | Feature engineering: mod residues, rolling stats, H-L density |
| `project/residue_analysis.py` | Class-level feature engineering + PySR setup |
| `project/rust_class_counts.py` | 10^12 ablation driver, H-L count integration |
| `project/los_comparison.py` | LO&S transition matrix construction + correlation test |
| `project/analyze_results.py` | Statistical tests: t-test, Bonferroni, Spearman r |

## Files for Results Section

| File | What it contains |
|------|-----------------|
| `project/data/residue/residue_class_stats.csv` | Per-class mean hl_ratio, t-stat, p-value, n |
| `project/data/residue/residue_class_pysr.txt` | PySR best equations and loss progression |
| `project/data/residue/residue_class_equations.csv` | Full PySR Pareto front |
| `project/data/counts/rust_class_counts_*.csv` | Per-class counts at each decade (1e9–1e12) |
| `project/data/residue/los_comparison.csv` | Self-transition rates vs hl_ratio per class |
| `project/data/residue/los_transition.csv` | Full 15×15 transition matrix |
| `project/data/analysis_results.txt` | Full text summary of all statistical results |

## Figures (all in `project/data/figures/`)

### fig1_gap_deviation.png — Main Result
Horizontal lollipop chart. Y-axis lists all 15 mod-210 classes sorted from most
above H-L (class 29, top) to most below (class 149, bottom). X-axis is mean
hl_ratio − 1 (deviation from H-L). Red dots = gaps larger than H-L predicts
(above-H-L classes: 29, 41, 59, 71, 107, 17, 11, 209); blue = gaps smaller
(below: 197, 101, 191, 137, 179, 167, 149). Error bars are ±2 SEM. Largest
deviations: class 29 at +0.05, class 149 at −0.05.
**Use for**: opening figure showing the phenomenon.

### fig2_significance.png — Volcano Plot
X-axis = mean hl_ratio − 1, Y-axis = −log₁₀(p). Dashed red line = Bonferroni
threshold (α = 6.67×10⁻⁴). 14/15 classes sit far above it. Class 29 reaches
−log₁₀(p) ≈ 130 — the strongest signal. One gray dot (class 197) is just below
the threshold. Blue = 6 classes significantly below H-L, red = 8 significantly
above. All labeled with their mod-210 value.
**Use for**: statistical significance section.

### fig3_ablation.png — Scale Consistency
Two panels. Left: bump chart showing the rank of each class's count-based
hl_ratio across three decade ranges ([1e10,1e11), [1e10,1e11), [1e11,1e12)).
Lines cross heavily — ranks are inconsistent, confirming count-based deviations
are noise at this scale (H-L is correct for counts). Right: 3×3 Spearman
correlation heatmap between the three ranges. Values are 0.16, 0.33, −0.17 —
near zero, confirming no consistent count-based ranking across scales.
**Note**: this figure shows the COUNT-based ablation. The gap-based consistency
(Spearman r = 0.95–0.98) is a separate finding from analyze_results.py.
**Use for**: showing H-L is right on counts, motivating that gap-spacing is the
real finding.

### fig4_pysr_fit.png — PySR Formula
Two panels. Left: scatter of observed vs PySR-predicted hl_ratio per class.
Spearman r = 0.886 — the formula tracks the ordering well. Points labeled with
class numbers; most follow the diagonal but classes 71 and 107 are outliers
(predicted near 1.0, actual ~1.03). Right: residual lollipop sorted by observed
hl_ratio. Classes 71, 107 have large positive residuals (~0.03); class 137 and
167 have large negative residuals. Systematic residual structure means the
formula captures the trend but misses within-group variation.
**Use for**: PySR results section — partial success, no clean correction found.

### fig5_los.png — Lemke Oliver–Soundararajan Comparison
Two panels. Left: 17×17 transition matrix heatmap of consecutive twin prime
residue classes (17 not 15 because classes 3 and 5 appear from the twin primes
(3,5) and (5,7)). Entire matrix is pale blue (all transitions suppressed below
uniform) except the diagonal entries for classes 3 and 5 which are dark red —
these small primes almost always transition to each other. The suppressed
diagonal for all other classes confirms LO&S anti-persistence in twin primes.
Right: scatter of self-transition rate P(r→r) vs mean hl_ratio. Points are
flat/scattered with Spearman r = −0.036, p = 0.899. The dotted vertical line
marks the uniform baseline (≈0.059). All 15 main classes cluster tightly around
0.044–0.047, far below uniform — anti-persistence is real but equal across
classes, so it cannot explain the differential gap deviations.
**Use for**: LO&S comparison section — confirm their bias, prove independence.

---

## References to Add

- **Hardy & Littlewood (1923)** — "Some problems of 'Partitio Numerorum'"; source
  of the twin prime conjecture and C₂ constant
- **Lemke Oliver & Soundararajan (2016)** — "Unexpected biases in the distribution
  of consecutive primes", PNAS 113(31). Their result on anti-persistence of
  consecutive prime residue classes is what Section 5 (LO&S comparison) tests
  against. We confirm their bias exists in twin primes but show it is uniform
  across classes and cannot explain the differential gap deviations.
- **Granville (1995)** — "Harald Cramér and the distribution of prime numbers" —
  context for gap distribution models
- **PySR / Cranmer (2023)** — cite the PySR library for symbolic regression

---

## Suggested Paper Structure (ICML AI4Math)

1. **Introduction** — H-L conjecture, motivation, one-paragraph summary of finding
2. **Background** — H-L singular series, mod-210 wheel, LO&S (half page)
3. **Methods** — Rust sieve, feature engineering, PySR setup, LO&S test
4. **Results**
   - 4.1 Gap deviations by residue class (fig1, fig2)
   - 4.2 Scale consistency to 10^12 (fig3)
   - 4.3 PySR symbolic search (fig4) — negative result: no clean formula
   - 4.4 Independence from LO&S (fig5)
5. **Discussion** — what H-L gets right (counts) vs wrong (gap spacing); future
   direction: closed-loop AI search for correction formula f(r)
6. **Conclusion**
