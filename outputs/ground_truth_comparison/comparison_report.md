# Ground-Truth Fidelity Comparison Report

**Generated:** 2025-08-10 18:04:18

## Dataset Information

- **Generated Dataset:** 51 files
- **Ground Truth Dataset:** 161 files

## Metric Comparison

| Metric | Ground Truth (Mean ± Std) | Generated (Mean ± Std) | Wasserstein Distance | Quality |
|--------|---------------------------|------------------------|---------------------|----------|
| Ssm Correlation | 0.238 ± 0.166 | 0.162 ± 0.131 | 0.076 | 🔵 Good |
| Novelty Correlation | 0.343 ± 0.284 | 0.029 ± 0.221 | 0.320 | 🔴 Poor |
| Boundary F Score | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 | 🟢 Excellent |
| Rms Correlation | 0.436 ± 0.353 | 0.020 ± 0.293 | 0.416 | 🔴 Poor |
| Onset Correlation | 0.055 ± 0.437 | 0.090 ± 0.219 | 0.183 | 🔴 Poor |
| Beat Peak Alignment | 0.041 ± 0.058 | 0.049 ± 0.065 | 0.012 | 🟢 Excellent |
| Beat Valley Alignment | 0.047 ± 0.067 | 0.052 ± 0.072 | 0.009 | 🟢 Excellent |
| Intensity Variance | 0.227 ± 0.087 | 0.299 ± 0.054 | 0.075 | 🔵 Good |
| Color Variance | 0.080 ± 0.072 | 0.281 ± 0.047 | 0.201 | 🔴 Poor |

*Lower Wasserstein Distance indicates higher similarity between distributions.*

## Statistical Tests

### Kolmogorov-Smirnov Test
Tests if two samples come from the same distribution.

| Metric | KS Statistic | p-value | Interpretation |
|--------|--------------|---------|----------------|
| Ssm Correlation | 0.249 | 0.013 | Different distributions |
| Novelty Correlation | 0.515 | 0.000 | Different distributions |
| Boundary F Score | 0.000 | 1.000 | Similar distributions ✓ |
| Rms Correlation | 0.565 | 0.000 | Different distributions |
| Onset Correlation | 0.218 | 0.042 | Different distributions |
| Beat Peak Alignment | 0.158 | 0.258 | Similar distributions ✓ |
| Beat Valley Alignment | 0.117 | 0.613 | Similar distributions ✓ |
| Intensity Variance | 0.527 | 0.000 | Different distributions |
| Color Variance | 0.876 | 0.000 | Different distributions |

## Key Findings

- **Best Match:** Boundary F Score (W-distance: 0.000)
- **Worst Match:** Rms Correlation (W-distance: 0.416)

### Overall Fidelity Score: 0.283
🔴 **Poor**: The model's outputs differ significantly from the training data's structural properties.

## Recommendations

