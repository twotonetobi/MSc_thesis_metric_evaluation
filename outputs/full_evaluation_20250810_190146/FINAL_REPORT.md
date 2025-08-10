# Complete Evaluation Report

**Generated:** 2025-08-10 19:02:04
**Output Directory:** `outputs/full_evaluation_20250810_190146`

## 1. Intention-Based Evaluation (9 Metrics)

| Metric | Mean ± Std |
|--------|------------|
| Ssm Correlation | 0.397 ± 0.217 |
| Novelty Correlation | 0.022 ± 0.307 |
| Boundary F Score | 0.000 ± 0.000 |
| Rms Correlation | -0.096 ± 0.631 |
| Onset Correlation | 0.031 ± 0.545 |
| Beat Peak Alignment | 0.046 ± 0.095 |
| Beat Valley Alignment | 0.028 ± 0.087 |
| Intensity Variance | 0.224 ± 0.105 |
| Color Variance | 0.187 ± 0.101 |
| Structure Score | 0.140 ± 0.126 |
| Rhythm Score | 0.037 ± 0.065 |
| Dynamics Score | -0.032 ± 0.449 |
| Overall Score | 0.048 ± 0.156 |

**Files Evaluated:** 51

## 2. Hybrid Wave Type Evaluation

See detailed report: `hybrid/evaluation_report.md`

| Metric | Score |
|--------|-------|
| **Overall Score** | 0.679 |
| Consistency | 0.593 |
| Musical Coherence | 0.732 |
| Transition Smoothness | 0.556 |
| Distribution Match | 0.834 |

## 4. Summary

**Evaluations Completed:**
- ✅ Intention-based (9 metrics)
- ✅ Hybrid wave type

**Total Processing Time:** 0:00:18.545330

## 5. Output Files

```
outputs/full_evaluation_20250810_190146/
├── intention_based/
│   ├── reports/
│   │   ├── metrics.csv
│   │   ├── metrics.json
│   │   └── evaluation_report.md
│   └── plots/
├── hybrid/
│   ├── wave_reconstruction.pkl
│   ├── evaluation_report.md
│   └── plots/
└── ground_truth/
    ├── comparison_report.md
    ├── distribution_distances.json
    └── plots/
        └── comprehensive_dashboard.png
```
