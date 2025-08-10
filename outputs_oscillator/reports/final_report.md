# Oscillator-Based Lighting Generation Evaluation Report
**Generated:** 2025-08-10 11:03:18

## Executive Summary
This report evaluates the oscillator-based lighting generation model against training data distributions and baseline methods.

### Key Findings:
- Analyzed 4 different segment types
- Evaluated 263 files

### Performance vs Baselines:
- Detailed baseline comparisons in sections below

## Parameter Distribution Analysis
### Global Parameter Statistics

| Parameter | Training Mean ± Std | Model Mean ± Std | Difference |
|-----------|---------------------|------------------|------------|
| pan_activity | 0.069 ± 0.173 | TBD | TBD |
| tilt_activity | 0.080 ± 0.183 | TBD | TBD |
| wave_type_a | 0.100 ± 0.000 | TBD | TBD |
| wave_type_b | 0.000 ± 0.000 | TBD | TBD |
| frequency | 0.216 ± 0.295 | TBD | TBD |
| amplitude | 0.440 ± 0.437 | TBD | TBD |
| offset | 0.112 ± 0.198 | TBD | TBD |
| phase | 0.492 ± 0.322 | TBD | TBD |
| col_hue | 0.122 ± 0.233 | TBD | TBD |
| col_sat | 0.298 ± 0.436 | TBD | TBD |

*Note: Model statistics extraction to be implemented based on evaluation results structure.*

## Musical Convention Adherence
### Wave Type Usage by Segment

#### Intro
- **Wave Types Used:**
  - sine: 100.0%
- **Mean Amplitude:** 0.260
- **Mean Frequency:** 0.349
- **Movement Activity:** 0.301

#### Verse
- **Wave Types Used:**
  - sine: 100.0%
- **Mean Amplitude:** 0.268
- **Mean Frequency:** 0.377
- **Movement Activity:** 0.327

#### Chorus
- **Wave Types Used:**
  - sine: 100.0%
- **Mean Amplitude:** 0.252
- **Mean Frequency:** 0.418
- **Movement Activity:** 0.271

#### Instrumental
- **Wave Types Used:**
  - sine: 100.0%
- **Mean Amplitude:** 0.248
- **Mean Frequency:** 0.317
- **Movement Activity:** 0.205


## Baseline Comparison
### Performance Comparison

| Metric | Model | Random | Beat-Sync | Constant |
|--------|-------|--------|-----------|----------|
| Plausibility | TBD | TBD | TBD | TBD |
| Consistency | TBD | TBD | TBD | TBD |
| Musical Coherence | TBD | TBD | TBD | TBD |

*Note: Baseline comparison metrics to be populated from evaluation results.*

## Inter-Group Coordination
### Group Coordination Analysis

- **Mean Inter-Group Correlation:** 0.166 ± 0.417
- **Mean Phase Difference:** 0.083 radians
- **Complementary Dynamics Score:** -0.236

### Correlation Distribution
- Strong (>0.7): 13.4%
- Moderate (0.3-0.7): 38.0%
- Independent (<0.3): 48.6%

### Interpretation
The lighting groups operate **mostly independently**, providing diverse visual elements.
