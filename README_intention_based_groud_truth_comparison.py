# Ground-Truth Comparison Extension

## Overview

This extension adds comprehensive ground-truth comparison capabilities to the existing evaluation framework. It enables distributional comparison between generated light shows and human-designed training data, answering the key question: **"Does the model generate light shows with the same structural properties as the training data?"**

## System Architecture

```
Ground-Truth Comparison System
├── Phase 1: Reusable Evaluation Pipeline
│   ├── run_evaluation_pipeline.py
│   └── Evaluates any audio-light dataset pair
│
├── Phase 2: Comparison Orchestration
│   ├── compare_to_ground_truth.py
│   ├── Runs pipeline on both datasets
│   ├── Calculates distribution distances
│   └── Generates initial visualizations
│
└── Phase 3: Enhanced Visualization
    ├── ground_truth_visualizer.py
    ├── Creates density plots
    ├── Generates radar charts
    └── Builds comprehensive dashboard
```

## Installation

1. **Setup Directory Structure**
```bash
python scripts/setup_ground_truth_comparison.py
```

This creates the required directory structure:
```
data/edge_intention/
├── audio/                  # Generated audio features
├── light/                  # Generated light shows
├── audio_ground_truth/     # Ground-truth audio features
└── light_ground_truth/     # Ground-truth light shows
```

2. **Add Your Data**
- Place generated model outputs in `audio/` and `light/`
- Place human-designed training data in `audio_ground_truth/` and `light_ground_truth/`
- Files should be pickle (.pkl) format with matching names

## Usage

### Quick Start

Run the complete comparison pipeline:

```bash
python scripts/compare_to_ground_truth.py
```

### Step-by-Step Workflow

1. **Evaluate Generated Data Only**
```bash
python scripts/run_evaluation_pipeline.py \
  --audio_dir data/edge_intention/audio \
  --light_dir data/edge_intention/light \
  --output_csv outputs/generated_metrics.csv
```

2. **Evaluate Ground Truth Only**
```bash
python scripts/evaluate_ground_truth_only.py
```

3. **Run Full Comparison**
```bash
python scripts/compare_to_ground_truth.py \
  --data_dir data/edge_intention \
  --output_dir outputs/ground_truth_comparison
```

4. **Generate Enhanced Visualizations**
```bash
python scripts/ground_truth_visualizer.py \
  --output_dir outputs/ground_truth_comparison
```

### Advanced Options

**Use tuned beat alignment parameters:**
```bash
python scripts/compare_to_ground_truth.py \
  --config data/beat_configs/evaluator_config_20250808_185625.json
```

**Test with limited files:**
```bash
python scripts/compare_to_ground_truth.py --max_files 10
```

## Metrics and Interpretation

### Distribution Distance Metrics

1. **Wasserstein Distance (Earth Mover's Distance)**
   - Measures the "work" needed to transform one distribution into another
   - Lower values = better match
   - Interpretation:
     - < 0.05: 🟢 Excellent match
     - 0.05-0.10: 🔵 Good match
     - 0.10-0.15: 🟡 Moderate match
     - > 0.15: 🔴 Poor match

2. **Kolmogorov-Smirnov Test**
   - Tests if samples come from the same distribution
   - p-value > 0.05 suggests similar distributions

3. **Mann-Whitney U Test**
   - Non-parametric test for distribution differences
   - p-value > 0.05 suggests similar central tendencies

### Overall Fidelity Score

Calculated as: `1 - (avg_wasserstein / 0.2)`

- **> 0.8**: Excellent fidelity - model closely matches training data
- **0.6-0.8**: Good fidelity - captures most structural properties
- **0.4-0.6**: Moderate fidelity - some notable differences
- **< 0.4**: Poor fidelity - significant differences from training

## Output Files

### Reports and Data
- `comparison_report.md` - Main analysis report with findings
- `combined_metrics.csv` - All evaluation metrics for both datasets
- `distribution_distances.json` - Statistical distance measurements
- `ground_truth_metrics.csv` - Ground truth evaluation results
- `generated_metrics.csv` - Generated data evaluation results

### Visualizations
- `comparison_boxplot.png` - Side-by-side metric distributions
- `distribution_violin.png` - Distribution shape comparison
- `density_comparison.png` - Probability density overlays
- `radar_comparison.png` - Multi-metric radar chart
- `comparison_heatmap.png` - Distance metrics heatmap
- `comprehensive_dashboard.png` - All-in-one analysis view

## Key Features

### 1. Modular Design
- Reusable evaluation pipeline can process any dataset
- Easy to extend with new metrics or visualizations
- Clean separation of concerns

### 2. Statistical Rigor
- Multiple distance metrics for robust comparison
- Statistical significance testing
- Comprehensive summary statistics

### 3. Visual Excellence
- Consistent styling with hybrid evaluation system
- Publication-ready plots
- Interactive dashboard view

### 4. Smart Defaults
- Auto-detects tuned parameters
- Handles missing files gracefully
- Informative error messages

## Technical Details

### Evaluated Metrics (9 Total)

**Structural Metrics:**
- SSM Correlation (Γ_structure)
- Novelty Correlation (Γ_novelty)
- Boundary F-Score (Γ_boundary)

**Dynamic Metrics:**
- RMS-Brightness Correlation (Γ_loud↔bright)
- Onset-Change Correlation (Γ_change)

**Rhythmic Metrics:**
- Beat-Peak Alignment (Γ_beat↔peak)
- Beat-Valley Alignment (Γ_beat↔valley)

**Variance Metrics:**
- Intensity Variance (Ψ_intensity)
- Color Variance (Ψ_color)

### Aggregate Scores

- **Structure Score**: Mean of structural metrics
- **Rhythm Score**: Mean of rhythmic metrics
- **Dynamics Score**: Mean of dynamic metrics
- **Overall Score**: Mean of all aggregate scores

## Troubleshooting

### Common Issues

1. **"No light file found for audio"**
   - Ensure audio and light files have matching names
   - Check file extensions are .pkl

2. **"Distribution distances very high"**
   - Verify ground truth is representative of training data
   - Check if generated data is from the same model version
   - Ensure sufficient sample size (20+ files recommended)

3. **"KS test shows different distributions"**
   - This may be expected for some metrics
   - Focus on Wasserstein distance for similarity measure
   - Check individual metric reports for insights

### Debug Mode

Run with verbose output:
```bash
python scripts/compare_to_ground_truth.py --verbose
```

## Integration with Existing Framework

This extension seamlessly integrates with:

1. **Intention-Based Evaluation** - Uses same StructuralEvaluator
2. **Hybrid Wave Type System** - Adopts visualization style
3. **Tuned Parameters** - Auto-detects and uses beat configs

## Citation

If using this comparison framework in research:

```bibtex
@mastersthesis{wursthorn2025generative,
  title={Generative Synthesis of Music-Driven Light Shows: 
         A Framework for Co-Creative Stage Lighting},
  author={Wursthorn, Tobias},
  year={2025},
  school={HAW Hamburg, Department of Media Technology}
}
```

## License

See LICENSE file. For scientific and educational purposes only.

---

**Version:** 1.0.0  
**Author:** Extension developed based on Tobias Wursthorn's evaluation framework  
**Date:** 2025