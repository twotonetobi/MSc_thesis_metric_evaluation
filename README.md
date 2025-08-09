# Evaluation Framework for Music-Driven Light Show Generation

## Master Thesis Context

This repository contains the evaluation framework developed for the master thesis:

**"Generative Synthesis of Music-Driven Light Shows: A Framework for Co-Creative Stage Lighting"**  
*Author: Tobias Wursthorn*  
*HAW Hamburg, Department of Media Technology, 2025*

This framework implements the quantitative evaluation methodology described in Chapter 3.5 and Chapter 4 of the thesis, evaluating TWO different generative approaches:

1. **Intention-Based (Diffusion Model)**: Continuous parameter representation (72-dimensional)
2. **Oscillator-Based (Conformer Model)**: Function generator approach with wave-type classification (60-dimensional)

## Overview

This evaluation framework provides comprehensive metrics for assessing AI-generated lighting sequences across two paradigms:

### A. Intention-Based Evaluation (Diffusion Model)
Measures structural and temporal correspondence between generated lighting and driving audio:
- **Structural Coherence**: SSM correlation, novelty detection, boundary alignment
- **Rhythmic Synchronization**: Context-aware beat-to-peak/valley alignment
- **Dynamic Response**: RMS-brightness and onset-change correlations
- **Variance Metrics**: Intensity and color variation analysis

### B. Oscillator-Based Evaluation (Conformer Model)
Assesses generative quality without ground truth through:
- **Plausibility**: Parameter distributions compared to training data
- **Musical Coherence**: Segment-appropriate patterns and conventions
- **Internal Consistency**: Stability within segments, dynamics across
- **Inter-Group Coordination**: Relationships between lighting groups

## Repository Structure

```
evaluation/
├── data/
│   ├── edge_intention/              # Intention-based dataset
│   │   ├── audio/                   # Audio feature extractions (*.pkl)
│   │   └── light/                   # Lighting intention sequences (*.pkl)
│   │
│   ├── conformer_osci/              # Oscillator-based dataset
│   │   ├── audio_90s/               # 90-second audio features
│   │   ├── audio_segments_information_jsons/  # Segment boundaries & BPM
│   │   └── light_segments/          # Predicted oscillator parameters
│   │
│   ├── training_data/               # For oscillator comparison
│   │   ├── oscillator_params/       # Training data oscillator params
│   │   └── statistics/              # Pre-computed statistics
│   │
│   ├── baselines/                   # Baseline predictions
│   │   ├── random/                  # Random parameters
│   │   ├── beat_sync/               # Simple beat-synchronized
│   │   └── constant/                # Constant parameters
│   │
│   └── beat_configs/                # Tuned parameter configurations
│
├── scripts/
│   ├── # Intention-based evaluation
│   ├── structural_evaluator.py      # Core metrics for intention-based
│   ├── evaluate_dataset.py          # Dataset-wide evaluation runner
│   ├── evaluate_dataset_with_tuned_params.py  # With tuned parameters
│   ├── enhanced_tuner.py            # Interactive GUI for parameter tuning
│   ├── visualizer.py                # Plotting utilities
│   ├── generate_final_plots.py      # Summary visualization generator
│   │
│   ├── # Oscillator-based evaluation
│   ├── oscillator_evaluator.py      # Core metrics for oscillator-based
│   ├── training_stats_extractor.py  # Extract training data statistics
│   ├── baseline_generators.py       # Generate baseline predictions
│   ├── inter_group_analyzer.py      # Analyze inter-group correlations
│   └── oscillator_report_generator.py  # Generate comprehensive report
│
├── outputs/                         # Intention-based results
├── outputs_oscillator/              # Oscillator-based results
├── extracted_formulas_intention_based.md  # Mathematical formulas
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Usage restrictions
└── README.md                        # This file
```

## Installation & Setup

### 1. Create Virtual Environment

```bash
# Navigate to the evaluation directory
cd evaluation/

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test that the evaluators load correctly
python scripts/structural_evaluator.py
python scripts/oscillator_evaluator.py
```

## Evaluation Workflows

### Part A: Intention-Based Evaluation (Diffusion Model)

This evaluates the 72-dimensional continuous representation from the diffusion model.

#### Quick Start (Default Parameters)

```bash
# Evaluate the entire dataset with default parameters
python scripts/evaluate_dataset.py \
    --data_dir data/edge_intention \
    --output_dir outputs

# Generate summary visualizations
python scripts/generate_final_plots.py
```

#### Advanced: Parameter Tuning

For optimal results, tune the evaluation parameters using the interactive GUI:

```bash
# Launch the interactive tuner
python scripts/enhanced_tuner.py
```

This allows you to:
- Load audio/light pairs and visualize correspondence
- Adjust rhythmic detection thresholds
- Fine-tune beat alignment parameters
- Save optimized configurations

#### Evaluation with Tuned Parameters

```bash
# Use your tuned configuration
python scripts/evaluate_dataset_with_tuned_params.py \
    data/beat_configs/your_config.json \
    --data_dir data/edge_intention \
    --output_dir outputs_tuned
```

### Part B: Oscillator-Based Evaluation (Conformer Model)

This evaluates the 60-dimensional oscillator parameters (3 groups × 20 params) from the conformer model.

#### Step 1: Extract Training Statistics (Run Once)

First, extract statistics from your training data to establish baselines:

```bash
python scripts/training_stats_extractor.py \
    --training_dir /path/to/your/training/oscillator/data \
    --segment_dir data/conformer_osci/audio_segments_information_jsons \
    --output_dir data/training_data/statistics
```

This creates distributions and conventions from your training set for comparison.

#### Step 2: Generate Baseline Predictions

Create simple baseline predictions for comparison:

```bash
python scripts/baseline_generators.py \
    --audio_dir data/conformer_osci/audio_segments_information_jsons \
    --stats_path data/training_data/statistics/parameter_distributions.pkl \
    --output_dir data/baselines
```

This generates three baseline types:
- **Random**: Parameters sampled from training distributions
- **Beat-sync**: Simple on-beat flashing patterns
- **Constant**: Static, unchanging parameters

#### Step 3: Evaluate Model Predictions

Evaluate your conformer model's predictions:

```bash
python scripts/oscillator_evaluator.py \
    --pred_dir data/conformer_osci/light_segments \
    --audio_dir data/conformer_osci/audio_segments_information_jsons \
    --stats_path data/training_data/statistics/parameter_distributions.pkl \
    --output_dir outputs_oscillator
```

#### Step 4: Evaluate Baselines for Comparison

Run the same evaluation on baselines:

```bash
# Evaluate random baseline
python scripts/oscillator_evaluator.py \
    --pred_dir data/baselines/random \
    --audio_dir data/conformer_osci/audio_segments_information_jsons \
    --output_dir outputs_oscillator/baselines/random

# Evaluate beat-sync baseline
python scripts/oscillator_evaluator.py \
    --pred_dir data/baselines/beat_sync \
    --audio_dir data/conformer_osci/audio_segments_information_jsons \
    --output_dir outputs_oscillator/baselines/beat_sync

# Evaluate constant baseline
python scripts/oscillator_evaluator.py \
    --pred_dir data/baselines/constant \
    --audio_dir data/conformer_osci/audio_segments_information_jsons \
    --output_dir outputs_oscillator/baselines/constant
```

#### Step 5: Analyze Inter-Group Correlations

Examine how the three lighting groups coordinate:

```bash
python scripts/inter_group_analyzer.py \
    --pred_dir data/conformer_osci/light_segments \
    --output_dir outputs_oscillator/inter_group
```

#### Step 6: Generate Comprehensive Report

Combine all analyses into a final report:

```bash
python scripts/oscillator_report_generator.py \
    --model_dir outputs_oscillator \
    --baseline_dir outputs_oscillator/baselines \
    --output_path outputs_oscillator/reports/final_report.md
```

## Output Files

### Intention-Based Outputs

Located in `outputs/`:
- `reports/metrics.csv` - Per-file metric results
- `reports/evaluation_report.md` - Comprehensive markdown report
- `plots/ssm/` - Self-similarity matrix visualizations
- `plots/novelty/` - Novelty function comparisons
- `plots/metrics/` - Summary statistics and distributions

### Oscillator-Based Outputs

Located in `outputs_oscillator/`:
- `metrics/` - Parameter distribution comparisons
- `plots/wave_distributions.png` - Wave type usage by segment
- `plots/parameter_comparison.png` - Parameter statistics across segments
- `inter_group/` - Inter-group correlation analyses
- `reports/final_report.md` - Comprehensive evaluation report

## Evaluation Metrics

### Intention-Based Metrics (72-dim continuous)

| Metric | Symbol | Description |
|--------|--------|-------------|
| SSM Correlation | Γ_structure | Structural correspondence via self-similarity |
| Novelty Correlation | Γ_novelty | Alignment of structural transitions |
| Boundary F-Score | Γ_boundary | Musical segment boundary detection |
| RMS↔Brightness | Γ_loud↔bright | Audio energy to lighting intensity |
| Onset↔Change | Γ_change | Musical onsets to lighting changes |
| Beat↔Peak | Γ_beat↔peak | Beat alignment with intensity peaks |
| Beat↔Valley | Γ_beat↔valley | Beat alignment with intensity valleys |
| Intensity Variance | Ψ_intensity | Variation in lighting intensity |
| Color Variance | Ψ_color | Variation in color parameters |

### Oscillator-Based Metrics (60-dim wave parameters)

| Metric | Description |
|--------|-------------|
| Parameter Plausibility | Distribution similarity to training data (KL divergence) |
| Wave Type Convention | Adherence to learned segment-wave associations |
| Segment Consistency | Parameter stability within musical segments |
| Inter-Group Correlation | Coordination between 3 lighting groups |
| MAI (Movement Activity) | Combined pan/tilt activity measure |
| Baseline Comparison | Performance vs random/simple approaches |

## Data Formats

### Intention-Based Format
- **Audio**: Pickle files with feature dictionaries (chroma_stft, onset_beat, etc.)
- **Light**: NumPy arrays of shape `(T, 72)` where T = time frames
  - 72 dims = 12 groups × 6 parameters (intensity, position, density, minima, hue, saturation)

### Oscillator-Based Format
- **Audio**: 90-second segments with JSON metadata (BPM, beats, segments)
- **Light**: NumPy arrays of shape `(2700, 60)` for 90s at 30fps
  - 60 dims = 3 groups × 20 params (10 standard + 10 highlight)
  - Parameters: pan/tilt activity, wave types, frequency, amplitude, offset, phase, hue, saturation

### Wave Type Mappings (Oscillator)

**Wave Type A:**
- sine: 0.1
- saw_up: 0.3
- saw_down: 0.5
- square: 0.7
- linear: 0.9

**Wave Type B:**
- other: 0.125
- plateau: 0.375
- gaussian_single: 0.625
- gaussian_double: 0.875

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all packages from `requirements.txt` are installed
2. **Memory errors**: Reduce batch size or process fewer files at once
3. **Missing files**: Check that light pickle filenames match audio file stems
4. **No training data**: For oscillator evaluation, you must first extract training statistics

### Data Requirements

- **Intention-based**: Requires paired audio/light pickles with matching stems
- **Oscillator-based**: Requires 90s segments with corresponding JSON metadata

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@mastersthesis{wursthorn2025generative,
  title={Generative Synthesis of Music-Driven Light Shows: 
         A Framework for Co-Creative Stage Lighting},
  author={Wursthorn, Tobias},
  year={2025},
  school={HAW Hamburg, Department of Media Technology}
}
```

## License & Usage

This framework is provided for SCIENTIFIC and EDUCATIONAL purposes only. See LICENSE file for full restrictions.

## Acknowledgments

This work was supported by:
- Prof. Dr. Larissa Putzar (Primary Supervisor)
- Prof. Dr. Kai von Luck (Secondary Supervisor)
- Anonymous lighting designers who provided training data
- MA Lighting (https://www.malighting.com/)
- The open-source community for essential libraries

---

**Note**: This is research software provided as-is for academic purposes. The evaluation framework demonstrates two complementary approaches to lighting generation evaluation, each suited to its respective model architecture.