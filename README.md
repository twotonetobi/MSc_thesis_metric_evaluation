# Evaluation Framework for Music-Driven Light Show Generation

## Master Thesis Context

This repository contains the evaluation framework developed for the master thesis:

**"Generative Synthesis of Music-Driven Light Shows: A Framework for Co-Creative Stage Lighting"**  
*Author: Tobias Wursthorn*  
*HAW Hamburg, Department of Media Technology, 2025*

This framework implements the quantitative evaluation methodology described in Chapter 3.5 and Chapter 4 of the thesis, specifically designed to assess the structural and temporal correspondence between generated lighting sequences and their driving audio.

## Purpose

This evaluation framework measures how well AI-generated lighting sequences align with musical features across multiple dimensions:
- **Structural Coherence**: SSM correlation, novelty detection, boundary alignment
- **Rhythmic Synchronization**: Beat-to-peak/valley alignment with rhythmic intent detection
- **Dynamic Response**: RMS-brightness and onset-change correlations
- **Variance Metrics**: Intensity and color variation analysis

The framework operates on the intention-based abstraction layer (72-dimensional lighting representation) as described in Section 3.3.3 of the thesis.

## Repository Structure

```
evaluation/
├── data/
│   ├── edge_intention/         # Primary dataset (intention-based)
│   │   ├── audio/              # Audio feature extractions (*.pkl)
│   │   └── light/              # Lighting intention sequences (*.pkl)
│   └── beat_configs/           # Tuned parameter configurations
├── scripts/
│   ├── structural_evaluator.py # Core evaluation metrics implementation
│   ├── evaluate_dataset.py     # Dataset-wide evaluation runner
│   ├── evaluate_dataset_with_tuned_params.py # Evaluation with tuned parameters
│   ├── enhanced_tuner.py       # Interactive GUI for parameter tuning
│   ├── visualizer.py           # Plotting utilities
│   └── generate_final_plots.py # Summary visualization generator
├── outputs/                    # Generated results (created on first run)
├── formulas/                   # Mathematical formulas documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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
# Test that the evaluator loads correctly
python scripts/structural_evaluator.py
```

## Usage

### Quick Start: Run Evaluation with Default Parameters

```bash
# Evaluate the entire dataset
python scripts/evaluate_dataset.py \
    --data_dir data/edge_intention \
    --output_dir outputs
```

### Advanced: Interactive Parameter Tuning

The framework includes an interactive GUI for tuning evaluation parameters to your specific dataset:

```bash
# Launch the interactive tuner
python scripts/enhanced_tuner.py
```

This allows you to:
- Load audio/light pairs and visualize their correspondence
- Adjust rhythmic detection thresholds
- Fine-tune beat alignment parameters
- Save optimized configurations

### Run Evaluation with Tuned Parameters

After tuning, use your saved configuration:

```bash
# Use tuned parameters for evaluation
python scripts/evaluate_dataset_with_tuned_params.py \
    data/beat_configs/your_config.json \
    --data_dir data/edge_intention \
    --output_dir outputs_tuned
```

### Generate Summary Visualizations

```bash
# Create metric distribution plots and correlation heatmaps
python scripts/generate_final_plots.py
```

## Output Files

The evaluation generates several types of outputs:

- `outputs/reports/metrics.csv` - Per-file metric results
- `outputs/reports/evaluation_report.md` - Comprehensive markdown report
- `outputs/plots/ssm/` - Self-similarity matrix visualizations
- `outputs/plots/novelty/` - Novelty function comparisons
- `outputs/plots/metrics/` - Summary statistics and distributions

## Evaluation Metrics

The framework computes the following metrics (as defined in Section 3.5.2 of the thesis):

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

## Data Format

### Audio Features (pickle format)
- `chroma_stft`: Chroma features for harmonic content
- `onset_beat`: Binary beat positions
- `onset_env`: Onset strength envelope
- `rms` or `melspe_db`: Energy/loudness features

### Lighting Features (pickle format)
- NumPy array of shape `(T, 72)` where T = time frames
- 72 dimensions = 12 groups × 6 parameters per group
- Parameters: intensity peak, slope, density, minima, hue, saturation

---

## COPYRIGHT NOTICE & DATA USAGE

### Audio Feature Extractions

This repository contains **audio feature extractions** (not original audio files) derived from copyrighted musical works. These extractions are:

1. **Transformative**: The data consists solely of numerical feature representations (MFCCs, chroma, beat positions, etc.) that cannot be used to reconstruct the original audio
2. **For Scientific Research**: Used exclusively for academic research purposes as part of a master thesis
3. **Non-Commercial**: Not intended for any commercial use or distribution
4. **Educational Fair Use**: Provided to enable reproducibility and understanding of the evaluation methodology

The author (Tobias Wursthorn) has legally obtained the original audio content through commercial purchases. The feature extractions are shared here solely to:
- Enable scientific reproducibility of the thesis results
- Allow understanding of the evaluation pipeline
- Facilitate academic peer review

### Lighting Data

The lighting control data represents original creative work by professional lighting designers who have provided their data under strict non-disclosure agreements (NDAs) with guaranteed anonymity. This data is:
- Proprietary intellectual property of the respective designers
- Shared only in abstracted form (intention layer)
- Not to be used for any commercial purposes
- Subject to the licensing terms below

---

## LICENSE & USAGE RESTRICTIONS

### Scientific Research License

This evaluation framework and associated data are released under the following terms:

**Permitted Uses:**
- Academic and scientific research
- Educational purposes in academic institutions
- Reproducibility studies of the thesis results
- Non-commercial evaluation of music-to-light generation systems

**Prohibited Uses:**
- Any commercial use or application
- Training machine learning models for commercial products
- Redistribution of the data without explicit permission
- Reverse engineering to identify anonymous contributors
- Any use that violates copyright or intellectual property rights

**Attribution Requirement:**
Any use of this framework or data must cite:
```
Wursthorn, T. (2025). "Generative Synthesis of Music-Driven Light Shows: 
A Framework for Co-Creative Stage Lighting." Master Thesis, 
HAW Hamburg, Department of Media Technology.
```

**Data Access:**
For access to the full dataset or any questions regarding usage, please contact:
- Author: Tobias Wursthorn
- Institution: HAW Hamburg
- [tobias.wursthorn@haw-hamburg.de](mailto:tobias.wursthorn@haw-hamburg.de)

### Third-Party Components

This framework uses open-source libraries (numpy, scipy, pandas, matplotlib, etc.) under their respective licenses. See `requirements.txt` for the complete list.

---

## Technical Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- Storage: ~500MB for code and example data

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all packages from `requirements.txt` are installed
2. **Memory errors**: Reduce batch size or downsample data
3. **No light file found**: Check that light pickle filenames contain the audio file stem
4. **mir_eval missing**: Install separately with `pip install mir_eval` if needed

### Parameter Tuning Tips

- Start with the default configuration
- Use the interactive tuner on representative samples
- Rhythmic threshold typically ranges from 0.03-0.08
- Beat sigma of 0.5 works well for most genres

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

## Acknowledgments

This work was supported by:
- Prof. Dr. Larissa Putzar (Primary Supervisor)
- Prof. Dr. Kai von Luck (Secondary Supervisor)
- The anonymous lighting designers who provided their creative work for research
- MA Lighting (https://www.malighting.com/)
- The open-source community for the essential libraries used in this framework

---

**Note**: This is research software. While efforts have been made to ensure correctness, it is provided as-is for academic purposes.
