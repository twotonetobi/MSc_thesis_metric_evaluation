# Evaluation Framework for Music-Driven Light Show Generation

## Master Thesis Implementation by Tobias Wursthorn

This repository contains the comprehensive evaluation framework developed for the master thesis "Generative Synthesis of Music-Driven Light Shows: A Framework for Co-Creative Stage Lighting" by Tobias Wursthorn, HAW Hamburg, 2025.

## Overview

Assessing generative systems for creative applications such as lighting design presents inherent challenges. Quantitative assessments provide objective measurements and systematic model comparisons, but may fail to capture subjective artistic characteristics that define successful light shows. Conversely, relying solely on qualitative feedback reduces the replicability necessary for rigorous scientific research.

This framework employs a comprehensive, mixed-method evaluation methodology that addresses these challenges through three distinct quantitative methods, each exploring different facets of generative quality. The quantitative methods are supplemented by qualitative analysis derived from expert interviews.

## Three-Component Evaluation Methodology

### I. Intention-Based Structural and Temporal Analysis

Assesses the internal coherence and musical alignment of generated output by analyzing the correspondence between continuous lighting parameters and input audio features. This evaluation operates without reference to ground truth data.

**Metrics evaluated:**
- **Structural Correspondence**: SSM Correlation, Novelty Correlation (functional quality approach)
- **Rhythmic and Temporal Alignment**: Onset-Change Correlation, Beat-Peak Alignment, Beat-Valley Alignment
- **Dynamic Variation**: RMS-Brightness Correlation (functional quality approach), Intensity Variance

**Key methodological consideration**: The framework introduces "functional quality" metrics for novelty and RMS correlation. Traditional Pearson correlation is sensitive to phase differences—professional lighting designers often implement artistically valid temporal offsets (anticipatory or delayed beats). The functional quality approach evaluates temporal coupling of events and correlation of magnitudes, tolerating minor temporal offsets that reflect artistic intent.

### II. Intention-Based Ground Truth Comparison

Benchmarks the functional quality of generated light shows against human-designed ground truth data, concentrating on the achievement of artistic objectives rather than statistical replication. Different designers produce different solutions that are equally valid; the model may produce coherent patterns that differ statistically from training data.

**Evaluation approach:**
- Utilizes the full suite of metrics from Section I
- Calculates achievement ratios: `median(generated) / median(ground_truth)`
- Ratios > 100% indicate stronger performance on that dimension than median human-designed shows
- May reflect training-emphasized features rather than superior artistic quality

**Quality achievement metrics:**
- Rhythmic Alignment Ratios
- Structural Correspondence Ratios
- Dynamic Variation Ratios

### III. Segment-Based Hybrid Oscillator Evaluation

Evaluates the appropriateness, consistency, and musical coherence of discrete, high-level decisions (e.g., wave type selection) made by the oscillator-based model. Rather than evaluating continuous signal fidelity, this component focuses on categorical decisions for each musical segment.

**Hybrid Dynamic Scoring Method:**
Synthesizes information from continuous intention-based streams and oscillator-based data to determine single, artistically meaningful decisions. Maps composite dynamic scores to discrete wave type labels via empirically tuned thresholds.

**Quality metrics:**
- **Consistency**: Temporal stability of wave type decisions within musical segments
- **Musical Coherence**: Appropriateness of wave type for calculated dynamic score (tests music-to-visual complexity mapping)
- **Transition Smoothness**: Quality of transitions between segments
- **Distribution Match**: Alignment of generated wave type distribution with ground truth patterns

**Wave complexity hierarchy:**
```
Still (0.0) → Sine (0.2) → Odd/Even (0.3) → Square (0.5) →
PWM Basic (0.6) → PWM Extended (0.8) → Random (1.0)
```

## Overall Quality Score

The framework calculates a consolidated Overall Quality Score as a weighted aggregate of achievement ratios from the three evaluation areas:

**Weighting structure:**
- 16% weight: Intention-Based Structural and Temporal Analysis (internal musical alignment without ground truth reference)
- 42% weight: Intention-Based Ground Truth Comparison (benchmarks continuous features against professional designs)
- 42% weight: Segment-Based Hybrid Oscillator Evaluation (evaluates discrete decisions against ground truth)

**Calculation:**
```
Score_quality = Σ(w_i × min(1.0, Ratio_i))
```

Where individual ratios are capped at 1.0 to prevent single metrics with exceptionally high ratios from disproportionately inflating the overall score.

**Classification levels:**
- Excellent (≥ 0.9): Performance closely matches or exceeds ground truth median
- Good (≥ 0.7): Strong performance, slightly below median
- Moderate (≥ 0.5): Acceptable performance with noticeable deviation
- Acceptable (≥ 0.3): Meets minimum criteria, requires improvement
- Needs Improvement (< 0.3): Significant divergence from professional standards

## Methodological Considerations

### Functional Quality Approach

Traditional correlation metrics assume parallel relationships and are sensitive to phase differences. The functional quality approach addresses these limitations:

**For Novelty Correlation:**
- Piecewise transformation based on correlation strength
- Strong correlation (|ρ| ≥ 0.15): Generous scaling with 0.8 cap
- Moderate coupling (0.05 ≤ |ρ| < 0.15): Baseline 0.4 with bonus
- Minimal coupling (< 0.05): Guaranteed minimum 0.1 to avoid penalizing intentional artistic choices

**For RMS-Brightness Correlation:**
- Evaluates both temporal coupling and magnitude correlation
- Temporal Coupling Rate: Proportion of audio energy changes met with brightness change within ±0.3s window
- Final score: 60% temporal coupling + 40% magnitude correlation

### Artistic Validity vs. Correlation

Professional lighting design involves creative choices that may dissociate from musical structure:
- Atmospheric suggestions with slow washes may purposefully disrupt musical structures
- Functions without beat synchronization as separate signals to performers
- Counterpoint: loud music with dim light for dramatic effect

These artistic intentions may yield lower correlation scores but do not represent lower quality. The framework must be interpreted cautiously, underscoring why comparison to ground truth data is necessary for complete assessment.

## Requirements

**This project requires a virtual environment.**

### Installing Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Main Dependencies

- numpy >= 2.0.0 - Core numerical computing (NumPy 2.x compatible)
- scipy >= 1.11.0 - Scientific computing
- pandas >= 2.0.0 - Data analysis
- matplotlib >= 3.8.0 - Visualization
- librosa >= 0.10.0 - Audio processing
- mir_eval >= 0.7 - Music information retrieval evaluation
- scikit-learn >= 1.3.0 - Machine learning utilities

See `requirements.txt` for complete list.

## Data Structure

The evaluation framework expects the following directory structure:

```
data/
├── edge_intention/
│   ├── audio/                      # Generated audio features (pkl)
│   ├── light/                      # Generated light parameters (pkl)
│   ├── audio_ground_truth/         # Training audio features (pkl)
│   └── light_ground_truth/         # Training light parameters (pkl)
└── conformer_osci/                 # For hybrid evaluation
    ├── light_segments/             # Oscillator parameters (pkl)
    └── audio_segments_information_jsons/  # Audio metadata (json)
```

## Running the Evaluation

### Complete Evaluation

Execute all three evaluation methodologies:

```bash
python scripts/thesis_workflow.py --data_dir data/edge_intention
```

### Individual Evaluations

**Intention-Based Structural Analysis:**
```bash
python scripts/intention_based/evaluate_dataset.py \
    --data_dir data/edge_intention \
    --output_dir outputs/intention_only
```

**Ground Truth Comparison:**
```bash
python scripts/intention_based_ground_truth_comparison/quality_based_comparator_optimized.py \
    --data_dir data/edge_intention \
    --output_dir outputs/quality_only
```

**Hybrid Oscillator Evaluation:**
```bash
# Step 1: Reconstruct wave types
python scripts/segment_based_hybrid_oscillator_evaluation/wave_type_reconstructor.py \
    --pas_dir data/edge_intention/light \
    --geo_dir data/conformer_osci/light_segments \
    --config configs/final_optimal.json

# Step 2: Evaluate decisions
python scripts/segment_based_hybrid_oscillator_evaluation/hybrid_evaluator.py

# Step 3: Visualize results
python scripts/segment_based_hybrid_oscillator_evaluation/wave_type_visualizer.py
```

## Output Structure

```
outputs/thesis_complete/run_YYYYMMDD_HHMMSS/
├── data/
│   ├── intention_based_metrics.csv
│   ├── ground_truth_comparison.json
│   └── hybrid_oscillator_results.pkl
│
├── plots/
│   ├── I_intention_based/
│   │   ├── structural_correspondence/
│   │   ├── rhythmic_temporal_alignment/
│   │   └── dynamic_variation/
│   │
│   ├── II_ground_truth_comparison/
│   │   ├── achievement_ratios.png
│   │   └── quality_breakdown.png
│   │
│   └── III_hybrid_oscillator/
│       ├── consistency.png
│       ├── musical_coherence.png
│       ├── transition_smoothness.png
│       └── wave_distribution.png
│
└── reports/
    ├── comprehensive_evaluation_report.md
    └── evaluation_metrics.json
```

## Repository Structure

```
scripts/
├── thesis_workflow.py                      # Main orchestrator
│
├── intention_based/                        # Section I evaluation
│   ├── structural_evaluator.py
│   ├── evaluate_dataset.py
│   ├── enhanced_tuner.py
│   └── boundary_tuner.py
│
├── intention_based_ground_truth_comparison/  # Section II evaluation
│   ├── quality_based_comparator_optimized.py
│   ├── ground_truth_visualizer.py
│   └── visualize_paradigm_comparison.py
│
├── segment_based_hybrid_oscillator_evaluation/  # Section III evaluation
│   ├── wave_type_reconstructor.py
│   ├── hybrid_evaluator.py
│   └── wave_type_visualizer.py
│
└── helpers/                                # Utilities
    ├── thesis_plot_generator.py
    └── visualizer.py
```

## Mathematical Foundations

### Self-Similarity Matrix (SSM) Computation

**Similarity calculation:**
```
S(i,j) = 1 - ||V_i - V_j||_2 / √d
```

Where V_i and V_j are feature vectors (chroma for audio, 72-dimensional intention for lighting) at frames i and j, and d is feature space dimensionality.

**SSM Correlation:**
```
Γ_structure = Pearson(S_audio.flatten(), S_light.flatten())
```

### Novelty Function

**Gaussian Checkerboard Kernel:**
```
K(i,j) = sign(i) × sign(j) × exp(-(i² + j²)/(2(L×σ)²))
```

**Novelty computation:**
```
nov(n) = Σ S_padded[n-L:n+L+1, n-L:n+L+1] ⊙ K
```

### Beat Alignment

**Gaussian alignment score:**
```
score = Σ exp(-(d(peak, nearest_beat)²)/(2σ²))
```

For each peak in rhythmic sections, where d is distance in frames and σ is alignment tolerance (0.1 seconds ≈ 3 frames at 30fps).

### Achievement Ratio

**For any metric M:**
```
Achievement_Ratio_M = median(M_generated) / median(M_ground_truth)
```

## Technical Notes

### Execution Time

- Complete workflow: 15-30 minutes (full dataset)
- Intention-based evaluation: ~20 minutes
- Ground truth comparison: ~5 minutes
- Hybrid oscillator evaluation: ~10 minutes

### Important Implementation Details

- **Dataset processing**: Never use `max_files` parameter unless explicitly testing; processes full dataset for valid distribution analysis
- **NumPy compatibility**: Framework requires NumPy 2.x; all dependencies updated accordingly
- **Memory considerations**: For large datasets, process in smaller batches using `--batch-size` parameter

## Methodological Artifacts

Definition: Measurement errors arising from evaluation methodology flaws rather than system performance issues.

**Identified artifacts:**
- Low novelty correlation due to phase sensitivity → addressed with functional quality approach
- Negative RMS-brightness correlation interpreted as failure → actually indicates sophisticated counterpoint usage
- Boundary detection issues in segmentation algorithms → metric excluded from final evaluation

## License

This framework is provided for scientific and educational purposes. Commercial use is prohibited.

## Citation

```bibtex
@mastersthesis{wursthorn2025generative,
  title={Generative Synthesis of Music-Driven Light Shows:
         A Framework for Co-Creative Stage Lighting},
  author={Wursthorn, Tobias},
  year={2025},
  school={HAW Hamburg, Department of Media Technology}
}
```

## Acknowledgements

**Academic Supervision:**
- Prof. Dr. Larissa Putzar (Primary Supervisor)
- Prof. Dr. Kai von Luck (Secondary Supervisor)

**Industry Collaboration:**
- MA Lighting for professional lighting expertise
- Professional lighting designers who provided training data

## Additional Documentation

- `metrics.md` - Comprehensive mathematical formulas and implementation details
- `metrics_functional_quality_explained.md` - Detailed explanation of functional quality approach
- `configs/` - Configuration files for wave type classification thresholds
