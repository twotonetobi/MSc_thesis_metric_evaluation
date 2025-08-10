# Evaluation Framework for Hybrid Music-Driven Light Show Generation

## Master Thesis Context

This repository contains the evaluation framework developed for the master thesis:

**"Generative Synthesis of Music-Driven Light Shows: A Framework for Co-Creative Stage Lighting"**  
*Author: Tobias Wursthorn*  
*HAW Hamburg, Department of Media Technology, 2025*

## ⚠️ CRITICAL DISCOVERY: Hybrid Wave Type Architecture

**Date: August 10, 2025**

During evaluation implementation, we discovered that the wave type parameters in the oscillator representation are **placeholders** (constant values: wave_type_a=0.1, wave_type_b=0.0 in training data). The actual wave type decisions are made through a **hybrid post-processing system** that combines metrics from both the intention-based (PAS) and oscillator-based (Geo) approaches.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Audio Input                              │
└────────────────┬───────────────────────────┬────────────────────┘
                 │                           │
                 ▼                           ▼
    ┌────────────────────────┐ ┌────────────────────────┐
    │  Intention-Based (PAS)  │ │ Oscillator-Based (Geo)  │
    │    72 dimensions        │ │    60 dimensions         │
    │    12 groups × 6        │ │    3 groups × 20         │
    └────────────┬────────────┘ └────────────┬────────────┘
                 │                           │
                 └─────────┬─────────────────┘
                           │
                           ▼
              ┌─────────────────────────────┐
              │ Hybrid Wave Type Decision   │
              │ (Post-Processing Function)  │
              └─────────────────────────────┘
                           │
                           ▼
                  [Actual Wave Types]
                  - still
                  - sine
                  - pwm_basic
                  - pwm_extended
                  - odd_even
                  - square
                  - random
```

## Current Implementation State

### ✅ Completed Components

1. **Basic Framework Structure** - All directories and base files created
2. **Intention-Based Evaluation** (`structural_evaluator.py`)
   - SSM correlation metrics
   - Novelty detection
   - Beat alignment with rhythmic filtering
   - Variance metrics
   - Interactive parameter tuner GUI

3. **Training Statistics Extraction** (`training_stats_extractor.py`)
   - Fixed to handle actual data format
   - Handles 60/61 dimension issues

4. **Inter-Group Analysis** (`inter_group_analyzer.py`)
   - Analyzes coordination between 3 oscillator groups
   - Fixed type hints and dimension handling

5. **Oscillator Evaluator** (`oscillator_evaluator.py`)
   - Basic metrics computation
   - Handles 61-dimension predictions (uses first 60)
   - **BUT: Currently shows all wave types as 100% sine due to placeholder values**

6. **Baseline Generators** (`baseline_generators.py`)
   - Random, beat-sync, and constant baselines

7. **Report Generator** (`oscillator_report_generator.py`)
   - Basic structure works
   - **BUT: Shows incomplete metrics due to wave type issue**

### ❌ Critical Issue Discovered

**Wave Type Parameters are Placeholders:**
- Training data: wave_type_a = 0.1 (constant), wave_type_b = 0.0 (constant)
- Model predictions: wave_type_a ≈ 0.002, wave_type_b ≈ 0.332
- Both map to constant wave types (sine, plateau) in current evaluation

## Group Mapping Configuration

### Intention-Based to Oscillator-Based Mapping

The system maps 12 intention-based groups to 3 oscillator-based groups:

```python
GROUP_MAPPING_CONFIG = {
    'default': {
        # PAS groups -> Oscillator group (default PAS group used)
        'oscillator_group_0': {
            'pas_groups': [0, 1, 2],      # PAS groups 1-3
            'default_pas': 2,              # Use group 3 (index 2) as default
            'description': 'Front/main lighting'
        },
        'oscillator_group_1': {
            'pas_groups': [3, 4, 5],      # PAS groups 4-6
            'default_pas': 4,              # Use group 5 (index 4) as default
            'description': 'Side/fill lighting'
        },
        'oscillator_group_2': {
            'pas_groups': [6, 7, 8, 9],   # PAS groups 7-10
            'default_pas': 7,              # Use group 8 (index 7) as default
            'description': 'Back/effect lighting'
        }
    },
    'alternative_mappings': {
        # Can define alternative mapping strategies here for experimentation
    }
}
```

## Hybrid Wave Type Reconstruction Functions

### Core Decision Function

```python
import numpy as np
from scipy.signal import find_peaks

def select_waveform_for_segment(luminaire_dict, config):
    """
    Extended decision function that uses both PAS and Geo approach parameters.
    
    PAS-based metrics (from luminaire_dict['PASv02_allframes']):
      - target_max: maximum intensity from PAS (column 0)
      - target_min: minimum inverse minima from PAS (column 3)
      - oscillation_count: number of peaks in the PAS intensity signal
    
    Geo-based metrics (from luminaire_dict):
      - phase_range: variation of the phase over time (max - min)
      - freq_range: variation of the frequency over time (max - min)
      - offset_range: variation of the offset over time (max - min)
    """
    # ---------------------------
    # PAS-based Metrics:
    # ---------------------------
    PAS_all = luminaire_dict['PASv02_allframes']
    intensityPeakPAS = PAS_all[:, 0]
    intensityInverseMinimaPAS = PAS_all[:, 3]
    
    # Count peaks in PAS intensity as a measure of oscillation
    peaks, _ = find_peaks(intensityPeakPAS, height=0.6)
    oscillation_count = len(peaks)
    
    target_max = np.max(intensityPeakPAS)
    target_min = 1.0 - np.max(intensityInverseMinimaPAS)
    if target_min > target_max:
        intensity_range = target_max
    else:
        intensity_range = target_max - target_min

    # Normalize PAS oscillation
    pas_dynamic_score = oscillation_count / config["oscillation_threshold"]

    # Modify target_max and target_min to be in a not too small range
    amplitude_geo = luminaire_dict['standard_amplitude']
    target_max_geo = np.max(amplitude_geo)
    target_min_geo = np.min(amplitude_geo)
    if target_max_geo > target_max:
        target_max_modified = target_max_geo
    else:
        target_max_modified = target_max

    if target_min_geo < target_min:
        target_min_modified = target_min_geo
    else:    
        target_min_modified = target_min

    # ---------------------------
    # Geo-based Metrics:
    # ---------------------------
    # Phase variation
    phase_geo = luminaire_dict['phase']
    geo_phase_range = np.max(phase_geo) - np.min(phase_geo)
    geo_phase_norm = geo_phase_range / config.get("geo_phase_threshold", 0.15)
    
    # Frequency variation
    freq_geo = luminaire_dict.get('freq')
    if freq_geo is not None and hasattr(freq_geo, '__len__') and np.ndim(freq_geo) > 0:
        geo_freq_range = np.max(freq_geo) - np.min(freq_geo)
    else:
        geo_freq_range = 0.0
    geo_freq_norm = geo_freq_range / config.get("geo_freq_threshold", 0.15)
    
    # Offset variation
    offset_geo = luminaire_dict.get('offset')
    if offset_geo is not None and hasattr(offset_geo, '__len__') and np.ndim(offset_geo) > 0:
        geo_offset_range = np.max(offset_geo) - np.min(offset_geo)
    else:
        geo_offset_range = 0.0
    geo_offset_norm = geo_offset_range / config.get("geo_offset_threshold", 0.15)
    
    # Overall geo dynamic score
    overall_geo_dynamic = (geo_phase_norm + geo_freq_norm + geo_offset_norm) / 3.0

    # ---------------------------
    # BPM:
    # ---------------------------
    bpm = luminaire_dict.get('bpm', 120)

    # ---------------------------
    # Combine PAS and Geo Dynamics:
    # ---------------------------
    overall_dynamic = (overall_geo_dynamic + pas_dynamic_score) / 2.0

    # ---------------------------
    # Decision Rules:
    # ---------------------------
    # Note: decision_boundary values should be loaded from config or tuned
    decision_boundary_01 = config.get('decision_boundary_01', 0.1)
    decision_boundary_02 = config.get('decision_boundary_02', 0.3)
    decision_boundary_03 = config.get('decision_boundary_03', 0.5)
    decision_boundary_04 = config.get('decision_boundary_04', 0.7)
    decision_boundary_05 = config.get('decision_boundary_05', 0.9)
    
    decision = None
    if intensity_range < decision_boundary_01:
        decision = "still"
    else:
        if overall_dynamic < decision_boundary_02:
            decision = "sine"
        elif overall_dynamic < decision_boundary_03:
            decision = "pwm_basic"
        elif overall_dynamic < decision_boundary_04:
            decision = "pwm_extended"
        elif overall_dynamic < decision_boundary_05:
            decision = "odd_even"
        else:
            # For very high overall dynamics
            if bpm > config['bpm_thresholds']['high']:
                decision = "square"
            else:
                decision = "random"
    
    print(f"Waveform decision: {decision}")
    print(f"  PAS Intensity Range: {intensity_range:.2f}")
    print(f"  Overall Dynamic Score: {overall_dynamic:.2f}")
    
    return decision, overall_dynamic

def construct_array_decision(luminaire_dict, config):
    """
    For a given musical segment, decide on waveform generator type and parameters.
    """
    frames = luminaire_dict['frames']
    bpm = luminaire_dict.get('bpm', 120)
    
    # Geo parameters
    freq_geo = luminaire_dict['freq']
    phase_geo = luminaire_dict['phase']
    col_hue_Geo = luminaire_dict['col_hue']
    col_sat_Geo = luminaire_dict['col_sat']
    col_hue_PAS = np.mean(luminaire_dict['PASv02_allframes'][:, 4]) / 1.0
    col_sat_PAS = np.mean(luminaire_dict['PASv02_allframes'][:, 5]) / 1.0
    mirroring_active = luminaire_dict['mirroring_active']
    moving_direction = luminaire_dict['moving_direction']
    highlights = luminaire_dict['highlights']

    col_dict = {
        'col_hue_GEO': col_hue_Geo,
        'col_sat_GEO': col_sat_Geo,
        'col_hue_PAS': col_hue_PAS,
        'col_sat_PAS': col_sat_PAS
    }

    # Color decision logic (simplified for example)
    col_hue = (col_hue_Geo + col_hue_PAS) / 2.0
    col_sat = (col_sat_Geo + col_sat_PAS) / 2.0

    # BPM scaling
    if bpm < config['bpm_thresholds']['low']:
        bpm_scale = 0.5
    elif bpm > config['bpm_thresholds']['high']:
        bpm_scale = 2.0
    else:
        bpm_scale = 1.0
    
    f0 = freq_geo * bpm_scale

    mean_phase = np.mean(phase_geo)
    phase_direction = 1 if moving_direction < 0.5 else -1
    phase_movement = (mean_phase * config["max_phase_cycles_per_second"] / frames) * phase_direction

    decision, overall_dynamic = select_waveform_for_segment(luminaire_dict, config)

    array_decision_dict = {
        'decision': decision,
        'overall_dynamic': overall_dynamic,
        'f0': f0,
        'phase_movement': phase_movement,
        'col_hue': col_hue,
        'col_sat': col_sat,
        'mirroring_active': mirroring_active,
        'moving_direction': moving_direction,
        'highlights': highlights
    }

    return array_decision_dict

# Configuration used in back-processing
CONFIG = {
    "max_cycles_per_second": 4.0,
    "max_phase_cycles_per_second": 8.0,
    "led_count": 33,
    "virtual_led_count": 8,
    "fps": 30,
    "mode": "hard",
    "bpm_thresholds": {
        "low": 80,
        "high": 135
    },
    "optimization": {
        "alpha": 1.0,
        "beta": 1.0,
        "delta": 1.0
    },
    "oscillation_threshold": 10,
    "geo_phase_threshold": 0.15,
    "geo_freq_threshold": 0.15,
    "geo_offset_threshold": 0.15,
    # Decision boundaries (to be tuned)
    "decision_boundary_01": 0.1,
    "decision_boundary_02": 0.3,
    "decision_boundary_03": 0.5,
    "decision_boundary_04": 0.7,
    "decision_boundary_05": 0.9
}
```

## Implementation Plan for Hybrid Evaluation

### Phase 1: Wave Type Reconstruction Module

**File to create:** `scripts/wave_type_reconstructor.py`

```python
"""
This module will:
1. Load both PAS (intention-based) and Geo (oscillator-based) data
2. Apply group mapping configuration
3. Reconstruct wave type decisions using the hybrid function
4. Output reconstructed decisions for evaluation
"""

class WaveTypeReconstructor:
    def __init__(self, config, group_mapping):
        self.config = config
        self.group_mapping = group_mapping
    
    def load_pas_data(self, pas_file):
        """Load 72-dim intention-based data"""
        pass
    
    def load_geo_data(self, geo_file):
        """Load 60-dim oscillator-based data"""
        pass
    
    def extract_pas_metrics(self, pas_data, group_idx):
        """Extract PAS metrics for specified group"""
        # Use group_mapping to get correct PAS group
        pas_group = self.group_mapping[f'oscillator_group_{group_idx}']['default_pas']
        # Extract columns for this PAS group (6 params per group)
        start_idx = pas_group * 6
        return pas_data[:, start_idx:start_idx+6]
    
    def extract_geo_metrics(self, geo_data, group_idx):
        """Extract Geo metrics for specified oscillator group"""
        start_idx = group_idx * 20
        return geo_data[:, start_idx:start_idx+20]
    
    def reconstruct_wave_types(self, pas_data, geo_data, audio_info):
        """Main reconstruction function"""
        results = []
        for group_idx in range(3):
            # Get mapped data
            pas_metrics = self.extract_pas_metrics(pas_data, group_idx)
            geo_metrics = self.extract_geo_metrics(geo_data, group_idx)
            
            # Prepare luminaire_dict
            luminaire_dict = self.prepare_luminaire_dict(
                pas_metrics, geo_metrics, audio_info
            )
            
            # Get wave type decision
            decision, dynamic_score = select_waveform_for_segment(
                luminaire_dict, self.config
            )
            
            results.append({
                'group': group_idx,
                'decision': decision,
                'dynamic_score': dynamic_score
            })
        
        return results
```

### Phase 2: Update Oscillator Evaluator

**Updates needed in** `scripts/oscillator_evaluator.py`:

1. Skip wave_type_a and wave_type_b in parameter statistics
2. Add wave type reconstruction step
3. Compare reconstructed decisions instead of raw values
4. Add hybrid metrics

### Phase 3: Create Hybrid Evaluation Pipeline

**New file:** `scripts/hybrid_evaluator.py`

```python
"""
Main evaluation pipeline that:
1. Loads both data types
2. Reconstructs wave types
3. Computes all metrics
4. Generates comprehensive report
"""

class HybridEvaluator:
    def __init__(self, config, group_mapping):
        self.reconstructor = WaveTypeReconstructor(config, group_mapping)
        self.metrics = {}
    
    def evaluate_file(self, pas_file, geo_file, audio_info):
        """Evaluate a single file pair"""
        # Load data
        pas_data = self.reconstructor.load_pas_data(pas_file)
        geo_data = self.reconstructor.load_geo_data(geo_file)
        
        # Reconstruct wave types
        wave_decisions = self.reconstructor.reconstruct_wave_types(
            pas_data, geo_data, audio_info
        )
        
        # Compute metrics
        metrics = self.compute_metrics(wave_decisions, pas_data, geo_data)
        
        return metrics
    
    def compute_metrics(self, wave_decisions, pas_data, geo_data):
        """Compute all evaluation metrics"""
        return {
            'wave_type_distribution': self.get_wave_distribution(wave_decisions),
            'dynamic_scores': self.get_dynamic_stats(wave_decisions),
            'parameter_fidelity': self.compute_param_metrics(geo_data),
            'pas_geo_alignment': self.compute_alignment(pas_data, geo_data)
        }
```

### Phase 4: Process All Data

**Workflow script:** `scripts/run_hybrid_evaluation.py`

```python
"""
Main execution script that processes all data through the hybrid pipeline
"""

def main():
    # Load configurations
    config = CONFIG
    group_mapping = GROUP_MAPPING_CONFIG['default']
    
    # Initialize evaluator
    evaluator = HybridEvaluator(config, group_mapping)
    
    # Process training data
    training_results = process_dataset(
        'data/training_data',
        evaluator,
        dataset_type='training'
    )
    
    # Process predictions
    prediction_results = process_dataset(
        'data/conformer_osci',
        evaluator,
        dataset_type='predictions'
    )
    
    # Compare and generate report
    generate_comparison_report(training_results, prediction_results)
```

## Metrics to Implement

### 1. Wave Type Metrics

```python
wave_metrics = {
    'distribution': {
        'still': 0.05,      # percentage
        'sine': 0.25,
        'pwm_basic': 0.20,
        'pwm_extended': 0.20,
        'odd_even': 0.15,
        'square': 0.10,
        'random': 0.05
    },
    'segment_appropriateness': {
        'intro': {'most_common': 'sine', 'diversity': 0.3},
        'verse': {'most_common': 'pwm_basic', 'diversity': 0.5},
        'chorus': {'most_common': 'pwm_extended', 'diversity': 0.6},
        'drop': {'most_common': 'square', 'diversity': 0.4}
    },
    'transition_smoothness': 0.75,  # How smooth are wave type transitions
    'bpm_coherence': 0.82           # Do wave types match BPM appropriately
}
```

### 2. Dynamic Score Metrics

```python
dynamic_metrics = {
    'overall_mean': 0.45,
    'overall_std': 0.22,
    'pas_contribution': 0.48,
    'geo_contribution': 0.42,
    'by_segment': {
        'intro': {'mean': 0.3, 'std': 0.15},
        'verse': {'mean': 0.4, 'std': 0.18},
        'chorus': {'mean': 0.6, 'std': 0.20},
        'drop': {'mean': 0.8, 'std': 0.15}
    }
}
```

### 3. Hybrid Alignment Metrics

```python
alignment_metrics = {
    'pas_geo_correlation': {
        'intensity_amplitude': 0.75,    # PAS intensity vs Geo amplitude
        'variation_correlation': 0.68,   # Overall activity correlation
        'group_coordination': [0.8, 0.7, 0.75]  # Per-group alignment
    },
    'decision_consistency': {
        'agreement_rate': 0.85,  # How often PAS and Geo suggest same decision
        'conflict_resolution': {  # When they disagree, which dominates
            'pas_dominant': 0.45,
            'geo_dominant': 0.35,
            'balanced': 0.20
        }
    }
}
```

## Next Steps Implementation Order

1. **Create `wave_type_reconstructor.py`** - Core reconstruction logic
2. **Create `hybrid_evaluator.py`** - Main evaluation pipeline
3. **Update `oscillator_evaluator.py`** - Skip wave type params, add reconstruction
4. **Create `run_hybrid_evaluation.py`** - Main execution script
5. **Update report generator** - Include hybrid metrics
6. **Test with subset** - Validate on 10 files first
7. **Full evaluation** - Process all data
8. **Generate final report** - Complete metrics with wave type distributions

## Data Requirements

### For Each Prediction File

Need access to:
1. **PAS data** (72-dim): `data/edge_intention/light/*.pkl`
2. **Geo data** (60-dim): `data/conformer_osci/light_segments/*.pkl`
3. **Audio info**: `data/conformer_osci/audio_segments_information_jsons/*.json`

### For Training Data

Need access to:
1. **Original training oscillator params**: `data/training_data/oscillator_params/*.pkl`
2. **Corresponding PAS data**: Need to identify location
3. **Audio segment info**: In same directory as predictions

## Configuration Files Needed

### 1. Group Mapping Config
`configs/group_mapping.json`

### 2. Wave Type Decision Config
`configs/wave_decision_thresholds.json`

### 3. Evaluation Metrics Config
`configs/evaluation_metrics.json`

## Expected Outcomes After Implementation

1. **Wave Type Distributions** will show realistic variety (not 100% sine)
2. **Dynamic Scores** will reflect actual musical energy
3. **Segment-Appropriate Decisions** will align with musical structure
4. **Comparison Metrics** will be meaningful between training and predictions
5. **Hybrid System Performance** will be properly evaluated

## Critical Notes

1. **Wave type parameters (0.1, 0.0) are placeholders** - Do not evaluate them directly
2. **Real decisions come from hybrid post-processing** - Must reconstruct
3. **Group mapping is configurable** - Can experiment with different mappings
4. **Decision boundaries need tuning** - Current values are estimates
5. **Both PAS and Geo data required** - Cannot evaluate with only one

## Troubleshooting

### If wave types still show as 100% sine:
- Check intensity_range calculation
- Verify PAS data is loaded correctly
- Check decision boundaries are reasonable

### If dynamic scores are all zero:
- Verify oscillation_count is computed
- Check threshold values in config
- Ensure phase/freq/offset arrays have variation

### If group mapping seems wrong:
- Verify PAS group indices (0-11)
- Check oscillator group indices (0-2)
- Confirm default selections make sense for your data

## File Structure After Full Implementation

```
evaluation/
├── configs/
│   ├── group_mapping.json
│   ├── wave_decision_thresholds.json
│   └── evaluation_metrics.json
├── scripts/
│   ├── # Existing scripts (updated)
│   ├── structural_evaluator.py
│   ├── oscillator_evaluator.py     # [UPDATE NEEDED]
│   ├── ...
│   │
│   ├── # New hybrid evaluation scripts
│   ├── wave_type_reconstructor.py  # [TO CREATE]
│   ├── hybrid_evaluator.py         # [TO CREATE]
│   ├── run_hybrid_evaluation.py    # [TO CREATE]
│   └── hybrid_report_generator.py  # [TO CREATE]
├── outputs_hybrid/                  # [NEW OUTPUT DIR]
│   ├── wave_distributions/
│   ├── dynamic_scores/
│   ├── alignment_metrics/
│   └── reports/
└── README.md                        # [THIS DOCUMENT]
```

## License & Usage

This framework is provided for SCIENTIFIC and EDUCATIONAL purposes only. See LICENSE file for full restrictions.

## Acknowledgments

- Prof. Dr. Larissa Putzar (Primary Supervisor)
- Prof. Dr. Kai von Luck (Secondary Supervisor)
- Lighting designers who provided training data and expertise
- Discovery of hybrid architecture: August 10, 2025

---

**IMPORTANT**: This document represents the current understanding of the system as of August 10, 2025. The discovery of the hybrid wave type decision system fundamentally changes how the evaluation must be performed. All previous evaluations showing 100% sine wave types are artifacts of not applying the reconstruction function.