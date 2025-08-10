# Evaluation Framework for Hybrid Music-Driven Light Show Generation

## Master Thesis Context

This repository contains the evaluation framework developed for the master thesis:

**"Generative Synthesis of Music-Driven Light Shows: A Framework for Co-Creative Stage Lighting"**  
*Author: Tobias Wursthorn*  
*HAW Hamburg, Department of Media Technology, 2025*

## 🔴 CRITICAL UPDATE: Hybrid Wave Type Architecture - RESOLVED

**Date: December 2024**

### ✅ What We've Discovered and Fixed

1. **Wave type parameters in oscillator data are placeholders** (constant values)
2. **Actual wave type decisions come from a hybrid post-processing system** that combines:
   - PAS (intention-based) data: 72 dimensions
   - Geo (oscillator-based) data: 60 dimensions
3. **Successfully reconstructed the decision logic** from TouchDesigner implementation
4. **Identified and fixed critical decision boundary values**

### ✅ Current Implementation Status

#### **COMPLETED** ✅

1. **Wave Type Reconstruction** (`wave_type_reconstructor.py`)
   - Correctly implements hybrid decision logic
   - Uses proper PAS group mappings (groups 2, 5, 8 → oscillator groups 0, 1, 2)
   - Fixed decision boundaries from TouchDesigner:
     - boundary_01: 0.22
     - boundary_02: 0.70
     - boundary_03: 1.10
     - boundary_04: 2.20
     - boundary_05: 7.00
   - Produces realistic wave type distributions

2. **Boundary Tuning Tools**
   - `boundary_tuner.py` - Interactive tool for distribution analysis
   - `custom_boundary_config.py` - Config generator for target distributions
   - Successfully tuned to achieve desired distributions

3. **Current Wave Type Distribution** (after tuning)
   ```
   odd_even:     37.1%
   still:        32.5%
   pwm_extended: 26.1%
   pwm_basic:     2.8%
   sine:          1.4%
   random:        0.1%
   ```

#### **IN PROGRESS** 🚧

1. **Hybrid Evaluator** (`hybrid_evaluator.py`)
   - Main evaluation pipeline combining both data types
   - Needs implementation based on wave_type_reconstructor

2. **Updated Oscillator Metrics**
   - Rewrite evaluation to consider reconstructed wave types
   - Compare reconstructed decisions vs training data patterns

#### **TO DO** 📝

1. **Complete Hybrid Evaluation Pipeline**
2. **Update Baseline Generators** for hybrid approach
3. **Rewrite Inter-Group Analysis** with wave type consideration
4. **Generate Final Report** with hybrid metrics

## System Architecture

```
Audio Input
    ├── Intention-Based (PAS) → 72 dims (12 groups × 6 params)
    └── Oscillator-Based (Geo) → 60 dims (3 groups × 20 params)
              ↓
    [Wave Type Reconstructor]
              ↓
    Hybrid Decision Function
              ↓
    Actual Wave Types:
    - still
    - sine
    - pwm_basic
    - pwm_extended
    - odd_even
    - square
    - random
```

## Current File Structure

```
evaluation/
├── configs/                          # NEW: Boundary configurations
│   ├── custom_recommended.json      # Tuned boundaries
│   ├── custom_max_sine.json
│   └── custom_min_random.json
│
├── scripts/
│   ├── # ✅ WORKING - Intention-based evaluation
│   ├── structural_evaluator.py      
│   ├── evaluate_dataset.py          
│   ├── evaluate_dataset_with_tuned_params.py
│   ├── enhanced_tuner.py            
│   ├── visualizer.py                
│   ├── generate_final_plots.py      
│   │
│   ├── # ✅ WORKING - Hybrid wave type reconstruction
│   ├── wave_type_reconstructor.py   # FIXED & WORKING
│   ├── boundary_tuner.py            # NEW tuning tool
│   ├── custom_boundary_config.py    # NEW config generator
│   │
│   ├── # ✅ UTILITY - Keep these
│   ├── inspect_pickle_audio.py      
│   ├── inspect_pickle_light.py      
│   │
│   ├── # ❌ DEPRECATED - To be replaced
│   ├── oscillator_evaluator.py      # Needs rewrite for hybrid
│   ├── baseline_generators.py       # Needs update for hybrid
│   ├── inter_group_analyzer.py      # Needs update for hybrid
│   ├── oscillator_report_generator.py # Needs update
│   ├── training_stats_extractor.py  # Needs update
│   ├── compute_model_stats.py       # Not needed
│   └── diagnostic_checker.py        # Debug tool, not needed
│
├── outputs_hybrid/                   # NEW: Hybrid evaluation results
│   ├── wave_reconstruction_fixed.pkl
│   └── wave_reconstruction_fixed.json
│
└── paste.txt                         # TouchDesigner reference code
```

## Quick Start Guide

### 1. Run Wave Type Reconstruction

```bash
# Test with custom boundaries
python scripts/wave_type_reconstructor.py --max_files 10 --config configs/custom_recommended.json

# Run full dataset when satisfied
python scripts/wave_type_reconstructor.py --config configs/custom_recommended.json
```

### 2. Tune Distribution (if needed)

```bash
# Interactive tuning
python scripts/boundary_tuner.py

# Generate custom configs
python scripts/custom_boundary_config.py
```

### 3. Current Workflow

```bash
# Step 1: Reconstruct wave types
python scripts/wave_type_reconstructor.py --config configs/custom_recommended.json

# Step 2: Analyze distribution
python scripts/boundary_tuner.py --analyze

# Step 3: (TODO) Run hybrid evaluation
python scripts/hybrid_evaluator.py  # To be implemented

# Step 4: (TODO) Generate report
python scripts/hybrid_report_generator.py  # To be implemented
```

## Key Discoveries & Fixes

### 1. Decision Boundaries (TouchDesigner values)
- **boundary_01**: 0.22 (controls still vs others)
- **boundary_02**: 0.70 (controls sine vs pwm_basic)
- **boundary_03**: 1.10 (controls pwm_basic vs pwm_extended)
- **boundary_04**: 2.20 (controls pwm_extended vs odd_even)
- **boundary_05**: 7.00 (controls odd_even vs random/square)

### 2. PAS to Oscillator Group Mapping
- Oscillator group 0 ← PAS group 2 (index 1)
- Oscillator group 1 ← PAS group 5 (index 4)
- Oscillator group 2 ← PAS group 8 (index 7)

### 3. Peak Detection
- Height threshold: 0.6 (works well, confirmed)
- Oscillation threshold: 10 (can be reduced to 8 for higher dynamics)

## Next Implementation Steps

### Phase 1: Complete Hybrid Evaluator ✅ (Partially Done)
- [x] Wave type reconstruction working
- [x] Boundary tuning tools created
- [ ] Main evaluation pipeline

### Phase 2: Update Metrics
- [ ] Rewrite oscillator_evaluator for hybrid approach
- [ ] Update baseline generators
- [ ] Fix inter-group analysis

### Phase 3: Generate Reports
- [ ] Create comprehensive report generator
- [ ] Compare with training data
- [ ] Visualize wave type distributions

## Scripts Cleanup Recommendation

### ✅ KEEP These Scripts:
```
# Intention-based evaluation (all working)
- structural_evaluator.py
- evaluate_dataset.py
- evaluate_dataset_with_tuned_params.py
- enhanced_tuner.py
- visualizer.py
- generate_final_plots.py

# Hybrid wave type (new & working)
- wave_type_reconstructor.py
- boundary_tuner.py
- custom_boundary_config.py

# Utilities
- inspect_pickle_audio.py
- inspect_pickle_light.py
```

### ❌ DELETE These Scripts:
```
# Deprecated/needs complete rewrite
- oscillator_evaluator.py
- baseline_generators.py
- inter_group_analyzer.py
- oscillator_report_generator.py
- training_stats_extractor.py
- compute_model_stats.py
- diagnostic_checker.py
```

### 📝 TO CREATE:
```
# New hybrid evaluation scripts
- hybrid_evaluator.py
- hybrid_baseline_generator.py
- hybrid_report_generator.py
- wave_type_analyzer.py
```

## Important Notes

1. **Wave type parameters in oscillator data are NOT real** - they're placeholders
2. **Real decisions come from hybrid post-processing** combining PAS and Geo data
3. **Decision boundaries are critical** - small changes have large effects
4. **Group mapping must be correct** - using wrong PAS groups breaks the system

## License & Usage

This framework is provided for SCIENTIFIC and EDUCATIONAL purposes only. See LICENSE file for full restrictions.

## Acknowledgments

- Prof. Dr. Larissa Putzar (Primary Supervisor)
- Prof. Dr. Kai von Luck (Secondary Supervisor)
- Discovery of hybrid architecture: August 2025
- Resolution of wave type reconstruction: December 2024

---

**CURRENT STATUS**: Wave type reconstruction working, boundaries tuned, ready for hybrid evaluation implementation.