# Import Changes Guide for Script Reorganization

This document provides detailed instructions for updating import statements after the scripts folder reorganization.

## Overview of Changes

The scripts directory has been reorganized into the following structure:

```
scripts/
├── intention_based/                            # NEW
│   ├── structural_evaluator.py
│   ├── evaluate_dataset.py
│   ├── evaluate_dataset_with_tuned_params.py
│   ├── enhanced_tuner.py
│   └── boundary_tuner.py
│
├── intention_based_ground_truth_comparison/      # NEW
│   ├── compare_to_ground_truth.py
│   ├── evaluate_ground_truth_only.py
│   ├── ground_truth_visualizer.py
│   ├── quality_based_comparator.py
│   ├── quality_based_comparator_optimized.py
│   ├── run_quality_comparison.py
│   └── visualize_paradigm_comparison.py
│
├── segment_based_hybrid_oscillator_evaluation/   # NEW
│   ├── wave_type_reconstructor.py
│   ├── hybrid_evaluator.py
│   ├── hybrid_report_generator.py
│   ├── wave_type_visualizer.py
│   ├── sine_boost_config.py
│   ├── square_booster.py
│   └── custom_boundary_config.py
│
├── helpers/                                      # NEW
│   ├── visualizer.py
│   ├── inspect_pickle_audio.py
│   ├── inspect_pickle_light.py
│   ├── test_baseline.py
│   ├── run_evaluation_pipeline.py
│   ├── generate_final_plots.py
│   ├── full_evaluation_workflow.py
│   ├── run_final_evaluation.py
│   ├── debug_workflow.py
│   ├── thesis_plot_generator.py
│   └── thesis_plot_generator_fixed.py
│
└── thesis_workflow.py                           # UPDATED
```

## Required Import Changes by File

### 1. Files in `intention_based/` folder

#### `structural_evaluator.py`
- **No changes needed** (this is a core class with no internal imports)

#### `evaluate_dataset.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
```

#### `evaluate_dataset_with_tuned_params.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
```

#### `enhanced_tuner.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator
from visualizer import create_all_plots

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
from helpers.visualizer import create_all_plots
```

#### `boundary_tuner.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

### 2. Files in `intention_based_ground_truth_comparison/` folder

#### `compare_to_ground_truth.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator
from visualizer import create_all_plots

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
from helpers.visualizer import create_all_plots
```

#### `evaluate_ground_truth_only.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
```

#### `ground_truth_visualizer.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
```

#### `quality_based_comparator.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
```

#### `quality_based_comparator_optimized.py`
**Current imports to update:**
```python
# FROM:
from quality_based_comparator import QualityBasedComparator

# TO:
from intention_based_ground_truth_comparison.quality_based_comparator import QualityBasedComparator
```

#### `run_quality_comparison.py`
**Current imports to update:**
```python
# FROM:
from quality_based_comparator_optimized import OptimizedQualityComparator

# TO:
from intention_based_ground_truth_comparison.quality_based_comparator_optimized import OptimizedQualityComparator
```

#### `visualize_paradigm_comparison.py`
**Current imports to update:**
```python
# FROM:
from quality_based_comparator_optimized import OptimizedQualityComparator

# TO:
from intention_based_ground_truth_comparison.quality_based_comparator_optimized import OptimizedQualityComparator
```

### 3. Files in `segment_based_hybrid_oscillator_evaluation/` folder

#### `wave_type_reconstructor.py`
- **No changes needed** (this is a core class with minimal dependencies)

#### `hybrid_evaluator.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

#### `wave_type_visualizer.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

#### `sine_boost_config.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

#### `square_booster.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

#### `custom_boundary_config.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

### 4. Files in `helpers/` folder

#### `run_evaluation_pipeline.py`
**Current imports to update:**
```python
# FROM:
from structural_evaluator import StructuralEvaluator
from visualizer import create_all_plots

# TO:
from intention_based.structural_evaluator import StructuralEvaluator
from helpers.visualizer import create_all_plots
```

#### `full_evaluation_workflow.py`
**Current imports to update:**
```python
# FROM:
from run_evaluation_pipeline import EvaluationPipeline
from quality_based_comparator_optimized import OptimizedQualityComparator
from wave_type_reconstructor import WaveTypeReconstructor

# TO:
from helpers.run_evaluation_pipeline import EvaluationPipeline
from intention_based_ground_truth_comparison.quality_based_comparator_optimized import OptimizedQualityComparator
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
```

#### `thesis_plot_generator.py`
**Current imports to update:**
```python
# FROM:
from quality_based_comparator_optimized import OptimizedQualityComparator

# TO:
from intention_based_ground_truth_comparison.quality_based_comparator_optimized import OptimizedQualityComparator
```

#### `debug_workflow.py`
**Current imports to update:**
```python
# FROM:
from wave_type_reconstructor import WaveTypeReconstructor
from thesis_plot_generator import ThesisPlotGenerator

# TO:
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor
from helpers.thesis_plot_generator import ThesisPlotGenerator
```

#### `generate_final_plots.py`
**Current imports to update:**
```python
# FROM:
from visualizer import create_summary_boxplot, create_correlation_heatmap

# TO:
from helpers.visualizer import create_summary_boxplot, create_correlation_heatmap
```

### 5. Root-level scripts (if any refer to moved files)

Any scripts that import the moved files need to be updated. The main workflow script (`thesis_workflow.py`) has already been updated in the new version.

## Additional Path Setup

For files that need to import from the new structure, you may need to add path setup at the beginning of the file:

```python
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent  # If file is in subfolder
sys.path.append(str(scripts_dir))

# OR if the file is at scripts root:
sys.path.append(str(Path(__file__).parent))
```

## Systematic Update Process

Follow these steps to update all imports systematically:

### Step 1: Update intention_based folder files
```bash
cd scripts/intention_based
# Update evaluate_dataset.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' evaluate_dataset.py

# Update evaluate_dataset_with_tuned_params.py  
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' evaluate_dataset_with_tuned_params.py

# Update enhanced_tuner.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' enhanced_tuner.py
sed -i 's/from visualizer import/from helpers.visualizer import/g' enhanced_tuner.py

# Update boundary_tuner.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' boundary_tuner.py
```

### Step 2: Update intention_based_ground_truth_comparison folder files
```bash
cd scripts/intention_based_ground_truth_comparison

# Update compare_to_ground_truth.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' compare_to_ground_truth.py
sed -i 's/from visualizer import/from helpers.visualizer import/g' compare_to_ground_truth.py

# Update evaluate_ground_truth_only.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' evaluate_ground_truth_only.py

# Update ground_truth_visualizer.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' ground_truth_visualizer.py

# Update quality_based_comparator.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' quality_based_comparator.py

# Update quality_based_comparator_optimized.py
sed -i 's/from quality_based_comparator import/from intention_based_ground_truth_comparison.quality_based_comparator import/g' quality_based_comparator_optimized.py

# Update run_quality_comparison.py
sed -i 's/from quality_based_comparator_optimized import/from intention_based_ground_truth_comparison.quality_based_comparator_optimized import/g' run_quality_comparison.py

# Update visualize_paradigm_comparison.py
sed -i 's/from quality_based_comparator_optimized import/from intention_based_ground_truth_comparison.quality_based_comparator_optimized import/g' visualize_paradigm_comparison.py
```

### Step 3: Update segment_based_hybrid_oscillator_evaluation folder files
```bash
cd scripts/segment_based_hybrid_oscillator_evaluation

# Update hybrid_evaluator.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' hybrid_evaluator.py

# Update wave_type_visualizer.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' wave_type_visualizer.py

# Update sine_boost_config.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' sine_boost_config.py

# Update square_booster.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' square_booster.py

# Update custom_boundary_config.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' custom_boundary_config.py
```

### Step 4: Update helpers folder files
```bash
cd scripts/helpers

# Update run_evaluation_pipeline.py
sed -i 's/from structural_evaluator import/from intention_based.structural_evaluator import/g' run_evaluation_pipeline.py
sed -i 's/from visualizer import/from helpers.visualizer import/g' run_evaluation_pipeline.py

# Update full_evaluation_workflow.py
sed -i 's/from run_evaluation_pipeline import/from helpers.run_evaluation_pipeline import/g' full_evaluation_workflow.py
sed -i 's/from quality_based_comparator_optimized import/from intention_based_ground_truth_comparison.quality_based_comparator_optimized import/g' full_evaluation_workflow.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' full_evaluation_workflow.py

# Update thesis_plot_generator.py
sed -i 's/from quality_based_comparator_optimized import/from intention_based_ground_truth_comparison.quality_based_comparator_optimized import/g' thesis_plot_generator.py

# Update debug_workflow.py
sed -i 's/from wave_type_reconstructor import/from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import/g' debug_workflow.py
sed -i 's/from thesis_plot_generator import/from helpers.thesis_plot_generator import/g' debug_workflow.py

# Update generate_final_plots.py
sed -i 's/from visualizer import/from helpers.visualizer import/g' generate_final_plots.py
```

## Testing the Changes

After making all import changes, test the new structure by:

1. **Test the main workflow:**
   ```bash
   cd scripts
   python thesis_workflow.py --help
   ```

2. **Test individual components:**
   ```bash
   python -c "from intention_based.structural_evaluator import StructuralEvaluator; print('✓ Intention-based imports work')"
   python -c "from intention_based_ground_truth_comparison.quality_based_comparator_optimized import OptimizedQualityComparator; print('✓ Ground truth comparison imports work')"
   python -c "from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor; print('✓ Hybrid oscillator imports work')"
   python -c "from helpers.run_evaluation_pipeline import EvaluationPipeline; print('✓ Helper imports work')"
   ```

3. **Run a basic evaluation test:**
   ```bash
   python thesis_workflow.py --data_dir data/edge_intention --output_dir outputs/test_run
   ```

## Troubleshooting Common Issues

### Issue 1: "No module named" errors
**Solution:** Add path setup at the beginning of the file:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

### Issue 2: Circular imports
**Solution:** Move the import inside the function where it's used, or refactor to avoid circular dependencies.

### Issue 3: Relative imports not working
**Solution:** Use absolute imports from the scripts root, e.g.:
```python
from intention_based.structural_evaluator import StructuralEvaluator
```

### Issue 4: Path issues when running from different directories
**Solution:** Add robust path detection:
```python
import sys
from pathlib import Path

# Get the scripts directory regardless of where the script is run from
scripts_dir = Path(__file__).parent
while not (scripts_dir / "thesis_workflow.py").exists():
    scripts_dir = scripts_dir.parent
    if scripts_dir.name == "":  # Reached filesystem root
        break
sys.path.insert(0, str(scripts_dir))
```

## Verification Checklist

After completing all changes, verify:

- [ ] All files import correctly without errors
- [ ] Main thesis_workflow.py runs without import errors
- [ ] Individual evaluation components can be imported
- [ ] Tests pass (if you have any)
- [ ] Scripts can be run from both scripts/ directory and project root
- [ ] No circular import dependencies exist

## Summary

This reorganization improves code organization and follows the structure of your thesis evaluation methodology. The new `thesis_workflow.py` provides comprehensive reporting with:

- Detailed metric formulas from actual code
- Extended interpretations addressing >100% achievement concerns
- Separate distribution overlay plots
- Missing hybrid oscillator metrics visualizations
- Quality-adjusted novelty correlation analysis

All import changes preserve the existing functionality while providing a much cleaner and more maintainable code structure.