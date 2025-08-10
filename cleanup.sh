#!/bin/bash

# Cleanup and organize evaluation framework scripts
# Run this from the evaluation/ directory

echo "=========================================="
echo "Evaluation Framework Cleanup & Organization"
echo "=========================================="

# Create backup directory
echo "Creating backup directory..."
mkdir -p scripts_backup_deprecated
echo "✓ Backup directory created: scripts_backup_deprecated/"

# Scripts to KEEP (don't move)
keep_scripts=(
    # Intention-based evaluation
    "structural_evaluator.py"
    "evaluate_dataset.py"
    "evaluate_dataset_with_tuned_params.py"
    "enhanced_tuner.py"
    "visualizer.py"
    "generate_final_plots.py"
    
    # Hybrid wave type reconstruction
    "wave_type_reconstructor.py"
    "boundary_tuner.py"
    "custom_boundary_config.py"
    
    # Utilities
    "inspect_pickle_audio.py"
    "inspect_pickle_light.py"
)

# Scripts to MOVE to deprecated
deprecated_scripts=(
    "oscillator_evaluator.py"
    "baseline_generators.py"
    "inter_group_analyzer.py"
    "oscillator_report_generator.py"
    "training_stats_extractor.py"
    "compute_model_stats.py"
    "diagnostic_checker.py"
)

echo ""
echo "Moving deprecated scripts to backup..."
echo "----------------------------------------"

for script in "${deprecated_scripts[@]}"; do
    if [ -f "scripts/$script" ]; then
        mv "scripts/$script" "scripts_backup_deprecated/"
        echo "  ✓ Moved: $script"
    else
        echo "  ⚠ Not found: $script"
    fi
done

echo ""
echo "Keeping active scripts..."
echo "----------------------------------------"

for script in "${keep_scripts[@]}"; do
    if [ -f "scripts/$script" ]; then
        echo "  ✓ Kept: $script"
    else
        echo "  ⚠ Not found: $script"
    fi
done

# Create placeholder for new scripts to be developed
echo ""
echo "Creating placeholders for new scripts..."
echo "----------------------------------------"

cat > scripts/hybrid_evaluator.py << 'EOF'
#!/usr/bin/env python
"""
Hybrid Evaluator - Main evaluation pipeline combining PAS and Geo data
Status: TO BE IMPLEMENTED

This will:
1. Load both PAS and Geo data
2. Reconstruct wave types using wave_type_reconstructor
3. Compute hybrid metrics
4. Generate comprehensive evaluation
"""

print("Hybrid evaluator - to be implemented")
print("Use wave_type_reconstructor.py as reference")
EOF

cat > scripts/hybrid_report_generator.py << 'EOF'
#!/usr/bin/env python
"""
Hybrid Report Generator - Generate comprehensive evaluation reports
Status: TO BE IMPLEMENTED

This will:
1. Load reconstruction results
2. Compare with training patterns
3. Generate visualizations
4. Create markdown report
"""

print("Hybrid report generator - to be implemented")
EOF

echo "  ✓ Created: hybrid_evaluator.py (placeholder)"
echo "  ✓ Created: hybrid_report_generator.py (placeholder)"

# Update README
echo ""
echo "Updating documentation..."
echo "----------------------------------------"

if [ -f "README_update_osci.md" ]; then
    mv README_update_osci.md README_update_osci_old.md
    echo "  ✓ Backed up old README to README_update_osci_old.md"
fi

# The new README is already created as an artifact
echo "  ℹ Copy the new README content from the artifact"

# Create a summary file
cat > CLEANUP_SUMMARY.txt << 'EOF'
EVALUATION FRAMEWORK CLEANUP SUMMARY
=====================================
Date: $(date)

KEPT SCRIPTS (Active):
----------------------
Intention-based:
- structural_evaluator.py
- evaluate_dataset.py
- evaluate_dataset_with_tuned_params.py
- enhanced_tuner.py
- visualizer.py
- generate_final_plots.py

Hybrid Wave Type:
- wave_type_reconstructor.py
- boundary_tuner.py
- custom_boundary_config.py

Utilities:
- inspect_pickle_audio.py
- inspect_pickle_light.py

MOVED TO DEPRECATED:
--------------------
- oscillator_evaluator.py (needs rewrite for hybrid)
- baseline_generators.py (needs update)
- inter_group_analyzer.py (needs update)
- oscillator_report_generator.py (needs update)
- training_stats_extractor.py (needs update)
- compute_model_stats.py (not needed)
- diagnostic_checker.py (debug tool, not needed)

NEW PLACEHOLDERS CREATED:
-------------------------
- hybrid_evaluator.py (to be implemented)
- hybrid_report_generator.py (to be implemented)

NEXT STEPS:
-----------
1. Run: python scripts/custom_boundary_config.py
2. Test: python scripts/wave_type_reconstructor.py --config configs/custom_recommended.json
3. Implement hybrid_evaluator.py based on wave_type_reconstructor.py
4. Generate final evaluation reports
EOF

echo "  ✓ Created: CLEANUP_SUMMARY.txt"

echo ""
echo "=========================================="
echo "CLEANUP COMPLETE!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Deprecated scripts moved to: scripts_backup_deprecated/"
echo "  - Active scripts remain in: scripts/"
echo "  - New placeholders created for hybrid evaluation"
echo "  - See CLEANUP_SUMMARY.txt for details"
echo ""
echo "Next steps:"
echo "  1. Generate custom config: python scripts/custom_boundary_config.py"
echo "  2. Test reconstruction: python scripts/wave_type_reconstructor.py --config configs/custom_recommended.json"
echo "  3. Implement hybrid evaluation pipeline"