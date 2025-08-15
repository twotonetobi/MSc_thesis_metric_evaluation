#!/usr/bin/env python
"""
Debug Script for Thesis Workflow
=================================
Diagnoses configuration and distribution issues
"""

import json
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def diagnose_configuration():
    """Check which configuration is actually being used."""
    
    print("\n" + "="*60)
    print("CONFIGURATION DIAGNOSIS")
    print("="*60)
    
    # Check for config files
    config_paths = [
        Path('configs/final_optimal.json'),
        Path('configs/custom_recommended.json'),
        Path('configs/tuned_boundaries.json')
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            print(f"\n✓ Found: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for decision boundaries
            boundaries = []
            for i in range(1, 6):
                key = f'decision_boundary_0{i}'
                if key in config:
                    boundaries.append(config[key])
            
            if boundaries:
                print(f"  Boundaries: {boundaries}")
            else:
                print("  ⚠️ No decision boundaries found!")
        else:
            print(f"\n✗ Missing: {config_path}")
    
    # Check what the reconstructor would use
    print("\n" + "-"*60)
    print("TESTING WAVE TYPE RECONSTRUCTOR")
    print("-"*60)
    
    try:
        from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor, CONFIG
        
        print("\nDefault CONFIG boundaries:")
        for i in range(1, 6):
            key = f'decision_boundary_0{i}'
            print(f"  {key}: {CONFIG.get(key, 'NOT FOUND')}")
        
        # Try loading with final_optimal
        if Path('configs/final_optimal.json').exists():
            with open('configs/final_optimal.json', 'r') as f:
                optimal_config = json.load(f)
            
            print("\nfinal_optimal.json boundaries:")
            for i in range(1, 6):
                key = f'decision_boundary_0{i}'
                print(f"  {key}: {optimal_config.get(key, 'NOT FOUND')}")
            
            # Test reconstruction with both configs
            print("\n" + "-"*60)
            print("TESTING DECISION LOGIC")
            print("-"*60)
            
            test_cases = [
                ("Low intensity", 0.05, 0.5),  # Should be "still"
                ("Medium dynamic", 0.3, 1.5),   # Should be "sine" or "pwm"
                ("High dynamic", 0.5, 4.0),     # Should be "odd_even" or higher
            ]
            
            for desc, intensity_range, dynamic_score in test_cases:
                print(f"\n{desc}: intensity={intensity_range:.2f}, dynamic={dynamic_score:.2f}")
                
                # With default config
                decision_default = get_decision(intensity_range, dynamic_score, CONFIG)
                print(f"  Default config → {decision_default}")
                
                # With optimal config
                decision_optimal = get_decision(intensity_range, dynamic_score, optimal_config)
                print(f"  Optimal config → {decision_optimal}")
    
    except ImportError as e:
        print(f"✗ Cannot import wave_type_reconstructor: {e}")

def get_decision(intensity_range, overall_dynamic, config):
    """Simplified decision logic for testing."""
    
    b1 = config.get('decision_boundary_01', 0.1)
    b2 = config.get('decision_boundary_02', 1.0)
    b3 = config.get('decision_boundary_03', 1.5)
    b4 = config.get('decision_boundary_04', 2.0)
    b5 = config.get('decision_boundary_05', 3.0)
    
    if intensity_range < b1:
        return "still"
    elif overall_dynamic < b2:
        return "sine"
    elif overall_dynamic < b3:
        return "pwm_basic"
    elif overall_dynamic < b4:
        return "pwm_extended"
    elif overall_dynamic < b5:
        return "odd_even"
    else:
        return "random/square"

def check_plot_generator():
    """Check which plot generator methods exist."""
    
    print("\n" + "="*60)
    print("PLOT GENERATOR DIAGNOSIS")
    print("="*60)
    
    try:
        from helpers.thesis_plot_generator import ThesisPlotGenerator
        generator = ThesisPlotGenerator()
        
        methods = [m for m in dir(generator) if not m.startswith('_')]
        
        print("\nAvailable methods:")
        for method in sorted(methods):
            if 'generate' in method or 'plot' in method:
                print(f"  ✓ {method}")
        
        # Check specifically for the missing method
        if hasattr(generator, 'generate_hybrid_wave_distribution'):
            print("\n✓ generate_hybrid_wave_distribution EXISTS")
        else:
            print("\n✗ generate_hybrid_wave_distribution MISSING")
            print("  This is why the workflow fails!")
    
    except ImportError as e:
        print(f"✗ Cannot import thesis_plot_generator: {e}")

def check_recent_results():
    """Check the most recent wave reconstruction results."""
    
    print("\n" + "="*60)
    print("RECENT RESULTS CHECK")
    print("="*60)
    
    result_paths = [
        Path('outputs_hybrid/wave_reconstruction.pkl'),
        Path('outputs_hybrid/wave_reconstruction_fixed.pkl'),
        Path('outputs/thesis_complete') / 'data' / 'wave_reconstruction.pkl'
    ]
    
    for result_path in result_paths:
        if result_path.exists():
            print(f"\n✓ Found: {result_path}")
            with open(result_path, 'rb') as f:
                results = pickle.load(f)
            
            if 'wave_type_distribution' in results:
                print("  Distribution:")
                for wave, pct in sorted(results['wave_type_distribution'].items(), 
                                       key=lambda x: -x[1]):
                    print(f"    {wave:15s}: {pct*100:5.1f}%")

def main():
    """Run all diagnostics."""
    
    print("\n" + "🔍"*30)
    print("\nTHESIS WORKFLOW DIAGNOSTICS")
    print("\n" + "🔍"*30)
    
    # Run all checks
    diagnose_configuration()
    check_plot_generator()
    check_recent_results()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    print("\n🎯 KEY FINDINGS:")
    print("1. Check if decision boundaries match expected values")
    print("2. Verify the plot generator has all required methods")
    print("3. Compare distributions across different result files")
    
    print("\n💡 LIKELY ISSUE:")
    print("The workflow is not passing the correct config to the")
    print("wave type reconstructor, causing it to use defaults.")

if __name__ == '__main__':
    main()