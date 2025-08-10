#!/usr/bin/env python
"""
Interactive Boundary Tuner for Wave Type Distribution
Helps find optimal decision boundaries for even distribution
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_current_distribution(results_file: Path = Path('outputs_hybrid/wave_reconstruction_fixed.json')):
    """Load and analyze current distribution."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    dist = data['wave_type_distribution']
    counts = data['wave_type_counts']
    
    print("\n" + "="*60)
    print("CURRENT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Order by percentage
    for wt in ['still', 'sine', 'pwm_basic', 'pwm_extended', 'odd_even', 'square', 'random']:
        if wt in dist:
            pct = dist[wt] * 100
            count = counts.get(wt, 0)
            bar = '█' * int(pct/2)
            print(f"{wt:15s}: {pct:5.1f}% ({count:3d}) {bar}")
    
    return dist, counts

def suggest_boundary_adjustments(current_dist: Dict, target_dist: Dict = None):
    """Suggest boundary adjustments to achieve target distribution."""
    
    if target_dist is None:
        # Default target: more even distribution
        target_dist = {
            'still': 0.15,      # Reduce from 32.5% to 15%
            'sine': 0.15,       # Increase from 1.4% to 15%
            'pwm_basic': 0.15,  # Increase from 2.8% to 15%
            'pwm_extended': 0.20, # Slight reduction from 26.1% to 20%
            'odd_even': 0.20,   # Reduce from 37.1% to 20%
            'square': 0.05,     # Add some square waves
            'random': 0.10      # Increase from 0.1% to 10%
        }
    
    print("\n" + "="*60)
    print("SUGGESTED BOUNDARY ADJUSTMENTS")
    print("="*60)
    
    print("\nTarget Distribution:")
    for wt, pct in target_dist.items():
        current_pct = current_dist.get(wt, 0)
        diff = (pct - current_pct) * 100
        sign = "+" if diff > 0 else ""
        print(f"  {wt:15s}: {current_pct*100:5.1f}% → {pct*100:5.1f}% ({sign}{diff:+.1f}%)")
    
    # Current boundaries
    current_boundaries = {
        'boundary_01': 0.22,  # Controls still vs others
        'boundary_02': 0.7,   # Controls sine vs pwm_basic
        'boundary_03': 1.1,   # Controls pwm_basic vs pwm_extended
        'boundary_04': 2.2,   # Controls pwm_extended vs odd_even
        'boundary_05': 7.0    # Controls odd_even vs random/square
    }
    
    # Suggested adjustments based on distribution analysis
    suggested_boundaries = {
        'boundary_01': 0.15,  # Lower to reduce "still" (32.5% → 15%)
        'boundary_02': 0.9,   # Raise to increase "sine" range (1.4% → 15%)
        'boundary_03': 1.3,   # Raise to increase "pwm_basic" range (2.8% → 15%)
        'boundary_04': 1.8,   # Lower to reduce "odd_even" range
        'boundary_05': 3.5    # Much lower to increase "random" (0.1% → 10%)
    }
    
    print("\n" + "="*60)
    print("RECOMMENDED BOUNDARY CHANGES")
    print("="*60)
    
    for key in current_boundaries:
        current = current_boundaries[key]
        suggested = suggested_boundaries[key]
        change = suggested - current
        print(f"{key}: {current:.2f} → {suggested:.2f} (change: {change:+.2f})")
    
    # Additional tuning suggestions
    print("\n" + "="*60)
    print("ADDITIONAL TUNING PARAMETERS")
    print("="*60)
    
    print("\n1. To influence dynamic scores (if boundaries alone don't work):")
    print("   - oscillation_threshold: 10 → 8 (makes dynamics higher)")
    print("   - peak_height: 0.6 → 0.4 (detects more peaks → higher dynamics)")
    print("   - geo_phase_threshold: 0.15 → 0.2 (reduces geo contribution)")
    
    print("\n2. Alternative boundary sets to try:")
    
    # Alternative 1: More aggressive
    print("\n   Alternative A (More Aggressive):")
    alt_a = {
        'boundary_01': 0.10,  # Very low for minimal "still"
        'boundary_02': 1.0,   # Higher for more "sine"
        'boundary_03': 1.5,   # Higher for more "pwm_basic"
        'boundary_04': 2.0,   # 
        'boundary_05': 3.0    # Much lower for more "random"
    }
    for key, val in alt_a.items():
        print(f"     {key}: {val:.2f}")
    
    # Alternative 2: Conservative
    print("\n   Alternative B (Conservative):")
    alt_b = {
        'boundary_01': 0.18,
        'boundary_02': 0.8,
        'boundary_03': 1.2,
        'boundary_04': 2.0,
        'boundary_05': 4.0
    }
    for key, val in alt_b.items():
        print(f"     {key}: {val:.2f}")
    
    return suggested_boundaries

def generate_config_file(boundaries: Dict, output_path: Path = Path('configs/tuned_boundaries.json')):
    """Generate a config file with new boundaries."""
    
    config = {
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
        "oscillation_threshold": 8,  # Reduced from 10
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15,
        "decision_boundary_01": boundaries['boundary_01'],
        "decision_boundary_02": boundaries['boundary_02'],
        "decision_boundary_03": boundaries['boundary_03'],
        "decision_boundary_04": boundaries['boundary_04'],
        "decision_boundary_05": boundaries['boundary_05']
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {output_path}")
    return config

def simulate_distribution(dynamic_values: np.ndarray, intensity_values: np.ndarray, 
                         boundaries: Dict) -> Dict:
    """Simulate what distribution would result from given boundaries."""
    
    decisions = []
    
    for intensity, dynamic in zip(intensity_values, dynamic_values):
        if intensity < boundaries['boundary_01']:
            decision = 'still'
        elif dynamic < boundaries['boundary_02']:
            decision = 'sine'
        elif dynamic < boundaries['boundary_03']:
            decision = 'pwm_basic'
        elif dynamic < boundaries['boundary_04']:
            decision = 'pwm_extended'
        elif dynamic < boundaries['boundary_05']:
            decision = 'odd_even'
        else:
            # Simplified: assume 50/50 split between square and random
            decision = 'random' if np.random.random() > 0.5 else 'square'
        
        decisions.append(decision)
    
    # Calculate distribution
    from collections import Counter
    counts = Counter(decisions)
    total = len(decisions)
    
    distribution = {k: v/total for k, v in counts.items()}
    
    return distribution

def interactive_tuner():
    """Interactive boundary tuning interface."""
    
    print("\n" + "="*60)
    print("INTERACTIVE BOUNDARY TUNER")
    print("="*60)
    
    # Load current results if available
    results_path = Path('outputs_hybrid/wave_reconstruction_fixed.json')
    if results_path.exists():
        current_dist, counts = analyze_current_distribution(results_path)
    else:
        print("No results file found. Using default distribution.")
        current_dist = {
            'odd_even': 0.371,
            'still': 0.325,
            'pwm_extended': 0.261,
            'pwm_basic': 0.028,
            'sine': 0.014,
            'random': 0.001
        }
    
    # Get suggestions
    suggested = suggest_boundary_adjustments(current_dist)
    
    # Ask user for preference
    print("\n" + "="*60)
    print("CHOOSE CONFIGURATION")
    print("="*60)
    print("\n1. Use SUGGESTED boundaries (balanced distribution)")
    print("2. Use ALTERNATIVE A (more aggressive)")
    print("3. Use ALTERNATIVE B (conservative)")
    print("4. Enter CUSTOM boundaries")
    print("5. Keep CURRENT boundaries")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        boundaries = suggested
    elif choice == '2':
        boundaries = {
            'boundary_01': 0.10,
            'boundary_02': 1.0,
            'boundary_03': 1.5,
            'boundary_04': 2.0,
            'boundary_05': 3.0
        }
    elif choice == '3':
        boundaries = {
            'boundary_01': 0.18,
            'boundary_02': 0.8,
            'boundary_03': 1.2,
            'boundary_04': 2.0,
            'boundary_05': 4.0
        }
    elif choice == '4':
        boundaries = {}
        for i in range(1, 6):
            key = f'boundary_0{i}'
            default = suggested[key]
            val = input(f"Enter {key} (default {default:.2f}): ").strip()
            boundaries[key] = float(val) if val else default
    else:
        boundaries = {
            'boundary_01': 0.22,
            'boundary_02': 0.7,
            'boundary_03': 1.1,
            'boundary_04': 2.2,
            'boundary_05': 7.0
        }
    
    # Generate config
    config_path = Path('configs/tuned_boundaries.json')
    generate_config_file(boundaries, config_path)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Test with a few files first:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/tuned_boundaries.json")
    print("\n2. If distribution looks good, run full dataset:")
    print("   python scripts/wave_type_reconstructor.py --config configs/tuned_boundaries.json")
    print("\n3. Update TouchDesigner boundaries if needed:")
    for key, val in boundaries.items():
        print(f"   Set {key} to {val:.2f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune decision boundaries for wave type distribution')
    parser.add_argument('--analyze', action='store_true', 
                       help='Only analyze current distribution')
    parser.add_argument('--suggest', action='store_true',
                       help='Only show suggestions')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive tuner')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_current_distribution()
    elif args.suggest:
        results_path = Path('outputs_hybrid/wave_reconstruction_fixed.json')
        if results_path.exists():
            with open(results_path, 'r') as f:
                data = json.load(f)
            current_dist = data['wave_type_distribution']
            suggest_boundary_adjustments(current_dist)
        else:
            print("No results file found.")
    else:
        # Default: run interactive tuner
        interactive_tuner()