#!/usr/bin/env python
"""
Interactive Boundary Tuner for Wave Type Distribution - FIXED VERSION
Uses correct decision_boundary_XX key names
Helps find optimal decision boundaries for even distribution
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_current_distribution(results_file: Path = Path('outputs_hybrid/wave_reconstruction_fixed.json')):
    """Load and analyze current distribution."""
    
    if not results_file.exists():
        print(f"Warning: {results_file} not found. Using default distribution.")
        dist = {
            'still': 0.31,
            'sine': 0.038, 
            'pwm_basic': 0.06,
            'pwm_extended': 0.113,
            'odd_even': 0.275,
            'square': 0.011,
            'random': 0.193
        }
        counts = {k: int(v * 945) for k, v in dist.items()}
    else:
        with open(results_file, 'r') as f:
            data = json.load(f)
        dist = data['wave_type_distribution']
        counts = data.get('wave_type_counts', {k: int(v * 945) for k, v in dist.items()})
    
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
            'still': 0.15,      # Reduce from ~31% to 15%
            'sine': 0.20,       # Increase from ~4% to 20%
            'pwm_basic': 0.10,  # Increase from ~6% to 10%
            'pwm_extended': 0.15, # Increase from ~11% to 15%
            'odd_even': 0.25,   # Reduce from ~28% to 25%
            'square': 0.05,     # Increase from ~1% to 5%
            'random': 0.10      # Reduce from ~19% to 10%
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
    
    # Current boundaries (from TouchDesigner)
    current_boundaries = {
        'decision_boundary_01': 0.22,  # Controls still vs others
        'decision_boundary_02': 0.7,   # Controls sine vs pwm_basic
        'decision_boundary_03': 1.1,   # Controls pwm_basic vs pwm_extended
        'decision_boundary_04': 2.2,   # Controls pwm_extended vs odd_even
        'decision_boundary_05': 7.0    # Controls odd_even vs random/square
    }
    
    # Suggested adjustments based on distribution analysis
    suggested_boundaries = {
        'decision_boundary_01': 0.06,  # Lower to reduce "still" (31% → 15%)
        'decision_boundary_02': 1.35,  # Raise to increase "sine" range (4% → 20%)
        'decision_boundary_03': 1.65,  # Raise to increase "pwm_basic" range (6% → 10%)
        'decision_boundary_04': 1.95,  # Lower slightly
        'decision_boundary_05': 4.2    # Lower to reduce "random" (19% → 10%)
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
        'decision_boundary_01': 0.05,  # Very low for minimal "still"
        'decision_boundary_02': 1.4,   # Higher for more "sine"
        'decision_boundary_03': 1.7,   # Higher for more "pwm_basic"
        'decision_boundary_04': 2.0,   # 
        'decision_boundary_05': 4.0    # Lower for less "random"
    }
    for key, val in alt_a.items():
        print(f"     {key}: {val:.2f}")
    
    # Alternative 2: Conservative
    print("\n   Alternative B (Conservative):")
    alt_b = {
        'decision_boundary_01': 0.08,
        'decision_boundary_02': 1.3,
        'decision_boundary_03': 1.6,
        'decision_boundary_04': 1.9,
        'decision_boundary_05': 5.0
    }
    for key, val in alt_b.items():
        print(f"     {key}: {val:.2f}")
    
    return suggested_boundaries

def generate_config_file(boundaries: Dict, output_path: Path = Path('configs/tuned_boundaries.json')):
    """Generate a config file with new boundaries using CORRECT key names."""
    
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
        "geo_offset_threshold": 0.15
    }
    
    # Add boundaries with correct key names
    for key, value in boundaries.items():
        config[key] = value
    
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
        if intensity < boundaries['decision_boundary_01']:
            decision = 'still'
        elif dynamic < boundaries['decision_boundary_02']:
            decision = 'sine'
        elif dynamic < boundaries['decision_boundary_03']:
            decision = 'pwm_basic'
        elif dynamic < boundaries['decision_boundary_04']:
            decision = 'pwm_extended'
        elif dynamic < boundaries['decision_boundary_05']:
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
    print("INTERACTIVE BOUNDARY TUNER - FIXED VERSION")
    print("="*60)
    print("Now using correct 'decision_boundary_XX' key names!")
    
    # Load current results if available
    results_path = Path('outputs_hybrid/wave_reconstruction_fixed.json')
    current_dist, counts = analyze_current_distribution(results_path)
    
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
    print("5. Keep CURRENT boundaries (TouchDesigner defaults)")
    print("6. Load from existing config file")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        boundaries = suggested
    elif choice == '2':
        boundaries = {
            'decision_boundary_01': 0.05,
            'decision_boundary_02': 1.4,
            'decision_boundary_03': 1.7,
            'decision_boundary_04': 2.0,
            'decision_boundary_05': 4.0
        }
    elif choice == '3':
        boundaries = {
            'decision_boundary_01': 0.08,
            'decision_boundary_02': 1.3,
            'decision_boundary_03': 1.6,
            'decision_boundary_04': 1.9,
            'decision_boundary_05': 5.0
        }
    elif choice == '4':
        boundaries = {}
        for i in range(1, 6):
            key = f'decision_boundary_0{i}'
            default = suggested[key]
            val = input(f"Enter {key} (default {default:.2f}): ").strip()
            boundaries[key] = float(val) if val else default
    elif choice == '6':
        config_file = input("Enter config file path: ").strip()
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            boundaries = {k: v for k, v in loaded_config.items() if k.startswith('decision_boundary')}
            print(f"Loaded boundaries from {config_file}")
        except Exception as e:
            print(f"Error loading config: {e}")
            boundaries = suggested
    else:
        # Keep current TouchDesigner defaults
        boundaries = {
            'decision_boundary_01': 0.22,
            'decision_boundary_02': 0.7,
            'decision_boundary_03': 1.1,
            'decision_boundary_04': 2.2,
            'decision_boundary_05': 7.0
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
    print("\n3. Check results:")
    print("   cat outputs_hybrid/wave_reconstruction_fixed.json")
    print("\n4. If needed, run this tuner again to adjust boundaries")

def analyze_results_and_suggest_adjustments():
    """Analyze the latest results and suggest specific adjustments."""
    
    results_path = Path('outputs_hybrid/wave_reconstruction_fixed.json')
    if not results_path.exists():
        print("No results found. Run wave_type_reconstructor.py first.")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    dist = data['wave_type_distribution']
    
    print("\n" + "="*60)
    print("SPECIFIC ADJUSTMENT RECOMMENDATIONS")
    print("="*60)
    
    # Analyze each wave type
    adjustments = []
    
    if dist.get('still', 0) > 0.20:  # More than 20%
        adjustments.append("• STILL too high: Lower decision_boundary_01 by 0.02-0.04")
    elif dist.get('still', 0) < 0.10:  # Less than 10%
        adjustments.append("• STILL too low: Raise decision_boundary_01 by 0.01-0.02")
    
    if dist.get('sine', 0) < 0.15:  # Less than 15%
        adjustments.append("• SINE too low: Raise decision_boundary_02 by 0.1-0.2")
    elif dist.get('sine', 0) > 0.25:  # More than 25%
        adjustments.append("• SINE too high: Lower decision_boundary_02 by 0.1")
    
    if dist.get('random', 0) > 0.15:  # More than 15%
        adjustments.append("• RANDOM too high: Raise decision_boundary_05 by 0.3-0.5")
    elif dist.get('random', 0) < 0.05:  # Less than 5%
        adjustments.append("• RANDOM too low: Lower decision_boundary_05 by 0.2-0.3")
    
    if adjustments:
        for adj in adjustments:
            print(adj)
    else:
        print("Distribution looks good! No major adjustments needed.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune decision boundaries for wave type distribution')
    parser.add_argument('--analyze', action='store_true', 
                       help='Only analyze current distribution')
    parser.add_argument('--suggest', action='store_true',
                       help='Only show suggestions')
    parser.add_argument('--adjust', action='store_true',
                       help='Analyze results and suggest specific adjustments')
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
            print("No results file found. Using default distribution.")
            suggest_boundary_adjustments({'still': 0.31, 'sine': 0.038, 'pwm_basic': 0.06, 
                                         'pwm_extended': 0.113, 'odd_even': 0.275, 
                                         'square': 0.011, 'random': 0.193})
    elif args.adjust:
        analyze_results_and_suggest_adjustments()
    else:
        # Default: run interactive tuner
        interactive_tuner()