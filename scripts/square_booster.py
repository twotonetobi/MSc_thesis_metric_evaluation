#!/usr/bin/env python
"""
Square Booster & Random Reducer Configuration
Targets: random <10%, square ~5%, while keeping sine and odd_even good
"""

import json
from pathlib import Path

def analyze_current_distribution():
    """Analyze the current improved distribution."""
    
    print("="*60)
    print("CURRENT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    current = {
        'still': 29.8,         # ✅ PERFECT
        'random': 18.4,        # ❌ TOO HIGH (need <10%)
        'sine': 17.6,          # ✅ PERFECT (target 15-20%)
        'odd_even': 15.0,      # ✅ PERFECT (target <30%)
        'pwm_basic': 11.1,     # ✅ GOOD
        'pwm_extended': 7.0,   # ✅ GOOD
        'square': 1.1          # ❌ TOO LOW (need ~5%)
    }
    
    target = {
        'still': 29.8,         # Keep as is
        'random': 9.0,         # Reduce from 18.4% to 9%
        'sine': 17.6,          # Keep as is (perfect!)
        'odd_even': 15.0,      # Keep as is (perfect!)
        'pwm_basic': 11.1,     # Keep as is
        'pwm_extended': 7.0,   # Keep as is
        'square': 5.0          # Increase from 1.1% to 5%
    }
    
    print("\nCurrent → Target:")
    for key in ['still', 'sine', 'pwm_basic', 'pwm_extended', 'odd_even', 'square', 'random']:
        curr = current[key]
        targ = target[key]
        diff = targ - curr
        
        if key == 'random':
            status = "❌ TOO HIGH"
        elif key == 'square':
            status = "❌ TOO LOW"
        elif key in ['sine', 'odd_even', 'still']:
            status = "✅ PERFECT"
        else:
            status = "✅ GOOD"
            
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {key:15s}: {curr:5.1f}% → {targ:5.1f}% ({arrow} {abs(diff):4.1f}%) {status}")
    
    return current, target

def generate_square_boost_configs():
    """Generate configs to boost square and reduce random."""
    
    print("\n" + "="*60)
    print("SOLUTION STRATEGY")
    print("="*60)
    
    print("\nThe issue:")
    print("• Random + Square = 19.5% total (when dynamic > boundary_05)")
    print("• Currently: 18.4% random, 1.1% square")
    print("• Need: ~9% random, ~5% square = 14% total")
    print("\nTwo-part solution:")
    print("1. RAISE boundary_05 to reduce total going to random/square")
    print("2. LOWER BPM threshold to convert more random → square")
    
    # Base configuration
    base_config = {
        "max_cycles_per_second": 4.0,
        "max_phase_cycles_per_second": 8.0,
        "led_count": 33,
        "virtual_led_count": 8,
        "fps": 30,
        "mode": "hard",
        "optimization": {
            "alpha": 1.0,
            "beta": 1.0,
            "delta": 1.0
        },
        "oscillation_threshold": 8,
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15
    }
    
    print("\n" + "="*60)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*60)
    
    configs = {}
    
    # Configuration 1: Balanced approach
    print("\n1. BALANCED APPROACH (Try this first):")
    config1 = {
        'decision_boundary_01': 0.06,   # Keep (still is perfect)
        'decision_boundary_02': 1.85,   # Keep (sine is perfect)
        'decision_boundary_03': 2.15,   # Keep (pwm_basic is good)
        'decision_boundary_04': 2.35,   # Keep (pwm_extended is good)
        'decision_boundary_05': 3.60,   # RAISE from 3.1 to 3.6 (reduce total random+square)
        'bpm_thresholds': {
            'low': 80,
            'high': 110    # LOWER from 135 to 110 (more square vs random)
        }
    }
    configs['final_balanced.json'] = config1
    for key, val in config1.items():
        if key != 'bpm_thresholds':
            print(f"  {key}: {val:.2f}")
    print(f"  BPM threshold: {config1['bpm_thresholds']['high']} (lowered from 135)")
    print("  Expected: random ~9%, square ~5%")
    
    # Configuration 2: More aggressive square boost
    print("\n2. AGGRESSIVE SQUARE BOOST:")
    config2 = {
        'decision_boundary_01': 0.06,
        'decision_boundary_02': 1.85,
        'decision_boundary_03': 2.15,
        'decision_boundary_04': 2.35,
        'decision_boundary_05': 3.80,   # Higher boundary (less total random+square)
        'bpm_thresholds': {
            'low': 80,
            'high': 100    # Even lower BPM threshold (more square)
        }
    }
    configs['final_aggressive_square.json'] = config2
    for key, val in config2.items():
        if key != 'bpm_thresholds':
            print(f"  {key}: {val:.2f}")
    print(f"  BPM threshold: {config2['bpm_thresholds']['high']}")
    print("  Expected: random ~7%, square ~6%")
    
    # Configuration 3: Conservative approach
    print("\n3. CONSERVATIVE APPROACH:")
    config3 = {
        'decision_boundary_01': 0.06,
        'decision_boundary_02': 1.85,
        'decision_boundary_03': 2.15,
        'decision_boundary_04': 2.35,
        'decision_boundary_05': 3.50,   # Moderate raise
        'bpm_thresholds': {
            'low': 80,
            'high': 115    # Moderate BPM reduction
        }
    }
    configs['final_conservative.json'] = config3
    for key, val in config3.items():
        if key != 'bpm_thresholds':
            print(f"  {key}: {val:.2f}")
    print(f"  BPM threshold: {config3['bpm_thresholds']['high']}")
    print("  Expected: random ~10%, square ~4%")
    
    # Configuration 4: Maximum square
    print("\n4. MAXIMUM SQUARE (If you want 7%+ square):")
    config4 = {
        'decision_boundary_01': 0.06,
        'decision_boundary_02': 1.85,
        'decision_boundary_03': 2.15,
        'decision_boundary_04': 2.35,
        'decision_boundary_05': 3.70,
        'bpm_thresholds': {
            'low': 80,
            'high': 90     # Very low BPM threshold (maximum square)
        }
    }
    configs['final_max_square.json'] = config4
    for key, val in config4.items():
        if key != 'bpm_thresholds':
            print(f"  {key}: {val:.2f}")
    print(f"  BPM threshold: {config4['bpm_thresholds']['high']}")
    print("  Expected: random ~6%, square ~8%")
    
    # Save all configurations
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    for filename, config_dict in configs.items():
        config = base_config.copy()
        # Handle the bpm_thresholds separately
        bpm = config_dict.pop('bpm_thresholds', {'low': 80, 'high': 135})
        config['bpm_thresholds'] = bpm
        config.update(config_dict)
        
        output_path = configs_dir / filename
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved: {output_path}")
    
    return configs

def create_optimal_config():
    """Create the optimal config based on all requirements."""
    
    print("\n" + "="*60)
    print("OPTIMAL FINAL CONFIG")
    print("="*60)
    
    print("\nBased on ALL your requirements:")
    print("• sine: 15-20% ✅ (currently 17.6%)")
    print("• odd_even: <30% ✅ (currently 15.0%)")
    print("• still: ~30% ✅ (currently 29.8%)")
    print("• random: <10% (currently 18.4%)")
    print("• square: ~5% (currently 1.1%)")
    
    optimal = {
        'decision_boundary_01': 0.06,   # Perfect for still
        'decision_boundary_02': 1.85,   # Perfect for sine
        'decision_boundary_03': 2.15,   # Good for pwm_basic
        'decision_boundary_04': 2.35,   # Good for pwm_extended
        'decision_boundary_05': 3.65,   # Raised to reduce random
        'bpm_thresholds': {
            'low': 80,
            'high': 108    # Lowered to boost square
        }
    }
    
    print("\nOPTIMAL CONFIG:")
    for key, val in optimal.items():
        if key != 'bpm_thresholds':
            print(f"  {key}: {val:.2f}")
    print(f"  BPM threshold high: {optimal['bpm_thresholds']['high']} (was 135)")
    
    # Save it
    base_config = {
        "max_cycles_per_second": 4.0,
        "max_phase_cycles_per_second": 8.0,
        "led_count": 33,
        "virtual_led_count": 8,
        "fps": 30,
        "mode": "hard",
        "optimization": {
            "alpha": 1.0,
            "beta": 1.0,
            "delta": 1.0
        },
        "oscillation_threshold": 8,
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15
    }
    
    config = base_config.copy()
    config.update({k: v for k, v in optimal.items() if k != 'bpm_thresholds'})
    config['bpm_thresholds'] = optimal['bpm_thresholds']
    
    output_path = Path('configs') / 'final_optimal.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved as: {output_path}")
    
    print("\nEXPECTED DISTRIBUTION:")
    print("  • still: ~30%")
    print("  • sine: ~18%")
    print("  • odd_even: ~15%")
    print("  • pwm_basic: ~11%")
    print("  • pwm_extended: ~7%")
    print("  • random: ~8-9%")
    print("  • square: ~5-6%")
    
    return optimal

def print_testing_instructions():
    """Print clear testing instructions."""
    
    print("\n" + "="*60)
    print("TESTING INSTRUCTIONS")
    print("="*60)
    
    print("\n1. TEST THE OPTIMAL CONFIG FIRST:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/final_optimal.json")
    
    print("\n2. If you need more square, try aggressive:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/final_aggressive_square.json")
    
    print("\n3. If random is still too high, try balanced:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/final_balanced.json")
    
    print("\n4. Once satisfied, run full dataset:")
    print("   python scripts/wave_type_reconstructor.py --config configs/final_optimal.json")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    print("\n• Random vs Square is determined by BPM")
    print("• Songs with BPM > threshold → square")
    print("• Songs with BPM ≤ threshold → random")
    print("• By lowering the BPM threshold, more songs qualify for square")
    
    print("\n" + "="*60)
    print("FINE-TUNING TIPS")
    print("="*60)
    
    print("\nAfter testing, adjust if needed:")
    print("• Still too much random? → Raise boundary_05 by 0.1")
    print("• Need more square? → Lower BPM threshold by 5")
    print("• Too much square now? → Raise BPM threshold by 5")

if __name__ == '__main__':
    # Analyze current state
    current, target = analyze_current_distribution()
    
    # Generate solution configs
    configs = generate_square_boost_configs()
    
    # Create optimal config
    optimal = create_optimal_config()
    
    # Print instructions
    print_testing_instructions()
    
    print("\n" + "="*60)
    print("🎯 QUICK START")
    print("="*60)
    print("\nRun this NOW to test the optimal config:")
    print("\n  python scripts/wave_type_reconstructor.py --max_files 10 --config configs/final_optimal.json")
    print("\nThis should give you the perfect distribution!")