#!/usr/bin/env python
"""
Custom Boundary Configuration Generator
Targets: sine ~20%, random ~7%, square ~5%, less still
"""

import json
from pathlib import Path
import numpy as np

def analyze_adjustments_needed():
    """Analyze what needs to change from aggressive config."""
    
    print("="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    current = {
        'still': 31.0,
        'sine': 3.8,
        'pwm_basic': 6.0,
        'pwm_extended': 11.3,
        'odd_even': 27.5,
        'square': 1.1,
        'random': 19.3
    }
    
    target = {
        'still': 15.0,      # Reduce from 31%
        'sine': 20.0,       # Increase from 3.8%
        'pwm_basic': 10.0,  # Increase from 6%
        'pwm_extended': 15.0,  # Slight increase from 11.3%
        'odd_even': 25.0,   # Slight reduction from 27.5%
        'square': 5.0,      # Increase from 1.1%
        'random': 7.0       # Reduce from 19.3%
    }
    
    print("\nCurrent → Target:")
    for key in ['still', 'sine', 'pwm_basic', 'pwm_extended', 'odd_even', 'square', 'random']:
        curr = current[key]
        targ = target[key]
        diff = targ - curr
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {key:15s}: {curr:5.1f}% → {targ:5.1f}% ({arrow} {abs(diff):4.1f}%)")
    
    return current, target

def generate_custom_config():
    """Generate custom configuration to achieve target distribution."""
    
    # Aggressive boundaries that gave us the current distribution
    aggressive = {
        'boundary_01': 0.10,   # Controls still vs others
        'boundary_02': 1.0,    # Controls sine vs pwm_basic
        'boundary_03': 1.5,    # Controls pwm_basic vs pwm_extended
        'boundary_04': 2.0,    # Controls pwm_extended vs odd_even
        'boundary_05': 3.0     # Controls odd_even vs random/square
    }
    
    # Custom boundaries to achieve target
    # Key insights:
    # - To increase sine from 3.8% to 20%: need MUCH higher boundary_02
    # - To reduce random from 19.3% to 7%: need higher boundary_05
    # - To reduce still from 31% to less: need lower boundary_01
    
    custom = {
        'boundary_01': 0.05,   # Lower to reduce "still" (31% → 15%)
        'boundary_02': 1.4,    # Much higher to increase "sine" (3.8% → 20%)
        'boundary_03': 1.7,    # Slightly higher for more "pwm_basic"
        'boundary_04': 2.0,    # Keep same
        'boundary_05': 4.0     # Higher to reduce "random" (19.3% → 7%)
    }
    
    print("\n" + "="*60)
    print("RECOMMENDED CUSTOM BOUNDARIES")
    print("="*60)
    
    print("\nAggressive → Custom:")
    for key in aggressive:
        agg = aggressive[key]
        cust = custom[key]
        diff = cust - agg
        print(f"  {key}: {agg:.2f} → {cust:.2f} (change: {diff:+.2f})")
    
    print("\n" + "="*60)
    print("ALTERNATIVE CONFIGURATIONS TO TRY")
    print("="*60)
    
    # Alternative 1: More aggressive for sine
    print("\nOption A - Maximum Sine (if 20% sine is critical):")
    option_a = {
        'boundary_01': 0.03,   # Very low for minimal still
        'boundary_02': 1.6,    # Very high for maximum sine range
        'boundary_03': 1.8,    # 
        'boundary_04': 2.1,    # 
        'boundary_05': 4.5     # Higher to minimize random
    }
    for key, val in option_a.items():
        print(f"  {key}: {val:.2f}")
    
    # Alternative 2: Balanced with focus on reducing random
    print("\nOption B - Minimal Random (if <7% random is critical):")
    option_b = {
        'boundary_01': 0.08,   # 
        'boundary_02': 1.3,    # 
        'boundary_03': 1.6,    # 
        'boundary_04': 1.9,    # 
        'boundary_05': 5.0     # Very high to minimize random
    }
    for key, val in option_b.items():
        print(f"  {key}: {val:.2f}")
    
    # Alternative 3: Fine-tuned based on dynamic score analysis
    print("\nOption C - Fine-tuned (recommended to try first):")
    option_c = {
        'boundary_01': 0.06,   # Reduce still moderately
        'boundary_02': 1.35,   # Increase sine significantly
        'boundary_03': 1.65,   # Slight increase for pwm_basic
        'boundary_04': 1.95,   # Slight reduction for odd_even
        'boundary_05': 4.2     # Increase to reduce random to ~7%
    }
    for key, val in option_c.items():
        print(f"  {key}: {val:.2f}")
    
    return custom, option_a, option_b, option_c

def save_configs():
    """Save all configuration options."""
    
    custom, option_a, option_b, option_c = generate_custom_config()
    
    # Base configuration template
    base_config = {
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
        "oscillation_threshold": 8,  # Reduced from 10 to increase dynamics
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15
    }
    
    # Save each configuration
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    configs_to_save = {
        'custom_balanced.json': custom,
        'custom_max_sine.json': option_a,
        'custom_min_random.json': option_b,
        'custom_recommended.json': option_c
    }
    
    for filename, boundaries in configs_to_save.items():
        config = base_config.copy()
        # Add the boundary values to the config
        for key, value in boundaries.items():
            config[key] = value
        
        output_path = configs_dir / filename
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved: {output_path}")
    
    print("\n" + "="*60)
    print("TESTING RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. Start with the recommended config:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/custom_recommended.json")
    
    print("\n2. If sine is still too low, try max_sine:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/custom_max_sine.json")
    
    print("\n3. If random is still too high, try min_random:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/custom_min_random.json")
    
    print("\n4. Once satisfied, run full dataset:")
    print("   python scripts/wave_type_reconstructor.py --config configs/custom_recommended.json")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    print("\nTo achieve your targets:")
    print("• SINE 20%: The dynamic range 1.0-1.35 was only 3.8% of decisions")
    print("  → Need to expand this range significantly (up to 1.6)")
    print("• RANDOM 7%: Dynamic > 3.0 was 19.3% of decisions")
    print("  → Need to raise threshold to 4.0-4.5")
    print("• STILL less: Intensity < 0.10 was 31% of decisions")
    print("  → Need to lower threshold to 0.05-0.06")
    
    print("\nNote about SQUARE vs RANDOM:")
    print("• Both occur when dynamic > boundary_05")
    print("• Decision between them is based on BPM > 135")
    print("• To get more square: Could lower BPM threshold in config")
    
    # Additional analysis
    print("\n" + "="*60)
    print("DYNAMIC SCORE DISTRIBUTION ESTIMATE")
    print("="*60)
    
    print("\nBased on your results, the dynamic scores roughly distribute as:")
    print("  0.0 - 1.0:  ~35% (mostly sine range)")
    print("  1.0 - 1.5:  ~10% (pwm_basic range)")
    print("  1.5 - 2.0:  ~20% (pwm_extended range)")
    print("  2.0 - 3.0:  ~15% (odd_even range)")
    print("  > 3.0:      ~20% (random/square range)")
    
    print("\nThis suggests your dynamics are clustered around 0-1 and >3")
    print("You may also want to adjust oscillation_threshold or peak_height")

if __name__ == '__main__':
    # Analyze what needs to change
    current, target = analyze_adjustments_needed()
    
    # Generate and save configurations
    save_configs()
    
    print("\n" + "="*60)
    print("QUICK ADJUSTMENT GUIDE")
    print("="*60)
    
    print("\nIf after testing you need quick adjustments:")
    print("• Too much STILL: Lower boundary_01 by 0.02")
    print("• Too little SINE: Raise boundary_02 by 0.1") 
    print("• Too much RANDOM: Raise boundary_05 by 0.3")
    print("• Too little SQUARE: Lower bpm_thresholds['high'] to 120")