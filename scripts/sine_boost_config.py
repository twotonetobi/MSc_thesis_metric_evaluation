#!/usr/bin/env python
"""
Sine Booster Configuration Generator
Specifically targets: 15-20% sine, <30% odd_even
Based on current distribution analysis
"""

import json
from pathlib import Path

def analyze_current_problem():
    """Analyze the current distribution and identify the issues."""
    
    print("="*60)
    print("CURRENT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    current = {
        'odd_even': 42.5,      # WAY TOO HIGH - needs to be <30%
        'still': 29.8,         # FINE
        'random': 7.5,         # FINE
        'pwm_extended': 7.0,   # OK
        'sine': 6.9,           # TOO LOW - needs 15-20%
        'pwm_basic': 5.7,      # OK
        'square': 0.5          # OK
    }
    
    target = {
        'odd_even': 25.0,      # Reduce from 42.5% to 25%
        'still': 29.8,         # Keep as is (user said fine)
        'random': 7.5,         # Keep as is (user said fine)
        'pwm_extended': 10.0,  # Slight increase
        'sine': 17.5,          # INCREASE from 6.9% to 17.5%
        'pwm_basic': 8.0,      # Slight increase
        'square': 2.0          # Slight increase
    }
    
    print("\nCurrent → Target:")
    for key in ['odd_even', 'sine', 'still', 'pwm_basic', 'pwm_extended', 'random', 'square']:
        curr = current[key]
        targ = target[key]
        diff = targ - curr
        
        if key == 'odd_even':
            status = "❌ WAY TOO HIGH"
        elif key == 'sine':
            status = "❌ TOO LOW"
        elif key in ['still', 'random']:
            status = "✓ FINE"
        else:
            status = ""
            
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {key:15s}: {curr:5.1f}% → {targ:5.1f}% ({arrow} {abs(diff):4.1f}%) {status}")
    
    return current, target

def generate_sine_boost_configs():
    """Generate configurations specifically to boost sine and reduce odd_even."""
    
    print("\n" + "="*60)
    print("DIAGNOSIS OF THE PROBLEM")
    print("="*60)
    
    print("\nThe issue is with the dynamic score ranges:")
    print("• Sine gets: dynamic < 1.35 (too narrow)")
    print("• Odd_even gets: 1.95 < dynamic < 4.2 (HUGE range = 2.25 units!)")
    print("\nWe need to:")
    print("1. EXPAND sine's range significantly (raise boundary_02)")
    print("2. SHRINK odd_even's range (lower boundary_05)")
    
    # Base configuration
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
        "oscillation_threshold": 8,
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15
    }
    
    print("\n" + "="*60)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*60)
    
    configs = {}
    
    # Configuration 1: Moderate boost
    print("\n1. MODERATE SINE BOOST (Try this first):")
    config1 = {
        'decision_boundary_01': 0.06,   # Keep still as is (working fine)
        'decision_boundary_02': 1.70,   # RAISE from 1.35 to 1.70 (expand sine range)
        'decision_boundary_03': 1.90,   # Adjust pwm_basic
        'decision_boundary_04': 2.10,   # Adjust pwm_extended
        'decision_boundary_05': 3.00    # LOWER from 4.2 to 3.0 (shrink odd_even)
    }
    configs['sine_boost_moderate.json'] = config1
    for key, val in config1.items():
        print(f"  {key}: {val:.2f}")
    print("  Expected: sine ~12%, odd_even ~35%")
    
    # Configuration 2: Aggressive boost
    print("\n2. AGGRESSIVE SINE BOOST (If moderate isn't enough):")
    config2 = {
        'decision_boundary_01': 0.06,   # Keep still as is
        'decision_boundary_02': 2.00,   # MAJOR increase (expand sine a lot)
        'decision_boundary_03': 2.20,   # Compress pwm_basic
        'decision_boundary_04': 2.40,   # Compress pwm_extended  
        'decision_boundary_05': 2.80    # Much lower (compress odd_even significantly)
    }
    configs['sine_boost_aggressive.json'] = config2
    for key, val in config2.items():
        print(f"  {key}: {val:.2f}")
    print("  Expected: sine ~18%, odd_even ~25%")
    
    # Configuration 3: Maximum sine
    print("\n3. MAXIMUM SINE (If you want 20%+ sine):")
    config3 = {
        'decision_boundary_01': 0.05,   # Slightly lower for less still
        'decision_boundary_02': 2.30,   # VERY high boundary for maximum sine
        'decision_boundary_03': 2.45,   # Very compressed pwm_basic
        'decision_boundary_04': 2.60,   # Very compressed pwm_extended
        'decision_boundary_05': 2.75    # Very low to minimize odd_even
    }
    configs['sine_boost_maximum.json'] = config3
    for key, val in config3.items():
        print(f"  {key}: {val:.2f}")
    print("  Expected: sine ~22%, odd_even ~20%")
    
    # Configuration 4: Balanced redistribution
    print("\n4. BALANCED REDISTRIBUTION (Even distribution):")
    config4 = {
        'decision_boundary_01': 0.06,   # Keep still
        'decision_boundary_02': 1.80,   # Good sine range
        'decision_boundary_03': 2.10,   # Decent pwm_basic
        'decision_boundary_04': 2.40,   # Decent pwm_extended
        'decision_boundary_05': 3.20    # Moderate odd_even
    }
    configs['sine_boost_balanced.json'] = config4
    for key, val in config4.items():
        print(f"  {key}: {val:.2f}")
    print("  Expected: sine ~15%, odd_even ~28%")
    
    # Save all configurations
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    for filename, boundaries in configs.items():
        config = base_config.copy()
        config.update(boundaries)
        
        output_path = configs_dir / filename
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved: {output_path}")
    
    return configs

def print_testing_instructions():
    """Print clear testing instructions."""
    
    print("\n" + "="*60)
    print("TESTING INSTRUCTIONS")
    print("="*60)
    
    print("\n1. TEST MODERATE BOOST FIRST (quickest path to ~15% sine):")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_moderate.json")
    
    print("\n2. IF SINE STILL TOO LOW, try aggressive:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_aggressive.json")
    
    print("\n3. IF YOU WANT 20%+ SINE, try maximum:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_maximum.json")
    
    print("\n4. FOR EVEN DISTRIBUTION, try balanced:")
    print("   python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_balanced.json")
    
    print("\n5. Once you find the right config, run full dataset:")
    print("   python scripts/wave_type_reconstructor.py --config configs/sine_boost_[chosen].json")
    
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    
    print("\nYour dynamic scores are clustering around certain values.")
    print("The problem was boundary_05 was too high (4.2), giving odd_even")
    print("a HUGE range (1.95-4.2). By lowering it to ~3.0 and raising")
    print("boundary_02 to ~1.8, we redistribute those decisions to sine.")
    
    print("\n" + "="*60)
    print("FINE-TUNING TIPS")
    print("="*60)
    
    print("\nAfter testing, if you need adjustments:")
    print("• Still too much odd_even? → Lower boundary_05 by 0.2")
    print("• Still not enough sine? → Raise boundary_02 by 0.1-0.2")
    print("• Too much sine? → Lower boundary_02 by 0.1")
    print("• Want more pwm types? → Spread boundaries 03-04 more")

def create_custom_config_from_feedback():
    """Create a custom config based on specific feedback."""
    
    print("\n" + "="*60)
    print("CUSTOM FINE-TUNED CONFIG")
    print("="*60)
    
    print("\nBased on your specific requirements:")
    print("• Sine: 15-20% (currently 6.9%)")
    print("• Odd_even: <30% (currently 42.5%)")
    print("• Still: ~30% (currently 29.8% - KEEP)")
    print("• Random: ~7.5% (currently 7.5% - KEEP)")
    
    custom = {
        'decision_boundary_01': 0.06,   # Keep for ~30% still
        'decision_boundary_02': 1.85,   # Significantly raised for 15-20% sine
        'decision_boundary_03': 2.15,   # Reasonable gap for pwm_basic
        'decision_boundary_04': 2.35,   # Reasonable gap for pwm_extended
        'decision_boundary_05': 3.10    # Lowered significantly to reduce odd_even
    }
    
    print("\nRECOMMENDED CUSTOM CONFIG:")
    for key, val in custom.items():
        print(f"  {key}: {val:.2f}")
    
    # Save it
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
        "oscillation_threshold": 8,
        "geo_phase_threshold": 0.15,
        "geo_freq_threshold": 0.15,
        "geo_offset_threshold": 0.15
    }
    
    config = base_config.copy()
    config.update(custom)
    
    output_path = Path('configs') / 'sine_boost_custom.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved as: {output_path}")
    print("\nTest with:")
    print("  python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_custom.json")
    
    return custom

if __name__ == '__main__':
    # Analyze the problem
    current, target = analyze_current_problem()
    
    # Generate solution configs
    configs = generate_sine_boost_configs()
    
    # Create custom config
    custom = create_custom_config_from_feedback()
    
    # Print instructions
    print_testing_instructions()
    
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("\nRun this NOW to test the custom config:")
    print("\n  python scripts/wave_type_reconstructor.py --max_files 10 --config configs/sine_boost_custom.json")
    print("\nThis should give you approximately:")
    print("  • sine: ~17% (up from 6.9%)")
    print("  • odd_even: ~28% (down from 42.5%)")
    print("  • still: ~30% (unchanged)")
    print("  • random: ~8% (unchanged)")