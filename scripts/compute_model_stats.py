#!/usr/bin/env python
"""
Compute model statistics from predictions for comparison with training data.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict

def compute_model_statistics(pred_dir: Path) -> Dict:
    """Extract statistics from model predictions matching training data format."""
    
    print(f"Computing statistics from {pred_dir}")
    
    # Parameter names
    param_names = [
        'pan_activity', 'tilt_activity', 'wave_type_a', 'wave_type_b',
        'frequency', 'amplitude', 'offset', 'phase', 'col_hue', 'col_sat'
    ]
    
    # Containers for all data
    all_params = {name: [] for name in param_names}
    segment_specific = {}
    
    # Process each prediction file
    pred_files = sorted(pred_dir.glob('*.pkl'))
    
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return {}
    
    print(f"Processing {len(pred_files)} prediction files...")
    
    for i, pred_file in enumerate(pred_files):
        if i % 50 == 0:
            print(f"  Processed {i} files...")
        
        # Load predictions
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle 61 dimensions
        if data.shape[1] == 61:
            data = data[:, :60]
        elif data.shape[1] != 60:
            print(f"  Warning: Skipping {pred_file.stem} - unexpected shape {data.shape}")
            continue
        
        # Extract segment type from filename
        filename = pred_file.stem
        if '_part_' in filename:
            parts = filename.split('_part_')
            if len(parts) > 1:
                # Last part after part_ contains segment type
                segment_part = parts[-1]
                # Extract segment type (e.g., "01_intro" -> "intro")
                if '_' in segment_part:
                    seg_type = segment_part.split('_', 1)[1].lower()
                else:
                    seg_type = segment_part.lower()
                
                # Normalize segment types
                if 'vers' in seg_type:
                    seg_type = 'verse'
                elif 'choru' in seg_type:
                    seg_type = 'chorus'
                elif 'intro' in seg_type:
                    seg_type = 'intro'
                elif 'outro' in seg_type:
                    seg_type = 'outro'
                elif 'bridg' in seg_type:
                    seg_type = 'bridge'
                elif 'inst' in seg_type:
                    seg_type = 'instrumental'
                
                if seg_type not in segment_specific:
                    segment_specific[seg_type] = {name: [] for name in param_names}
        else:
            seg_type = 'unknown'
        
        # Process standard parameters (3 groups)
        for group_idx in range(3):
            start_idx = group_idx * 20
            standard_params = data[:, start_idx:start_idx+10]
            
            # Collect parameters
            for param_idx, param_name in enumerate(param_names):
                values = standard_params[:, param_idx].flatten()
                all_params[param_name].extend(values)
                
                if seg_type != 'unknown' and seg_type in segment_specific:
                    segment_specific[seg_type][param_name].extend(values)
    
    print(f"Processed {len(pred_files)} files")
    
    # Compute statistics in same format as training data
    statistics = {'global': {}, 'per_segment': {}}
    
    # Global statistics
    print("\nComputing global statistics...")
    for param_name, values in all_params.items():
        if len(values) > 0:
            values = np.array(values)
            statistics['global'][param_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q25': float(np.percentile(values, 25)),
                'q50': float(np.percentile(values, 50)),
                'q75': float(np.percentile(values, 75))
            }
            print(f"  {param_name}: mean={statistics['global'][param_name]['mean']:.3f}, "
                  f"std={statistics['global'][param_name]['std']:.3f}")
    
    # Per-segment statistics
    print("\nComputing per-segment statistics...")
    for seg_type, params in segment_specific.items():
        statistics['per_segment'][seg_type] = {}
        print(f"  {seg_type}:")
        for param_name, values in params.items():
            if len(values) > 0:
                values = np.array(values)
                statistics['per_segment'][seg_type][param_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values)
                }
                if param_name in ['amplitude', 'frequency', 'wave_type_a']:
                    print(f"    {param_name}: mean={statistics['per_segment'][seg_type][param_name]['mean']:.3f}")
    
    # Analyze wave type actual usage
    print("\nWave Type Analysis:")
    
    # Wave type boundaries
    wave_a_boundaries = {
        'sine': (0.0, 0.2),
        'saw_up': (0.2, 0.4),
        'saw_down': (0.4, 0.6),
        'square': (0.6, 0.8),
        'linear': (0.8, 1.0)
    }
    
    wave_b_boundaries = {
        'other': (0.0, 0.25),
        'plateau': (0.25, 0.5),
        'gaussian_single': (0.5, 0.75),
        'gaussian_double': (0.75, 1.0)
    }
    
    # Classify wave types
    wave_a_values = all_params['wave_type_a']
    wave_a_classified = []
    for val in wave_a_values:
        val = np.clip(val, 0.0, 1.0)
        for wave_name, (min_v, max_v) in wave_a_boundaries.items():
            if min_v <= val <= max_v:
                wave_a_classified.append(wave_name)
                break
    
    wave_b_values = all_params['wave_type_b']
    wave_b_classified = []
    for val in wave_b_values:
        val = np.clip(val, 0.0, 1.0)
        for wave_name, (min_v, max_v) in wave_b_boundaries.items():
            if min_v <= val <= max_v:
                wave_b_classified.append(wave_name)
                break
    
    # Count distributions
    from collections import Counter
    wave_a_dist = Counter(wave_a_classified)
    wave_b_dist = Counter(wave_b_classified)
    
    print(f"  Wave Type A distribution:")
    total_a = sum(wave_a_dist.values())
    for wave, count in wave_a_dist.most_common():
        print(f"    {wave}: {count} ({100*count/total_a:.1f}%)")
    
    print(f"  Wave Type B distribution:")
    total_b = sum(wave_b_dist.values())
    for wave, count in wave_b_dist.most_common():
        print(f"    {wave}: {count} ({100*count/total_b:.1f}%)")
    
    # Add wave distributions to statistics
    statistics['wave_distributions'] = {
        'wave_type_a': {k: v/total_a for k, v in wave_a_dist.items()},
        'wave_type_b': {k: v/total_b for k, v in wave_b_dist.items()}
    }
    
    return statistics

def save_model_statistics(stats: Dict, output_path: Path):
    """Save model statistics for report generation."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # Save as JSON
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nModel statistics saved to:")
    print(f"  {output_path.with_suffix('.pkl')}")
    print(f"  {output_path.with_suffix('.json')}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute model statistics')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with model predictions')
    parser.add_argument('--output_path', type=str,
                       default='outputs_oscillator/model_statistics',
                       help='Output path for statistics (without extension)')
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {pred_dir}")
        return
    
    # Compute statistics
    stats = compute_model_statistics(pred_dir)
    
    if stats:
        # Save statistics
        save_model_statistics(stats, Path(args.output_path))
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Show key issues
        if 'global' in stats:
            wave_a_mean = stats['global']['wave_type_a']['mean']
            wave_a_std = stats['global']['wave_type_a']['std']
            print(f"\nWave Type A: mean={wave_a_mean:.3f}, std={wave_a_std:.3f}")
            
            if wave_a_std < 0.01:
                print("  ⚠️  WARNING: Model outputs nearly constant wave_type_a values!")
                print("     This explains why all segments show 100% sine.")
                print("     The model needs retraining with proper diversity.")
            
            wave_b_mean = stats['global']['wave_type_b']['mean']
            wave_b_std = stats['global']['wave_type_b']['std']
            print(f"\nWave Type B: mean={wave_b_mean:.3f}, std={wave_b_std:.3f}")
            
            if wave_b_std < 0.01:
                print("  ⚠️  WARNING: Model outputs nearly constant wave_type_b values!")

if __name__ == '__main__':
    main()