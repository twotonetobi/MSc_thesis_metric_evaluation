#!/usr/bin/env python
"""
Diagnostic script to check what the model is actually outputting
"""

import numpy as np
import pickle
from pathlib import Path
import sys

def check_training_data(train_dir: Path, num_files: int = 5):
    """Check training data files to see what values they contain."""
    
    train_files = sorted(train_dir.glob('*.pkl'))[:num_files]
    
    if not train_files:
        print(f"No training files found in {train_dir}")
        return
    
    print(f"\n{'='*60}")
    print("TRAINING DATA CHECK")
    print('='*60)
    print(f"Checking {len(train_files)} training files...\n")
    
    for train_file in train_files:
        print(f"\nFile: {train_file.name}")
        
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Shape: {data.shape}")
        
        if data.shape[1] != 60:
            print(f"WARNING: Expected 60 dimensions, got {data.shape[1]}")
            continue
        
        # Check wave type values across all groups
        wave_a_values = []
        wave_b_values = []
        
        for group_idx in range(3):
            start = group_idx * 20
            wave_a_values.extend(data[:, start + 2].tolist())
            wave_b_values.extend(data[:, start + 3].tolist())
        
        print(f"  Wave Type A: min={np.min(wave_a_values):.3f}, max={np.max(wave_a_values):.3f}, "
              f"mean={np.mean(wave_a_values):.3f}, std={np.std(wave_a_values):.3f}")
        
        unique_a = np.unique(wave_a_values)
        if len(unique_a) < 10:
            print(f"    -> Unique values: {unique_a}")
        
        print(f"  Wave Type B: min={np.min(wave_b_values):.3f}, max={np.max(wave_b_values):.3f}, "
              f"mean={np.mean(wave_b_values):.3f}, std={np.std(wave_b_values):.3f}")
        
        unique_b = np.unique(wave_b_values)
        if len(unique_b) < 10:
            print(f"    -> Unique values: {unique_b}")
    
    # Summary across all training files
    print(f"\n{'='*60}")
    print("TRAINING DATA SUMMARY")
    print('='*60)
    
    all_wave_a = []
    all_wave_b = []
    
    for train_file in train_dir.glob('*.pkl'):
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
        
        if data.shape[1] == 60:
            for group_idx in range(3):
                start = group_idx * 20
                all_wave_a.extend(data[:, start + 2].tolist())
                all_wave_b.extend(data[:, start + 3].tolist())
    
    if all_wave_a:
        print(f"\nWave Type A across ALL training files:")
        print(f"  Range: [{np.min(all_wave_a):.3f}, {np.max(all_wave_a):.3f}]")
        print(f"  Mean: {np.mean(all_wave_a):.3f}, Std: {np.std(all_wave_a):.3f}")
        
        # Check distribution
        hist, bins = np.histogram(all_wave_a, bins=10)
        print(f"  Distribution (10 bins):")
        for i in range(len(hist)):
            print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}]: {hist[i]} samples")
    
    if all_wave_b:
        print(f"\nWave Type B across ALL training files:")
        print(f"  Range: [{np.min(all_wave_b):.3f}, {np.max(all_wave_b):.3f}]")
        print(f"  Mean: {np.mean(all_wave_b):.3f}, Std: {np.std(all_wave_b):.3f}")

def check_predictions(pred_dir: Path, num_files: int = 5):
    """Check a few prediction files to see what values they contain."""
    
    pred_files = sorted(pred_dir.glob('*.pkl'))[:num_files]
    
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return
    
    print(f"Checking {len(pred_files)} prediction files...\n")
    
    for pred_file in pred_files:
        print(f"\n{'='*60}")
        print(f"File: {pred_file.name}")
        print('='*60)
        
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle 61 dimensions
        if data.shape[1] == 61:
            data = data[:, :60]
        
        print(f"Shape: {data.shape}")
        
        # Check each group
        for group_idx in range(3):
            print(f"\nGroup {group_idx}:")
            start = group_idx * 20
            
            # Standard parameters (first 10)
            params = data[:, start:start+10]
            
            param_names = ['pan_activity', 'tilt_activity', 'wave_type_a', 'wave_type_b',
                          'frequency', 'amplitude', 'offset', 'phase', 'col_hue', 'col_sat']
            
            for i, name in enumerate(param_names):
                values = params[:, i]
                print(f"  {name:15s}: min={values.min():.3f}, max={values.max():.3f}, "
                      f"mean={values.mean():.3f}, std={values.std():.3f}")
                
                # Special check for wave types
                if name == 'wave_type_a':
                    unique = np.unique(values)
                    if len(unique) < 10:
                        print(f"    -> Unique values: {unique}")
                    
                    # Check if all values are the same
                    if values.std() < 0.001:
                        print(f"    -> WARNING: All values are essentially the same!")
                
                if name == 'wave_type_b':
                    unique = np.unique(values)
                    if len(unique) < 10:
                        print(f"    -> Unique values: {unique}")
            
            # Check highlight parameters (next 10)
            highlight_params = data[:, start+10:start+20]
            if np.any(highlight_params != 0):
                print(f"  Highlight params: Active (non-zero values found)")
            else:
                print(f"  Highlight params: Inactive (all zeros)")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    # Check all files for wave type distribution
    all_wave_a = []
    all_wave_b = []
    
    for pred_file in sorted(pred_dir.glob('*.pkl'))[:50]:  # Check first 50 files
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
        
        if data.shape[1] == 61:
            data = data[:, :60]
        
        for group_idx in range(3):
            start = group_idx * 20
            all_wave_a.extend(data[:, start + 2].tolist())
            all_wave_b.extend(data[:, start + 3].tolist())
    
    print(f"\nWave Type A distribution (across first 50 files):")
    unique_a = np.unique(all_wave_a)
    if len(unique_a) < 20:
        print(f"  Unique values: {unique_a}")
    else:
        print(f"  Range: [{np.min(all_wave_a):.3f}, {np.max(all_wave_a):.3f}]")
    print(f"  Mean: {np.mean(all_wave_a):.3f}, Std: {np.std(all_wave_a):.3f}")
    
    print(f"\nWave Type B distribution (across first 50 files):")
    unique_b = np.unique(all_wave_b)
    if len(unique_b) < 20:
        print(f"  Unique values: {unique_b}")
    else:
        print(f"  Range: [{np.min(all_wave_b):.3f}, {np.max(all_wave_b):.3f}]")
    print(f"  Mean: {np.mean(all_wave_b):.3f}, Std: {np.std(all_wave_b):.3f}")
    
    # Classification test
    print("\nWave Type Classification Test:")
    test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    wave_a_boundaries = {
        'sine': (0.0, 0.2),
        'saw_up': (0.2, 0.4),
        'saw_down': (0.4, 0.6),
        'square': (0.6, 0.8),
        'linear': (0.8, 1.0)
    }
    
    for val in test_values:
        for wave_name, (min_v, max_v) in wave_a_boundaries.items():
            if min_v <= val <= max_v:
                print(f"  {val:.1f} -> {wave_name}")
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check model predictions')
    parser.add_argument('--pred_dir', type=str, 
                       default='data/conformer_osci/light_segments',
                       help='Directory with predictions')
    parser.add_argument('--training_dir', type=str,
                       default='data/training_data/oscillator_params',
                       help='Directory with training data')
    parser.add_argument('--mode', type=str, choices=['pred', 'train', 'both'],
                       default='pred',
                       help='Check predictions, training data, or both')
    args = parser.parse_args()
    
    if args.mode in ['pred', 'both']:
        check_predictions(Path(args.pred_dir))
    
    if args.mode in ['train', 'both']:
        check_training_data(Path(args.training_dir))