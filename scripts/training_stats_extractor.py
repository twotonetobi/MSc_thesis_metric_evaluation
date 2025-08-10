#!/usr/bin/env python
"""
Extract statistics from training data oscillator parameters.
This creates the baseline distributions for comparison.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List
from scipy import stats
import matplotlib.pyplot as plt

class TrainingStatsExtractor:
    """Extract and save statistics from training oscillator data."""
    
    def __init__(self):
        self.param_names = [
            'pan_activity', 'tilt_activity', 'wave_type_a', 'wave_type_b',
            'frequency', 'amplitude', 'offset', 'phase', 'col_hue', 'col_sat'
        ]
        
        # Wave type mappings with complete coverage (no gaps)
        self.wave_a_boundaries = {
            'sine': (0.0, 0.2),      # 0.0-0.2 → sine (center: 0.1)
            'saw_up': (0.2, 0.4),    # 0.2-0.4 → saw_up (center: 0.3)
            'saw_down': (0.4, 0.6),  # 0.4-0.6 → saw_down (center: 0.5)
            'square': (0.6, 0.8),    # 0.6-0.8 → square (center: 0.7)
            'linear': (0.8, 1.0)     # 0.8-1.0 → linear (center: 0.9)
        }
        
        self.wave_b_boundaries = {
            'other': (0.0, 0.25),           # 0.0-0.25 → other (center: 0.125)
            'plateau': (0.25, 0.5),         # 0.25-0.5 → plateau (center: 0.375)
            'gaussian_single': (0.5, 0.75), # 0.5-0.75 → gaussian_single (center: 0.625)
            'gaussian_double': (0.75, 1.0)  # 0.75-1.0 → gaussian_double (center: 0.875)
        }
        
    def extract_from_directory(self, training_dir: Path, segment_info_dir: Path) -> Dict:
        """Extract statistics from all training files."""
        
        print(f"Extracting statistics from {training_dir}")
        
        # Containers for all data
        all_params = {name: [] for name in self.param_names}
        
        # FIXED: Initialize segment_specific with all parameter names
        segment_types = ['verse', 'chorus', 'bridge', 'intro', 'outro', 'drop', 'buildup', 'breakdown']
        segment_specific = {}
        for seg_type in segment_types:
            segment_specific[seg_type] = {name: [] for name in self.param_names}
        
        wave_conventions = {}
        
        # Process each training file
        pkl_files = sorted(training_dir.glob('*.pkl'))
        
        if not pkl_files:
            print(f"Warning: No .pkl files found in {training_dir}")
            return {}
        
        for pkl_file in pkl_files:
            print(f"  Processing {pkl_file.stem}")
            
            # Load oscillator params (should be numpy array of shape (frames, 60))
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Verify data shape
            if not isinstance(data, np.ndarray):
                print(f"    Warning: Expected numpy array, got {type(data)}")
                continue
                
            if data.shape[1] != 60:
                print(f"    Warning: Expected 60 dimensions, got {data.shape[1]}")
                continue
            
            print(f"    Loaded data shape: {data.shape}")
            
            # Find corresponding segment info (might not exist for all files)
            base_name = pkl_file.stem
            # Try different naming patterns
            possible_json_names = [
                f"{base_name}.json",
                f"{base_name.split('-')[0]}.json",  # Try first part before dash
                f"{base_name.replace('-', '_')}.json"  # Try replacing dashes
            ]
            
            segments = None
            for json_name in possible_json_names:
                segment_file = segment_info_dir / json_name
                if segment_file.exists():
                    with open(segment_file, 'r') as f:
                        seg_data = json.load(f)
                        segments = seg_data.get('segments', [])
                    print(f"    Found segment info with {len(segments)} segments")
                    break
            
            if not segments:
                print(f"    No segment info found, processing entire file")
            
            # Process standard parameters (3 groups)
            for group_idx in range(3):
                start_idx = group_idx * 20
                standard_params = data[:, start_idx:start_idx+10]
                
                # Collect global parameters (entire file)
                for param_idx, param_name in enumerate(self.param_names):
                    all_params[param_name].extend(standard_params[:, param_idx].flatten())
                
                # Segment-specific analysis if available
                if segments:
                    for segment in segments:
                        seg_label = segment.get('label', '').lower()
                        
                        # Skip non-music segments
                        if seg_label in ['start', 'end', 'silence']:
                            continue
                        
                        # Map to known segment types or use 'other'
                        if seg_label not in segment_specific:
                            # Try to map common variations
                            if 'vers' in seg_label:
                                seg_type = 'verse'
                            elif 'choru' in seg_label or 'refrain' in seg_label:
                                seg_type = 'chorus'
                            elif 'bridge' in seg_label:
                                seg_type = 'bridge'
                            elif 'intro' in seg_label:
                                seg_type = 'intro'
                            elif 'outro' in seg_label or 'ending' in seg_label:
                                seg_type = 'outro'
                            elif 'drop' in seg_label:
                                seg_type = 'drop'
                            elif 'build' in seg_label:
                                seg_type = 'buildup'
                            elif 'break' in seg_label:
                                seg_type = 'breakdown'
                            else:
                                # Add new segment type if needed
                                if seg_label not in segment_specific:
                                    segment_specific[seg_label] = {name: [] for name in self.param_names}
                                seg_type = seg_label
                        else:
                            seg_type = seg_label
                        
                        # Extract segment frames
                        start_frame = int(segment['start'] * 30)  # 30 fps
                        end_frame = min(int(segment['end'] * 30), len(data))
                        
                        if start_frame < len(data) and end_frame > start_frame:
                            seg_data = standard_params[start_frame:end_frame]
                            
                            if len(seg_data) > 0:
                                for param_idx, param_name in enumerate(self.param_names):
                                    segment_specific[seg_type][param_name].extend(
                                        seg_data[:, param_idx].flatten()
                                    )
        
        # Remove empty segment types
        segment_specific = {k: v for k, v in segment_specific.items() 
                          if any(len(vals) > 0 for vals in v.values())}
        
        print(f"\nProcessed {len(pkl_files)} files")
        print(f"Segment types found: {list(segment_specific.keys())}")
        
        # Compute statistics
        statistics = self._compute_statistics(all_params, segment_specific)
        
        # Analyze wave type conventions
        statistics['wave_conventions'] = self._analyze_wave_conventions(segment_specific)
        
        return statistics
    
    def _compute_statistics(self, all_params: Dict, segment_specific: Dict) -> Dict:
        """Compute statistical measures for each parameter."""
        
        stats = {'global': {}, 'per_segment': {}}
        
        # Global statistics
        print("\nComputing global statistics...")
        for param_name, values in all_params.items():
            if len(values) > 0:
                values = np.array(values)
                stats['global'][param_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q50': float(np.percentile(values, 50)),
                    'q75': float(np.percentile(values, 75)),
                    'distribution': np.histogram(values, bins=50)[0].tolist(),
                    'bin_edges': np.histogram(values, bins=50)[1].tolist()
                }
                print(f"  {param_name}: mean={stats['global'][param_name]['mean']:.3f}, "
                      f"std={stats['global'][param_name]['std']:.3f}")
        
        # Per-segment statistics
        print("\nComputing per-segment statistics...")
        for seg_type, params in segment_specific.items():
            stats['per_segment'][seg_type] = {}
            print(f"  {seg_type}:")
            for param_name, values in params.items():
                if len(values) > 0:
                    values = np.array(values)
                    stats['per_segment'][seg_type][param_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
                    if param_name in ['amplitude', 'frequency']:  # Show key params
                        print(f"    {param_name}: mean={stats['per_segment'][seg_type][param_name]['mean']:.3f}")
        
        return stats
    
    def _analyze_wave_conventions(self, segment_specific: Dict) -> Dict:
        """Analyze which wave types are used for which segment types."""
        
        conventions = {}
        
        for seg_type, params in segment_specific.items():
            if 'wave_type_a' in params and len(params['wave_type_a']) > 0:
                wave_types = []
                for val in params['wave_type_a']:
                    val = np.clip(val, 0.0, 1.0)  # Ensure valid range
                    # Find which boundary contains this value
                    for wave_name, (min_v, max_v) in self.wave_a_boundaries.items():
                        if min_v <= val <= max_v:
                            wave_types.append(wave_name)
                            break
                
                if wave_types:
                    wave_dist = pd.Series(wave_types).value_counts(normalize=True)
                    conventions[seg_type] = wave_dist.to_dict()
        
        return conventions
    
    def save_statistics(self, stats: Dict, output_dir: Path):
        """Save computed statistics."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main statistics as pickle
        with open(output_dir / 'parameter_distributions.pkl', 'wb') as f:
            pickle.dump(stats, f)
        
        # Save wave conventions as JSON for readability
        with open(output_dir / 'wave_type_conventions.json', 'w') as f:
            json.dump(stats.get('wave_conventions', {}), f, indent=2)
        
        # Save summary as readable text
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("Training Data Statistics Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Global Statistics:\n")
            for param, pstats in stats['global'].items():
                f.write(f"  {param}:\n")
                f.write(f"    Mean: {pstats['mean']:.4f}\n")
                f.write(f"    Std:  {pstats['std']:.4f}\n")
                f.write(f"    Range: [{pstats['min']:.4f}, {pstats['max']:.4f}]\n\n")
            
            f.write("\nSegment-specific Statistics:\n")
            for seg_type, seg_stats in stats['per_segment'].items():
                f.write(f"  {seg_type}:\n")
                if 'amplitude' in seg_stats:
                    f.write(f"    Amplitude: {seg_stats['amplitude']['mean']:.3f} ± {seg_stats['amplitude']['std']:.3f}\n")
                if 'frequency' in seg_stats:
                    f.write(f"    Frequency: {seg_stats['frequency']['mean']:.3f} ± {seg_stats['frequency']['std']:.3f}\n")
        
        # Create summary plots
        self._create_summary_plots(stats, output_dir)
        
        print(f"\nStatistics saved to {output_dir}")
    
    def _create_summary_plots(self, stats: Dict, output_dir: Path):
        """Create visualization of training data distributions."""
        
        # Parameter distributions
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, param_name in enumerate(self.param_names):
            if param_name in stats['global'] and 'distribution' in stats['global'][param_name]:
                ax = axes[idx]
                param_stats = stats['global'][param_name]
                bins = param_stats['bin_edges'][:-1]
                values = param_stats['distribution']
                
                # Bar plot
                width = (bins[1] - bins[0]) if len(bins) > 1 else 1
                ax.bar(bins, values, width=width, alpha=0.7, color='steelblue')
                
                # Add mean line
                ax.axvline(param_stats['mean'], color='red', linestyle='--', 
                          label=f"Mean: {param_stats['mean']:.2f}")
                
                ax.set_title(param_name.replace('_', ' ').title())
                ax.set_xlabel('Value')
                ax.set_ylabel('Count')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Data Parameter Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Created distribution plot: training_distributions.png")


def main():
    """Extract training statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract training data statistics')
    parser.add_argument('--training_dir', type=str, required=True,
                       help='Directory with training oscillator params')
    parser.add_argument('--segment_dir', type=str, required=True,
                       help='Directory with segment info JSONs')
    parser.add_argument('--output_dir', type=str, 
                       default='data/training_data/statistics',
                       help='Output directory for statistics')
    
    args = parser.parse_args()
    
    # Validate directories
    training_path = Path(args.training_dir)
    segment_path = Path(args.segment_dir)
    
    if not training_path.exists():
        print(f"Error: Training directory not found: {training_path}")
        return
    
    if not segment_path.exists():
        print(f"Warning: Segment directory not found: {segment_path}")
        print("Will process without segment information")
    
    extractor = TrainingStatsExtractor()
    stats = extractor.extract_from_directory(training_path, segment_path)
    
    if stats:
        extractor.save_statistics(stats, Path(args.output_dir))
        print("\nExtraction complete!")
    else:
        print("\nNo statistics extracted. Check your input directories.")

if __name__ == '__main__':
    main()