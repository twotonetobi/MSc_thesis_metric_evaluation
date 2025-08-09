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
        
        # Wave type mappings
        self.wave_a_ranges = {
            'sine': (0.05, 0.15),
            'saw_up': (0.25, 0.35),
            'saw_down': (0.45, 0.55),
            'square': (0.65, 0.75),
            'linear': (0.85, 0.95)
        }
        
        self.wave_b_ranges = {
            'other': (0.0625, 0.1875),
            'plateau': (0.3125, 0.4375),
            'gaussian_single': (0.5625, 0.6875),
            'gaussian_double': (0.8125, 0.9375)
        }
        
    def extract_from_directory(self, training_dir: Path, segment_info_dir: Path) -> Dict:
        """Extract statistics from all training files."""
        
        print(f"Extracting statistics from {training_dir}")
        
        # Containers for all data
        all_params = {name: [] for name in self.param_names}
        segment_specific = {'verse': {}, 'chorus': {}, 'bridge': {}, 'intro': {}, 'outro': {}}
        wave_conventions = {}
        
        # Process each training file
        pkl_files = sorted(training_dir.glob('*.pkl'))
        
        for pkl_file in pkl_files:
            print(f"  Processing {pkl_file.stem}")
            
            # Load oscillator params
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Find corresponding segment info
            base_name = pkl_file.stem.split('_')[0]  # Adjust based on naming
            segment_file = segment_info_dir / f"{base_name}.json"
            
            segments = None
            if segment_file.exists():
                with open(segment_file, 'r') as f:
                    seg_data = json.load(f)
                    segments = seg_data.get('segments', [])
            
            # Process standard parameters (3 groups)
            for group_idx in range(3):
                start_idx = group_idx * 20
                standard_params = data[:, start_idx:start_idx+10]
                
                # Collect raw parameters
                for param_idx, param_name in enumerate(self.param_names):
                    all_params[param_name].extend(standard_params[:, param_idx].flatten())
                
                # Segment-specific analysis if available
                if segments:
                    for segment in segments:
                        if segment['label'] in segment_specific:
                            start_frame = int(segment['start'] * 30)
                            end_frame = int(segment['end'] * 30)
                            seg_data = standard_params[start_frame:end_frame]
                            
                            if len(seg_data) > 0:
                                seg_type = segment['label']
                                if seg_type not in segment_specific:
                                    segment_specific[seg_type] = {name: [] for name in self.param_names}
                                
                                for param_idx, param_name in enumerate(self.param_names):
                                    segment_specific[seg_type][param_name].extend(
                                        seg_data[:, param_idx].flatten()
                                    )
        
        # Compute statistics
        statistics = self._compute_statistics(all_params, segment_specific)
        
        # Analyze wave type conventions
        statistics['wave_conventions'] = self._analyze_wave_conventions(segment_specific)
        
        return statistics
    
    def _compute_statistics(self, all_params: Dict, segment_specific: Dict) -> Dict:
        """Compute statistical measures for each parameter."""
        
        stats = {'global': {}, 'per_segment': {}}
        
        # Global statistics
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
        
        # Per-segment statistics
        for seg_type, params in segment_specific.items():
            stats['per_segment'][seg_type] = {}
            for param_name, values in params.items():
                if len(values) > 0:
                    values = np.array(values)
                    stats['per_segment'][seg_type][param_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
        
        return stats
    
    def _analyze_wave_conventions(self, segment_specific: Dict) -> Dict:
        """Analyze which wave types are used for which segment types."""
        
        conventions = {}
        
        # Use the complete coverage boundaries
        wave_a_boundaries = {
            'sine': (0.0, 0.2),
            'saw_up': (0.2, 0.4),
            'saw_down': (0.4, 0.6),
            'square': (0.6, 0.8),
            'linear': (0.8, 1.0)
        }
        
        for seg_type, params in segment_specific.items():
            if 'wave_type_a' in params and len(params['wave_type_a']) > 0:
                wave_types = []
                for val in params['wave_type_a']:
                    val = np.clip(val, 0.0, 1.0)  # Ensure valid range
                    # Find which boundary contains this value
                    for wave_name, (min_v, max_v) in wave_a_boundaries.items():
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
        
        # Create summary plots
        self._create_summary_plots(stats, output_dir)
        
        print(f"Statistics saved to {output_dir}")
    
    def _create_summary_plots(self, stats: Dict, output_dir: Path):
        """Create visualization of training data distributions."""
        
        # Parameter distributions
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (param_name, param_stats) in enumerate(stats['global'].items()):
            if idx < 10 and 'distribution' in param_stats:
                ax = axes[idx]
                bins = param_stats['bin_edges'][:-1]
                values = param_stats['distribution']
                ax.bar(bins, values, width=bins[1]-bins[0])
                ax.set_title(param_name)
                ax.set_xlabel('Value')
                ax.set_ylabel('Count')
        
        plt.suptitle('Training Data Parameter Distributions')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_distributions.png', dpi=150)
        plt.close()


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
    
    extractor = TrainingStatsExtractor()
    stats = extractor.extract_from_directory(
        Path(args.training_dir),
        Path(args.segment_dir)
    )
    extractor.save_statistics(stats, Path(args.output_dir))

if __name__ == '__main__':
    main()