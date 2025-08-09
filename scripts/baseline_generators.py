#!/usr/bin/env python
"""
Generate baseline predictions for comparison.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple

class BaselineGenerator:
    """Generate various baseline predictions."""
    
    def __init__(self, stats_path: Path = None):
        """Initialize with optional training statistics."""
        
        self.stats = None
        if stats_path and stats_path.exists():
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
        
        # Default ranges if no stats
        self.default_ranges = {
            'pan_activity': (0.0, 0.5),
            'tilt_activity': (0.0, 0.5),
            'wave_type_a': (0.1, 0.9),  # Full range
            'wave_type_b': (0.125, 0.875),
            'frequency': (0.1, 2.0),
            'amplitude': (0.2, 0.8),
            'offset': (0.0, 0.3),
            'phase': (0.0, 2*np.pi),
            'col_hue': (0.0, 1.0),
            'col_sat': (0.3, 0.8)
        }
    
    def generate_random(self, num_frames: int, num_groups: int = 3) -> np.ndarray:
        """Generate completely random parameters."""
        
        # 60 dims: 3 groups × (10 standard + 10 highlight)
        output = np.zeros((num_frames, 60))
        
        for group_idx in range(num_groups):
            start_idx = group_idx * 20
            
            # Standard parameters (always active)
            for frame in range(num_frames):
                for param_idx, (param_name, (min_v, max_v)) in enumerate(self.default_ranges.items()):
                    if self.stats and 'global' in self.stats:
                        # Use training data ranges
                        param_stats = self.stats['global'].get(param_name, {})
                        min_v = param_stats.get('q25', min_v)
                        max_v = param_stats.get('q75', max_v)
                    
                    output[frame, start_idx + param_idx] = np.random.uniform(min_v, max_v)
            
            # Highlight parameters (sparse - 10% active)
            if np.random.random() < 0.1:
                highlight_start = start_idx + 10
                for frame in range(num_frames):
                    for param_idx in range(10):
                        output[frame, highlight_start + param_idx] = output[frame, start_idx + param_idx] * 1.2
        
        return output
    
    def generate_beat_sync(self, num_frames: int, audio_info: Dict) -> np.ndarray:
        """Generate simple beat-synchronized patterns."""
        
        output = np.zeros((num_frames, 60))
        beats = audio_info.get('beats', [])
        beat_frames = [int(b * 30) for b in beats]  # Convert to frames
        
        for group_idx in range(3):
            start_idx = group_idx * 20
            
            # Create on-beat flashes
            for frame in range(num_frames):
                # Check if near a beat
                is_beat = any(abs(frame - bf) < 3 for bf in beat_frames)
                
                # Standard parameters
                output[frame, start_idx + 0] = 0.1  # pan_activity
                output[frame, start_idx + 1] = 0.1  # tilt_activity
                output[frame, start_idx + 2] = 0.7  # wave_type_a: square
                output[frame, start_idx + 3] = 0.375  # wave_type_b: plateau
                output[frame, start_idx + 4] = 1.0  # frequency
                output[frame, start_idx + 5] = 0.8 if is_beat else 0.2  # amplitude
                output[frame, start_idx + 6] = 0.1  # offset
                output[frame, start_idx + 7] = 0.0  # phase
                output[frame, start_idx + 8] = 0.5 + group_idx * 0.15  # hue (different per group)
                output[frame, start_idx + 9] = 0.7  # saturation
        
        return output
    
    def generate_constant(self, num_frames: int) -> np.ndarray:
        """Generate constant parameters (minimal variation)."""
        
        output = np.zeros((num_frames, 60))
        
        for group_idx in range(3):
            start_idx = group_idx * 20
            
            # Constant mild lighting
            for frame in range(num_frames):
                output[frame, start_idx + 0] = 0.05  # pan_activity
                output[frame, start_idx + 1] = 0.05  # tilt_activity
                output[frame, start_idx + 2] = 0.1   # wave_type_a: sine
                output[frame, start_idx + 3] = 0.375  # wave_type_b: plateau
                output[frame, start_idx + 4] = 0.5   # frequency
                output[frame, start_idx + 5] = 0.4   # amplitude
                output[frame, start_idx + 6] = 0.2   # offset
                output[frame, start_idx + 7] = 0.0   # phase
                output[frame, start_idx + 8] = 0.6   # hue
                output[frame, start_idx + 9] = 0.3   # saturation
        
        return output
    
    def generate_all_baselines(self, audio_dir: Path, output_dir: Path):
        """Generate all baseline types for the dataset."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'random').mkdir(exist_ok=True)
        (output_dir / 'beat_sync').mkdir(exist_ok=True)
        (output_dir / 'constant').mkdir(exist_ok=True)
        
        # Process each audio file
        json_files = sorted(audio_dir.glob('*.json'))
        
        for json_file in json_files:
            print(f"Generating baselines for {json_file.stem}")
            
            with open(json_file, 'r') as f:
                audio_info = json.load(f)
            
            # All methods generate 90 seconds (2700 frames)
            num_frames = 2700
            
            # Generate each baseline type
            random_pred = self.generate_random(num_frames)
            beat_sync_pred = self.generate_beat_sync(num_frames, audio_info)
            constant_pred = self.generate_constant(num_frames)
            
            # Save
            base_name = json_file.stem
            with open(output_dir / 'random' / f'{base_name}.pkl', 'wb') as f:
                pickle.dump(random_pred, f)
            with open(output_dir / 'beat_sync' / f'{base_name}.pkl', 'wb') as f:
                pickle.dump(beat_sync_pred, f)
            with open(output_dir / 'constant' / f'{base_name}.pkl', 'wb') as f:
                pickle.dump(constant_pred, f)
        
        print(f"Baselines saved to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate baseline predictions')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory with audio segment JSONs')
    parser.add_argument('--stats_path', type=str,
                       default='data/training_data/statistics/parameter_distributions.pkl',
                       help='Path to training statistics')
    parser.add_argument('--output_dir', type=str,
                       default='data/baselines',
                       help='Output directory for baselines')
    
    args = parser.parse_args()
    
    generator = BaselineGenerator(Path(args.stats_path) if args.stats_path else None)
    generator.generate_all_baselines(Path(args.audio_dir), Path(args.output_dir))

if __name__ == '__main__':
    main()