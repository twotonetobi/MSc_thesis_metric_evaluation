#!/usr/bin/env python
"""
Oscillator-based lighting evaluation without ground truth.
Properly handles both 60 and 61 dimension inputs.
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class OscillatorEvaluator:
    """Evaluates oscillator-based lighting generation without ground truth."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # FIXED: Complete coverage with no gaps for wave type boundaries
        # For wave_type_a (0.0 to 1.0 range)
        self.wave_a_boundaries = {
            'sine': (0.0, 0.2),      # 0.0-0.2 → sine (center: 0.1)
            'saw_up': (0.2, 0.4),    # 0.2-0.4 → saw_up (center: 0.3)
            'saw_down': (0.4, 0.6),  # 0.4-0.6 → saw_down (center: 0.5)
            'square': (0.6, 0.8),    # 0.6-0.8 → square (center: 0.7)
            'linear': (0.8, 1.0)     # 0.8-1.0 → linear (center: 0.9)
        }
        
        # For wave_type_b (0.0 to 1.0 range)
        self.wave_b_boundaries = {
            'other': (0.0, 0.25),           # 0.0-0.25 → other (center: 0.125)
            'plateau': (0.25, 0.5),         # 0.25-0.5 → plateau (center: 0.375)
            'gaussian_single': (0.5, 0.75), # 0.5-0.75 → gaussian_single (center: 0.625)
            'gaussian_double': (0.75, 1.0)  # 0.75-1.0 → gaussian_double (center: 0.875)
        }
    
        # Parameter indices in 10-dim blocks
        self.param_indices = {
            'pan_activity': 0,
            'tilt_activity': 1,
            'wave_type_a': 2,
            'wave_type_b': 3,
            'frequency': 4,
            'amplitude': 5,
            'offset': 6,
            'phase': 7,
            'col_hue': 8,
            'col_sat': 9
        }
        
        # Training data statistics (to be loaded)
        self.training_stats = None
        
    def load_training_statistics(self, stats_path: Path):
        """Load pre-computed statistics from training data."""
        with open(stats_path, 'rb') as f:
            self.training_stats = pickle.load(f)
            print(f"Loaded training statistics from {stats_path}")

    def classify_wave_type(self, value: float, wave_boundaries: Dict) -> str:
        """Classify any value to the appropriate wave type."""
        # Clamp to valid range
        value = np.clip(value, 0.0, 1.0)
        
        # Find which boundary contains this value
        for wave_name, (min_val, max_val) in wave_boundaries.items():
            if min_val <= value <= max_val:
                return wave_name
        
        # Fallback (shouldn't happen with complete coverage)
        return list(wave_boundaries.keys())[0]
    
    def compute_MAI(self, pan_activity: np.ndarray, tilt_activity: np.ndarray) -> float:
        """Combine pan and tilt into Movement Activity Index."""
        return np.sqrt(pan_activity**2 + tilt_activity**2).mean()
    
    def handle_dimensions(self, data: np.ndarray) -> np.ndarray:
        """Handle both 60 and 61 dimension inputs."""
        if data.shape[1] == 61:
            # Use first 60 columns, skip the last one
            return data[:, :60]
        elif data.shape[1] == 60:
            return data
        else:
            raise ValueError(f"Expected 60 or 61 dimensions, got {data.shape[1]}")
    
    def analyze_segment(self, segment_data: np.ndarray, segment_info: Dict, 
                       audio_features: Dict) -> Dict:
        """Analyze a single segment's oscillator parameters."""
        
        # Ensure we have 60 dimensions
        segment_data = self.handle_dimensions(segment_data)
        
        results = {
            'segment_type': segment_info['label'],
            'duration': segment_info['end'] - segment_info['start'],
            'metrics': {}
        }
        
        # Process standard parameters (first 3 groups)
        for group_idx in range(3):
            group_start = group_idx * 20  # Each group is 20 dims (10 standard + 10 highlight)
            standard_params = segment_data[:, group_start:group_start+10]
            
            # Extract parameters
            pan_act = standard_params[:, 0]
            tilt_act = standard_params[:, 1]
            wave_a = standard_params[:, 2]
            wave_b = standard_params[:, 3]
            freq = standard_params[:, 4]
            amp = standard_params[:, 5]
            offset = standard_params[:, 6]
            phase = standard_params[:, 7]
            hue = standard_params[:, 8]
            sat = standard_params[:, 9]
            
            # Movement Activity Index
            mai = self.compute_MAI(pan_act, tilt_act)
            
            # Wave type consistency
            wave_a_types = [self.classify_wave_type(v, self.wave_a_boundaries) for v in wave_a]
            wave_a_consistency = pd.Series(wave_a_types).value_counts(normalize=True).max()
            dominant_wave_a = pd.Series(wave_a_types).mode()[0] if len(wave_a_types) > 0 else 'sine'
            
            # Frequency-BPM coherence
            bpm = audio_features.get('bpm', 120)
            freq_bpm_ratio = freq.mean() / (bpm / 60) if bpm > 0 else 0
            
            # Parameter stability (low std within segment = stable)
            param_stability = {
                'freq_std': freq.std(),
                'amp_std': amp.std(),
                'hue_std': hue.std()
            }
            
            results['metrics'][f'group_{group_idx}'] = {
                'MAI': mai,
                'dominant_wave': dominant_wave_a,
                'wave_consistency': wave_a_consistency,
                'freq_bpm_ratio': freq_bpm_ratio,
                'mean_amplitude': amp.mean(),
                'mean_frequency': freq.mean(),
                'stability': param_stability
            }
        
        return results
    
    def evaluate_predictions(self, predictions_dir: Path, audio_info_dir: Path, 
                            output_dir: Path) -> Dict:
        """Main evaluation pipeline."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        
        # Collect all predictions
        pred_files = sorted(predictions_dir.glob('*.pkl'))
        
        if not pred_files:
            print(f"No prediction files found in {predictions_dir}")
            return {}
        
        print(f"Found {len(pred_files)} prediction files")
        
        # Check first file for dimensions
        with open(pred_files[0], 'rb') as f:
            sample_data = pickle.load(f)
        print(f"First file shape: {sample_data.shape}")
        if sample_data.shape[1] == 61:
            print("  -> Detected 61 dimensions, will use columns 0-59")
        
        processed_count = 0
        skipped_count = 0
        
        for pred_file in pred_files:
            # Load prediction
            with open(pred_file, 'rb') as f:
                pred_data = pickle.load(f)
            
            # Handle dimensions
            try:
                pred_data = self.handle_dimensions(pred_data)
            except ValueError as e:
                print(f"  Skipping {pred_file.stem}: {e}")
                skipped_count += 1
                continue
            
            # Find corresponding audio info
            base_name = pred_file.stem.split('_seed')[0]  # Remove seed suffix if present
            
            # Try different naming patterns for audio info
            audio_info_file = None
            possible_names = [
                f"{base_name}.json",
                f"{pred_file.stem}.json"
            ]
            
            # Handle part-based naming
            if '_part_' in base_name:
                parts = base_name.split('_part_')
                if len(parts) > 1:
                    possible_names.append(f"{parts[0]}.json")
            
            for name in possible_names:
                candidate = audio_info_dir / name
                if candidate.exists():
                    audio_info_file = candidate
                    break
            
            if not audio_info_file:
                print(f"  Warning: No audio info for {pred_file.stem}, using defaults")
                # Create default audio info
                audio_info = {
                    'bpm': 120,
                    'segments': [{'label': 'unknown', 'start': 0, 'end': 90}]
                }
            else:
                with open(audio_info_file, 'r') as f:
                    audio_info = json.load(f)
            
            # Analyze by segment
            segment_results = []
            for segment in audio_info.get('segments', []):
                if segment['label'] == 'start':  # Skip start marker
                    continue
                    
                # Get frames for this segment
                start_frame = int(segment['start'] * 30)  # 30 fps
                end_frame = min(int(segment['end'] * 30), len(pred_data))
                
                if end_frame > start_frame:
                    segment_frames = pred_data[start_frame:end_frame]
                    
                    if len(segment_frames) > 0:
                        seg_analysis = self.analyze_segment(
                            segment_frames, segment, audio_info
                        )
                        segment_results.append(seg_analysis)
            
            if segment_results:
                all_results.append({
                    'file': pred_file.stem,
                    'bpm': audio_info.get('bpm', 120),
                    'segments': segment_results
                })
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"  Processed {processed_count} files...")
        
        print(f"\nProcessing complete: {processed_count} files processed, {skipped_count} skipped")
        
        if not all_results:
            print("No results to aggregate")
            return {}
        
        # Aggregate statistics
        aggregated = self.aggregate_results(all_results)
        
        # Generate visualizations
        self.create_visualizations(aggregated, output_dir)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save results as JSON (with converted types)
        json_safe_aggregated = convert_numpy_types(aggregated)
        with open(output_dir / 'oscillator_evaluation.json', 'w') as f:
            json.dump(json_safe_aggregated, f, indent=2)
        
        # Save results as pickle (preserves numpy types)
        with open(output_dir / 'oscillator_evaluation.pkl', 'wb') as f:
            pickle.dump(aggregated, f)
        
        print(f"Results saved to {output_dir}")
        
        return aggregated
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all files."""
        
        # Collect by segment type
        segment_stats = {}
        
        for file_result in all_results:
            for segment in file_result['segments']:
                seg_type = segment['segment_type'].lower()
                
                # Normalize segment type names
                if 'vers' in seg_type:
                    seg_type = 'verse'
                elif 'choru' in seg_type or 'refrain' in seg_type:
                    seg_type = 'chorus'
                elif 'bridg' in seg_type:
                    seg_type = 'bridge'
                elif 'intro' in seg_type:
                    seg_type = 'intro'
                elif 'outro' in seg_type or 'ending' in seg_type:
                    seg_type = 'outro'
                elif 'inst' in seg_type:
                    seg_type = 'instrumental'
                
                if seg_type not in segment_stats:
                    segment_stats[seg_type] = {
                        'wave_types': [],
                        'amplitudes': [],
                        'frequencies': [],
                        'mai_values': [],
                        'freq_bpm_ratios': [],
                        'consistencies': []
                    }
                
                # Aggregate from all groups
                for group_key, group_metrics in segment['metrics'].items():
                    segment_stats[seg_type]['wave_types'].append(group_metrics['dominant_wave'])
                    segment_stats[seg_type]['amplitudes'].append(group_metrics['mean_amplitude'])
                    segment_stats[seg_type]['frequencies'].append(group_metrics['mean_frequency'])
                    segment_stats[seg_type]['mai_values'].append(group_metrics['MAI'])
                    segment_stats[seg_type]['freq_bpm_ratios'].append(group_metrics['freq_bpm_ratio'])
                    segment_stats[seg_type]['consistencies'].append(group_metrics['wave_consistency'])
        
        # Compute statistics
        final_stats = {}
        for seg_type, data in segment_stats.items():
            if len(data['wave_types']) > 0:
                wave_dist = pd.Series(data['wave_types']).value_counts(normalize=True).to_dict()
                
                final_stats[seg_type] = {
                    'wave_distribution': wave_dist,
                    'mean_amplitude': np.mean(data['amplitudes']),
                    'std_amplitude': np.std(data['amplitudes']),
                    'mean_frequency': np.mean(data['frequencies']),
                    'std_frequency': np.std(data['frequencies']),
                    'mean_mai': np.mean(data['mai_values']),
                    'std_mai': np.std(data['mai_values']),
                    'mean_consistency': np.mean(data['consistencies']),
                    'freq_bpm_ratio_distribution': {
                        'mean': np.mean(data['freq_bpm_ratios']),
                        'std': np.std(data['freq_bpm_ratios'])
                    },
                    'sample_count': len(data['amplitudes'])
                }
        
        return {
            'segment_statistics': final_stats,
            'num_files': len(all_results),
            'total_segments': sum(len(r['segments']) for r in all_results)
        }
    
    def create_visualizations(self, aggregated: Dict, output_dir: Path):
        """Create evaluation plots."""
        
        if 'segment_statistics' not in aggregated or not aggregated['segment_statistics']:
            print("No segment statistics to visualize")
            return
        
        seg_stats = aggregated['segment_statistics']
        
        # 1. Wave type distribution by segment
        num_segments = len(seg_stats)
        if num_segments > 0:
            fig, axes = plt.subplots(1, min(num_segments, 4), figsize=(min(15, num_segments*4), 5))
            if num_segments == 1:
                axes = [axes]
            elif num_segments > 4:
                # Only show first 4 segment types
                seg_stats = dict(list(seg_stats.items())[:4])
                
            for idx, (seg_type, stats) in enumerate(seg_stats.items()):
                if idx < len(axes):
                    wave_dist = stats['wave_distribution']
                    if wave_dist:
                        axes[idx].bar(wave_dist.keys(), wave_dist.values())
                        axes[idx].set_title(f'{seg_type.capitalize()}')
                        axes[idx].set_ylabel('Frequency')
                        axes[idx].tick_params(axis='x', rotation=45)
                        axes[idx].set_ylim(0, 1)
            
            plt.suptitle('Wave Type Distribution by Segment Type', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'wave_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Parameter comparison across segment types
        if len(seg_stats) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            seg_types = list(seg_stats.keys())
            
            # Amplitude comparison
            amplitudes = [seg_stats[st]['mean_amplitude'] for st in seg_types]
            amp_errors = [seg_stats[st]['std_amplitude'] for st in seg_types]
            axes[0, 0].bar(seg_types, amplitudes, yerr=amp_errors, capsize=5)
            axes[0, 0].set_title('Mean Amplitude by Segment Type')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Frequency comparison
            frequencies = [seg_stats[st]['mean_frequency'] for st in seg_types]
            freq_errors = [seg_stats[st]['std_frequency'] for st in seg_types]
            axes[0, 1].bar(seg_types, frequencies, yerr=freq_errors, capsize=5)
            axes[0, 1].set_title('Mean Frequency by Segment Type')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # MAI comparison
            mai_values = [seg_stats[st]['mean_mai'] for st in seg_types]
            mai_errors = [seg_stats[st]['std_mai'] for st in seg_types]
            axes[1, 0].bar(seg_types, mai_values, yerr=mai_errors, capsize=5)
            axes[1, 0].set_title('Mean Movement Activity by Segment Type')
            axes[1, 0].set_ylabel('MAI')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Consistency comparison
            consistencies = [seg_stats[st]['mean_consistency'] for st in seg_types]
            axes[1, 1].bar(seg_types, consistencies)
            axes[1, 1].set_title('Wave Type Consistency by Segment Type')
            axes[1, 1].set_ylabel('Consistency')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Parameter Analysis Across Segment Types', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate oscillator-based predictions')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with prediction pkl files')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory with audio segment JSON files')
    parser.add_argument('--stats_path', type=str,
                       default='data/training_data/statistics/parameter_distributions.pkl',
                       help='Path to training statistics')
    parser.add_argument('--output_dir', type=str,
                       default='outputs_oscillator',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluator = OscillatorEvaluator()
    
    # Load training statistics if available
    stats_path = Path(args.stats_path)
    if stats_path.exists():
        evaluator.load_training_statistics(stats_path)
    else:
        print(f"Warning: Training statistics not found at {stats_path}")
    
    # Run evaluation
    results = evaluator.evaluate_predictions(
        Path(args.pred_dir),
        Path(args.audio_dir),
        Path(args.output_dir)
    )
    
    if results:
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Files processed: {results['num_files']}")
        print(f"Total segments: {results['total_segments']}")
        
        if 'segment_statistics' in results:
            print("\nSegment types analyzed:")
            for seg_type, stats in results['segment_statistics'].items():
                print(f"  {seg_type}: {stats['sample_count']} samples")

if __name__ == '__main__':
    main()