
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
        
        # FIXED: Complete coverage with no gaps
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

    """
    def classify_wave_type(self, value: float, wave_ranges: Dict) -> str:
        for wave_name, (min_val, max_val) in wave_ranges.items():
            if min_val <= value <= max_val:
                return wave_name
        # If not in any range, find closest
        min_dist = float('inf')
        closest = 'other'
        for wave_name, (min_val, max_val) in wave_ranges.items():
            mid = (min_val + max_val) / 2
            dist = abs(value - mid)
            if dist < min_dist:
                min_dist = dist
                closest = wave_name
        return closest
    """
    
    def compute_MAI(self, pan_activity: np.ndarray, tilt_activity: np.ndarray) -> float:
        """Combine pan and tilt into Movement Activity Index."""
        return np.sqrt(pan_activity**2 + tilt_activity**2).mean()
    
    def analyze_segment(self, segment_data: np.ndarray, segment_info: Dict, 
                       audio_features: Dict) -> Dict:
        """Analyze a single segment's oscillator parameters."""
        
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
            wave_a_types = [self.classify_wave_type(v, self.wave_a_ranges) for v in wave_a]
            wave_a_consistency = pd.Series(wave_a_types).value_counts(normalize=True).max()
            dominant_wave_a = pd.Series(wave_a_types).mode()[0]
            
            # Frequency-BPM coherence
            bpm = audio_features.get('bpm', 120)
            freq_bpm_ratio = freq.mean() / (bpm / 60)
            
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
        
        for pred_file in pred_files:
            # Load prediction
            with open(pred_file, 'rb') as f:
                pred_data = pickle.load(f)
            
            # Find corresponding audio info
            base_name = pred_file.stem.split('_seed')[0]  # Remove seed suffix
            audio_info_file = audio_info_dir / f"{base_name}.json"
            
            if not audio_info_file.exists():
                print(f"Warning: No audio info for {pred_file.stem}")
                continue
                
            with open(audio_info_file, 'r') as f:
                audio_info = json.load(f)
            
            # Analyze by segment
            segment_results = []
            for segment in audio_info['segments']:
                if segment['label'] == 'start':  # Skip start marker
                    continue
                    
                # Get frames for this segment
                start_frame = int(segment['start'] * 30)  # 30 fps
                end_frame = int(segment['end'] * 30)
                segment_frames = pred_data[start_frame:end_frame]
                
                if len(segment_frames) > 0:
                    seg_analysis = self.analyze_segment(
                        segment_frames, segment, audio_info
                    )
                    segment_results.append(seg_analysis)
            
            all_results.append({
                'file': pred_file.stem,
                'bpm': audio_info['bpm'],
                'segments': segment_results
            })
        
        # Aggregate statistics
        aggregated = self.aggregate_results(all_results)
        
        # Generate visualizations
        self.create_visualizations(aggregated, output_dir)
        
        # Save results
        with open(output_dir / 'oscillator_evaluation.json', 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        return aggregated
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all files."""
        
        # Collect by segment type
        segment_stats = {}
        
        for file_result in all_results:
            for segment in file_result['segments']:
                seg_type = segment['segment_type']
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
            wave_dist = pd.Series(data['wave_types']).value_counts(normalize=True).to_dict()
            
            final_stats[seg_type] = {
                'wave_distribution': wave_dist,
                'mean_amplitude': np.mean(data['amplitudes']),
                'std_amplitude': np.std(data['amplitudes']),
                'mean_frequency': np.mean(data['frequencies']),
                'mean_mai': np.mean(data['mai_values']),
                'mean_consistency': np.mean(data['consistencies']),
                'freq_bpm_ratio_distribution': {
                    'mean': np.mean(data['freq_bpm_ratios']),
                    'std': np.std(data['freq_bpm_ratios'])
                }
            }
        
        return {
            'segment_statistics': final_stats,
            'num_files': len(all_results),
            'total_segments': sum(len(r['segments']) for r in all_results)
        }
    
    def create_visualizations(self, aggregated: Dict, output_dir: Path):
        """Create evaluation plots."""
        
        seg_stats = aggregated['segment_statistics']
        
        # 1. Wave type distribution by segment
        fig, axes = plt.subplots(1, len(seg_stats), figsize=(15, 5))
        if len(seg_stats) == 1:
            axes = [axes]
            
        for idx, (seg_type, stats) in enumerate(seg_stats.items()):
            wave_dist = stats['wave_distribution']
            axes[idx].bar(wave_dist.keys(), wave_dist.values())
            axes[idx].set_title(f'{seg_type.capitalize()} - Wave Types')
            axes[idx].set_ylabel('Frequency')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Wave Type Distribution by Segment Type')
        plt.tight_layout()
        plt.savefig(output_dir / 'wave_distributions.png', dpi=150)
        plt.close()
        
        # 2. Parameter comparison across segment types
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Amplitude comparison
        seg_types = list(seg_stats.keys())
        amplitudes = [seg_stats[st]['mean_amplitude'] for st in seg_types]
        axes[0, 0].bar(seg_types, amplitudes)
        axes[0, 0].set_title('Mean Amplitude by Segment Type')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Frequency comparison
        frequencies = [seg_stats[st]['mean_frequency'] for st in seg_types]
        axes[0, 1].bar(seg_types, frequencies)
        axes[0, 1].set_title('Mean Frequency by Segment Type')
        axes[0, 1].set_ylabel('Frequency')
        
        # MAI comparison
        mai_values = [seg_stats[st]['mean_mai'] for st in seg_types]
        axes[1, 0].bar(seg_types, mai_values)
        axes[1, 0].set_title('Mean Movement Activity by Segment Type')
        axes[1, 0].set_ylabel('MAI')
        
        # Consistency comparison
        consistencies = [seg_stats[st]['mean_consistency'] for st in seg_types]
        axes[1, 1].bar(seg_types, consistencies)
        axes[1, 1].set_title('Wave Type Consistency by Segment Type')
        axes[1, 1].set_ylabel('Consistency')
        
        plt.suptitle('Parameter Analysis Across Segment Types')
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_comparison.png', dpi=150)
        plt.close()