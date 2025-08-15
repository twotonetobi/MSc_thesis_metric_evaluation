#!/usr/bin/env python
"""
Wave Type Reconstructor Module - FIXED VERSION
With correct decision boundaries and PAS mapping from TouchDesigner

This module reconstructs actual wave type decisions by combining:
- PAS (intention-based) data: 72 dimensions (12 groups × 6 params)
- Geo (oscillator-based) data: 60 dimensions (3 groups × 20 params)

Author: Tobias Wursthorn
Date: August 2025
"""

import numpy as np
import pickle
import json
from pathlib import Path
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION - FIXED WITH ACTUAL TOUCHDESIGNER VALUES
# ============================================================================

CONFIG = {
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
    "oscillation_threshold": 10,
    "geo_phase_threshold": 0.15,
    "geo_freq_threshold": 0.15,
    "geo_offset_threshold": 0.15,
    # FIXED: Using actual TouchDesigner decision boundaries
    "decision_boundary_01": 0.22,   # was 0.1
    "decision_boundary_02": 0.7,    # was 0.3
    "decision_boundary_03": 1.1,    # was 0.5
    "decision_boundary_04": 2.2,    # was 0.7
    "decision_boundary_05": 7.0     # was 0.9 - THIS IS THE KEY FIX!
}

# FIXED: Correct PAS to Oscillator mapping based on your TouchDesigner setup
GROUP_MAPPING_CONFIG = {
    'default': {
        # PAS groups -> Oscillator group mapping
        'oscillator_group_0': {
            'pas_groups': [1],  # Intention group 2 (index 1)
            'default_pas': 1,   # Use group 2 (index 1)
            'description': 'Front/main lighting (LX1)'
        },
        'oscillator_group_1': {
            'pas_groups': [4],  # Intention group 5 (index 4)
            'default_pas': 4,   # Use group 5 (index 4)
            'description': 'Side/fill lighting (LX2)'
        },
        'oscillator_group_2': {
            'pas_groups': [7],  # Intention group 8 (index 7)
            'default_pas': 7,   # Use group 8 (index 7)
            'description': 'Back/effect lighting (LX3)'
        }
    }
}

# ============================================================================
# CORE DECISION FUNCTIONS (from back-processing framework)
# ============================================================================

def select_waveform_for_segment(luminaire_dict: Dict, config: Dict) -> Tuple[str, float]:
    """
    Extended decision function that uses both PAS and Geo approach parameters.
    Matches TouchDesigner's select_waveform_for_segment logic.
    
    Returns:
        Tuple of (decision, overall_dynamic_score)
    """
    # ---------------------------
    # PAS-based Metrics:
    # ---------------------------
    PAS_all = luminaire_dict['PASv02_allframes']
    intensityPeakPAS = PAS_all[:, 0]
    intensityInverseMinimaPAS = PAS_all[:, 3]
    
    # Count peaks in PAS intensity as a measure of oscillation
    # KEEPING height=0.6 as you confirmed it works well
    peaks, _ = find_peaks(intensityPeakPAS, height=0.6)
    oscillation_count = len(peaks)
    
    target_max = np.max(intensityPeakPAS)
    target_min = 1.0 - np.max(intensityInverseMinimaPAS)
    if target_min > target_max:
        intensity_range = target_max
    else:
        intensity_range = target_max - target_min

    # Normalize PAS oscillation
    pas_dynamic_score = oscillation_count / config["oscillation_threshold"]

    # Modify target_max and target_min to be in a not too small range
    amplitude_geo = luminaire_dict['standard_amplitude']
    target_max_geo = np.max(amplitude_geo)
    target_min_geo = np.min(amplitude_geo)
    if target_max_geo > target_max:
        target_max_modified = target_max_geo
    else:
        target_max_modified = target_max

    if target_min_geo < target_min:
        target_min_modified = target_min_geo
    else:    
        target_min_modified = target_min

    # ---------------------------
    # Geo-based Metrics:
    # ---------------------------
    # Phase variation
    phase_geo = luminaire_dict['phase']
    geo_phase_range = np.max(phase_geo) - np.min(phase_geo)
    geo_phase_norm = geo_phase_range / config.get("geo_phase_threshold", 0.15)
    
    # Frequency variation
    freq_geo = luminaire_dict.get('freq')
    if freq_geo is not None and hasattr(freq_geo, '__len__') and np.ndim(freq_geo) > 0:
        geo_freq_range = np.max(freq_geo) - np.min(freq_geo)
    else:
        geo_freq_range = 0.0
    geo_freq_norm = geo_freq_range / config.get("geo_freq_threshold", 0.15)
    
    # Offset variation
    offset_geo = luminaire_dict.get('offset')
    if offset_geo is not None and hasattr(offset_geo, '__len__') and np.ndim(offset_geo) > 0:
        geo_offset_range = np.max(offset_geo) - np.min(offset_geo)
    else:
        geo_offset_range = 0.0
    geo_offset_norm = geo_offset_range / config.get("geo_offset_threshold", 0.15)
    
    # Overall geo dynamic score
    overall_geo_dynamic = (geo_phase_norm + geo_freq_norm + geo_offset_norm) / 3.0

    # ---------------------------
    # BPM:
    # ---------------------------
    bpm = luminaire_dict.get('bpm', 120)

    # ---------------------------
    # Combine PAS and Geo Dynamics:
    # ---------------------------
    overall_dynamic = (overall_geo_dynamic + pas_dynamic_score) / 2.0

    # ---------------------------
    # Decision Rules with CORRECT boundaries:
    # ---------------------------
    decision_boundary_01 = config.get('decision_boundary_01', 0.22)
    decision_boundary_02 = config.get('decision_boundary_02', 0.7)
    decision_boundary_03 = config.get('decision_boundary_03', 1.1)
    decision_boundary_04 = config.get('decision_boundary_04', 2.2)
    decision_boundary_05 = config.get('decision_boundary_05', 7.0)
    
    decision = None
    if intensity_range < decision_boundary_01:
        decision = "still"
    else:
        if overall_dynamic < decision_boundary_02:
            decision = "sine"
        elif overall_dynamic < decision_boundary_03:
            decision = "pwm_basic"
        elif overall_dynamic < decision_boundary_04:
            decision = "pwm_extended"
        elif overall_dynamic < decision_boundary_05:
            decision = "odd_even"
        else:
            # For very high overall dynamics (>7.0 - rare!)
            if bpm > config['bpm_thresholds']['high']:
                decision = "square"
            else:
                decision = "random"
    
    # Debug output to understand the decision
    print(f"    Decision: {decision}")
    print(f"      Intensity range: {intensity_range:.3f} (boundary: {decision_boundary_01})")
    print(f"      Overall dynamic: {overall_dynamic:.3f}")
    print(f"      - PAS dynamic: {pas_dynamic_score:.3f} (peaks: {oscillation_count})")
    print(f"      - Geo dynamic: {overall_geo_dynamic:.3f}")
    print(f"      Boundaries: [{decision_boundary_01}, {decision_boundary_02}, {decision_boundary_03}, {decision_boundary_04}, {decision_boundary_05}]")
    
    return decision, overall_dynamic

# ============================================================================
# WAVE TYPE RECONSTRUCTOR CLASS
# ============================================================================

class WaveTypeReconstructor:
    """
    Reconstructs wave type decisions from PAS and Geo data.
    """
    
    def __init__(self, config: Optional[Dict] = None, 
                 group_mapping: Optional[Dict] = None,
                 verbose: bool = True):
        """
        Initialize reconstructor with configuration.
        
        Args:
            config: Configuration dictionary (uses default if None)
            group_mapping: Group mapping configuration (uses default if None)
            verbose: Print detailed information during processing
        """
        self.config = config if config is not None else CONFIG
        self.group_mapping = (group_mapping if group_mapping is not None 
                             else GROUP_MAPPING_CONFIG['default'])
        self.verbose = verbose
        
        if self.verbose:
            print("="*60)
            print("Wave Type Reconstructor initialized (FIXED VERSION)")
            print("="*60)
            print(f"Decision boundaries: {[self.config[f'decision_boundary_0{i}'] for i in range(1,6)]}")
            print(f"Group mapping:")
            for key, val in self.group_mapping.items():
                print(f"  {key}: PAS group {val['default_pas']+1} (index {val['default_pas']})")
    
    def load_pas_data(self, pas_file: Path) -> np.ndarray:
        """
        Load 72-dimensional intention-based (PAS) data.
        
        Args:
            pas_file: Path to PAS pickle file
            
        Returns:
            numpy array of shape (frames, 72)
        """
        with open(pas_file, 'rb') as f:
            pas_data = pickle.load(f)
        
        if self.verbose:
            print(f"  Loaded PAS data: shape {pas_data.shape}")
        
        # Verify dimensions
        if pas_data.shape[1] != 72:
            raise ValueError(f"Expected 72 PAS dimensions, got {pas_data.shape[1]}")
        
        return pas_data
    
    def load_geo_data(self, geo_file: Path) -> np.ndarray:
        """
        Load 60-dimensional oscillator-based (Geo) data.
        
        Args:
            geo_file: Path to Geo pickle file
            
        Returns:
            numpy array of shape (frames, 60)
        """
        with open(geo_file, 'rb') as f:
            geo_data = pickle.load(f)
        
        # Handle 61 dimensions (skip last column)
        if geo_data.shape[1] == 61:
            geo_data = geo_data[:, :60]
            if self.verbose:
                print(f"  Loaded Geo data: shape {geo_data.shape} (trimmed from 61)")
        elif geo_data.shape[1] == 60:
            if self.verbose:
                print(f"  Loaded Geo data: shape {geo_data.shape}")
        else:
            raise ValueError(f"Expected 60 or 61 Geo dimensions, got {geo_data.shape[1]}")
        
        return geo_data
    
    def extract_pas_metrics_for_group(self, pas_data: np.ndarray, 
                                      oscillator_group_idx: int) -> np.ndarray:
        """
        Extract PAS metrics for a specific oscillator group using mapping.
        
        Args:
            pas_data: Full PAS data (frames, 72)
            oscillator_group_idx: Oscillator group index (0-2)
            
        Returns:
            PAS data for the mapped group (frames, 6)
        """
        # Get the PAS group for this oscillator group
        mapping_key = f'oscillator_group_{oscillator_group_idx}'
        if mapping_key not in self.group_mapping:
            raise ValueError(f"No mapping found for {mapping_key}")
        
        pas_group_idx = self.group_mapping[mapping_key]['default_pas']
        
        # Extract the 6 parameters for this PAS group
        start_idx = pas_group_idx * 6
        end_idx = start_idx + 6
        
        if self.verbose:
            print(f"    Oscillator group {oscillator_group_idx} -> PAS group {pas_group_idx+1} (index {pas_group_idx})")
        
        return pas_data[:, start_idx:end_idx]
    
    def extract_geo_metrics_for_group(self, geo_data: np.ndarray, 
                                      oscillator_group_idx: int) -> Dict:
        """
        Extract Geo metrics for a specific oscillator group.
        
        Args:
            geo_data: Full Geo data (frames, 60)
            oscillator_group_idx: Oscillator group index (0-2)
            
        Returns:
            Dictionary with extracted Geo parameters
        """
        # Each oscillator group has 20 parameters
        start_idx = oscillator_group_idx * 20
        
        # Standard parameters (first 10)
        standard_params = geo_data[:, start_idx:start_idx+10]
        
        # Highlight parameters (next 10)
        highlight_params = geo_data[:, start_idx+10:start_idx+20]
        
        return {
            'pan_activity': standard_params[:, 0],
            'tilt_activity': standard_params[:, 1],
            'wave_type_a': standard_params[:, 2],  # Note: these are placeholders
            'wave_type_b': standard_params[:, 3],  # Note: these are placeholders
            'frequency': standard_params[:, 4],
            'amplitude': standard_params[:, 5],
            'offset': standard_params[:, 6],
            'phase': standard_params[:, 7],
            'col_hue': standard_params[:, 8],
            'col_sat': standard_params[:, 9],
            'highlights': highlight_params
        }
    
    def prepare_luminaire_dict(self, pas_metrics: np.ndarray, 
                               geo_metrics: Dict,
                               audio_info: Dict,
                               frames: int) -> Dict:
        """
        Prepare the luminaire dictionary for wave type decision.
        
        Args:
            pas_metrics: PAS data for one group (frames, 6)
            geo_metrics: Geo parameters for one group
            audio_info: Audio information (BPM, segments, etc.)
            frames: Number of frames
            
        Returns:
            Dictionary formatted for select_waveform_for_segment()
        """
        # PAS columns: 0=intensity_peak, 1=slope, 2=density, 3=inverse_minima, 4=hue, 5=saturation
        luminaire_dict = {
            'PASv02_allframes': pas_metrics,
            'standard_amplitude': geo_metrics['amplitude'],
            'freq': geo_metrics['frequency'],
            'phase': geo_metrics['phase'],
            'offset': geo_metrics['offset'],
            'col_hue': geo_metrics['col_hue'],
            'col_sat': geo_metrics['col_sat'],
            'bpm': audio_info.get('bpm', 120),
            'frames': frames,
            'highlights': geo_metrics['highlights'],
            # Additional parameters (set defaults for now)
            'mirroring_active': 0.5,
            'moving_direction': 0.5
        }
        
        return luminaire_dict
    
    def reconstruct_single_file(self, pas_file: Path, geo_file: Path, 
                               audio_info: Optional[Dict] = None) -> List[Dict]:
        """
        Reconstruct wave type decisions for a single file pair.
        
        Args:
            pas_file: Path to PAS data file
            geo_file: Path to Geo data file
            audio_info: Optional audio information (BPM, segments)
            
        Returns:
            List of dictionaries with wave type decisions for each group
        """
        if self.verbose:
            print(f"\nReconstructing: {pas_file.stem}")
            print("-"*60)
        
        # Load data
        pas_data = self.load_pas_data(pas_file)
        geo_data = self.load_geo_data(geo_file)
        
        # Ensure same number of frames
        min_frames = min(pas_data.shape[0], geo_data.shape[0])
        if pas_data.shape[0] != geo_data.shape[0]:
            if self.verbose:
                print(f"  Warning: Frame mismatch (PAS: {pas_data.shape[0]}, Geo: {geo_data.shape[0]})")
                print(f"  Using first {min_frames} frames")
            pas_data = pas_data[:min_frames]
            geo_data = geo_data[:min_frames]
        
        # Default audio info if not provided
        if audio_info is None:
            audio_info = {'bpm': 120, 'segment_type': 'unknown'}
        
        # Reconstruct for each oscillator group
        results = []
        for group_idx in range(3):
            if self.verbose:
                print(f"\n  Processing oscillator group {group_idx}:")
            
            # Extract metrics using group mapping
            pas_metrics = self.extract_pas_metrics_for_group(pas_data, group_idx)
            geo_metrics = self.extract_geo_metrics_for_group(geo_data, group_idx)
            
            # Prepare data for decision function
            luminaire_dict = self.prepare_luminaire_dict(
                pas_metrics, geo_metrics, audio_info, min_frames
            )
            
            # Get wave type decision
            decision, dynamic_score = select_waveform_for_segment(luminaire_dict, self.config)
            
            # Calculate additional metrics
            intensity_range = np.max(pas_metrics[:, 0]) - (1.0 - np.max(pas_metrics[:, 3]))
            
            result = {
                'group_idx': group_idx,
                'decision': decision,
                'dynamic_score': dynamic_score,
                'intensity_range': intensity_range,
                'bpm': audio_info.get('bpm', 120),
                'segment_type': audio_info.get('segment_type', 'unknown'),
                'frames': min_frames
            }
            
            results.append(result)
        
        return results
    
    def reconstruct_dataset(self, pas_dir: Path, geo_dir: Path, 
                           audio_dir: Optional[Path] = None,
                           max_files: Optional[int] = None) -> Dict:
        """
        Reconstruct wave types for an entire dataset.
        
        Args:
            pas_dir: Directory with PAS pickle files
            geo_dir: Directory with Geo pickle files
            audio_dir: Optional directory with audio JSON files
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            Dictionary with all reconstruction results
        """
        print(f"\nReconstructing dataset:")
        print(f"  PAS dir: {pas_dir}")
        print(f"  Geo dir: {geo_dir}")
        if audio_dir:
            print(f"  Audio dir: {audio_dir}")
        
        # Find matching files
        pas_files = sorted(pas_dir.glob('*.pkl'))
        if max_files:
            pas_files = pas_files[:max_files]
        print(f"  Processing {len(pas_files)} PAS files")
        
        all_results = []
        wave_type_counts = {}
        
        for pas_file in pas_files:
            # Find corresponding Geo file
            geo_file = geo_dir / pas_file.name
            if not geo_file.exists():
                # Try without seed suffix
                base_name = pas_file.stem.split('_seed')[0]
                geo_candidates = list(geo_dir.glob(f"{base_name}*.pkl"))
                if geo_candidates:
                    geo_file = geo_candidates[0]
                else:
                    print(f"  Warning: No Geo file for {pas_file.stem}")
                    continue
            
            # Load audio info if available
            audio_info = None
            if audio_dir:
                # Try different naming patterns
                audio_candidates = [
                    audio_dir / f"{pas_file.stem}.json",
                    audio_dir / f"{pas_file.stem.split('_seed')[0]}.json"
                ]
                for audio_file in audio_candidates:
                    if audio_file.exists():
                        with open(audio_file, 'r') as f:
                            audio_info = json.load(f)
                        break
            
            # Reconstruct wave types
            file_results = self.reconstruct_single_file(pas_file, geo_file, audio_info)
            
            # Collect statistics
            for result in file_results:
                wave_type = result['decision']
                if wave_type not in wave_type_counts:
                    wave_type_counts[wave_type] = 0
                wave_type_counts[wave_type] += 1
            
            all_results.append({
                'file': pas_file.stem,
                'results': file_results
            })
        
        # Calculate distribution
        total_decisions = sum(wave_type_counts.values())
        wave_type_distribution = {
            wt: count / total_decisions 
            for wt, count in wave_type_counts.items()
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL Wave Type Distribution:")
        print(f"{'='*60}")
        for wt, pct in sorted(wave_type_distribution.items(), key=lambda x: -x[1]):
            print(f"  {wt:15s}: {pct*100:5.1f}% ({wave_type_counts[wt]} occurrences)")
        
        return {
            'files': all_results,
            'wave_type_distribution': wave_type_distribution,
            'wave_type_counts': wave_type_counts,
            'total_files': len(all_results),
            'total_decisions': total_decisions
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Test the wave type reconstructor with corrected parameters."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconstruct wave type decisions')
    parser.add_argument('--pas_dir', type=str, 
                       default='data/edge_intention/light',
                       help='Directory with PAS (intention-based) data')
    parser.add_argument('--geo_dir', type=str,
                       default='data/conformer_osci/light_segments',
                       help='Directory with Geo (oscillator-based) data')
    parser.add_argument('--audio_dir', type=str,
                       default='data/conformer_osci/audio_segments_information_jsons',
                       help='Directory with audio information JSONs')
    parser.add_argument('--single_file', type=str,
                       help='Process single file (stem name) for testing')
    parser.add_argument('--max_files', type=int,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--output', type=str,
                       default='outputs_hybrid/wave_reconstruction_fixed.pkl',
                       help='Output file for results')
    parser.add_argument('--config', type=str,
                       help='Path to custom config JSON file with boundaries')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = CONFIG  # Default
    if args.config:
        print(f"Loading custom config from: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Custom boundaries loaded: [{config['decision_boundary_01']}, {config['decision_boundary_02']}, {config['decision_boundary_03']}, {config['decision_boundary_04']}, {config['decision_boundary_05']}]")
    
    # Initialize reconstructor with config
    reconstructor = WaveTypeReconstructor(config=config, verbose=True)
    
    if args.single_file:
        # Test with single file
        pas_file = Path(args.pas_dir) / f"{args.single_file}.pkl"
        geo_file = Path(args.geo_dir) / f"{args.single_file}.pkl"
        
        if not pas_file.exists():
            print(f"Error: PAS file not found: {pas_file}")
            return
        if not geo_file.exists():
            print(f"Error: Geo file not found: {geo_file}")
            return
        
        # Load audio info if available
        audio_info = None
        if args.audio_dir:
            audio_file = Path(args.audio_dir) / f"{args.single_file}.json"
            if audio_file.exists():
                with open(audio_file, 'r') as f:
                    audio_info = json.load(f)
        
        results = reconstructor.reconstruct_single_file(
            pas_file, geo_file, audio_info
        )
        
        print("\n" + "="*60)
        print("Single File Reconstruction Results:")
        print("="*60)
        for r in results:
            print(f"  Group {r['group_idx']}: {r['decision']:15s} "
                  f"(dynamic={r['dynamic_score']:.3f}, intensity={r['intensity_range']:.3f})")
    
    else:
        # Process entire dataset
        results = reconstructor.reconstruct_dataset(
            Path(args.pas_dir),
            Path(args.geo_dir),
            Path(args.audio_dir) if args.audio_dir else None,
            max_files=args.max_files
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save as JSON for readability
        json_path = output_path.with_suffix('.json')
        # Convert for JSON serialization
        json_results = {
            'wave_type_distribution': results['wave_type_distribution'],
            'wave_type_counts': results['wave_type_counts'],
            'total_files': results['total_files'],
            'total_decisions': results['total_decisions']
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Summary saved to: {json_path}")


if __name__ == '__main__':
    main()