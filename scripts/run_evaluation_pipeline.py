#!/usr/bin/env python
"""
Reusable Evaluation Pipeline Runner
Runs the 9-metric structural evaluation on any given pair of audio and light directories.
This is a modular version that can be used for both generated and ground-truth data.
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from structural_evaluator import StructuralEvaluator


class EvaluationPipeline:
    """Reusable pipeline for evaluating audio-light pairs."""
    
    def __init__(self, config: Optional[Dict] = None, verbose: bool = True):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Optional configuration for StructuralEvaluator
            verbose: Whether to print detailed progress
        """
        self.config = config or {}
        self.verbose = verbose
        self.evaluator = StructuralEvaluator(config)
        
    def find_matching_files(self, audio_dir: Path, light_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Find matching audio-light file pairs.
        
        Args:
            audio_dir: Directory containing audio pickle files
            light_dir: Directory containing light pickle files
            
        Returns:
            List of (audio_file, light_file) tuples
        """
        audio_files = sorted(audio_dir.glob('*.pkl'))
        matched_pairs = []
        
        for audio_file in audio_files:
            # Look for exact match first
            light_file = light_dir / audio_file.name
            
            if not light_file.exists():
                # Try finding with pattern matching
                light_candidates = list(light_dir.glob(f'*{audio_file.stem}*.pkl'))
                if light_candidates:
                    light_file = light_candidates[0]
                else:
                    if self.verbose:
                        print(f"  ⚠️  No light file found for: {audio_file.stem}")
                    continue
            
            matched_pairs.append((audio_file, light_file))
        
        return matched_pairs
    
    def evaluate_file_pair(self, audio_file: Path, light_file: Path) -> Optional[Dict]:
        """
        Evaluate a single audio-light file pair.
        
        Args:
            audio_file: Path to audio pickle
            light_file: Path to light pickle
            
        Returns:
            Dictionary with evaluation metrics, or None if evaluation failed
        """
        try:
            # Run evaluation
            metrics, viz_data = self.evaluator.evaluate_single_file(audio_file, light_file)
            
            # Prepare result dictionary
            result = {
                'file': audio_file.stem,
                'audio_path': str(audio_file),
                'light_path': str(light_file),
                # Structure metrics
                'ssm_correlation': metrics.get('ssm_correlation', 0),
                'novelty_correlation': metrics.get('novelty_correlation', 0),
                'boundary_f_score': metrics.get('boundary_f_score', 0),
                'boundary_precision': metrics.get('boundary_precision', 0),
                'boundary_recall': metrics.get('boundary_recall', 0),
                # Dynamic metrics
                'rms_correlation': metrics.get('rms_correlation', 0),
                'onset_correlation': metrics.get('onset_correlation', 0),
                # Beat alignment
                'beat_peak_alignment': metrics.get('beat_peak_alignment', 0),
                'beat_valley_alignment': metrics.get('beat_valley_alignment', 0),
                # Variance metrics
                'intensity_variance': metrics.get('intensity_variance', 0),
                'color_variance': metrics.get('color_variance', 0),
            }
            
            # Calculate aggregate scores
            result['structure_score'] = np.mean([
                result['ssm_correlation'],
                result['novelty_correlation'],
                result['boundary_f_score']
            ])
            
            result['rhythm_score'] = np.mean([
                result['beat_peak_alignment'],
                result['beat_valley_alignment']
            ])
            
            result['dynamics_score'] = np.mean([
                result['rms_correlation'],
                result['onset_correlation']
            ])
            
            result['overall_score'] = np.mean([
                result['structure_score'],
                result['rhythm_score'],
                result['dynamics_score']
            ])
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error evaluating {audio_file.stem}: {e}")
                traceback.print_exc()
            return None
    
    def run_evaluation(self, audio_dir: Path, light_dir: Path, 
                       output_csv: Optional[Path] = None,
                       max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Run evaluation on all matching files in the directories.
        
        Args:
            audio_dir: Directory with audio pickle files
            light_dir: Directory with light pickle files
            output_csv: Optional path to save results CSV
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION PIPELINE")
        print("="*60)
        print(f"Audio directory: {audio_dir}")
        print(f"Light directory: {light_dir}")
        
        # Find matching files
        matched_pairs = self.find_matching_files(audio_dir, light_dir)
        
        if max_files:
            matched_pairs = matched_pairs[:max_files]
        
        print(f"Found {len(matched_pairs)} matching file pairs")
        print("-"*60)
        
        # Evaluate each pair
        results = []
        failed_count = 0
        
        for i, (audio_file, light_file) in enumerate(matched_pairs, 1):
            if self.verbose:
                print(f"\n[{i}/{len(matched_pairs)}] Processing: {audio_file.stem}")
            
            result = self.evaluate_file_pair(audio_file, light_file)
            
            if result:
                results.append(result)
                if self.verbose and i % 10 == 0:
                    print(f"  ✓ Completed {i}/{len(matched_pairs)} files")
            else:
                failed_count += 1
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save if output path provided
        if output_csv and len(df) > 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\n📊 Results saved to: {output_csv}")
        
        # Print summary statistics
        if len(df) > 0:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Successfully processed: {len(results)} files")
            print(f"Failed: {failed_count} files")
            print("\nMetric Averages:")
            
            metrics_to_report = [
                ('overall_score', 'Overall Score'),
                ('structure_score', 'Structure Score'),
                ('rhythm_score', 'Rhythm Score'),
                ('dynamics_score', 'Dynamics Score'),
                ('ssm_correlation', 'SSM Correlation'),
                ('beat_peak_alignment', 'Beat Peak Alignment'),
                ('rms_correlation', 'RMS Correlation')
            ]
            
            for metric, name in metrics_to_report:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    print(f"  {name:25s}: {mean_val:.3f} ± {std_val:.3f}")
        
        return df
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive summary statistics.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if len(df) == 0:
            return {}
        
        summary = {
            'num_files': len(df),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate statistics for each metric
        metrics = [
            'ssm_correlation', 'novelty_correlation', 'boundary_f_score',
            'rms_correlation', 'onset_correlation',
            'beat_peak_alignment', 'beat_valley_alignment',
            'intensity_variance', 'color_variance',
            'structure_score', 'rhythm_score', 'dynamics_score', 'overall_score'
        ]
        
        for metric in metrics:
            if metric in df.columns:
                summary[metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'median': float(df[metric].median()),
                    'q25': float(df[metric].quantile(0.25)),
                    'q75': float(df[metric].quantile(0.75))
                }
        
        # Find best and worst files
        if 'overall_score' in df.columns:
            best_idx = df['overall_score'].idxmax()
            worst_idx = df['overall_score'].idxmin()
            
            summary['best_file'] = {
                'name': df.loc[best_idx, 'file'],
                'score': float(df.loc[best_idx, 'overall_score'])
            }
            
            summary['worst_file'] = {
                'name': df.loc[worst_idx, 'file'],
                'score': float(df.loc[worst_idx, 'overall_score'])
            }
        
        return summary


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run structural evaluation pipeline on audio-light pairs'
    )
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing audio pickle files')
    parser.add_argument('--light_dir', type=str, required=True,
                       help='Directory containing light pickle files')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file path for results')
    parser.add_argument('--config', type=str,
                       help='Optional JSON config file for evaluator')
    parser.add_argument('--max_files', type=int,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config=config, verbose=not args.quiet)
    
    # Run evaluation
    df = pipeline.run_evaluation(
        audio_dir=Path(args.audio_dir),
        light_dir=Path(args.light_dir),
        output_csv=Path(args.output_csv),
        max_files=args.max_files
    )
    
    # Calculate and save summary statistics
    if len(df) > 0:
        summary = pipeline.calculate_summary_statistics(df)
        
        # Save summary as JSON
        summary_path = Path(args.output_csv).with_suffix('.summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📈 Summary statistics saved to: {summary_path}")
    
    return df


if __name__ == '__main__':
    main()