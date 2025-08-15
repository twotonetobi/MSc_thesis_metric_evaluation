#!/usr/bin/env python
"""
Hybrid Evaluator - Simple Working Version
Evaluates the quality of wave type decisions from reconstruction
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import argparse

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the wave type reconstructor
from segment_based_hybrid_oscillator_evaluation.wave_type_reconstructor import WaveTypeReconstructor

class HybridEvaluator:
    """Evaluates hybrid lighting generation quality."""
    
    def __init__(self, config_path: str = 'configs/final_optimal.json', verbose: bool = True):
        """Initialize evaluator with config."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.reconstructor = WaveTypeReconstructor(self.config, verbose=False)
        self.verbose = verbose
        
    def evaluate_wave_consistency(self, decisions: List[Dict]) -> Dict:
        """
        Evaluate if wave types are consistent within segments.
        
        Returns:
            Consistency metrics
        """
        # For now, calculate overall consistency
        wave_types = [d['decision'] for d in decisions]
        
        if not wave_types:
            return {'consistency': 0.0, 'dominant_wave': None}
        
        # Find dominant wave type
        wave_counts = Counter(wave_types)
        dominant_wave, dominant_count = wave_counts.most_common(1)[0]
        
        # Consistency = how much the dominant wave dominates
        consistency = dominant_count / len(wave_types)
        
        return {
            'consistency': consistency,
            'dominant_wave': dominant_wave,
            'wave_counts': dict(wave_counts)
        }
    
    def evaluate_musical_coherence(self, decisions: List[Dict]) -> float:
        """
        Evaluate if wave types match their expected dynamic ranges.
        
        Returns:
            Coherence score (0-1)
        """
        coherence_scores = []
        
        for decision in decisions:
            wave = decision['decision']
            dynamic = decision['dynamic_score']
            
            # Check if dynamic score is in expected range for wave type
            in_range = False
            
            if wave == 'still':
                # Still should have very low dynamics
                in_range = dynamic < 1.0
            elif wave == 'sine':
                # Sine should be in lower-mid range
                in_range = 0.5 < dynamic < 2.0
            elif wave == 'pwm_basic':
                # PWM basic in mid range
                in_range = 1.5 < dynamic < 2.5
            elif wave == 'pwm_extended':
                # PWM extended in mid-high range
                in_range = 2.0 < dynamic < 3.0
            elif wave == 'odd_even':
                # Odd even in high range
                in_range = 2.5 < dynamic < 4.0
            elif wave in ['random', 'square']:
                # Random/square in very high range
                in_range = dynamic > 3.0
            
            coherence_scores.append(float(in_range))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def evaluate_transition_smoothness(self, decisions: List[Dict]) -> Dict:
        """
        Evaluate smoothness of transitions between wave types.
        
        Returns:
            Transition metrics
        """
        if len(decisions) < 2:
            return {
                'num_transitions': 0,
                'smooth_ratio': 1.0,
                'avg_dynamic_jump': 0.0
            }
        
        transitions = []
        smooth_transitions = 0
        
        for i in range(1, len(decisions)):
            prev = decisions[i-1]
            curr = decisions[i]
            
            if prev['decision'] != curr['decision']:
                # There's a transition
                dynamic_jump = abs(curr['dynamic_score'] - prev['dynamic_score'])
                
                # Consider smooth if dynamic jump is < 1.0
                is_smooth = dynamic_jump < 1.0
                if is_smooth:
                    smooth_transitions += 1
                
                transitions.append({
                    'from': prev['decision'],
                    'to': curr['decision'],
                    'dynamic_jump': dynamic_jump,
                    'smooth': is_smooth
                })
        
        if transitions:
            smooth_ratio = smooth_transitions / len(transitions)
            avg_jump = np.mean([t['dynamic_jump'] for t in transitions])
        else:
            smooth_ratio = 1.0  # No transitions = perfectly smooth
            avg_jump = 0.0
        
        return {
            'num_transitions': len(transitions),
            'smooth_ratio': smooth_ratio,
            'avg_dynamic_jump': avg_jump,
            'transitions': transitions[:5]  # Keep first 5 for inspection
        }
    
    def evaluate_distribution_match(self, decisions: List[Dict]) -> float:
        """
        Compare distribution to our target distribution.
        
        Returns:
            Distribution match score (0-1)
        """
        target_dist = {
            'still': 0.298,
            'odd_even': 0.219,
            'sine': 0.176,
            'square': 0.116,
            'pwm_basic': 0.111,
            'pwm_extended': 0.070,
            'random': 0.010
        }
        
        # Calculate actual distribution
        wave_types = [d['decision'] for d in decisions]
        if not wave_types:
            return 0.0
        
        actual_counts = Counter(wave_types)
        total = len(wave_types)
        actual_dist = {w: actual_counts.get(w, 0) / total for w in target_dist.keys()}
        
        # Calculate similarity (1 - average absolute difference)
        diffs = [abs(target_dist[w] - actual_dist[w]) for w in target_dist.keys()]
        avg_diff = np.mean(diffs)
        
        return max(0.0, 1.0 - avg_diff)
    
    def evaluate_single_file(self, pas_file: Path, geo_file: Path, 
                            audio_file: Path = None) -> Dict:
        """
        Evaluate a single file triplet.
        
        Returns:
            Evaluation metrics for the file
        """
        # Load audio info if available
        audio_info = {}
        if audio_file and audio_file.exists():
            with open(audio_file, 'r') as f:
                audio_info = json.load(f)
        
        # Reconstruct wave types
        try:
            decisions = self.reconstructor.reconstruct_single_file(
                pas_file, geo_file, audio_info
            )
        except Exception as e:
            print(f"Error reconstructing {pas_file.stem}: {e}")
            return None
        
        # Evaluate different aspects
        consistency = self.evaluate_wave_consistency(decisions)
        coherence = self.evaluate_musical_coherence(decisions)
        transitions = self.evaluate_transition_smoothness(decisions)
        distribution_match = self.evaluate_distribution_match(decisions)
        
        # Calculate overall score
        overall_score = np.mean([
            consistency['consistency'],
            coherence,
            transitions['smooth_ratio'],
            distribution_match
        ])
        
        return {
            'file': pas_file.stem,
            'consistency': consistency,
            'coherence': coherence,
            'transitions': transitions,
            'distribution_match': distribution_match,
            'overall_score': overall_score,
            'num_decisions': len(decisions)
        }
    
    def evaluate_dataset(self, pas_dir: Path, geo_dir: Path, 
                        audio_dir: Path = None,
                        max_files: int = None) -> Dict:
        """
        Evaluate entire dataset.
        
        Returns:
            Aggregated evaluation results
        """
        pas_files = sorted(pas_dir.glob('*.pkl'))
        
        if max_files:
            pas_files = pas_files[:max_files]
        
        print(f"\nEvaluating {len(pas_files)} files...")
        print("="*60)
        
        results = []
        failed = 0
        
        for i, pas_file in enumerate(pas_files, 1):
            # Find corresponding geo file
            geo_file = geo_dir / pas_file.name
            if not geo_file.exists():
                # Try without seed suffix
                base_name = pas_file.stem.split('_seed')[0]
                geo_candidates = list(geo_dir.glob(f"{base_name}*.pkl"))
                if geo_candidates:
                    geo_file = geo_candidates[0]
                else:
                    if self.verbose:
                        print(f"  [{i}/{len(pas_files)}] No geo file for {pas_file.stem}")
                    failed += 1
                    continue
            
            # Find audio file if available
            audio_file = None
            if audio_dir:
                audio_candidates = [
                    audio_dir / f"{pas_file.stem}.json",
                    audio_dir / f"{pas_file.stem.split('_seed')[0]}.json"
                ]
                for candidate in audio_candidates:
                    if candidate.exists():
                        audio_file = candidate
                        break
            
            if self.verbose and i % 10 == 0:
                print(f"  Processing file {i}/{len(pas_files)}...")
            
            # Evaluate
            result = self.evaluate_single_file(pas_file, geo_file, audio_file)
            if result:
                results.append(result)
        
        if not results:
            print("No files successfully evaluated!")
            return None
        
        # Aggregate results
        print(f"\nSuccessfully evaluated {len(results)} files")
        print(f"Failed: {failed} files")
        
        # Calculate aggregate metrics
        avg_consistency = np.mean([r['consistency']['consistency'] for r in results])
        avg_coherence = np.mean([r['coherence'] for r in results])
        avg_smooth = np.mean([r['transitions']['smooth_ratio'] for r in results])
        avg_dist_match = np.mean([r['distribution_match'] for r in results])
        avg_overall = np.mean([r['overall_score'] for r in results])
        
        # Find best and worst files
        results_sorted = sorted(results, key=lambda x: x['overall_score'], reverse=True)
        best_files = results_sorted[:5]
        worst_files = results_sorted[-5:]
        
        return {
            'num_files': len(results),
            'failed_files': failed,
            'aggregate_metrics': {
                'avg_consistency': avg_consistency,
                'avg_coherence': avg_coherence,
                'avg_smoothness': avg_smooth,
                'avg_distribution_match': avg_dist_match,
                'avg_overall_score': avg_overall
            },
            'best_files': best_files,
            'worst_files': worst_files,
            'all_results': results
        }
    
    def generate_report(self, evaluation_results: Dict, 
                       output_path: Path = Path('outputs_hybrid/evaluation_report.md')):
        """Generate a markdown report from evaluation results."""
        
        if not evaluation_results:
            print("No results to report!")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# Hybrid Wave Type Evaluation Report\n\n")
        report.append(f"**Date:** {np.datetime64('now')}\n")
        report.append(f"**Files Evaluated:** {evaluation_results['num_files']}\n")
        report.append(f"**Failed:** {evaluation_results['failed_files']}\n\n")
        
        # Aggregate metrics
        report.append("## Overall Performance\n\n")
        metrics = evaluation_results['aggregate_metrics']
        
        report.append("| Metric | Score |\n")
        report.append("|--------|-------|\n")
        report.append(f"| **Overall Score** | {metrics['avg_overall_score']:.3f} |\n")
        report.append(f"| Consistency | {metrics['avg_consistency']:.3f} |\n")
        report.append(f"| Musical Coherence | {metrics['avg_coherence']:.3f} |\n")
        report.append(f"| Transition Smoothness | {metrics['avg_smoothness']:.3f} |\n")
        report.append(f"| Distribution Match | {metrics['avg_distribution_match']:.3f} |\n\n")
        
        # Interpretation
        report.append("## Interpretation\n\n")
        
        overall = metrics['avg_overall_score']
        if overall > 0.8:
            quality = "Excellent"
        elif overall > 0.6:
            quality = "Good"
        elif overall > 0.4:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        report.append(f"**Quality Assessment: {quality}**\n\n")
        
        if metrics['avg_consistency'] < 0.5:
            report.append("- ⚠️ Low consistency: Wave types are changing too frequently\n")
        if metrics['avg_coherence'] < 0.5:
            report.append("- ⚠️ Low coherence: Wave types don't match their expected dynamic ranges\n")
        if metrics['avg_smoothness'] < 0.5:
            report.append("- ⚠️ Rough transitions: Large jumps in dynamics between wave changes\n")
        if metrics['avg_distribution_match'] < 0.5:
            report.append("- ⚠️ Distribution mismatch: Individual files deviate from target distribution\n")
        
        if overall > 0.7:
            report.append("- ✅ Overall system is performing well!\n")
        
        # Best and worst files
        report.append("\n## Top Performing Files\n\n")
        for i, file_result in enumerate(evaluation_results['best_files'][:3], 1):
            report.append(f"{i}. **{file_result['file']}** (score: {file_result['overall_score']:.3f})\n")
        
        report.append("\n## Worst Performing Files\n\n")
        for i, file_result in enumerate(evaluation_results['worst_files'][:3], 1):
            report.append(f"{i}. **{file_result['file']}** (score: {file_result['overall_score']:.3f})\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        print(f"\n📄 Report saved to: {output_path}")
        
        # Also save raw results as JSON
        json_path = output_path.with_suffix('.json')
        results_for_json = {
            'num_files': evaluation_results['num_files'],
            'failed_files': evaluation_results['failed_files'],
            'aggregate_metrics': evaluation_results['aggregate_metrics']
        }
        with open(json_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"📊 Raw results saved to: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate hybrid wave type reconstruction')
    parser.add_argument('--pas_dir', type=str, 
                       default='data/edge_intention/light',
                       help='Directory with PAS data')
    parser.add_argument('--geo_dir', type=str,
                       default='data/conformer_osci/light_segments',
                       help='Directory with Geo data')
    parser.add_argument('--audio_dir', type=str,
                       default='data/conformer_osci/audio_segments_information_jsons',
                       help='Directory with audio JSONs')
    parser.add_argument('--config', type=str,
                       default='configs/final_optimal.json',
                       help='Configuration file')
    parser.add_argument('--max_files', type=int,
                       help='Maximum files to evaluate (for testing)')
    parser.add_argument('--output', type=str,
                       default='outputs_hybrid/evaluation_report.md',
                       help='Output report path')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HybridEvaluator(args.config, verbose=not args.quiet)
    
    # Run evaluation
    print("\n" + "="*60)
    print("HYBRID WAVE TYPE EVALUATION")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"PAS dir: {args.pas_dir}")
    print(f"Geo dir: {args.geo_dir}")
    
    results = evaluator.evaluate_dataset(
        Path(args.pas_dir),
        Path(args.geo_dir),
        Path(args.audio_dir) if args.audio_dir else None,
        max_files=args.max_files
    )
    
    if results:
        # Generate report
        evaluator.generate_report(results, Path(args.output))
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        metrics = results['aggregate_metrics']
        print(f"Overall Score:        {metrics['avg_overall_score']:.3f}")
        print(f"Consistency:          {metrics['avg_consistency']:.3f}")
        print(f"Musical Coherence:    {metrics['avg_coherence']:.3f}")
        print(f"Transition Smoothness: {metrics['avg_smoothness']:.3f}")
        print(f"Distribution Match:   {metrics['avg_distribution_match']:.3f}")


if __name__ == '__main__':
    main()