#!/usr/bin/env python
"""
Generate comprehensive evaluation report for oscillator-based predictions.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional  # FIXED: Added missing imports

class OscillatorReportGenerator:
    """Generate evaluation report combining all metrics."""
    
    def generate_report(self, 
                       model_results: Dict,
                       baseline_results: Dict,
                       inter_group_results: Dict,
                       training_stats: Dict,
                       output_path: Path):
        """Generate comprehensive markdown report."""
        
        report = []
        report.append("# Oscillator-Based Lighting Generation Evaluation Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(self._generate_summary(model_results, baseline_results))
        
        # Distribution Comparison
        report.append("\n## Parameter Distribution Analysis\n")
        report.append(self._compare_distributions(model_results, training_stats))
        
        # Musical Convention Analysis
        report.append("\n## Musical Convention Adherence\n")
        report.append(self._analyze_conventions(model_results))
        
        # Baseline Comparison
        report.append("\n## Baseline Comparison\n")
        report.append(self._compare_baselines(model_results, baseline_results))
        
        # Inter-Group Dynamics
        report.append("\n## Inter-Group Coordination\n")
        report.append(self._analyze_inter_group(inter_group_results))
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        print(f"Report saved to {output_path}")
    
    def _generate_summary(self, model_results: Dict, baseline_results: Dict) -> str:
        """Generate executive summary."""
        summary = []
        
        # Check if we have results
        if not model_results:
            summary.append("No model results available.\n")
            return ''.join(summary)
        
        summary.append("This report evaluates the oscillator-based lighting generation model ")
        summary.append("against training data distributions and baseline methods.\n\n")
        
        summary.append("### Key Findings:\n")
        
        # Add key metrics if available
        if 'segment_statistics' in model_results:
            num_segments = len(model_results['segment_statistics'])
            summary.append(f"- Analyzed {num_segments} different segment types\n")
        
        if 'num_files' in model_results:
            summary.append(f"- Evaluated {model_results['num_files']} files\n")
        
        # Baseline comparison summary
        if baseline_results:
            summary.append("\n### Performance vs Baselines:\n")
            # Would add specific comparison metrics here
            summary.append("- Detailed baseline comparisons in sections below\n")
        
        return ''.join(summary)
    
    def _compare_distributions(self, model_results: Dict, training_stats: Dict) -> str:
        """Compare parameter distributions between model and training."""
        comparison = []
        
        if not training_stats or 'global' not in training_stats:
            comparison.append("Training statistics not available for comparison.\n")
            return ''.join(comparison)
        
        comparison.append("### Global Parameter Statistics\n\n")
        comparison.append("| Parameter | Training Mean ± Std | Model Mean ± Std | Difference |\n")
        comparison.append("|-----------|---------------------|------------------|------------|\n")
        
        # Compare each parameter if data is available
        if 'global' in training_stats:
            for param_name, train_stats in training_stats['global'].items():
                train_mean = train_stats.get('mean', 0)
                train_std = train_stats.get('std', 0)
                
                # Would get model stats from model_results if structured appropriately
                comparison.append(f"| {param_name} | {train_mean:.3f} ± {train_std:.3f} | TBD | TBD |\n")
        
        comparison.append("\n*Note: Model statistics extraction to be implemented based on evaluation results structure.*\n")
        
        return ''.join(comparison)
    
    def _analyze_conventions(self, model_results: Dict) -> str:
        """Analyze adherence to musical conventions."""
        conventions = []
        
        if 'segment_statistics' not in model_results:
            conventions.append("Segment statistics not available.\n")
            return ''.join(conventions)
        
        conventions.append("### Wave Type Usage by Segment\n\n")
        
        for seg_type, stats in model_results['segment_statistics'].items():
            conventions.append(f"#### {seg_type.capitalize()}\n")
            
            if 'wave_distribution' in stats:
                conventions.append("- **Wave Types Used:**\n")
                for wave, freq in stats['wave_distribution'].items():
                    conventions.append(f"  - {wave}: {freq*100:.1f}%\n")
            
            if 'mean_amplitude' in stats:
                conventions.append(f"- **Mean Amplitude:** {stats['mean_amplitude']:.3f}\n")
            
            if 'mean_frequency' in stats:
                conventions.append(f"- **Mean Frequency:** {stats['mean_frequency']:.3f}\n")
            
            if 'mean_mai' in stats:
                conventions.append(f"- **Movement Activity:** {stats['mean_mai']:.3f}\n")
            
            conventions.append("\n")
        
        return ''.join(conventions)
    
    def _compare_baselines(self, model_results: Dict, baseline_results: Dict) -> str:
        """Compare model performance against baselines."""
        comparison = []
        
        if not baseline_results:
            comparison.append("Baseline results not available for comparison.\n")
            return ''.join(comparison)
        
        comparison.append("### Performance Comparison\n\n")
        comparison.append("| Metric | Model | Random | Beat-Sync | Constant |\n")
        comparison.append("|--------|-------|--------|-----------|----------|\n")
        
        # Extract metrics from results
        # This would be populated based on actual baseline evaluation results
        comparison.append("| Plausibility | TBD | TBD | TBD | TBD |\n")
        comparison.append("| Consistency | TBD | TBD | TBD | TBD |\n")
        comparison.append("| Musical Coherence | TBD | TBD | TBD | TBD |\n")
        
        comparison.append("\n*Note: Baseline comparison metrics to be populated from evaluation results.*\n")
        
        return ''.join(comparison)
    
    def _analyze_inter_group(self, inter_group_results: Dict) -> str:
        """Analyze inter-group coordination."""
        analysis = []
        
        if not inter_group_results:
            analysis.append("Inter-group analysis not available.\n")
            return ''.join(analysis)
        
        analysis.append("### Group Coordination Analysis\n\n")
        
        if 'mean_correlation' in inter_group_results:
            mean_corr = inter_group_results['mean_correlation']
            std_corr = inter_group_results.get('std_correlation', 0)
            analysis.append(f"- **Mean Inter-Group Correlation:** {mean_corr:.3f} ± {std_corr:.3f}\n")
        
        if 'mean_phase_diff' in inter_group_results:
            phase_diff = inter_group_results['mean_phase_diff']
            analysis.append(f"- **Mean Phase Difference:** {phase_diff:.3f} radians\n")
        
        if 'mean_complementary' in inter_group_results:
            comp = inter_group_results['mean_complementary']
            analysis.append(f"- **Complementary Dynamics Score:** {comp:.3f}\n")
        
        analysis.append("\n### Correlation Distribution\n")
        if 'strong_correlations' in inter_group_results:
            strong = inter_group_results['strong_correlations'] * 100
            moderate = inter_group_results.get('moderate_correlations', 0) * 100
            independent = inter_group_results.get('independent_groups', 0) * 100
            
            analysis.append(f"- Strong (>0.7): {strong:.1f}%\n")
            analysis.append(f"- Moderate (0.3-0.7): {moderate:.1f}%\n")
            analysis.append(f"- Independent (<0.3): {independent:.1f}%\n")
        
        analysis.append("\n### Interpretation\n")
        if 'mean_correlation' in inter_group_results:
            if inter_group_results['mean_correlation'] > 0.5:
                analysis.append("The lighting groups show **coordinated behavior**, ")
                analysis.append("suggesting the model learned to synchronize groups effectively.\n")
            elif inter_group_results['mean_correlation'] < 0.3:
                analysis.append("The lighting groups operate **mostly independently**, ")
                analysis.append("providing diverse visual elements.\n")
            else:
                analysis.append("The lighting groups show **moderate coordination**, ")
                analysis.append("balancing synchronization with independence.\n")
        
        return ''.join(analysis)


def load_results(directory: Path, filename: str) -> Optional[Dict]:
    """Load results from pickle or JSON file."""
    # Try pickle first
    pkl_path = directory / filename if filename.endswith('.pkl') else directory / f"{filename}.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    # Try JSON
    json_path = directory / filename if filename.endswith('.json') else directory / f"{filename}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    
    return None


def main():
    """Main execution logic."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate oscillator evaluation report')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory with model evaluation results')
    parser.add_argument('--baseline_dir', type=str,
                       help='Directory with baseline evaluation results')
    parser.add_argument('--inter_group_dir', type=str,
                       help='Directory with inter-group analysis results')
    parser.add_argument('--training_stats', type=str,
                       help='Path to training statistics file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for the report (markdown file)')
    
    args = parser.parse_args()
    
    # Load all results
    model_results = {}
    baseline_results = {}
    inter_group_results = {}
    training_stats = {}
    
    # Load model results
    model_dir = Path(args.model_dir)
    if model_dir.exists():
        # Try to load oscillator evaluation results
        model_eval = load_results(model_dir, 'oscillator_evaluation')
        if model_eval:
            model_results = model_eval
            print(f"Loaded model evaluation results from {model_dir}")
        else:
            print(f"Warning: No oscillator_evaluation file found in {model_dir}")
    
    # Load baseline results if provided
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        if baseline_dir.exists():
            # Load results for each baseline type
            for baseline_type in ['random', 'beat_sync', 'constant']:
                baseline_path = baseline_dir / baseline_type
                if baseline_path.exists():
                    baseline_eval = load_results(baseline_path, 'oscillator_evaluation')
                    if baseline_eval:
                        baseline_results[baseline_type] = baseline_eval
                        print(f"Loaded {baseline_type} baseline results")
    
    # Load inter-group results
    inter_group_dir = Path(args.inter_group_dir) if args.inter_group_dir else model_dir / 'inter_group'
    if inter_group_dir.exists():
        inter_group = load_results(inter_group_dir, 'inter_group_analysis')
        if not inter_group:
            inter_group = load_results(inter_group_dir, 'inter_group_summary')
        if inter_group:
            inter_group_results = inter_group
            print(f"Loaded inter-group analysis from {inter_group_dir}")
    
    # Load training statistics
    if args.training_stats:
        stats_path = Path(args.training_stats)
        if stats_path.exists():
            training_stats = load_results(stats_path.parent, stats_path.name)
            if training_stats:
                print(f"Loaded training statistics from {stats_path}")
    else:
        # Try default location
        default_stats = Path('data/training_data/statistics/parameter_distributions.pkl')
        if default_stats.exists():
            with open(default_stats, 'rb') as f:
                training_stats = pickle.load(f)
                print(f"Loaded training statistics from default location")
    
    # Generate report
    generator = OscillatorReportGenerator()
    output_path = Path(args.output_path)
    
    print("\nGenerating report...")
    generator.generate_report(
        model_results=model_results,
        baseline_results=baseline_results,
        inter_group_results=inter_group_results,
        training_stats=training_stats,
        output_path=output_path
    )
    
    print(f"\n✓ Report generated successfully: {output_path}")
    
    # Summary of what was included
    print("\nReport includes:")
    if model_results:
        print("  ✓ Model evaluation results")
    if baseline_results:
        print(f"  ✓ Baseline comparisons ({len(baseline_results)} baselines)")
    if inter_group_results:
        print("  ✓ Inter-group coordination analysis")
    if training_stats:
        print("  ✓ Training data distribution comparison")

if __name__ == '__main__':
    main()