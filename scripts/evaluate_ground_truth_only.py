#!/usr/bin/env python
"""
Evaluate Ground Truth Only
Quick script to evaluate just the ground truth data for validation
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import the pipeline
import sys
sys.path.append('scripts')
from run_evaluation_pipeline import EvaluationPipeline

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def evaluate_ground_truth(data_dir: Path = Path('data/edge_intention'),
                         output_dir: Path = Path('outputs/ground_truth_only'),
                         max_files: int = None) -> pd.DataFrame:
    """
    Evaluate only the ground truth dataset.
    
    Args:
        data_dir: Base data directory
        output_dir: Output directory for results
        max_files: Maximum files to process
        
    Returns:
        DataFrame with evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    audio_dir = data_dir / 'audio_ground_truth'
    light_dir = data_dir / 'light_ground_truth'
    
    print("\n" + "="*60)
    print("GROUND TRUTH EVALUATION")
    print("="*60)
    print(f"Audio dir: {audio_dir}")
    print(f"Light dir: {light_dir}")
    
    # Check directories exist
    if not audio_dir.exists() or not light_dir.exists():
        print("❌ Ground truth directories not found!")
        print("Please ensure you have:")
        print(f"  • {audio_dir}")
        print(f"  • {light_dir}")
        return pd.DataFrame()
    
    # Load config if available
    config = None
    config_path = Path('data/beat_configs/evaluator_config_20250808_185625.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Using tuned config: {config_path}")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config=config, verbose=True)
    
    # Run evaluation
    output_csv = output_dir / 'ground_truth_metrics.csv'
    df = pipeline.run_evaluation(
        audio_dir=audio_dir,
        light_dir=light_dir,
        output_csv=output_csv,
        max_files=max_files
    )
    
    if len(df) == 0:
        print("❌ No results obtained")
        return df
    
    # Generate summary statistics
    summary = pipeline.calculate_summary_statistics(df)
    
    # Save summary
    summary_path = output_dir / 'ground_truth_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to: {summary_path}")
    
    # Generate visualization
    create_ground_truth_visualization(df, output_dir)
    
    # Generate report
    generate_ground_truth_report(df, summary, output_dir)
    
    return df


def create_ground_truth_visualization(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualization of ground truth metrics."""
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    metrics = [
        ('ssm_correlation', 'SSM Correlation'),
        ('novelty_correlation', 'Novelty Correlation'),
        ('boundary_f_score', 'Boundary F-Score'),
        ('rms_correlation', 'RMS-Brightness'),
        ('onset_correlation', 'Onset-Change'),
        ('beat_peak_alignment', 'Beat-Peak Alignment'),
        ('beat_valley_alignment', 'Beat-Valley Alignment'),
        ('intensity_variance', 'Intensity Variance'),
        ('color_variance', 'Color Variance')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        if metric not in df.columns:
            axes[i].set_visible(False)
            continue
        
        ax = axes[i]
        
        # Create histogram with KDE
        data = df[metric].values
        
        # Histogram
        n, bins, patches = ax.hist(data, bins=15, density=True, 
                                   alpha=0.7, color='skyblue', 
                                   edgecolor='black')
        
        # KDE overlay
        from scipy.stats import gaussian_kde
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Statistics
        mean_val = data.mean()
        std_val = data.std()
        median_val = np.median(data)
        
        # Add vertical lines
        ax.axvline(mean_val, color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', 
                  linewidth=2, label=f'Median: {median_val:.3f}')
        
        # Formatting
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{title}\n(σ = {std_val:.3f})', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ground Truth Metrics Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = plots_dir / 'ground_truth_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization to: {output_path}")


def generate_ground_truth_report(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    """Generate markdown report for ground truth evaluation."""
    
    report = []
    report.append("# Ground Truth Evaluation Report\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Files Evaluated:** {len(df)}\n\n")
    
    report.append("## Purpose\n\n")
    report.append("This report analyzes the ground truth (human-designed) light shows ")
    report.append("to establish baseline performance metrics. These values represent ")
    report.append("the target distribution that generated models should aim to match.\n\n")
    
    report.append("## Summary Statistics\n\n")
    report.append("| Metric | Mean ± Std | Min | Median | Max |\n")
    report.append("|--------|------------|-----|--------|-----|\n")
    
    for metric_name, metric_stats in summary.items():
        if isinstance(metric_stats, dict) and 'mean' in metric_stats:
            report.append(f"| {metric_name.replace('_', ' ').title()} | "
                         f"{metric_stats['mean']:.3f} ± {metric_stats['std']:.3f} | "
                         f"{metric_stats['min']:.3f} | "
                         f"{metric_stats['median']:.3f} | "
                         f"{metric_stats['max']:.3f} |\n")
    
    report.append("\n## Key Observations\n\n")
    
    # Identify characteristics
    high_metrics = []
    low_metrics = []
    
    for metric_name, metric_stats in summary.items():
        if isinstance(metric_stats, dict) and 'mean' in metric_stats:
            if metric_stats['mean'] > 0.7:
                high_metrics.append((metric_name, metric_stats['mean']))
            elif metric_stats['mean'] < 0.3:
                low_metrics.append((metric_name, metric_stats['mean']))
    
    if high_metrics:
        report.append("### Strong Performance Areas\n")
        report.append("Human designers excel at:\n\n")
        for metric, value in high_metrics:
            report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n")
    
    if low_metrics:
        report.append("\n### Challenging Areas\n")
        report.append("Lower scores (possibly by design):\n\n")
        for metric, value in low_metrics:
            report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n")
    
    report.append("\n## Distribution Characteristics\n\n")
    report.append("The ground truth shows the following patterns:\n\n")
    
    # Analyze variance
    if 'intensity_variance' in summary:
        int_var = summary['intensity_variance']['mean']
        if int_var > 0.5:
            report.append("- **High intensity variation**: Dynamic, expressive lighting\n")
        else:
            report.append("- **Moderate intensity variation**: Balanced dynamics\n")
    
    if 'beat_peak_alignment' in summary and 'beat_valley_alignment' in summary:
        beat_avg = (summary['beat_peak_alignment']['mean'] + 
                   summary['beat_valley_alignment']['mean']) / 2
        if beat_avg > 0.5:
            report.append("- **Strong beat synchronization**: Tightly coupled to rhythm\n")
        else:
            report.append("- **Flexible beat alignment**: Not strictly beat-locked\n")
    
    # Best and worst files
    if 'best_file' in summary:
        report.append(f"\n### Best Performing File\n")
        report.append(f"- **{summary['best_file']['name']}**: {summary['best_file']['score']:.3f}\n")
    
    if 'worst_file' in summary:
        report.append(f"\n### Lowest Performing File\n")
        report.append(f"- **{summary['worst_file']['name']}**: {summary['worst_file']['score']:.3f}\n")
    
    report.append("\n## Usage Notes\n\n")
    report.append("These ground truth metrics serve as the target distribution ")
    report.append("for evaluating generated light shows. A good generative model ")
    report.append("should produce metrics with similar mean values and distributions ")
    report.append("to these ground truth measurements.\n")
    
    # Save report
    report_path = output_dir / 'ground_truth_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"📄 Report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate ground truth light shows'
    )
    parser.add_argument('--data_dir', type=str,
                       default='data/edge_intention',
                       help='Base data directory')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/ground_truth_only',
                       help='Output directory')
    parser.add_argument('--max_files', type=int,
                       help='Maximum files to process')
    
    args = parser.parse_args()
    
    # Run evaluation
    df = evaluate_ground_truth(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_files=args.max_files
    )
    
    if len(df) > 0:
        print("\n" + "="*60)
        print("✅ GROUND TRUTH EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print("\nFiles created:")
        print("- ground_truth_metrics.csv: Raw metrics data")
        print("- ground_truth_summary.json: Statistical summary")
        print("- ground_truth_report.md: Analysis report")
        print("- plots/ground_truth_distributions.png: Visualizations")


if __name__ == '__main__':
    main()