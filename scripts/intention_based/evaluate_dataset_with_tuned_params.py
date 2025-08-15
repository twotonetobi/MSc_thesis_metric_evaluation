#!/usr/bin/env python
"""
Evaluation script using parameters tuned with the Interactive GUI
This replaces evaluate_dataset.py when you have tuned parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from intention_based.structural_evaluator import StructuralEvaluator
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import traceback


def evaluate_dataset_with_tuned_params(config_file, data_dir, output_dir='outputs', 
                                       create_plots=True, verbose=True):
    """
    Evaluate entire dataset using tuned parameters from GUI.
    
    Args:
        config_file: Path to JSON config saved from the Interactive Tuner
        data_dir: Path to edge_intention directory with audio/ and light/ subdirs
        output_dir: Where to save results
        create_plots: Whether to generate visualization plots
        verbose: Print detailed progress
    """
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'reports').mkdir(exist_ok=True)
    
    # Load tuned configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("="*60)
    print("EVALUATION WITH TUNED PARAMETERS")
    print("="*60)
    print(f"Configuration file: {config_file}")
    print(f"Tuned on: {config.get('tuned_on', 'unknown')}")
    print(f"Tuned with: {config.get('tuned_with_file', 'unknown')}")
    print("-"*60)
    print("Key Parameters:")
    print(f"  Rhythmic window: {config.get('rhythmic_window', 'N/A')} frames")
    print(f"  STD threshold: {config.get('rhythmic_threshold', 'N/A'):.3f}")
    print(f"  Beat sigma: {config.get('beat_align_sigma', 'N/A'):.2f}")
    print(f"  Peak prominence: {config.get('peak_prominence', 'N/A'):.3f}")
    print(f"  Use rhythmic filter: {config.get('use_rhythmic_filter', True)}")
    print("="*60)
    
    # Create evaluator with tuned params
    if verbose:
        config['verbose'] = True
    evaluator = StructuralEvaluator(config)
    
    # Process all files
    audio_dir = Path(data_dir) / 'audio'
    light_dir = Path(data_dir) / 'light'
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    audio_files = sorted(audio_dir.glob('*.pkl'))
    print(f"\nFound {len(audio_files)} audio files to process")
    
    results = []
    failed_files = []
    
    for i, audio_file in enumerate(audio_files, 1):
        # Find corresponding light file
        light_files = list(light_dir.rglob(f'*{audio_file.stem}*.pkl'))
        
        if not light_files:
            print(f"  [{i}/{len(audio_files)}] ⚠️  No light file for: {audio_file.stem}")
            failed_files.append(audio_file.stem)
            continue
        
        light_file = light_files[0]
        
        try:
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.stem}")
            
            # Evaluate with tuned parameters
            metrics, viz_data = evaluator.evaluate_single_file(audio_file, light_file)
            
            # Collect all metrics
            result = {
                'file': audio_file.stem,
                # Structure metrics
                'ssm_correlation': metrics.get('ssm_correlation', 0),
                'novelty_correlation': metrics.get('novelty_correlation', 0),
                'boundary_f_score': metrics.get('boundary_f_score', 0),
                # Dynamic metrics
                'rms_correlation': metrics.get('rms_correlation', 0),
                'onset_correlation': metrics.get('onset_correlation', 0),
                # Beat alignment (main focus)
                'beat_peak_alignment': metrics.get('beat_peak_alignment', 0),
                'beat_valley_alignment': metrics.get('beat_valley_alignment', 0),
                # Variance metrics
                'intensity_variance': metrics.get('intensity_variance', 0),
                'color_variance': metrics.get('color_variance', 0),
                # Aggregate scores
                'structure_score': np.mean([
                    metrics.get('ssm_correlation', 0),
                    metrics.get('novelty_correlation', 0),
                    metrics.get('boundary_f_score', 0)
                ]),
                'rhythm_score': np.mean([
                    metrics.get('beat_peak_alignment', 0),
                    metrics.get('beat_valley_alignment', 0)
                ]),
                'dynamics_score': np.mean([
                    metrics.get('rms_correlation', 0),
                    metrics.get('onset_correlation', 0)
                ])
            }
            
            # Overall score
            result['overall_score'] = np.mean([
                result['structure_score'],
                result['rhythm_score'],
                result['dynamics_score']
            ])
            
            results.append(result)
            
            # Create plots if requested
            if create_plots and viz_data:
                create_file_plots(viz_data, audio_file.stem, output_path / 'plots')
            
        except Exception as e:
            print(f"  ❌ Error processing {audio_file.stem}: {e}")
            if verbose:
                traceback.print_exc()
            failed_files.append(audio_file.stem)
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Generate reports
    generate_evaluation_report(df, config, output_path / 'reports', failed_files)
    
    # Save raw results
    df.to_csv(output_path / 'reports' / 'metrics.csv', index=False)
    
    # Save as JSON for further processing
    with open(output_path / 'reports' / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary visualizations
    if create_plots and len(df) > 0:
        create_summary_plots(df, output_path / 'plots')
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Successfully processed: {len(results)} files")
    print(f"Failed: {len(failed_files)} files")
    
    if len(df) > 0:
        print("\n📊 Overall Statistics:")
        print(f"  Beat Peak Alignment:   {df['beat_peak_alignment'].mean():.3f} ± {df['beat_peak_alignment'].std():.3f}")
        print(f"  Beat Valley Alignment: {df['beat_valley_alignment'].mean():.3f} ± {df['beat_valley_alignment'].std():.3f}")
        print(f"  Rhythm Score:          {df['rhythm_score'].mean():.3f} ± {df['rhythm_score'].std():.3f}")
        print(f"  Structure Score:       {df['structure_score'].mean():.3f} ± {df['structure_score'].std():.3f}")
        print(f"  Dynamics Score:        {df['dynamics_score'].mean():.3f} ± {df['dynamics_score'].std():.3f}")
        print(f"  OVERALL SCORE:         {df['overall_score'].mean():.3f} ± {df['overall_score'].std():.3f}")
        
        print("\n🏆 Top 3 Files (by rhythm score):")
        top_files = df.nlargest(3, 'rhythm_score')[['file', 'rhythm_score']]
        for _, row in top_files.iterrows():
            print(f"  {row['file'][:40]:40s} : {row['rhythm_score']:.3f}")
    
    print(f"\n📁 Results saved to: {output_path}")
    
    return df


def create_file_plots(viz_data, file_name, plot_dir):
    """Create SSM and novelty plots for a single file."""
    
    # SSM plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Audio SSM
    im1 = axes[0].imshow(viz_data['audio_ssm'], cmap='hot', aspect='auto')
    axes[0].set_title(f"Audio SSM ({viz_data['audio_ssm'].shape[0]}×{viz_data['audio_ssm'].shape[0]})")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Light SSM
    im2 = axes[1].imshow(viz_data['light_ssm'], cmap='hot', aspect='auto')
    axes[1].set_title(f"Light SSM ({viz_data['light_ssm'].shape[0]}×{viz_data['light_ssm'].shape[0]})")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Difference
    diff = viz_data['audio_ssm'] - viz_data['light_ssm']
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[2].set_title('Difference')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.suptitle(f'SSM: {file_name}')
    plt.tight_layout()
    
    (plot_dir / 'ssm').mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'ssm' / f'{file_name}_ssm.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Novelty plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    time_axis = np.arange(len(viz_data['audio_novelty'])) * 10 / 30  # Downsampled
    
    axes[0].plot(time_axis, viz_data['audio_novelty'], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Audio Novelty')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, viz_data['light_novelty'], 'r-', linewidth=1.5)
    axes[1].set_ylabel('Light Novelty')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Novelty: {file_name}')
    plt.tight_layout()
    
    (plot_dir / 'novelty').mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'novelty' / f'{file_name}_novelty.png', dpi=100, bbox_inches='tight')
    plt.close()


def create_summary_plots(df, plot_dir):
    """Create summary visualizations."""
    
    # 1. Metric distribution boxplot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics = [
        ('beat_peak_alignment', 'Beat Peak Alignment'),
        ('beat_valley_alignment', 'Beat Valley Alignment'),
        ('rhythm_score', 'Rhythm Score'),
        ('ssm_correlation', 'SSM Correlation'),
        ('novelty_correlation', 'Novelty Correlation'),
        ('boundary_f_score', 'Boundary F-Score'),
        ('rms_correlation', 'RMS Correlation'),
        ('onset_correlation', 'Onset Correlation'),
        ('overall_score', 'Overall Score')
    ]
    
    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        bp = ax.boxplot(df[col].values, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        
        # Add individual points
        y = df[col].values
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=20, color='red')
        
        ax.set_title(title)
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])
    
    plt.suptitle('Metric Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['beat_peak_alignment', 'beat_valley_alignment', 'ssm_correlation', 
                 'novelty_correlation', 'rms_correlation', 'onset_correlation']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Metric Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / 'correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Rhythm score histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df['rhythm_score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['rhythm_score'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["rhythm_score"].mean():.3f}')
    ax.set_xlabel('Rhythm Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Rhythm Scores (Beat Alignment)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'rhythm_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(df, config, report_dir, failed_files):
    """Generate detailed markdown report."""
    
    report = []
    report.append("# Evaluation Report - Tuned Parameters\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Files Evaluated:** {len(df)}\n")
    report.append(f"**Files Failed:** {len(failed_files)}\n\n")
    
    # Configuration section
    report.append("## Configuration Used\n\n")
    report.append(f"- **Tuned on:** {config.get('tuned_on', 'N/A')}\n")
    report.append(f"- **Tuned with:** {config.get('tuned_with_file', 'N/A')}\n")
    report.append(f"- **Rhythmic Window:** {config.get('rhythmic_window', 'N/A')} frames\n")
    report.append(f"- **STD Threshold:** {config.get('rhythmic_threshold', 'N/A'):.3f}\n")
    report.append(f"- **Beat Sigma:** {config.get('beat_align_sigma', 'N/A'):.2f}\n")
    report.append(f"- **Peak Prominence:** {config.get('peak_prominence', 'N/A'):.3f}\n\n")
    
    # Summary statistics
    report.append("## Summary Statistics\n\n")
    report.append("| Metric | Mean ± Std | Min | Max |\n")
    report.append("|--------|------------|-----|-----|\n")
    
    metrics_to_report = [
        ('beat_peak_alignment', 'Beat Peak Alignment'),
        ('beat_valley_alignment', 'Beat Valley Alignment'),
        ('rhythm_score', 'Rhythm Score'),
        ('structure_score', 'Structure Score'),
        ('dynamics_score', 'Dynamics Score'),
        ('overall_score', 'Overall Score')
    ]
    
    for col, name in metrics_to_report:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            report.append(f"| {name} | {mean:.3f} ± {std:.3f} | {min_val:.3f} | {max_val:.3f} |\n")
    
    # Top performers
    report.append("\n## Top 5 Files by Rhythm Score\n\n")
    report.append("| Rank | File | Rhythm Score | Beat Peak | Beat Valley |\n")
    report.append("|------|------|--------------|-----------|-------------|\n")
    
    top_files = df.nlargest(5, 'rhythm_score')
    for i, (_, row) in enumerate(top_files.iterrows(), 1):
        file_short = row['file'][:30] + '...' if len(row['file']) > 30 else row['file']
        report.append(f"| {i} | {file_short} | {row['rhythm_score']:.3f} | "
                     f"{row['beat_peak_alignment']:.3f} | {row['beat_valley_alignment']:.3f} |\n")
    
    # Failed files
    if failed_files:
        report.append("\n## Failed Files\n\n")
        for file in failed_files:
            report.append(f"- {file}\n")
    
    # Save report
    report_path = report_dir / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"\n📄 Report saved to: {report_path}")


def main():
    """Main entry point with command line arguments."""
    
    parser = argparse.ArgumentParser(description='Evaluate dataset with tuned parameters')
    parser.add_argument('config_file', type=str, 
                       help='Path to JSON config file from Interactive Tuner')
    parser.add_argument('--data_dir', type=str, default='data/edge_intention',
                       help='Path to edge_intention directory')
    parser.add_argument('--output_dir', type=str, default='outputs_tuned',
                       help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation for faster processing')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_dataset_with_tuned_params(
        config_file=args.config_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        create_plots=not args.no_plots,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()