#!/usr/bin/env python
"""
Thesis Plot Generator
=====================
Generates individual, publication-ready plots from evaluation results.
Each plot is saved as a separate PNG for easy inclusion in thesis documents.

This module extracts and refines individual visualizations from the 
comprehensive dashboards, creating focused, single-metric plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 300  # High DPI for thesis
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ThesisPlotGenerator:
    """Generate individual plots for thesis inclusion."""
    
    def __init__(self, output_base: Path = Path('outputs/thesis_plots')):
        """Initialize the plot generator."""
        self.output_base = output_base
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.dirs = {
            'ground_truth': self.output_base / 'ground_truth',
            'hybrid': self.output_base / 'hybrid',
            'quality': self.output_base / 'quality',
            'combined': self.output_base / 'combined',
            'metrics': self.output_base / 'individual_metrics'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_individual_metric_boxplot(self, df: pd.DataFrame, metric: str, 
                                          metric_label: str, output_name: str,
                                          category: str = 'metrics') -> Path:
        """
        Generate a single boxplot for one metric.
        
        Args:
            df: DataFrame with 'source' column and metric values
            metric: Column name of the metric
            metric_label: Display label for the metric
            output_name: Name for the output file
            category: Category folder for organization
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create boxplot
        bp = sns.boxplot(
            x='source',
            y=metric,
            data=df,
            palette={'Generated': '#3498db', 'Ground Truth': '#95a5a6'},
            ax=ax
        )
        
        # Add individual points for transparency
        sns.stripplot(
            x='source',
            y=metric,
            data=df,
            color='red',
            alpha=0.2,
            size=2,
            ax=ax
        )
        
        # Calculate statistics
        gen_data = df[df['source'] == 'Generated'][metric]
        gt_data = df[df['source'] == 'Ground Truth'][metric]
        
        # Add statistical annotations
        stats_text = f"Generated: μ={gen_data.mean():.3f}, σ={gen_data.std():.3f}\n"
        stats_text += f"Ground Truth: μ={gt_data.mean():.3f}, σ={gt_data.std():.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        # Save
        output_path = self.dirs[category] / f'{output_name}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_individual_boxplots(self, df_combined: pd.DataFrame) -> Dict[str, Path]:
        """
        Generate individual boxplots for all metrics.
        
        Args:
            df_combined: Combined DataFrame with generated and ground truth data
            
        Returns:
            Dictionary mapping metric names to file paths
        """
        metrics = {
            'ssm_correlation': 'SSM Correlation',
            'novelty_correlation': 'Novelty Correlation',
            'boundary_f_score': 'Boundary F-Score',
            'rms_correlation': 'RMS-Brightness Correlation',
            'onset_correlation': 'Onset-Change Correlation',
            'beat_peak_alignment': 'Beat-Peak Alignment',
            'beat_valley_alignment': 'Beat-Valley Alignment',
            'intensity_variance': 'Intensity Variance',
            'color_variance': 'Color Variance'
        }
        
        paths = {}
        print("\n📊 Generating Individual Metric Boxplots")
        print("=" * 50)
        
        for metric, label in metrics.items():
            if metric not in df_combined.columns:
                print(f"  ⚠️  Skipping {metric} (not in data)")
                continue
            
            output_name = f'boxplot_{metric}'
            path = self.generate_individual_metric_boxplot(
                df_combined, metric, label, output_name
            )
            paths[metric] = path
            print(f"  ✓ {label}: {path.name}")
        
        return paths
    
    def generate_combined_boxplot_grid(self, df_combined: pd.DataFrame,
                                       title: str = "Metric Comparison") -> Path:
        """
        Generate a single figure with all boxplots in a grid.
        
        Args:
            df_combined: Combined DataFrame
            title: Overall title for the figure
            
        Returns:
            Path to saved plot
        """
        metrics = [
            ('ssm_correlation', 'SSM Correlation'),
            ('novelty_correlation', 'Novelty Correlation'),
            ('boundary_f_score', 'Boundary F-Score'),
            ('rms_correlation', 'RMS-Brightness'),
            ('onset_correlation', 'Onset-Change'),
            ('beat_peak_alignment', 'Beat-Peak'),
            ('beat_valley_alignment', 'Beat-Valley'),
            ('intensity_variance', 'Intensity Var.'),
            ('color_variance', 'Color Var.')
        ]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i]
            
            if metric not in df_combined.columns:
                ax.set_visible(False)
                continue
            
            # Create boxplot
            bp = sns.boxplot(
                x='source',
                y=metric,
                data=df_combined,
                palette={'Generated': '#3498db', 'Ground Truth': '#95a5a6'},
                ax=ax
            )
            
            # Add points
            sns.stripplot(
                x='source',
                y=metric,
                data=df_combined,
                color='red',
                alpha=0.15,
                size=1,
                ax=ax
            )
            
            # Calculate mean for annotation
            gen_mean = df_combined[df_combined['source'] == 'Generated'][metric].mean()
            gt_mean = df_combined[df_combined['source'] == 'Ground Truth'][metric].mean()
            
            # Add mean line
            ax.axhline(gen_mean, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(gt_mean, color='#95a5a6', linestyle='--', alpha=0.5, linewidth=1)
            
            # Formatting
            ax.set_xlabel('')
            ax.set_ylabel('Score', fontsize=9)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.set_xticklabels(['Gen.', 'GT'], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        output_path = self.dirs['combined'] / 'all_metrics_boxplot_grid.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_quality_achievement_bars(self, achievements: Dict) -> Path:
        """
        Generate individual bar chart for quality achievement scores.
        
        Args:
            achievements: Achievement dictionary from quality comparator
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(achievements.keys())
        scores = [achievements[m]['achievement_score'] for m in metrics]
        ratios = [achievements[m]['achievement_ratios']['median'] for m in metrics]
        
        # Create bars
        x = np.arange(len(metrics))
        bars = ax.bar(x, scores, color='steelblue', edgecolor='navy', linewidth=1.5)
        
        # Color by achievement level
        for bar, metric in zip(bars, metrics):
            level = achievements[metric]['achievement_level']
            if level == 'Excellent':
                bar.set_facecolor('#2ECC71')
            elif level == 'Good':
                bar.set_facecolor('#3498DB')
            elif level == 'Moderate':
                bar.set_facecolor('#F39C12')
            else:
                bar.set_facecolor('#E74C3C')
        
        # Add ratio labels on bars
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{ratio:.0%}', ha='center', va='bottom', fontsize=10)
        
        # Add threshold lines
        ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.3, 
                  label='Good (75%)', linewidth=1)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3,
                  label='Moderate (50%)', linewidth=1)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                          rotation=45, ha='right')
        ax.set_ylabel('Achievement Score', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title('Quality Achievement by Metric', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        output_path = self.dirs['quality'] / 'achievement_scores_bar.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_distribution_comparison(self, df_gen: pd.DataFrame, 
                                        df_gt: pd.DataFrame,
                                        metric: str) -> Path:
        """
        Generate distribution comparison plot for a single metric.
        
        Args:
            df_gen: Generated data
            df_gt: Ground truth data
            metric: Metric to visualize
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram comparison
        ax1.hist(df_gen[metric], bins=20, alpha=0.5, color='#3498db', 
                label='Generated', density=True, edgecolor='black')
        ax1.hist(df_gt[metric], bins=20, alpha=0.5, color='#95a5a6',
                label='Ground Truth', density=True, edgecolor='black')
        
        ax1.set_xlabel('Score', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'{metric.replace("_", " ").title()} Distribution',
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Violin plot comparison
        data_list = [df_gen[metric].values, df_gt[metric].values]
        parts = ax2.violinplot(data_list, positions=[1, 2],
                              showmeans=True, showmedians=True)
        
        # Color violins
        colors = ['#3498db', '#95a5a6']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['Generated', 'Ground Truth'])
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Distribution Shape Comparison',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{metric.replace("_", " ").title()} Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.dirs['metrics'] / f'distribution_{metric}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_hybrid_wave_distribution(self, wave_distribution: Dict) -> Path:
        """
        Generate clean bar chart for wave type distribution.
        
        Args:
            wave_distribution: Dictionary of wave type percentages
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Order wave types logically
        wave_order = ['still', 'sine', 'pwm_basic', 'pwm_extended', 
                     'odd_even', 'square', 'random']
        
        waves = [w for w in wave_order if w in wave_distribution]
        percentages = [wave_distribution[w] * 100 for w in waves]
        
        # Create bars with gradient colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(waves)))
        bars = ax.bar(range(len(waves)), percentages, color=colors,
                      edgecolor='black', linewidth=1.5)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        # Formatting
        ax.set_xticks(range(len(waves)))
        ax.set_xticklabels([w.replace('_', ' ').title() for w in waves])
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, max(percentages) * 1.15)
        ax.set_title('Wave Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at uniform distribution
        uniform = 100 / len(waves)
        ax.axhline(y=uniform, color='red', linestyle='--', alpha=0.5,
                  label=f'Uniform ({uniform:.1f}%)')
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = self.dirs['hybrid'] / 'wave_distribution_bar.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_summary_statistics_table(self, df: pd.DataFrame,
                                         source: str = "combined") -> Path:
        """
        Generate a clean statistics table as an image.
        
        Args:
            df: DataFrame with metrics
            source: Data source identifier
            
        Returns:
            Path to saved plot
        """
        metrics = ['ssm_correlation', 'novelty_correlation', 'boundary_f_score',
                  'rms_correlation', 'onset_correlation', 'beat_peak_alignment',
                  'beat_valley_alignment', 'intensity_variance', 'color_variance']
        
        # Calculate statistics
        stats_data = []
        for metric in metrics:
            if metric in df.columns:
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Mean': f"{df[metric].mean():.3f}",
                    'Std': f"{df[metric].std():.3f}",
                    'Min': f"{df[metric].min():.3f}",
                    'Max': f"{df[metric].max():.3f}",
                    'Median': f"{df[metric].median():.3f}"
                })
        
        # Create figure and table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = [[d['Metric'], d['Mean'], d['Std'], 
                      d['Min'], d['Max'], d['Median']] for d in stats_data]
        headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A4A4A')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(stats_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title(f'Summary Statistics - {source.title()}',
                 fontsize=14, fontweight='bold', pad=20)
        
        # Save
        output_path = self.dirs['metrics'] / f'statistics_table_{source}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_thesis_plots(self, df_gen: pd.DataFrame = None,
                                 df_gt: pd.DataFrame = None,
                                 df_combined: pd.DataFrame = None,
                                 achievements: Dict = None,
                                 wave_distribution: Dict = None) -> Dict[str, List[Path]]:
        """
        Generate all plots needed for thesis.
        
        Args:
            df_gen: Generated data
            df_gt: Ground truth data
            df_combined: Combined data with 'source' column
            achievements: Quality achievement data
            wave_distribution: Wave type distribution
            
        Returns:
            Dictionary categorizing all generated plot paths
        """
        all_plots = {
            'individual_metrics': [],
            'combined': [],
            'distributions': [],
            'quality': [],
            'hybrid': [],
            'tables': []
        }
        
        print("\n" + "🎨" * 30)
        print("\n📚 GENERATING THESIS PLOTS")
        print("\n" + "🎨" * 30)
        
        # Individual metric boxplots
        if df_combined is not None:
            print("\n1️⃣ Individual Metric Boxplots")
            paths = self.generate_all_individual_boxplots(df_combined)
            all_plots['individual_metrics'].extend(paths.values())
            
            print("\n2️⃣ Combined Boxplot Grid")
            grid_path = self.generate_combined_boxplot_grid(df_combined)
            all_plots['combined'].append(grid_path)
            print(f"  ✓ Combined grid: {grid_path.name}")
        
        # Distribution comparisons
        if df_gen is not None and df_gt is not None:
            print("\n3️⃣ Distribution Comparisons")
            key_metrics = ['ssm_correlation', 'beat_peak_alignment', 'onset_correlation']
            for metric in key_metrics:
                if metric in df_gen.columns and metric in df_gt.columns:
                    path = self.generate_distribution_comparison(df_gen, df_gt, metric)
                    all_plots['distributions'].append(path)
                    print(f"  ✓ {metric}: {path.name}")
        
        # Quality achievement
        if achievements:
            print("\n4️⃣ Quality Achievement")
            path = self.generate_quality_achievement_bars(achievements)
            all_plots['quality'].append(path)
            print(f"  ✓ Achievement bars: {path.name}")
        
        # Hybrid wave distribution
        if wave_distribution:
            print("\n5️⃣ Wave Type Distribution")
            path = self.generate_hybrid_wave_distribution(wave_distribution)
            all_plots['hybrid'].append(path)
            print(f"  ✓ Wave distribution: {path.name}")
        
        # Statistics tables
        if df_combined is not None:
            print("\n6️⃣ Statistics Tables")
            path = self.generate_summary_statistics_table(df_combined, "combined")
            all_plots['tables'].append(path)
            print(f"  ✓ Combined stats: {path.name}")
            
            if df_gen is not None:
                path_gen = self.generate_summary_statistics_table(df_gen, "generated")
                all_plots['tables'].append(path_gen)
                print(f"  ✓ Generated stats: {path_gen.name}")
            
            if df_gt is not None:
                path_gt = self.generate_summary_statistics_table(df_gt, "ground_truth")
                all_plots['tables'].append(path_gt)
                print(f"  ✓ Ground truth stats: {path_gt.name}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 THESIS PLOTS SUMMARY")
        print("=" * 50)
        total = sum(len(v) for v in all_plots.values())
        print(f"Total plots generated: {total}")
        for category, paths in all_plots.items():
            if paths:
                print(f"  • {category}: {len(paths)} plots")
        print(f"\nAll plots saved to: {self.output_base}")
        
        return all_plots


# Convenience function for quick generation
def generate_thesis_plots_from_csvs(gen_csv: str, gt_csv: str, 
                                   output_dir: str = None) -> Dict:
    """
    Quick function to generate all thesis plots from CSV files.
    
    Args:
        gen_csv: Path to generated metrics CSV
        gt_csv: Path to ground truth metrics CSV
        output_dir: Optional output directory
        
    Returns:
        Dictionary of generated plot paths
    """
    # Load data
    df_gen = pd.read_csv(gen_csv)
    df_gt = pd.read_csv(gt_csv)
    
    # Create combined dataframe
    df_gen['source'] = 'Generated'
    df_gt['source'] = 'Ground Truth'
    df_combined = pd.concat([df_gen, df_gt], ignore_index=True)
    
    # Initialize generator
    if output_dir:
        generator = ThesisPlotGenerator(Path(output_dir))
    else:
        generator = ThesisPlotGenerator()
    
    # Generate all plots
    return generator.generate_all_thesis_plots(
        df_gen=df_gen[df_gen['source'] == 'Generated'],
        df_gt=df_gt[df_gt['source'] == 'Ground Truth'],
        df_combined=df_combined
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate thesis plots')
    parser.add_argument('--gen_csv', type=str, help='Generated metrics CSV')
    parser.add_argument('--gt_csv', type=str, help='Ground truth metrics CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/thesis_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.gen_csv and args.gt_csv:
        plots = generate_thesis_plots_from_csvs(
            args.gen_csv, args.gt_csv, args.output_dir
        )
        print("\n✅ All thesis plots generated successfully!")
    else:
        print("Please provide both --gen_csv and --gt_csv paths")