#!/usr/bin/env python
"""
Enhanced Ground Truth Comparison Visualizer
Creates comprehensive visualizations and dashboard for ground-truth comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import wasserstein_distance, gaussian_kde

# Consistent style with hybrid approach
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class GroundTruthVisualizer:
    """Create enhanced visualizations for ground-truth comparison."""
    
    def __init__(self, output_dir: Path = Path('outputs/ground_truth_comparison')):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory containing comparison results
        """
        self.output_dir = output_dir
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data if available
        self.df_combined = self._load_combined_metrics()
        self.distances = self._load_distances()
    
    def _load_combined_metrics(self) -> Optional[pd.DataFrame]:
        """Load combined metrics CSV if available."""
        csv_path = self.output_dir / 'combined_metrics.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
    
    def _load_distances(self) -> Optional[Dict]:
        """Load distribution distances JSON if available."""
        json_path = self.output_dir / 'distribution_distances.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
    
    def create_density_comparison_plot(self) -> None:
        """Create density plots comparing distributions."""
        if self.df_combined is None:
            print("No data available for density plots")
            return
        
        metrics = [
            'ssm_correlation', 'novelty_correlation', 'boundary_f_score',
            'rms_correlation', 'onset_correlation', 'beat_peak_alignment'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric not in self.df_combined.columns:
                axes[i].set_visible(False)
                continue
            
            ax = axes[i]
            
            # Get data for each source
            gen_data = self.df_combined[self.df_combined['source'] == 'Generated'][metric].values
            gt_data = self.df_combined[self.df_combined['source'] == 'Ground Truth'][metric].values
            
            # Skip if insufficient data
            if len(gen_data) < 2 or len(gt_data) < 2:
                ax.set_visible(False)
                continue
            
            # Create density plots
            try:
                # Calculate KDE
                gen_kde = gaussian_kde(gen_data)
                gt_kde = gaussian_kde(gt_data)
                
                # Create x range
                x_min = min(gen_data.min(), gt_data.min())
                x_max = max(gen_data.max(), gt_data.max())
                x_range = np.linspace(x_min, x_max, 100)
                
                # Plot densities
                ax.fill_between(x_range, gen_kde(x_range), alpha=0.5, 
                               color='#3498db', label='Generated')
                ax.fill_between(x_range, gt_kde(x_range), alpha=0.5, 
                               color='#95a5a6', label='Ground Truth')
                
                # Add vertical lines for means
                ax.axvline(gen_data.mean(), color='#3498db', linestyle='--', 
                          linewidth=2, alpha=0.8)
                ax.axvline(gt_data.mean(), color='#95a5a6', linestyle='--', 
                          linewidth=2, alpha=0.8)
                
                # Add Wasserstein distance
                w_dist = wasserstein_distance(gen_data, gt_data)
                ax.text(0.05, 0.95, f'W-dist: {w_dist:.3f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            except Exception as e:
                print(f"Error creating density plot for {metric}: {e}")
                ax.set_visible(False)
                continue
            
            # Formatting
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution Density Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.plots_dir / 'density_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved density comparison to: {output_path}")
    
    def create_radar_comparison_plot(self) -> None:
        """Create radar chart comparing mean values."""
        if self.df_combined is None:
            print("No data available for radar plot")
            return
        
        metrics = [
            'ssm_correlation', 'novelty_correlation', 'boundary_f_score',
            'rms_correlation', 'onset_correlation', 'beat_peak_alignment',
            'beat_valley_alignment', 'intensity_variance'
        ]
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.df_combined.columns]
        
        if len(available_metrics) < 3:
            print("Insufficient metrics for radar plot")
            return
        
        # Calculate means
        gen_means = []
        gt_means = []
        
        for metric in available_metrics:
            gen_data = self.df_combined[self.df_combined['source'] == 'Generated'][metric]
            gt_data = self.df_combined[self.df_combined['source'] == 'Ground Truth'][metric]
            
            gen_means.append(gen_data.mean() if len(gen_data) > 0 else 0)
            gt_means.append(gt_data.mean() if len(gt_data) > 0 else 0)
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        
        # Close the plot
        gen_means += gen_means[:1]
        gt_means += gt_means[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, gen_means, 'o-', linewidth=2, color='#3498db', label='Generated')
        ax.fill(angles, gen_means, alpha=0.25, color='#3498db')
        
        ax.plot(angles, gt_means, 'o-', linewidth=2, color='#95a5a6', label='Ground Truth')
        ax.fill(angles, gt_means, alpha=0.25, color='#95a5a6')
        
        # Labels
        ax.set_xticks(angles[:-1])
        labels = [m.replace('_', '\n').title() for m in available_metrics]
        ax.set_xticklabels(labels, fontsize=10)
        
        # Scale
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        
        # Legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title('Mean Metric Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.plots_dir / 'radar_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved radar comparison to: {output_path}")
    
    def create_heatmap_comparison(self) -> None:
        """Create heatmap showing metric differences."""
        if self.distances is None:
            print("No distance data available for heatmap")
            return
        
        # Prepare data for heatmap
        metrics = list(self.distances.keys())
        measures = ['wasserstein', 'mean_diff', 'std_diff']
        
        # Create matrix
        data_matrix = []
        for measure in measures:
            row = []
            for metric in metrics:
                if measure in self.distances[metric]:
                    value = self.distances[metric][measure]
                    # Normalize differences for better visualization
                    if measure == 'wasserstein':
                        row.append(value)
                    else:
                        row.append(abs(value))
                else:
                    row.append(0)
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(measures)))
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
        ax.set_yticklabels(['Wasserstein\nDistance', 'Mean\nDifference', 'Std\nDifference'], 
                          fontsize=11)
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Distance/Difference', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(measures)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Metric Comparison Heatmap\n(Lower values = Better match)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.plots_dir / 'comparison_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comparison heatmap to: {output_path}")
    
    def create_comprehensive_dashboard(self) -> None:
        """Create a comprehensive dashboard combining all visualizations."""
        if self.df_combined is None or self.distances is None:
            print("Insufficient data for dashboard")
            return
        
        # Create figure with complex layout
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('🎯 Ground-Truth Comparison Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Summary statistics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        # Calculate overall fidelity score
        w_distances = [d['wasserstein'] for d in self.distances.values()]
        avg_w_dist = np.mean(w_distances)
        fidelity_score = max(0, 1 - (avg_w_dist / 0.2))
        
        # Determine quality
        if fidelity_score > 0.8:
            quality = "🟢 Excellent"
            quality_color = '#2ECC71'
        elif fidelity_score > 0.6:
            quality = "🔵 Good"
            quality_color = '#3498DB'
        elif fidelity_score > 0.4:
            quality = "🟡 Moderate"
            quality_color = '#F39C12'
        else:
            quality = "🔴 Poor"
            quality_color = '#E74C3C'
        
        summary_text = f"📊 SUMMARY STATISTICS\n{'='*30}\n\n"
        summary_text += f"Fidelity Score: {fidelity_score:.3f}\n"
        summary_text += f"Quality: {quality}\n\n"
        summary_text += f"Generated Files: {len(self.df_combined[self.df_combined['source'] == 'Generated'])}\n"
        summary_text += f"Ground Truth Files: {len(self.df_combined[self.df_combined['source'] == 'Ground Truth'])}\n\n"
        summary_text += f"Avg. Wasserstein: {avg_w_dist:.3f}\n"
        summary_text += f"Min. Wasserstein: {min(w_distances):.3f}\n"
        summary_text += f"Max. Wasserstein: {max(w_distances):.3f}"
        
        ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.2))
        
        # 2. Wasserstein distances bar chart (top middle-right, spanning 2 cols)
        ax2 = fig.add_subplot(gs[0, 1:3])
        
        metrics = list(self.distances.keys())
        w_dists = [self.distances[m]['wasserstein'] for m in metrics]
        
        bars = ax2.bar(range(len(metrics)), w_dists, color='steelblue', edgecolor='navy')
        
        # Color bars by quality
        for bar, dist in zip(bars, w_dists):
            if dist < 0.05:
                bar.set_facecolor('#2ECC71')
            elif dist < 0.1:
                bar.set_facecolor('#3498DB')
            elif dist < 0.15:
                bar.set_facecolor('#F39C12')
            else:
                bar.set_facecolor('#E74C3C')
        
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9, rotation=45)
        ax2.set_ylabel('Wasserstein Distance')
        ax2.set_title('Distribution Distances by Metric', fontsize=12, fontweight='bold')
        ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.3, label='Excellent')
        ax2.axhline(y=0.1, color='blue', linestyle='--', alpha=0.3, label='Good')
        ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.3, label='Moderate')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Best/Worst metrics (top right)
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        
        # Find best and worst
        best_metric = min(self.distances.keys(), key=lambda x: self.distances[x]['wasserstein'])
        worst_metric = max(self.distances.keys(), key=lambda x: self.distances[x]['wasserstein'])
        
        findings_text = f"🔍 KEY FINDINGS\n{'='*30}\n\n"
        findings_text += f"Best Match:\n{best_metric.replace('_', ' ').title()}\n"
        findings_text += f"W-dist: {self.distances[best_metric]['wasserstein']:.3f}\n\n"
        findings_text += f"Worst Match:\n{worst_metric.replace('_', ' ').title()}\n"
        findings_text += f"W-dist: {self.distances[worst_metric]['wasserstein']:.3f}\n\n"
        
        # Add specific insights
        if 'beat' in worst_metric:
            findings_text += "⚠️ Rhythmic alignment\nneeds improvement"
        elif 'variance' in worst_metric:
            findings_text += "⚠️ Dynamic range\nneeds adjustment"
        elif 'boundary' in worst_metric:
            findings_text += "⚠️ Structural transitions\nneed refinement"
        
        ax3.text(0.1, 0.9, findings_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 4-9. Metric comparison boxplots (middle section, 2x3 grid)
        key_metrics = ['overall_score', 'structure_score', 'rhythm_score',
                      'dynamics_score', 'beat_peak_alignment', 'rms_correlation']
        
        for i, metric in enumerate(key_metrics):
            row = 1 + (i // 3)
            col = i % 3
            
            if metric not in self.df_combined.columns:
                continue
            
            ax = fig.add_subplot(gs[row, col])
            
            # Create boxplot
            bp = sns.boxplot(
                x='source',
                y=metric,
                data=self.df_combined,
                ax=ax,
                palette={'Generated': '#3498db', 'Ground Truth': '#95a5a6'}
            )
            
            # Add strip plot for individual points
            sns.stripplot(
                x='source',
                y=metric,
                data=self.df_combined,
                ax=ax,
                color='red',
                alpha=0.2,
                size=2
            )
            
            # Calculate and display statistics
            gen_mean = self.df_combined[self.df_combined['source'] == 'Generated'][metric].mean()
            gt_mean = self.df_combined[self.df_combined['source'] == 'Ground Truth'][metric].mean()
            
            ax.text(0.5, 0.95, f'Gen: {gen_mean:.3f}\nGT: {gt_mean:.3f}',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Score', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 10. Density plot for selected metric (bottom left, spanning 2 cols)
        ax10 = fig.add_subplot(gs[3, :2])
        
        # Choose most interesting metric (highest wasserstein distance)
        interesting_metric = max(self.distances.keys(), 
                               key=lambda x: self.distances[x]['wasserstein'])
        
        if interesting_metric in self.df_combined.columns:
            gen_data = self.df_combined[self.df_combined['source'] == 'Generated'][interesting_metric].values
            gt_data = self.df_combined[self.df_combined['source'] == 'Ground Truth'][interesting_metric].values
            
            if len(gen_data) > 1 and len(gt_data) > 1:
                # Create violin plot
                parts = ax10.violinplot(
                    [gen_data, gt_data],
                    positions=[0, 1],
                    showmeans=True,
                    showmedians=True,
                    showextrema=True
                )
                
                # Color the violins
                colors = ['#3498db', '#95a5a6']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax10.set_xticks([0, 1])
                ax10.set_xticklabels(['Generated', 'Ground Truth'])
                ax10.set_ylabel('Score')
                ax10.set_title(f'Distribution Detail: {interesting_metric.replace("_", " ").title()}',
                              fontsize=11, fontweight='bold')
                ax10.grid(True, alpha=0.3)
        
        # 11. Statistical test results (bottom right)
        ax11 = fig.add_subplot(gs[3, 2:])
        
        # Create table of statistical tests
        test_data = []
        for metric in list(self.distances.keys())[:5]:  # Show top 5
            ks_p = self.distances[metric]['ks_pvalue']
            mw_p = self.distances[metric]['mw_pvalue']
            
            # Determine significance
            if ks_p > 0.05:
                ks_sig = '✓'
            else:
                ks_sig = '✗'
            
            if mw_p > 0.05:
                mw_sig = '✓'
            else:
                mw_sig = '✗'
            
            test_data.append([
                metric.replace('_', ' ').title()[:20],
                f'{ks_p:.3f} {ks_sig}',
                f'{mw_p:.3f} {mw_sig}'
            ])
        
        # Create table
        table = ax11.table(
            cellText=test_data,
            colLabels=['Metric', 'KS Test', 'MW Test'],
            cellLoc='center',
            loc='center',
            colWidths=[0.5, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(test_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4A4A4A')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax11.axis('off')
        ax11.set_title('Statistical Test Results (p-values)', 
                      fontsize=11, fontweight='bold', y=0.95)
        
        # Save dashboard
        output_path = self.plots_dir / 'comprehensive_dashboard.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comprehensive dashboard to: {output_path}")
    
    def generate_all_visualizations(self) -> None:
        """Generate all visualization types."""
        print("\n" + "="*60)
        print("GENERATING ENHANCED VISUALIZATIONS")
        print("="*60)
        
        if self.df_combined is None:
            print("❌ No combined metrics found. Run comparison first.")
            return
        
        print("Creating visualizations...")
        
        # Generate each visualization type
        self.create_density_comparison_plot()
        self.create_radar_comparison_plot()
        self.create_heatmap_comparison()
        self.create_comprehensive_dashboard()
        
        print("\n✅ All visualizations complete!")
        print(f"Check: {self.plots_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate enhanced visualizations for ground-truth comparison'
    )
    parser.add_argument('--output_dir', type=str,
                       default='outputs/ground_truth_comparison',
                       help='Directory containing comparison results')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = GroundTruthVisualizer(output_dir=Path(args.output_dir))
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()