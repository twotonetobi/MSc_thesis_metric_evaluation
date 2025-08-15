#!/usr/bin/env python
"""
Compare Generated Light Shows to Ground Truth
Main orchestrator for ground-truth comparison evaluation
"""

#!/usr/bin/env python
"""
Compare Generated Light Shows to Ground Truth
Main orchestrator for ground-truth comparison evaluation
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, mannwhitneyu
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent style with hybrid approach
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class GroundTruthComparator:
    """Compare generated light shows against ground-truth training data."""
    
    def __init__(self, data_dir: Path = Path('data/edge_intention'),
                 output_dir: Path = Path('outputs/ground_truth_comparison')):
        """
        Initialize comparator.
        
        Args:
            data_dir: Base directory containing audio/light and ground truth subdirs
            output_dir: Directory for output results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        self.generated_audio_dir = data_dir / 'audio'
        self.generated_light_dir = data_dir / 'light'
        self.ground_truth_audio_dir = data_dir / 'audio_ground_truth'
        self.ground_truth_light_dir = data_dir / 'light_ground_truth'
        
    def run_evaluation_pipeline(self, audio_dir: Path, light_dir: Path, 
                               output_csv: Path, label: str) -> pd.DataFrame:
        """
        Run the evaluation pipeline on a dataset.
        
        Args:
            audio_dir: Audio directory
            light_dir: Light directory
            output_csv: Output CSV path
            label: Label for this dataset ('generated' or 'ground_truth')
            
        Returns:
            DataFrame with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {label.upper()} DATASET")
        print(f"{'='*60}")
        
        # Check if directories exist
        if not audio_dir.exists():
            print(f"❌ Audio directory not found: {audio_dir}")
            return pd.DataFrame()
        
        if not light_dir.exists():
            print(f"❌ Light directory not found: {light_dir}")
            return pd.DataFrame()
        
        # Run the evaluation pipeline script
        cmd = [
            sys.executable,
            'scripts/run_evaluation_pipeline.py',
            '--audio_dir', str(audio_dir),
            '--light_dir', str(light_dir),
            '--output_csv', str(output_csv)
        ]
        
        # Add config if tuned parameters exist
        config_path = Path('data/beat_configs/evaluator_config_20250808_185625.json')
        if config_path.exists():
            cmd.extend(['--config', str(config_path)])
            print(f"Using tuned config: {config_path}")
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Load the resulting CSV
            if output_csv.exists():
                df = pd.read_csv(output_csv)
                return df
            else:
                print(f"❌ Output CSV not created: {output_csv}")
                return pd.DataFrame()
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running evaluation pipeline: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return pd.DataFrame()
        except FileNotFoundError:
            print("❌ run_evaluation_pipeline.py script not found")
            print("Please ensure the script is in the scripts/ directory")
            return pd.DataFrame()
    
    def calculate_distribution_distances(self, df_gen: pd.DataFrame, 
                                        df_gt: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate distribution distances between generated and ground truth.
        
        Args:
            df_gen: Generated dataset metrics
            df_gt: Ground truth dataset metrics
            
        Returns:
            Dictionary with distance metrics for each feature
        """
        distances = {}
        
        metrics = [
            'ssm_correlation', 'novelty_correlation', 'boundary_f_score',
            'rms_correlation', 'onset_correlation',
            'beat_peak_alignment', 'beat_valley_alignment',
            'intensity_variance', 'color_variance'
        ]
        
        for metric in metrics:
            if metric in df_gen.columns and metric in df_gt.columns:
                gen_values = df_gen[metric].values
                gt_values = df_gt[metric].values
                
                # Skip if either is empty
                if len(gen_values) == 0 or len(gt_values) == 0:
                    continue
                
                # Calculate various distance metrics
                distances[metric] = {
                    # Wasserstein distance (Earth Mover's Distance)
                    'wasserstein': float(wasserstein_distance(gen_values, gt_values)),
                    
                    # Kolmogorov-Smirnov test
                    'ks_statistic': float(ks_2samp(gen_values, gt_values).statistic),
                    'ks_pvalue': float(ks_2samp(gen_values, gt_values).pvalue),
                    
                    # Mann-Whitney U test
                    'mw_statistic': float(mannwhitneyu(gen_values, gt_values).statistic),
                    'mw_pvalue': float(mannwhitneyu(gen_values, gt_values).pvalue),
                    
                    # Simple statistics comparison
                    'mean_diff': float(np.mean(gen_values) - np.mean(gt_values)),
                    'std_diff': float(np.std(gen_values) - np.std(gt_values)),
                    
                    # Generated statistics
                    'gen_mean': float(np.mean(gen_values)),
                    'gen_std': float(np.std(gen_values)),
                    'gen_median': float(np.median(gen_values)),
                    
                    # Ground truth statistics
                    'gt_mean': float(np.mean(gt_values)),
                    'gt_std': float(np.std(gt_values)),
                    'gt_median': float(np.median(gt_values))
                }
        
        return distances
    
    def create_comparison_boxplot(self, df_combined: pd.DataFrame, 
                                  output_path: Path) -> None:
        """
        Create comparison boxplots matching the hybrid system style.
        
        Args:
            df_combined: Combined DataFrame with 'source' column
            output_path: Path to save the plot
        """
        metric_columns = [
            'ssm_correlation', 'novelty_correlation', 'boundary_f_score',
            'rms_correlation', 'onset_correlation', 'beat_peak_alignment',
            'beat_valley_alignment', 'intensity_variance', 'color_variance'
        ]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, metric in enumerate(metric_columns):
            if metric not in df_combined.columns:
                axes[i].set_visible(False)
                continue
            
            # Create boxplot
            bp = sns.boxplot(
                x='source', 
                y=metric, 
                data=df_combined, 
                ax=axes[i],
                palette={'Generated': '#3498db', 'Ground Truth': '#95a5a6'}
            )
            
            # Add individual points
            sns.stripplot(
                x='source',
                y=metric,
                data=df_combined,
                ax=axes[i],
                color='red',
                alpha=0.3,
                size=3
            )
            
            # Calculate statistics for annotation
            gen_data = df_combined[df_combined['source'] == 'Generated'][metric]
            gt_data = df_combined[df_combined['source'] == 'Ground Truth'][metric]
            
            if len(gen_data) > 0 and len(gt_data) > 0:
                wasserstein = wasserstein_distance(gen_data.values, gt_data.values)
                axes[i].text(0.5, 0.95, f'W-dist: {wasserstein:.3f}',
                           transform=axes[i].transAxes,
                           ha='center', va='top',
                           fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Formatting
            axes[i].set_title(metric.replace('_', ' ').title(), fontsize=14)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Ground-Truth vs Generated Metric Distributions', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comparison boxplot to: {output_path}")
    
    def create_distribution_violin_plot(self, df_combined: pd.DataFrame,
                                       output_path: Path) -> None:
        """
        Create violin plots for distribution comparison.
        
        Args:
            df_combined: Combined DataFrame
            output_path: Output path for plot
        """
        metrics = ['overall_score', 'structure_score', 'rhythm_score', 'dynamics_score']
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        for i, metric in enumerate(metrics):
            if metric not in df_combined.columns:
                axes[i].set_visible(False)
                continue
            
            # Create violin plot
            parts = axes[i].violinplot(
                [df_combined[df_combined['source'] == 'Generated'][metric].values,
                 df_combined[df_combined['source'] == 'Ground Truth'][metric].values],
                positions=[0, 1],
                showmeans=True,
                showmedians=True
            )
            
            # Color the violins
            colors = ['#3498db', '#95a5a6']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Labels and formatting
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Generated', 'Ground Truth'])
            axes[i].set_ylabel('Score')
            axes[i].set_title(metric.replace('_', ' ').title(), fontsize=14)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([0, 1])
        
        plt.suptitle('Score Distribution Comparison (Violin Plots)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved violin plot to: {output_path}")
    
    def generate_report(self, df_gen: pd.DataFrame, df_gt: pd.DataFrame,
                       distances: Dict, output_path: Path) -> None:
        """
        Generate markdown report with results.
        
        Args:
            df_gen: Generated dataset results
            df_gt: Ground truth dataset results
            distances: Distribution distance metrics
            output_path: Path for markdown report
        """
        report = []
        report.append("# Ground-Truth Fidelity Comparison Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset sizes
        report.append("## Dataset Information\n\n")
        report.append(f"- **Generated Dataset:** {len(df_gen)} files\n")
        report.append(f"- **Ground Truth Dataset:** {len(df_gt)} files\n\n")
        
        # Main comparison table
        report.append("## Metric Comparison\n\n")
        report.append("| Metric | Ground Truth (Mean ± Std) | Generated (Mean ± Std) | Wasserstein Distance | Quality |\n")
        report.append("|--------|---------------------------|------------------------|---------------------|----------|\n")
        
        for metric, dist_info in distances.items():
            gt_mean = dist_info['gt_mean']
            gt_std = dist_info['gt_std']
            gen_mean = dist_info['gen_mean']
            gen_std = dist_info['gen_std']
            w_dist = dist_info['wasserstein']
            
            # Quality assessment based on Wasserstein distance
            if w_dist < 0.05:
                quality = "🟢 Excellent"
            elif w_dist < 0.1:
                quality = "🔵 Good"
            elif w_dist < 0.15:
                quality = "🟡 Moderate"
            else:
                quality = "🔴 Poor"
            
            metric_name = metric.replace('_', ' ').title()
            report.append(f"| {metric_name} | {gt_mean:.3f} ± {gt_std:.3f} | "
                         f"{gen_mean:.3f} ± {gen_std:.3f} | {w_dist:.3f} | {quality} |\n")
        
        report.append("\n*Lower Wasserstein Distance indicates higher similarity between distributions.*\n\n")
        
        # Statistical tests section
        report.append("## Statistical Tests\n\n")
        report.append("### Kolmogorov-Smirnov Test\n")
        report.append("Tests if two samples come from the same distribution.\n\n")
        report.append("| Metric | KS Statistic | p-value | Interpretation |\n")
        report.append("|--------|--------------|---------|----------------|\n")
        
        for metric, dist_info in distances.items():
            ks_stat = dist_info['ks_statistic']
            ks_p = dist_info['ks_pvalue']
            
            if ks_p > 0.05:
                interp = "Similar distributions ✓"
            else:
                interp = "Different distributions"
            
            metric_name = metric.replace('_', ' ').title()
            report.append(f"| {metric_name} | {ks_stat:.3f} | {ks_p:.3f} | {interp} |\n")
        
        # Key findings
        report.append("\n## Key Findings\n\n")
        
        # Find best and worst matching metrics
        w_distances = {m: d['wasserstein'] for m, d in distances.items()}
        best_metric = min(w_distances, key=w_distances.get)
        worst_metric = max(w_distances, key=w_distances.get)
        
        report.append(f"- **Best Match:** {best_metric.replace('_', ' ').title()} "
                     f"(W-distance: {w_distances[best_metric]:.3f})\n")
        report.append(f"- **Worst Match:** {worst_metric.replace('_', ' ').title()} "
                     f"(W-distance: {w_distances[worst_metric]:.3f})\n")
        
        # Overall fidelity score (average of normalized W-distances)
        avg_w_dist = np.mean(list(w_distances.values()))
        fidelity_score = max(0, 1 - (avg_w_dist / 0.2))  # Normalize to 0-1
        
        report.append(f"\n### Overall Fidelity Score: {fidelity_score:.3f}\n")
        
        if fidelity_score > 0.8:
            report.append("✅ **Excellent**: The model generates light shows with very similar "
                         "structural properties to the training data.\n")
        elif fidelity_score > 0.6:
            report.append("🔵 **Good**: The model captures most structural properties "
                         "of the training data well.\n")
        elif fidelity_score > 0.4:
            report.append("🟡 **Moderate**: The model captures some structural properties "
                         "but shows notable differences from training data.\n")
        else:
            report.append("🔴 **Poor**: The model's outputs differ significantly "
                         "from the training data's structural properties.\n")
        
        # Recommendations
        report.append("\n## Recommendations\n\n")
        
        if worst_metric == 'beat_peak_alignment' or worst_metric == 'beat_valley_alignment':
            report.append("- Consider fine-tuning the model's rhythmic response to better "
                         "align with musical beats\n")
        
        if worst_metric == 'intensity_variance' or worst_metric == 'color_variance':
            report.append("- The model may need adjustment to match the dynamic range "
                         "of human-designed shows\n")
        
        if worst_metric in ['ssm_correlation', 'novelty_correlation', 'boundary_f_score']:
            report.append("- The model's structural understanding could be improved "
                         "to better match segment transitions\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        print(f"📄 Report saved to: {output_path}")
    
    def run_comparison(self, max_files: Optional[int] = None) -> Dict:
        """
        Run the complete comparison pipeline.
        
        Args:
            max_files: Maximum files to process (for testing)
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*60)
        print("GROUND-TRUTH COMPARISON PIPELINE")
        print("="*60)
        
        # Step 1: Evaluate generated dataset
        gen_csv = self.output_dir / 'generated_metrics.csv'
        df_gen = self.run_evaluation_pipeline(
            self.generated_audio_dir,
            self.generated_light_dir,
            gen_csv,
            'generated'
        )
        
        # Step 2: Evaluate ground truth dataset
        gt_csv = self.output_dir / 'ground_truth_metrics.csv'
        df_gt = self.run_evaluation_pipeline(
            self.ground_truth_audio_dir,
            self.ground_truth_light_dir,
            gt_csv,
            'ground_truth'
        )
        
        if len(df_gen) == 0 or len(df_gt) == 0:
            print("❌ Cannot proceed with comparison - missing data")
            return {}
        
        # Step 3: Combine datasets
        df_gen['source'] = 'Generated'
        df_gt['source'] = 'Ground Truth'
        df_combined = pd.concat([df_gen, df_gt], ignore_index=True)
        
        # Save combined dataset
        combined_csv = self.output_dir / 'combined_metrics.csv'
        df_combined.to_csv(combined_csv, index=False)
        print(f"📊 Combined metrics saved to: {combined_csv}")
        
        # Step 4: Calculate distribution distances
        distances = self.calculate_distribution_distances(df_gen, df_gt)
        
        # Save distances as JSON
        distances_json = self.output_dir / 'distribution_distances.json'
        with open(distances_json, 'w') as f:
            json.dump(distances, f, indent=2)
        print(f"📊 Distribution distances saved to: {distances_json}")
        
        # Step 5: Create visualizations
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        self.create_comparison_boxplot(df_combined, plots_dir / 'comparison_boxplot.png')
        self.create_distribution_violin_plot(df_combined, plots_dir / 'distribution_violin.png')
        
        # Step 6: Generate report
        report_path = self.output_dir / 'comparison_report.md'
        self.generate_report(df_gen, df_gt, distances, report_path)
        
        return {
            'generated': df_gen,
            'ground_truth': df_gt,
            'combined': df_combined,
            'distances': distances
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare generated light shows to ground truth'
    )
    parser.add_argument('--data_dir', type=str, 
                       default='data/edge_intention',
                       help='Base data directory')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/ground_truth_comparison',
                       help='Output directory for results')
    parser.add_argument('--max_files', type=int,
                       help='Maximum files to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = GroundTruthComparator(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir)
    )
    
    # Run comparison
    results = comparator.run_comparison(max_files=args.max_files)
    
    if results:
        print("\n" + "="*60)
        print("✅ COMPARISON COMPLETE")
        print("="*60)
        print(f"Results saved to: {comparator.output_dir}")
        print("\nCheck the following files:")
        print("- comparison_report.md: Full analysis report")
        print("- plots/comparison_boxplot.png: Visual comparison")
        print("- plots/distribution_violin.png: Distribution shapes")
        print("- distribution_distances.json: Statistical metrics")


if __name__ == '__main__':
    main()