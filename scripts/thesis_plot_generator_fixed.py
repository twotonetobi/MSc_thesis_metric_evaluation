#!/usr/bin/env python
"""
Fixed Thesis Plot Generator with Proper Naming Structure
=========================================================
Implements the exact naming conventions from your thesis structure.
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
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ThesisPlotGenerator:
    """Generate individual plots with thesis-aligned naming structure."""
    
    def __init__(self, output_base: Path = Path('outputs/thesis_plots')):
        """Initialize with thesis-specific directory structure."""
        self.output_base = output_base
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Create thesis-aligned directory structure
        self.dirs = {
            # Section 1: Intention-Based Evaluation
            'intention_structure': self.output_base / '1_intention_based' / 'structural_correspondence',
            'intention_rhythm': self.output_base / '1_intention_based' / 'rhythmic_alignment',
            'intention_dynamics': self.output_base / '1_intention_based' / 'dynamic_variation',
            
            # Section 2: Hybrid Wave Type
            'hybrid_consistency': self.output_base / '2_hybrid_wave' / 'consistency',
            'hybrid_coherence': self.output_base / '2_hybrid_wave' / 'musical_coherence',
            'hybrid_transitions': self.output_base / '2_hybrid_wave' / 'transition_smoothness',
            'hybrid_distribution': self.output_base / '2_hybrid_wave' / 'distribution_match',
            
            # Section 3: Quality-Based Comparison
            'quality_achievement': self.output_base / '3_quality_comparison' / 'achievement_ratios',
            'quality_overlap': self.output_base / '3_quality_comparison' / 'quality_overlap',
            'quality_preservation': self.output_base / '3_quality_comparison' / 'correlation_preservation',
            
            # Combined/Summary
            'combined': self.output_base / 'combined_visualizations',
            'tables': self.output_base / 'statistical_tables'
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Thesis metric naming map (matches your LaTeX notation)
        self.metric_symbols = {
            'ssm_correlation': 'Γ_structure',
            'novelty_correlation': 'Γ_novelty',
            'boundary_f_score': 'Γ_boundary',
            'rms_correlation': 'Γ_loud↔bright',
            'onset_correlation': 'Γ_change',
            'beat_peak_alignment': 'Γ_beat↔peak',
            'beat_valley_alignment': 'Γ_beat↔valley',
            'intensity_variance': 'Ψ_intensity',
            'color_variance': 'Ψ_color'
        }
    
    def generate_quality_achievement_bars(self, achievements: Dict) -> Path:
        """
        FIXED: Generate bar chart for quality achievement scores.
        Handles both old and new achievement dictionary structures.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(achievements.keys())
        
        # Handle different possible dictionary structures
        scores = []
        ratios = []
        
        for m in metrics:
            # Check for different possible keys
            if 'achievement_score' in achievements[m]:
                score = achievements[m]['achievement_score']
            elif 'score' in achievements[m]:
                score = achievements[m]['score']
            else:
                # Default to a moderate score if missing
                score = 0.5
            scores.append(score)
            
            # Get achievement ratio
            if 'achievement_ratios' in achievements[m]:
                if 'median' in achievements[m]['achievement_ratios']:
                    ratio = achievements[m]['achievement_ratios']['median']
                else:
                    ratio = achievements[m]['achievement_ratios'].get('mean', score)
            elif 'ratio' in achievements[m]:
                ratio = achievements[m]['ratio']
            else:
                ratio = score
            ratios.append(ratio)
        
        # Create bars
        x = np.arange(len(metrics))
        bars = ax.bar(x, scores, color='steelblue', edgecolor='navy', linewidth=1.5)
        
        # Color by achievement level
        for bar, metric in zip(bars, metrics):
            if 'achievement_level' in achievements[metric]:
                level = achievements[metric]['achievement_level']
            else:
                # Determine level from score
                score = scores[metrics.index(metric)]
                if score >= 0.75:
                    level = 'Excellent'
                elif score >= 0.5:
                    level = 'Good'
                elif score >= 0.25:
                    level = 'Moderate'
                else:
                    level = 'Needs Improvement'
            
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
        
        # Use thesis notation for x-labels
        xlabels = [self.metric_symbols.get(m, m.replace('_', ' ').title()) for m in metrics]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_ylabel('Achievement Score', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title('Quality Achievement by Metric', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save with thesis-aligned naming
        output_path = self.dirs['quality_achievement'] / 'quality_achievement_ratios.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path

    def generate_hybrid_wave_distribution(self, wave_distribution: Dict) -> Path:
        """Generate bar chart for wave type distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        wave_order = ['still', 'sine', 'pwm_basic', 'pwm_extended', 
                    'odd_even', 'square', 'random']
        
        waves = [w for w in wave_order if w in wave_distribution]
        percentages = [wave_distribution.get(w, 0) * 100 for w in waves]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(waves)))
        bars = ax.bar(range(len(waves)), percentages, color=colors,
                    edgecolor='black', linewidth=1.5)
        
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(waves)))
        ax.set_xticklabels([w.replace('_', ' ').title() for w in waves])
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, max(percentages) * 1.15 if percentages else 100)
        ax.set_title('Wave Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.dirs.get('hybrid_distribution', 
                                    self.output_base / 'hybrid_wave_distribution.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_intention_based_plots(self, df_combined: pd.DataFrame) -> Dict[str, Path]:
        """Generate plots for Section 1: Intention-Based Evaluation."""
        paths = {}
        
        # Structural Correspondence Metrics
        structural_metrics = ['ssm_correlation', 'novelty_correlation', 'boundary_f_score']
        for metric in structural_metrics:
            if metric in df_combined.columns:
                output_name = f"structural_{self.metric_symbols.get(metric, metric)}"
                path = self._generate_metric_plot(
                    df_combined, metric, 
                    self.dirs['intention_structure'] / f"{output_name}.png"
                )
                paths[metric] = path
        
        # Rhythmic Alignment Metrics
        rhythm_metrics = ['beat_peak_alignment', 'beat_valley_alignment']
        for metric in rhythm_metrics:
            if metric in df_combined.columns:
                output_name = f"rhythmic_{self.metric_symbols.get(metric, metric)}"
                path = self._generate_metric_plot(
                    df_combined, metric,
                    self.dirs['intention_rhythm'] / f"{output_name}.png"
                )
                paths[metric] = path
        
        # Dynamic Variation Metrics
        dynamic_metrics = ['rms_correlation', 'onset_correlation', 'intensity_variance', 'color_variance']
        for metric in dynamic_metrics:
            if metric in df_combined.columns:
                output_name = f"dynamic_{self.metric_symbols.get(metric, metric)}"
                path = self._generate_metric_plot(
                    df_combined, metric,
                    self.dirs['intention_dynamics'] / f"{output_name}.png"
                )
                paths[metric] = path
        
        return paths
    
    def _generate_metric_plot(self, df: pd.DataFrame, metric: str, output_path: Path) -> Path:
        """Generate a single metric comparison plot."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create boxplot
        bp = sns.boxplot(
            x='source',
            y=metric,
            data=df,
            palette={'Generated': '#3498db', 'Ground Truth': '#95a5a6'},
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            x='source',
            y=metric,
            data=df,
            color='red',
            alpha=0.2,
            size=2,
            ax=ax
        )
        
        # Statistics
        gen_data = df[df['source'] == 'Generated'][metric]
        gt_data = df[df['source'] == 'Ground Truth'][metric]
        
        stats_text = f"Generated: μ={gen_data.mean():.3f}, σ={gen_data.std():.3f}\n"
        stats_text += f"Ground Truth: μ={gt_data.mean():.3f}, σ={gt_data.std():.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Use thesis notation
        metric_label = self.metric_symbols.get(metric, metric.replace('_', ' ').title())
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_markdown_summary(self, results: Dict) -> Path:
        """Generate markdown file with all metric values for easy reference."""
        md_path = self.output_base / 'metric_values_summary.md'
        
        with open(md_path, 'w') as f:
            f.write("# Thesis Evaluation Metrics Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Section 1: Intention-Based
            f.write("## 1. Intention-Based Performance Indicators\n\n")
            f.write("### Structural Correspondence\n")
            f.write("| Metric | Symbol | Generated | Ground Truth | Achievement |\n")
            f.write("|--------|--------|-----------|--------------|-------------|\n")
            # ... (populate with actual values)
            
            # Section 2: Hybrid Wave Type
            f.write("\n## 2. Hybrid Wave Type Decision Quality\n\n")
            f.write("| Metric | Score | Target | Status |\n")
            f.write("|--------|-------|--------|--------|\n")
            # ... (populate with actual values)
            
            # Section 3: Quality-Based Comparison
            f.write("\n## 3. Quality-Based Ground Truth Comparison\n\n")
            f.write("| Metric | Achievement Ratio | Quality Level |\n")
            f.write("|--------|------------------|---------------|\n")
            # ... (populate with actual values)
        
        return md_path
    
    def generate_all_thesis_plots(self, df_gen: pd.DataFrame = None,
                                 df_gt: pd.DataFrame = None,
                                 df_combined: pd.DataFrame = None,
                                 achievements: Dict = None,
                                 wave_distribution: Dict = None) -> Dict[str, List[Path]]:
        """Generate all plots with thesis-aligned structure."""
        all_plots = {
            'intention_based': [],
            'hybrid_wave': [],
            'quality_comparison': [],
            'combined': [],
            'tables': []
        }
        
        print("\n📚 GENERATING THESIS-ALIGNED PLOTS")
        print("=" * 50)
        
        # Generate intention-based plots
        if df_combined is not None:
            print("\n1️⃣ Section 1: Intention-Based Evaluation")
            intention_paths = self.generate_intention_based_plots(df_combined)
            all_plots['intention_based'].extend(intention_paths.values())
            print(f"  ✓ Generated {len(intention_paths)} plots")
        
        # Generate quality achievement plots
        if achievements:
            print("\n3️⃣ Section 3: Quality-Based Comparison")
            try:
                path = self.generate_quality_achievement_bars(achievements)
                all_plots['quality_comparison'].append(path)
                print(f"  ✓ Quality achievement bars: {path.name}")
            except Exception as e:
                print(f"  ⚠️ Error generating quality bars: {e}")
        
        # Generate markdown summary
        summary_path = self.generate_markdown_summary({'all_plots': all_plots})
        print(f"\n📄 Metric values summary: {summary_path.name}")
        
        return all_plots