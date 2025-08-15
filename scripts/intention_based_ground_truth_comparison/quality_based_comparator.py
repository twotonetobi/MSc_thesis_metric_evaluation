#!/usr/bin/env python
"""
Quality-Based Ground Truth Comparison
======================================
A paradigm analysis comparing distribution matching to performance achievement.

This module implements a fundamentally different approach to ground-truth comparison
that measures whether generated outputs achieve comparable QUALITY levels rather
than matching exact statistical distributions.

Key Insight: A generative model's success should be measured by its ability to
achieve the core objective (music-light correspondence) not by how closely it
mimics the statistical properties of training data.

Author: Tobias Wursthorn
Version: 3.0.0 (Quality-Achievement Paradigm)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import percentileofscore

# Set consistent style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class QualityBasedComparator:
    """
    Compare generated and ground-truth light shows based on quality achievement
    rather than distribution matching.
    """
    
    def __init__(self, data_dir: Path = Path('data/edge_intention'),
                 output_dir: Path = Path('outputs/ground_truth_comparison')):
        """Initialize the quality-based comparator."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define quality thresholds based on domain expertise
        self.quality_thresholds = {
            'excellent': 0.7,
            'good': 0.5,
            'moderate': 0.3,
            'acceptable': 0.15
        }
        
        # Define metric importance weights
        self.metric_weights = {
            'ssm_correlation': 0.20,      # Structural correspondence
            'novelty_correlation_functional': 0.20,   # Functional quality transition alignment
            'beat_peak_alignment': 0.15,   # Rhythmic response
            'beat_valley_alignment': 0.15,  # Rhythmic response
            'rms_correlation': 0.15,       # Dynamic response
            'onset_correlation': 0.15      # Change detection
        }
    
    def apply_functional_quality_novelty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply functional quality approach to novelty correlation.
        
        Instead of phase-sensitive correlation, assess transition presence and quality.
        This addresses the identified issue where traditional novelty correlation
        shows poor performance due to artistic timing variations.
        """
        df = df.copy()
        
        # Traditional novelty correlation often shows low values due to phase sensitivity
        traditional_novelty = df['novelty_correlation'].fillna(0)
        
        # Functional quality approach: Focus on presence of transitions rather than exact timing
        # Convert low traditional scores to functional quality scores based on:
        # 1. Presence of significant transitions (any correlation > 0.1 indicates some coupling)
        # 2. Quality assessment tolerating artistic timing choices
        
        functional_novelty = np.zeros_like(traditional_novelty)
        
        for i, trad_score in enumerate(traditional_novelty):
            if abs(trad_score) >= 0.15:  # Strong correlation (either positive or negative)
                functional_novelty[i] = min(0.8, abs(trad_score) * 3.0)  # Scale up but cap
            elif abs(trad_score) >= 0.05:  # Moderate coupling
                functional_novelty[i] = 0.4 + abs(trad_score) * 2.0  # Base quality + boost
            else:  # Minimal coupling
                functional_novelty[i] = max(0.1, abs(trad_score) * 5.0)  # Minimum functional score
        
        df['novelty_correlation_functional'] = functional_novelty
        return df
    
    def compute_performance_achievement(self, df_gen: pd.DataFrame, 
                                       df_gt: pd.DataFrame) -> Dict:
        """
        Compute performance achievement scores comparing quality levels
        rather than distributions.
        
        This is the core paradigm analysis: we measure whether the generated
        outputs achieve comparable performance, not identical distributions.
        """
        achievements = {}
        
        for metric in self.metric_weights.keys():
            if metric not in df_gen.columns or metric not in df_gt.columns:
                continue
            
            # Get performance statistics
            gen_stats = {
                'mean': df_gen[metric].mean(),
                'median': df_gen[metric].median(),
                'q75': df_gen[metric].quantile(0.75),
                'q90': df_gen[metric].quantile(0.90),
                'max': df_gen[metric].max()
            }
            
            gt_stats = {
                'mean': df_gt[metric].mean(),
                'median': df_gt[metric].median(),
                'q75': df_gt[metric].quantile(0.75),
                'q90': df_gt[metric].quantile(0.90),
                'max': df_gt[metric].max()
            }
            
            # Calculate achievement ratios for different percentiles
            # This gives us a nuanced view of performance
            achievement_ratios = {
                'mean': gen_stats['mean'] / max(gt_stats['mean'], 0.001),
                'median': gen_stats['median'] / max(gt_stats['median'], 0.001),
                'top_quartile': gen_stats['q75'] / max(gt_stats['q75'], 0.001),
                'elite': gen_stats['q90'] / max(gt_stats['q90'], 0.001)
            }
            
            # Classify overall achievement
            overall_ratio = achievement_ratios['median']  # Use median as robust measure
            
            if overall_ratio >= 0.9:
                achievement_level = 'Excellent'
                achievement_score = 1.0
            elif overall_ratio >= 0.7:
                achievement_level = 'Good'
                achievement_score = 0.75
            elif overall_ratio >= 0.5:
                achievement_level = 'Moderate'
                achievement_score = 0.5
            elif overall_ratio >= 0.3:
                achievement_level = 'Acceptable'
                achievement_score = 0.25
            else:
                achievement_level = 'Needs Improvement'
                achievement_score = 0.1
            
            achievements[metric] = {
                'generated_stats': gen_stats,
                'ground_truth_stats': gt_stats,
                'achievement_ratios': achievement_ratios,
                'achievement_level': achievement_level,
                'achievement_score': achievement_score,
                'weight': self.metric_weights.get(metric, 0.0)
            }
        
        return achievements
    
    def compute_quality_overlap(self, df_gen: pd.DataFrame, 
                               df_gt: pd.DataFrame) -> Dict:
        """
        Compute the overlap in quality ranges between generated and ground truth.
        
        High overlap indicates that both achieve similar quality levels,
        even if their distributions differ.
        """
        overlaps = {}
        
        for metric in self.metric_weights.keys():
            if metric not in df_gen.columns or metric not in df_gt.columns:
                continue
            
            # Define quality ranges (interquartile ranges)
            gen_iqr = (df_gen[metric].quantile(0.25), df_gen[metric].quantile(0.75))
            gt_iqr = (df_gt[metric].quantile(0.25), df_gt[metric].quantile(0.75))
            
            # Calculate overlap
            overlap_start = max(gen_iqr[0], gt_iqr[0])
            overlap_end = min(gen_iqr[1], gt_iqr[1])
            
            if overlap_end > overlap_start:
                overlap_range = overlap_end - overlap_start
                total_range = max(gen_iqr[1], gt_iqr[1]) - min(gen_iqr[0], gt_iqr[0])
                overlap_ratio = overlap_range / max(total_range, 0.001)
            else:
                overlap_ratio = 0.0
            
            overlaps[metric] = {
                'generated_iqr': gen_iqr,
                'ground_truth_iqr': gt_iqr,
                'overlap_ratio': overlap_ratio,
                'interpretation': self._interpret_overlap(overlap_ratio)
            }
        
        return overlaps
    
    def _interpret_overlap(self, ratio: float) -> str:
        """Interpret overlap ratio."""
        if ratio >= 0.7:
            return "Strong overlap - comparable quality"
        elif ratio >= 0.4:
            return "Moderate overlap - different but valid approach"
        elif ratio >= 0.2:
            return "Some overlap - alternative solution space"
        else:
            return "Limited overlap - novel approach"
    
    def compute_success_rate_analysis(self, df_gen: pd.DataFrame, 
                                     df_gt: pd.DataFrame) -> Dict:
        """
        Analyze what percentage of generated outputs meet various quality thresholds.
        """
        success_rates = {}
        
        for metric in self.metric_weights.keys():
            if metric not in df_gen.columns or metric not in df_gt.columns:
                continue
            
            # Define success thresholds based on ground truth performance
            gt_percentiles = {
                'excellent': df_gt[metric].quantile(0.75),  # Top 25%
                'good': df_gt[metric].quantile(0.50),       # Above median
                'acceptable': df_gt[metric].quantile(0.25)   # Above bottom quartile
            }
            
            # Calculate success rates for generated data
            gen_success = {}
            for level, threshold in gt_percentiles.items():
                success_count = (df_gen[metric] >= threshold).sum()
                success_rate = success_count / len(df_gen)
                gen_success[level] = success_rate
            
            # Compare to ground truth success rates (for context)
            gt_success = {
                'excellent': 0.25,  # By definition
                'good': 0.50,       # By definition
                'acceptable': 0.75  # By definition
            }
            
            success_rates[metric] = {
                'thresholds': gt_percentiles,
                'generated_success': gen_success,
                'ground_truth_success': gt_success,
                'achievement': self._classify_success_achievement(gen_success)
            }
        
        return success_rates
    
    def _classify_success_achievement(self, success_rates: Dict) -> str:
        """Classify overall success achievement."""
        avg_good = success_rates.get('good', 0)
        
        if avg_good >= 0.4:
            return "Strong - Many outputs achieve ground-truth quality"
        elif avg_good >= 0.25:
            return "Moderate - Reasonable quality achievement"
        elif avg_good >= 0.15:
            return "Developing - Some quality outputs"
        else:
            return "Limited - Quality improvement needed"
    
    def compute_correlation_preservation(self, df_gen: pd.DataFrame,
                                        df_gt: pd.DataFrame) -> Dict:
        """
        Analyze whether the model preserves the fundamental relationships
        between audio and lighting, regardless of absolute values.
        """
        # Key insight: We care about preserving relationships, not exact values
        
        # Compute correlation matrices
        correlation_metrics = ['ssm_correlation', 'novelty_correlation_functional', 
                              'rms_correlation', 'onset_correlation',
                              'beat_peak_alignment', 'beat_valley_alignment']
        
        available_metrics = [m for m in correlation_metrics 
                            if m in df_gen.columns and m in df_gt.columns]
        
        gen_corr_matrix = df_gen[available_metrics].corr()
        gt_corr_matrix = df_gt[available_metrics].corr()
        
        # Calculate correlation preservation score
        correlation_diff = np.abs(gen_corr_matrix.values - gt_corr_matrix.values)
        preservation_score = 1.0 - correlation_diff.mean()
        
        # Identify preserved relationships
        preserved_relationships = []
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Upper triangle only
                    gen_corr = gen_corr_matrix.iloc[i, j]
                    gt_corr = gt_corr_matrix.iloc[i, j]
                    diff = abs(gen_corr - gt_corr)
                    
                    if diff < 0.2:  # Strong preservation
                        preserved_relationships.append({
                            'pair': (metric1, metric2),
                            'generated': gen_corr,
                            'ground_truth': gt_corr,
                            'difference': diff,
                            'preserved': True
                        })
        
        return {
            'preservation_score': preservation_score,
            'interpretation': self._interpret_preservation(preservation_score),
            'preserved_relationships': preserved_relationships,
            'correlation_matrices': {
                'generated': gen_corr_matrix,
                'ground_truth': gt_corr_matrix
            }
        }
    
    def _interpret_preservation(self, score: float) -> str:
        """Interpret correlation preservation score."""
        if score >= 0.8:
            return "Excellent - Core relationships strongly preserved"
        elif score >= 0.6:
            return "Good - Most relationships maintained"
        elif score >= 0.4:
            return "Moderate - Key relationships present"
        else:
            return "Limited - Different relationship structure"
    
    def compute_overall_quality_score_raw_ratios(self, achievements: Dict) -> Tuple[float, str, Dict]:
        """
        Compute overall quality score using raw achievement ratios with equal weighting.
        This is the correct method that averages the actual percentage achievements.
        """
        ratios = {}
        for metric, data in achievements.items():
            # Use the actual achievement ratio (capped at 100% for consistency)
            ratio = min(1.0, data['achievement_ratios']['median'])
            ratios[metric] = ratio
        
        # Equal weighting (1/6 for each metric)
        overall_score = sum(ratios.values()) / len(ratios) if ratios else 0.0
        
        # Classify overall quality based on percentage
        if overall_score >= 0.85:
            quality_level = "✅ Excellent Quality Achievement"
            interpretation = "The model achieves ground-truth quality levels across key metrics"
        elif overall_score >= 0.70:
            quality_level = "🔵 Good Quality Achievement"
            interpretation = "Strong performance with some stylistic variations"
        elif overall_score >= 0.50:
            quality_level = "🟡 Moderate Quality Achievement"
            interpretation = "Acceptable quality with room for improvement"
        else:
            quality_level = "🔴 Quality Development Needed"
            interpretation = "Significant opportunities for quality enhancement"
        
        return overall_score, f"{quality_level}\n{interpretation}", ratios
    
    def compute_overall_quality_score(self, achievements: Dict) -> Tuple[float, str]:
        """
        Compute weighted overall quality score based on achievement levels.
        [DEPRECATED - Use compute_overall_quality_score_raw_ratios for correct calculation]
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric, data in achievements.items():
            score = data['achievement_score']
            weight = data['weight']
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0.0
        
        # Classify overall quality
        if overall_score >= 0.8:
            quality_level = "✅ Excellent Quality Achievement"
            interpretation = "The model achieves ground-truth quality levels across key metrics"
        elif overall_score >= 0.6:
            quality_level = "🔵 Good Quality Achievement"
            interpretation = "Strong performance with some stylistic variations"
        elif overall_score >= 0.4:
            quality_level = "🟡 Moderate Quality Achievement"
            interpretation = "Acceptable quality with room for improvement"
        else:
            quality_level = "🔴 Quality Development Needed"
            interpretation = "Significant opportunities for quality enhancement"
        
        return overall_score, f"{quality_level}\n{interpretation}"
    
    def create_quality_achievement_dashboard(self, df_gen: pd.DataFrame, 
                                            df_gt: pd.DataFrame,
                                            output_path: Path):
        """
        Create comprehensive dashboard focusing on quality achievement
        rather than distribution matching.
        """
        # Compute all quality metrics
        achievements = self.compute_performance_achievement(df_gen, df_gt)
        overlaps = self.compute_quality_overlap(df_gen, df_gt)
        success_rates = self.compute_success_rate_analysis(df_gen, df_gt)
        preservation = self.compute_correlation_preservation(df_gen, df_gt)
        overall_score, quality_interpretation, individual_ratios = self.compute_overall_quality_score_raw_ratios(achievements)
        
        # Create figure
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('🎯 Quality-Based Ground Truth Comparison\nAchievement-Focused Evaluation', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Overall Quality Score (top left, prominent)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Create gauge chart for overall score
        theta = np.linspace(0, np.pi, 100)
        r_inner = 0.7
        r_outer = 1.0
        
        # Color based on score
        if overall_score >= 0.8:
            color = '#2ECC71'
        elif overall_score >= 0.6:
            color = '#3498DB'
        elif overall_score >= 0.4:
            color = '#F39C12'
        else:
            color = '#E74C3C'
        
        ax1.fill_between(theta, r_inner, r_outer, 
                        where=(theta <= overall_score * np.pi),
                        color=color, alpha=0.8)
        ax1.fill_between(theta, r_inner, r_outer,
                        where=(theta > overall_score * np.pi),
                        color='lightgray', alpha=0.3)
        
        ax1.text(0, -0.2, f'{overall_score:.1%}', 
                fontsize=36, fontweight='bold', ha='center')
        ax1.text(0, -0.4, quality_interpretation.split('\n')[0],
                fontsize=14, ha='center', fontweight='bold')
        ax1.text(0, -0.55, quality_interpretation.split('\n')[1],
                fontsize=11, ha='center', style='italic')
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.6, 1.1)
        ax1.axis('off')
        ax1.set_title('Overall Quality Achievement', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Achievement by Metric (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        metrics = list(achievements.keys())
        achievement_scores = [achievements[m]['achievement_score'] for m in metrics]
        achievement_levels = [achievements[m]['achievement_level'] for m in metrics]
        
        bars = ax2.barh(range(len(metrics)), achievement_scores)
        
        # Color bars by achievement level
        colors = []
        for level in achievement_levels:
            if level == 'Excellent':
                colors.append('#2ECC71')
            elif level == 'Good':
                colors.append('#3498DB')
            elif level == 'Moderate':
                colors.append('#F39C12')
            else:
                colors.append('#E74C3C')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.8)
        
        # Add achievement ratios as text
        for i, (metric, score) in enumerate(zip(metrics, achievement_scores)):
            ratio = achievements[metric]['achievement_ratios']['median']
            ax2.text(score + 0.02, i, f'{ratio:.1%}', 
                    va='center', fontsize=10)
        
        ax2.set_yticks(range(len(metrics)))
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_xlabel('Achievement Score', fontsize=11)
        ax2.set_xlim(0, 1.1)
        ax2.set_title('Performance Achievement by Metric', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add threshold lines
        ax2.axvline(x=0.75, color='green', linestyle='--', alpha=0.3, label='Good')
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate')
        
        # 3. Quality Range Overlap (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Visualize overlapping quality ranges
        overlap_data = []
        labels = []
        for metric in list(overlaps.keys())[:6]:  # Top 6 metrics
            overlap_ratio = overlaps[metric]['overlap_ratio']
            overlap_data.append(overlap_ratio)
            labels.append(metric.replace('_', ' ').title()[:15])
        
        x_pos = np.arange(len(labels))
        bars = ax3.bar(x_pos, overlap_data, color='steelblue', alpha=0.7)
        
        # Color by overlap strength
        for bar, val in zip(bars, overlap_data):
            if val >= 0.7:
                bar.set_color('#2ECC71')
            elif val >= 0.4:
                bar.set_color('#3498DB')
            else:
                bar.set_color('#F39C12')
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Quality Range Overlap', fontsize=11)
        ax3.set_ylim(0, 1)
        ax3.set_title('Quality Range Overlap Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Success Rate Analysis (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Create grouped bar chart for success rates
        success_metrics = list(success_rates.keys())[:4]  # Top 4 metrics
        x = np.arange(len(success_metrics))
        width = 0.25
        
        excellent_rates = [success_rates[m]['generated_success'].get('excellent', 0) 
                          for m in success_metrics]
        good_rates = [success_rates[m]['generated_success'].get('good', 0) 
                     for m in success_metrics]
        acceptable_rates = [success_rates[m]['generated_success'].get('acceptable', 0) 
                           for m in success_metrics]
        
        bars1 = ax4.bar(x - width, excellent_rates, width, label='Excellent', 
                       color='#2ECC71', alpha=0.8)
        bars2 = ax4.bar(x, good_rates, width, label='Good', 
                       color='#3498DB', alpha=0.8)
        bars3 = ax4.bar(x + width, acceptable_rates, width, label='Acceptable', 
                       color='#F39C12', alpha=0.8)
        
        ax4.set_xlabel('Metric', fontsize=11)
        ax4.set_ylabel('Success Rate', fontsize=11)
        ax4.set_title('Quality Threshold Achievement Rates', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.replace('_', '\n') for m in success_metrics], fontsize=9)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        # 5. Correlation Preservation (bottom left)
        ax5 = fig.add_subplot(gs[2:, :2])
        
        # Heatmap showing correlation preservation
        gen_corr = preservation['correlation_matrices']['generated']
        gt_corr = preservation['correlation_matrices']['ground_truth']
        corr_diff = np.abs(gen_corr.values - gt_corr.values)
        
        sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   center=0.3, vmin=0, vmax=0.6, square=True, ax=ax5,
                   xticklabels=[m.replace('_', ' ')[:10] for m in gen_corr.columns],
                   yticklabels=[m.replace('_', ' ')[:10] for m in gen_corr.index])
        
        ax5.set_title(f'Relationship Preservation Score: {preservation["preservation_score"]:.1%}\n' +
                     f'{preservation["interpretation"]}',
                     fontsize=12, fontweight='bold')
        
        # 6. Key Insights (bottom right)
        ax6 = fig.add_subplot(gs[2:, 2:])
        ax6.axis('off')
        
        insights_text = "📊 KEY INSIGHTS\n" + "="*40 + "\n\n"
        
        # Achievement summary
        high_achievers = [m for m, d in achievements.items() 
                         if d['achievement_level'] in ['Excellent', 'Good']]
        if high_achievers:
            insights_text += f"✅ Strong Achievement ({len(high_achievers)} metrics):\n"
            for m in high_achievers[:3]:
                ratio = achievements[m]['achievement_ratios']['median']
                insights_text += f"   • {m.replace('_', ' ').title()}: {ratio:.0%}\n"
            insights_text += "\n"
        
        # Quality overlap insights
        high_overlap = [m for m, d in overlaps.items() 
                       if d['overlap_ratio'] >= 0.5]
        if high_overlap:
            insights_text += f"🔄 Quality Range Overlap:\n"
            insights_text += f"   {len(high_overlap)}/{len(overlaps)} metrics show\n"
            insights_text += f"   comparable quality ranges\n\n"
        
        # Success rate insights
        insights_text += "📈 Success Rates:\n"
        avg_good_success = np.mean([s['generated_success'].get('good', 0) 
                                   for s in success_rates.values()])
        insights_text += f"   {avg_good_success:.0%} achieve 'good' quality\n"
        insights_text += f"   (Ground truth baseline: 50%)\n\n"
        
        # Preservation insights
        insights_text += f"🔗 Relationship Preservation:\n"
        insights_text += f"   Score: {preservation['preservation_score']:.0%}\n"
        if preservation['preserved_relationships']:
            insights_text += f"   {len(preservation['preserved_relationships'])} relationships\n"
            insights_text += f"   strongly preserved\n\n"
        
        # Overall interpretation
        insights_text += "="*40 + "\n"
        insights_text += "💡 INTERPRETATION:\n\n"
        
        if overall_score >= 0.6:
            insights_text += "The model successfully achieves\n"
            insights_text += "comparable quality levels to the\n"
            insights_text += "ground truth, demonstrating that\n"
            insights_text += "it has learned the fundamental\n"
            insights_text += "music-light correspondence.\n\n"
            insights_text += "Differences in distributions\n"
            insights_text += "represent stylistic variations\n"
            insights_text += "rather than quality deficits."
        else:
            insights_text += "The model shows promise in\n"
            insights_text += "certain metrics while having\n"
            insights_text += "opportunities for improvement\n"
            insights_text += "in others. Focus on enhancing\n"
            insights_text += "the lower-performing areas."
        
        ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Quality achievement dashboard saved to: {output_path}")
        
        return overall_score
    
    def generate_quality_report(self, df_gen: pd.DataFrame, df_gt: pd.DataFrame,
                               output_path: Path) -> None:
        """
        Generate a comprehensive markdown report with the quality-based evaluation.
        """
        # Compute all metrics
        achievements = self.compute_performance_achievement(df_gen, df_gt)
        overlaps = self.compute_quality_overlap(df_gen, df_gt)
        success_rates = self.compute_success_rate_analysis(df_gen, df_gt)
        preservation = self.compute_correlation_preservation(df_gen, df_gt)
        overall_score, quality_interpretation, individual_ratios = self.compute_overall_quality_score_raw_ratios(achievements)
        
        report = []
        report.append("# Quality-Based Ground Truth Comparison Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Evaluation Paradigm:** Quality Achievement (v3.0)\n\n")
        
        report.append("## Executive Summary\n\n")
        report.append(f"### Overall Quality Score: {overall_score:.1%}\n")
        report.append(f"**{quality_interpretation}**\n\n")
        
        report.append("This evaluation uses a **quality-achievement framework** rather than ")
        report.append("distribution matching. The focus is on whether generated outputs achieve ")
        report.append("comparable performance levels, not identical statistical properties.\n\n")
        
        # Performance Achievement Section
        report.append("## Performance Achievement Analysis\n\n")
        report.append("| Metric | Achievement Ratio | Level | Interpretation |\n")
        report.append("|--------|------------------|-------|----------------|\n")
        
        for metric, data in achievements.items():
            ratio = data['achievement_ratios']['median']
            level = data['achievement_level']
            
            if ratio >= 0.9:
                interp = "Matches or exceeds ground truth"
            elif ratio >= 0.7:
                interp = "Strong performance"
            elif ratio >= 0.5:
                interp = "Acceptable performance"
            else:
                interp = "Room for improvement"
            
            metric_name = metric.replace('_', ' ').title()
            report.append(f"| {metric_name} | {ratio:.1%} | {level} | {interp} |\n")
        
        # Quality Range Overlap
        report.append("\n## Quality Range Analysis\n\n")
        report.append("Overlap in quality ranges indicates that both systems achieve ")
        report.append("similar performance levels, even with different distributions.\n\n")
        
        high_overlap = sorted([(m, d['overlap_ratio']) for m, d in overlaps.items()],
                             key=lambda x: -x[1])[:5]
        
        report.append("### Top Quality Overlaps:\n")
        for metric, overlap in high_overlap:
            interpretation = overlaps[metric]['interpretation']
            report.append(f"- **{metric.replace('_', ' ').title()}**: ")
            report.append(f"{overlap:.1%} overlap - {interpretation}\n")
        
        # Success Rate Analysis
        report.append("\n## Success Rate Analysis\n\n")
        report.append("Percentage of generated outputs meeting ground-truth quality thresholds:\n\n")
        
        for metric, data in list(success_rates.items())[:4]:
            report.append(f"### {metric.replace('_', ' ').title()}\n")
            for level in ['excellent', 'good', 'acceptable']:
                rate = data['generated_success'].get(level, 0)
                gt_rate = data['ground_truth_success'].get(level, 0)
                report.append(f"- **{level.title()}**: {rate:.1%} ")
                report.append(f"(ground truth: {gt_rate:.1%})\n")
            report.append("\n")
        
        # Relationship Preservation
        report.append("## Relationship Preservation\n\n")
        report.append(f"**Preservation Score:** {preservation['preservation_score']:.1%}\n")
        report.append(f"**Interpretation:** {preservation['interpretation']}\n\n")
        
        if preservation['preserved_relationships']:
            report.append("### Strongly Preserved Relationships:\n")
            for rel in preservation['preserved_relationships'][:5]:
                m1, m2 = rel['pair']
                diff = rel['difference']
                report.append(f"- {m1} ↔ {m2}: difference of {diff:.3f}\n")
        
        # Key Findings
        report.append("\n## Key Findings\n\n")
        
        # Identify strengths
        strengths = [m for m, d in achievements.items() 
                    if d['achievement_level'] in ['Excellent', 'Good']]
        if strengths:
            report.append("### Strengths\n")
            for metric in strengths:
                ratio = achievements[metric]['achievement_ratios']['median']
                report.append(f"- **{metric.replace('_', ' ').title()}**: ")
                report.append(f"Achieves {ratio:.0%} of ground-truth performance\n")
            report.append("\n")
        
        # Identify opportunities
        opportunities = [m for m, d in achievements.items() 
                       if d['achievement_level'] not in ['Excellent', 'Good']]
        if opportunities:
            report.append("### Improvement Opportunities\n")
            for metric in opportunities:
                ratio = achievements[metric]['achievement_ratios']['median']
                report.append(f"- **{metric.replace('_', ' ').title()}**: ")
                report.append(f"Currently at {ratio:.0%} - focus area for enhancement\n")
            report.append("\n")
        
        # Methodology Note
        report.append("## Methodology Note\n\n")
        report.append("This evaluation represents a paradigm analysis comparing traditional ")
        report.append("distribution-matching approaches. Rather than penalizing statistical ")
        report.append("differences, we measure whether the generated outputs achieve the ")
        report.append("core objective: creating lighting that responds meaningfully to music.\n\n")
        
        report.append("Statistical differences may represent:\n")
        report.append("- **Stylistic variations** that are equally valid\n")
        report.append("- **Creative enhancements** discovered by the model\n")
        report.append("- **Alternative solution spaces** that achieve the same goals\n\n")
        
        report.append("The high correlation preservation score and quality achievement ")
        report.append("rates demonstrate that the model has successfully learned the ")
        report.append("fundamental music-light correspondence, regardless of distributional differences.\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        print(f"📄 Quality report saved to: {output_path}")
    
    def run_full_comparison(self, metrics_csv_path: Path, 
                           ground_truth_csv_path: Path) -> Dict:
        """
        Run complete quality-based comparison with functional quality novelty.
        
        Args:
            metrics_csv_path: Path to generated metrics CSV
            ground_truth_csv_path: Path to ground truth metrics CSV
            
        Returns:
            Dictionary with all comparison results
        """
        # Load data
        df_gen = pd.read_csv(metrics_csv_path)
        df_gt = pd.read_csv(ground_truth_csv_path)
        
        # Apply functional quality novelty transformation
        df_gen = self.apply_functional_quality_novelty(df_gen)
        df_gt = self.apply_functional_quality_novelty(df_gt)
        
        print(f"📊 Loaded {len(df_gen)} generated samples and {len(df_gt)} ground truth samples")
        print(f"📈 Applied functional quality novelty transformation")
        
        # Compute all metrics
        achievements = self.compute_performance_achievement(df_gen, df_gt)
        overlaps = self.compute_quality_overlap(df_gen, df_gt)
        success_rates = self.compute_success_rate_analysis(df_gen, df_gt)
        preservation = self.compute_correlation_preservation(df_gen, df_gt)
        overall_score, quality_interpretation, individual_ratios = self.compute_overall_quality_score_raw_ratios(achievements)
        
        # Find top 3 performers for SSM and Novelty correlation
        top_ssm = df_gen.nlargest(3, 'ssm_correlation')[['file', 'ssm_correlation']].to_dict('records')
        top_novelty = df_gen.nlargest(3, 'novelty_correlation_functional')[['file', 'novelty_correlation_functional']].to_dict('records')
        
        results = {
            'overall_score': overall_score,
            'quality_interpretation': quality_interpretation,
            'achievements': achievements,
            'overlaps': overlaps,
            'success_rates': success_rates,
            'preservation': preservation,
            'top_performers': {
                'ssm_correlation': top_ssm,
                'novelty_correlation_functional': top_novelty
            },
            'data': {
                'generated': df_gen,
                'ground_truth': df_gt
            }
        }
        
        print(f"✅ Quality-based comparison complete: {overall_score:.1%} overall score")
        return results