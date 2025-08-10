#!/usr/bin/env python
"""
Run Quality-Based Ground Truth Comparison
This implements the new evaluation paradigm focusing on quality achievement
rather than distribution matching.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Import both the pipeline and the new comparator
sys.path.append('scripts')
from run_evaluation_pipeline import EvaluationPipeline
from quality_based_comparator import QualityBasedComparator


def main():
    parser = argparse.ArgumentParser(
        description='Quality-based ground truth comparison'
    )
    parser.add_argument('--data_dir', type=str,
                       default='data/edge_intention',
                       help='Base data directory')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/quality_comparison',
                       help='Output directory for quality-based results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("QUALITY-BASED GROUND TRUTH COMPARISON")
    print("="*60)
    print("Paradigm: Quality Achievement (not distribution matching)")
    print("-"*60)
    
    # Step 1: Run evaluations using existing pipeline
    pipeline = EvaluationPipeline(verbose=True)
    
    # Evaluate generated
    print("\n📊 Evaluating generated dataset...")
    gen_csv = output_dir / 'generated_metrics.csv'
    df_gen = pipeline.run_evaluation(
        audio_dir=data_dir / 'audio',
        light_dir=data_dir / 'light',
        output_csv=gen_csv
    )
    
    # Evaluate ground truth
    print("\n📊 Evaluating ground truth dataset...")
    gt_csv = output_dir / 'ground_truth_metrics.csv'
    df_gt = pipeline.run_evaluation(
        audio_dir=data_dir / 'audio_ground_truth',
        light_dir=data_dir / 'light_ground_truth',
        output_csv=gt_csv
    )
    
    # Step 2: Run quality-based comparison
    print("\n🎯 Running quality-based comparison...")
    comparator = QualityBasedComparator(data_dir, output_dir)
    
    # Create quality achievement dashboard
    dashboard_path = output_dir / 'quality_achievement_dashboard.png'
    overall_score = comparator.create_quality_achievement_dashboard(
        df_gen, df_gt, dashboard_path
    )
    
    # Generate quality report
    report_path = output_dir / 'quality_comparison_report.md'
    comparator.generate_quality_report(df_gen, df_gt, report_path)
    
    print("\n" + "="*60)
    print("✅ QUALITY-BASED COMPARISON COMPLETE")
    print("="*60)
    print(f"Overall Quality Score: {overall_score:.1%}")
    print(f"\nResults saved to: {output_dir}")
    print("\nKey outputs:")
    print("  • quality_achievement_dashboard.png - Visual summary")
    print("  • quality_comparison_report.md - Detailed analysis")
    print("  • generated_metrics.csv - Generated dataset metrics")
    print("  • ground_truth_metrics.csv - Ground truth metrics")


if __name__ == '__main__':
    main()