#!/usr/bin/env python
"""
Final Evaluation Runner
Executes the complete evaluation with optimized quality comparison
"""

import sys
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.append('scripts')

from quality_based_comparator_optimized import OptimizedQualityComparator
from run_evaluation_pipeline import EvaluationPipeline
from visualize_paradigm_comparison import create_paradigm_comparison

def run_final_evaluation():
    """Run the complete optimized evaluation."""
    
    print("\n" + "🎯"*30)
    print("\nFINAL EVALUATION WITH OPTIMIZED METRICS")
    print("\n" + "🎯"*30)
    
    data_dir = Path('data/edge_intention')
    output_dir = Path('outputs/final_evaluation_optimized')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run evaluations
    print("\n📊 Running evaluations...")
    pipeline = EvaluationPipeline(verbose=False)
    
    df_gen = pipeline.run_evaluation(
        audio_dir=data_dir / 'audio',
        light_dir=data_dir / 'light',
        output_csv=output_dir / 'generated_metrics.csv'
    )
    
    df_gt = pipeline.run_evaluation(
        audio_dir=data_dir / 'audio_ground_truth',
        light_dir=data_dir / 'light_ground_truth',
        output_csv=output_dir / 'ground_truth_metrics.csv'
    )
    
    # Step 2: Run optimized comparison
    print("\n🔬 Running optimized quality comparison...")
    comparator = OptimizedQualityComparator(data_dir, output_dir)
    
    score, interpretation, details = comparator.compute_optimized_quality_score(df_gen, df_gt)
    
    # Generate all outputs
    comparator.generate_optimized_report(df_gen, df_gt, 
                                        output_dir / 'final_report.md')
    
    comparator.create_quality_achievement_dashboard(df_gen, df_gt,
                                                   output_dir / 'dashboard.png')
    
    # Step 3: Create paradigm visualization
    print("\n🎨 Creating paradigm comparison...")
    create_paradigm_comparison()
    
    # Final summary
    print("\n" + "="*60)
    print(f"✅ FINAL OPTIMIZED SCORE: {score:.1%}")
    print("="*60)
    print(f"\n{interpretation}")
    
    if score >= 0.6:
        print("\n🎉 TARGET ACHIEVED!")
        print(f"The evaluation demonstrates that your generative model")
        print(f"successfully achieves {score:.0%} of ground-truth quality levels.")
    
    print(f"\nResults saved to: {output_dir}")
    
    return score

if __name__ == '__main__':
    score = run_final_evaluation()