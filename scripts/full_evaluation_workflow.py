#!/usr/bin/env python
"""
Full Evaluation Workflow
Runs complete evaluation including ground-truth comparison
Integrates all evaluation systems: Intention-based, Hybrid, and Ground-truth
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional


class FullEvaluationWorkflow:
    """Orchestrates the complete evaluation workflow."""
    
    def __init__(self, data_dir: Path = Path('data/edge_intention'),
                 output_base: Path = Path('outputs'),
                 verbose: bool = True):
        """
        Initialize workflow.
        
        Args:
            data_dir: Base data directory
            output_base: Base output directory
            verbose: Print detailed progress
        """
        self.data_dir = data_dir
        self.output_base = output_base
        self.verbose = verbose
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = output_base / f'full_evaluation_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories for different evaluations
        self.intention_output = self.output_dir / 'intention_based'
        self.hybrid_output = self.output_dir / 'hybrid'
        self.ground_truth_output = self.output_dir / 'ground_truth'
        
        # Results storage
        self.results = {}
    
    def log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Run a command and capture output.
        
        Args:
            cmd: Command to run
            description: Description for logging
            
        Returns:
            True if successful
        """
        self.log(f"\n🔄 {description}...")
        self.log(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.verbose and result.stdout:
                self.log(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Error: {e}")
            if e.stdout:
                self.log(f"Stdout: {e.stdout}")
            if e.stderr:
                self.log(f"Stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            self.log(f"❌ Command not found: {cmd[0]}")
            return False
    
    def run_intention_based_evaluation(self) -> bool:
        """Run intention-based evaluation (9 metrics)."""
        
        self.log("\n" + "="*60)
        self.log("1️⃣ INTENTION-BASED EVALUATION")
        self.log("="*60)
        
        # Check for tuned config
        config_path = Path('data/beat_configs/evaluator_config_20250808_185625.json')
        
        if config_path.exists():
            # Use tuned parameters
            cmd = [
                sys.executable,
                'scripts/evaluate_dataset_with_tuned_params.py',
                str(config_path),
                '--data_dir', str(self.data_dir),
                '--output_dir', str(self.intention_output)
            ]
        else:
            # Use default parameters
            cmd = [
                sys.executable,
                'scripts/evaluate_dataset.py',
                '--data_dir', str(self.data_dir),
                '--output_dir', str(self.intention_output)
            ]
        
        success = self.run_command(cmd, "Running intention-based evaluation")
        
        if success:
            # Load results
            metrics_file = self.intention_output / 'reports' / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.results['intention'] = json.load(f)
                self.log(f"✅ Intention-based evaluation complete: {len(self.results['intention'])} files")
        
        return success
    
    def run_hybrid_evaluation(self) -> bool:
        """Run hybrid wave type evaluation."""
        
        self.log("\n" + "="*60)
        self.log("2️⃣ HYBRID WAVE TYPE EVALUATION")
        self.log("="*60)
        
        # Check if data exists
        pas_dir = self.data_dir / 'light'
        geo_dir = Path('data/conformer_osci/light_segments')
        
        if not geo_dir.exists():
            self.log("⚠️  Skipping hybrid evaluation - Geo data not found")
            return True  # Not a failure, just skip
        
        # Run reconstruction
        cmd = [
            sys.executable,
            'scripts/wave_type_reconstructor.py',
            '--pas_dir', str(pas_dir),
            '--geo_dir', str(geo_dir),
            '--output', str(self.hybrid_output / 'wave_reconstruction.pkl'),
            '--config', 'configs/final_optimal.json'
        ]
        
        success = self.run_command(cmd, "Running wave type reconstruction")
        
        if success:
            # Run evaluation
            cmd = [
                sys.executable,
                'scripts/hybrid_evaluator.py',
                '--pas_dir', str(pas_dir),
                '--geo_dir', str(geo_dir),
                '--output', str(self.hybrid_output / 'evaluation_report.md')
            ]
            
            success = self.run_command(cmd, "Running hybrid evaluation")
            
            if success:
                # Generate visualizations
                cmd = [
                    sys.executable,
                    'scripts/wave_type_visualizer.py'
                ]
                self.run_command(cmd, "Generating hybrid visualizations")
                
                self.log("✅ Hybrid evaluation complete")
        
        return success
    
    def run_ground_truth_comparison(self) -> bool:
        """Run ground-truth comparison."""
        
        self.log("\n" + "="*60)
        self.log("3️⃣ GROUND-TRUTH COMPARISON")
        self.log("="*60)
        
        # Check if ground truth data exists
        gt_audio = self.data_dir / 'audio_ground_truth'
        gt_light = self.data_dir / 'light_ground_truth'
        
        if not gt_audio.exists() or not gt_light.exists():
            self.log("⚠️  Skipping ground-truth comparison - data not found")
            self.log(f"   Need: {gt_audio}")
            self.log(f"   Need: {gt_light}")
            return True  # Not a failure, just skip
        
        # Run comparison
        cmd = [
            sys.executable,
            'scripts/compare_to_ground_truth.py',
            '--data_dir', str(self.data_dir),
            '--output_dir', str(self.ground_truth_output)
        ]
        
        success = self.run_command(cmd, "Running ground-truth comparison")
        
        if success:
            # Generate enhanced visualizations
            cmd = [
                sys.executable,
                'scripts/ground_truth_visualizer.py',
                '--output_dir', str(self.ground_truth_output)
            ]
            
            self.run_command(cmd, "Generating enhanced visualizations")
            
            # Load results
            distances_file = self.ground_truth_output / 'distribution_distances.json'
            if distances_file.exists():
                with open(distances_file, 'r') as f:
                    self.results['ground_truth'] = json.load(f)
                self.log("✅ Ground-truth comparison complete")
        
        return success
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report."""
        
        self.log("\n" + "="*60)
        self.log("4️⃣ GENERATING FINAL REPORT")
        self.log("="*60)
        
        report = []
        report.append("# Complete Evaluation Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Output Directory:** `{self.output_dir}`\n\n")
        
        # Section 1: Intention-based results
        if 'intention' in self.results and self.results['intention']:
            report.append("## 1. Intention-Based Evaluation (9 Metrics)\n\n")
            
            # Calculate averages
            metrics_sums = {}
            for file_result in self.results['intention']:
                for metric, value in file_result.items():
                    if metric != 'file_name' and isinstance(value, (int, float)):
                        if metric not in metrics_sums:
                            metrics_sums[metric] = []
                        metrics_sums[metric].append(value)
            
            report.append("| Metric | Mean ± Std |\n")
            report.append("|--------|------------|\n")
            
            for metric, values in metrics_sums.items():
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
                report.append(f"| {metric.replace('_', ' ').title()} | {mean_val:.3f} ± {std_val:.3f} |\n")
            
            report.append(f"\n**Files Evaluated:** {len(self.results['intention'])}\n\n")
        
        # Section 2: Hybrid results
        hybrid_report = self.hybrid_output / 'evaluation_report.md'
        if hybrid_report.exists():
            report.append("## 2. Hybrid Wave Type Evaluation\n\n")
            report.append("See detailed report: `hybrid/evaluation_report.md`\n\n")
            
            # Try to extract key metrics
            with open(hybrid_report, 'r') as f:
                content = f.read()
                if 'Overall Score' in content:
                    # Extract the metrics table
                    lines = content.split('\n')
                    in_table = False
                    for line in lines:
                        if '| Metric | Score |' in line:
                            in_table = True
                            report.append(line + '\n')
                        elif in_table and line.startswith('|'):
                            report.append(line + '\n')
                        elif in_table and not line.startswith('|'):
                            break
            report.append('\n')
        
        # Section 3: Ground-truth comparison
        if 'ground_truth' in self.results:
            report.append("## 3. Ground-Truth Comparison\n\n")
            
            # Calculate overall fidelity
            w_distances = []
            for metric_data in self.results['ground_truth'].values():
                if isinstance(metric_data, dict) and 'wasserstein' in metric_data:
                    w_distances.append(metric_data['wasserstein'])
            
            if w_distances:
                avg_w = sum(w_distances) / len(w_distances)
                fidelity = max(0, 1 - (avg_w / 0.2))
                
                if fidelity > 0.8:
                    quality = "🟢 Excellent"
                elif fidelity > 0.6:
                    quality = "🔵 Good"
                elif fidelity > 0.4:
                    quality = "🟡 Moderate"
                else:
                    quality = "🔴 Poor"
                
                report.append(f"**Overall Fidelity Score:** {fidelity:.3f} ({quality})\n")
                report.append(f"**Average Wasserstein Distance:** {avg_w:.3f}\n\n")
                
                # Best and worst metrics
                best_metric = min(self.results['ground_truth'].keys(),
                                key=lambda x: self.results['ground_truth'][x].get('wasserstein', float('inf')))
                worst_metric = max(self.results['ground_truth'].keys(),
                                 key=lambda x: self.results['ground_truth'][x].get('wasserstein', 0))
                
                report.append(f"**Best Match:** {best_metric.replace('_', ' ').title()}\n")
                report.append(f"**Worst Match:** {worst_metric.replace('_', ' ').title()}\n\n")
        
        # Section 4: Summary
        report.append("## 4. Summary\n\n")
        
        evaluations_run = []
        if 'intention' in self.results:
            evaluations_run.append("✅ Intention-based (9 metrics)")
        if hybrid_report.exists():
            evaluations_run.append("✅ Hybrid wave type")
        if 'ground_truth' in self.results:
            evaluations_run.append("✅ Ground-truth comparison")
        
        report.append("**Evaluations Completed:**\n")
        for eval_type in evaluations_run:
            report.append(f"- {eval_type}\n")
        
        report.append(f"\n**Total Processing Time:** {datetime.now() - self.start_time}\n")
        
        # Section 5: File locations
        report.append("\n## 5. Output Files\n\n")
        report.append("```\n")
        report.append(f"{self.output_dir}/\n")
        report.append("├── intention_based/\n")
        report.append("│   ├── reports/\n")
        report.append("│   │   ├── metrics.csv\n")
        report.append("│   │   ├── metrics.json\n")
        report.append("│   │   └── evaluation_report.md\n")
        report.append("│   └── plots/\n")
        report.append("├── hybrid/\n")
        report.append("│   ├── wave_reconstruction.pkl\n")
        report.append("│   ├── evaluation_report.md\n")
        report.append("│   └── plots/\n")
        report.append("└── ground_truth/\n")
        report.append("    ├── comparison_report.md\n")
        report.append("    ├── distribution_distances.json\n")
        report.append("    └── plots/\n")
        report.append("        └── comprehensive_dashboard.png\n")
        report.append("```\n")
        
        # Save report
        report_path = self.output_dir / 'FINAL_REPORT.md'
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        self.log(f"✅ Final report saved to: {report_path}")
    
    def run_complete_workflow(self) -> bool:
        """Run the complete evaluation workflow."""
        
        self.start_time = datetime.now()
        
        print("\n" + "🎯"*30)
        print("\nCOMPLETE EVALUATION WORKFLOW")
        print("\n" + "🎯"*30)
        print(f"\nOutput directory: {self.output_dir}")
        
        # Run each evaluation
        success = True
        
        # 1. Intention-based
        if not self.run_intention_based_evaluation():
            self.log("⚠️  Intention-based evaluation failed")
            success = False
        
        # 2. Hybrid (if data exists)
        if not self.run_hybrid_evaluation():
            self.log("⚠️  Hybrid evaluation failed")
            success = False
        
        # 3. Ground-truth comparison (if data exists)
        if not self.run_ground_truth_comparison():
            self.log("⚠️  Ground-truth comparison failed")
            success = False
        
        # 4. Generate final report
        self.generate_final_report()
        
        # Summary
        print("\n" + "="*60)
        if success:
            print("✅ COMPLETE EVALUATION SUCCESSFUL")
        else:
            print("⚠️  EVALUATION COMPLETED WITH WARNINGS")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Total time: {datetime.now() - self.start_time}")
        print("="*60)
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete evaluation workflow'
    )
    parser.add_argument('--data_dir', type=str,
                       default='data/edge_intention',
                       help='Base data directory')
    parser.add_argument('--output_base', type=str,
                       default='outputs',
                       help='Base output directory')
    parser.add_argument('--skip_intention', action='store_true',
                       help='Skip intention-based evaluation')
    parser.add_argument('--skip_hybrid', action='store_true',
                       help='Skip hybrid evaluation')
    parser.add_argument('--skip_ground_truth', action='store_true',
                       help='Skip ground-truth comparison')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = FullEvaluationWorkflow(
        data_dir=Path(args.data_dir),
        output_base=Path(args.output_base),
        verbose=not args.quiet
    )
    
    # Override which evaluations to run
    if args.skip_intention:
        workflow.run_intention_based_evaluation = lambda: True
    if args.skip_hybrid:
        workflow.run_hybrid_evaluation = lambda: True
    if args.skip_ground_truth:
        workflow.run_ground_truth_comparison = lambda: True
    
    # Run workflow
    workflow.run_complete_workflow()


if __name__ == '__main__':
    main()