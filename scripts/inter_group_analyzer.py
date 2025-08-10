#!/usr/bin/env python
"""
Analyze correlations and relationships between the 3 lighting groups.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List  # FIXED: Added missing imports

class InterGroupAnalyzer:
    """Analyze relationships between lighting groups."""
    
    def analyze_file(self, data: np.ndarray) -> Dict:
        """Analyze inter-group relationships in one file."""
        
        results = {
            'correlations': {},
            'phase_relationships': {},
            'complementary_dynamics': {}
        }
        
        # Extract group parameters
        groups = []
        for g in range(3):
            start = g * 20
            groups.append({
                'amplitude': data[:, start + 5],
                'frequency': data[:, start + 4],
                'phase': data[:, start + 7],
                'hue': data[:, start + 8],
                'mai': np.sqrt(data[:, start]**2 + data[:, start+1]**2)  # Combined movement
            })
        
        # Compute correlations between groups
        for i in range(3):
            for j in range(i+1, 3):
                # Amplitude correlation
                corr, _ = pearsonr(groups[i]['amplitude'], groups[j]['amplitude'])
                results['correlations'][f'amp_g{i}_g{j}'] = corr
                
                # Hue correlation
                corr, _ = pearsonr(groups[i]['hue'], groups[j]['hue'])
                results['correlations'][f'hue_g{i}_g{j}'] = corr
                
                # MAI correlation
                corr, _ = pearsonr(groups[i]['mai'], groups[j]['mai'])
                results['correlations'][f'mai_g{i}_g{j}'] = corr
        
        # Phase relationships (are groups in sync or offset?)
        for i in range(3):
            for j in range(i+1, 3):
                phase_diff = np.mean(np.abs(groups[i]['phase'] - groups[j]['phase']))
                results['phase_relationships'][f'g{i}_g{j}'] = phase_diff
        
        # Complementary dynamics (when one is bright, is another dim?)
        for i in range(3):
            for j in range(i+1, 3):
                # Anti-correlation suggests complementary behavior
                anti_corr = -pearsonr(groups[i]['amplitude'], groups[j]['amplitude'])[0]
                results['complementary_dynamics'][f'g{i}_g{j}'] = anti_corr
        
        return results
    
    def analyze_directory(self, pred_dir: Path) -> Dict:
        """Analyze all predictions in directory."""
        
        all_results = []
        
        pkl_files = sorted(pred_dir.glob('*.pkl'))
        
        if not pkl_files:
            print(f"Warning: No .pkl files found in {pred_dir}")
            return {}
        
        print(f"Analyzing {len(pkl_files)} prediction files...")
        
        # Diagnostic: Check first file to understand structure
        first_file = pkl_files[0]
        with open(first_file, 'rb') as f:
            sample_data = pickle.load(f)
        
        print(f"\nDiagnostic - First file shape: {sample_data.shape}")
        if sample_data.shape[1] == 61:
            print(f"  Last column (col 60) stats: min={sample_data[:, -1].min():.2f}, max={sample_data[:, -1].max():.2f}")
            print(f"  -> Using columns 0-59 as oscillator parameters (skipping last column)\n")
        
        for pkl_file in pkl_files:
            print(f"  Processing {pkl_file.stem}")
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Verify shape
            if not isinstance(data, np.ndarray):
                print(f"    Warning: Expected numpy array, got {type(data)}")
                continue
                
            # Handle 61 dimensions (extra column at the end)
            if data.shape[1] == 61:
                # Use first 60 columns (0-59), skip the last one
                data = data[:, :60]
            elif data.shape[1] == 60:
                pass  # Perfect, use as is
            else:
                print(f"    Warning: Unexpected dimensions {data.shape[1]}, skipping file")
                continue
            
            file_results = self.analyze_file(data)
            file_results['file'] = pkl_file.stem
            all_results.append(file_results)
        
        # Aggregate statistics
        if all_results:
            aggregated = self.aggregate_results(all_results)
            # Add file count
            aggregated['num_files'] = len(all_results)
        else:
            aggregated = {}
        
        return aggregated
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate inter-group statistics."""
        
        correlations = []
        phase_diffs = []
        complementary = []
        
        # Collect all metrics
        for r in all_results:
            correlations.extend(list(r['correlations'].values()))
            phase_diffs.extend(list(r['phase_relationships'].values()))
            complementary.extend(list(r['complementary_dynamics'].values()))
        
        if not correlations:
            return {}
        
        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'mean_phase_diff': np.mean(phase_diffs),
            'std_phase_diff': np.std(phase_diffs),
            'mean_complementary': np.mean(complementary),
            'strong_correlations': sum(1 for c in correlations if abs(c) > 0.7) / len(correlations),
            'moderate_correlations': sum(1 for c in correlations if 0.3 <= abs(c) <= 0.7) / len(correlations),
            'independent_groups': sum(1 for c in correlations if abs(c) < 0.3) / len(correlations),
            'all_correlations': correlations,  # Keep raw data for visualization
            'all_phase_diffs': phase_diffs,
            'all_complementary': complementary
        }
    
    def create_visualization(self, results: Dict, output_path: Path):
        """Create correlation visualizations."""
        
        if not results or 'all_correlations' not in results:
            print("No data to visualize")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Correlation distribution histogram
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Amplitude/Hue/MAI correlations
        all_corr = results['all_correlations']
        axes[0].hist(all_corr, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(results['mean_correlation'], color='red', linestyle='--', 
                       label=f'Mean: {results["mean_correlation"]:.3f}')
        axes[0].set_xlabel('Correlation')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Inter-Group Correlations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Phase differences
        phase_diffs = results['all_phase_diffs']
        axes[1].hist(phase_diffs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[1].axvline(results['mean_phase_diff'], color='red', linestyle='--',
                       label=f'Mean: {results["mean_phase_diff"]:.3f}')
        axes[1].set_xlabel('Phase Difference (radians)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Phase Relationships')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Complementary dynamics
        comp = results['all_complementary']
        axes[2].hist(comp, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[2].axvline(results['mean_complementary'], color='red', linestyle='--',
                       label=f'Mean: {results["mean_complementary"]:.3f}')
        axes[2].set_xlabel('Anti-correlation')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Complementary Dynamics')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Inter-Group Relationship Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'inter_group_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Summary pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = ['Strong (>0.7)', 'Moderate (0.3-0.7)', 'Independent (<0.3)']
        sizes = [
            results['strong_correlations'] * 100,
            results['moderate_correlations'] * 100,
            results['independent_groups'] * 100
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
        ax.set_title('Inter-Group Correlation Distribution', fontsize=14, fontweight='bold')
        
        plt.savefig(output_path / 'correlation_categories.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def save_results(self, results: Dict, output_path: Path):
        """Save analysis results."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results as pickle
        with open(output_path / 'inter_group_analysis.pkl', 'wb') as f:
            # Remove raw data arrays for cleaner saving
            save_results = {k: v for k, v in results.items() 
                          if not k.startswith('all_')}
            pickle.dump(save_results, f)
        
        # Save summary as JSON
        import json
        with open(output_path / 'inter_group_summary.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if not key.startswith('all_'):  # Skip raw arrays
                    if isinstance(value, (np.float32, np.float64)):
                        json_results[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        json_results[key] = int(value)
                    else:
                        json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        # Save human-readable report
        with open(output_path / 'inter_group_report.txt', 'w') as f:
            f.write("Inter-Group Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            if 'num_files' in results:
                f.write(f"Files analyzed: {results['num_files']}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write(f"  Mean correlation: {results.get('mean_correlation', 0):.3f} ± {results.get('std_correlation', 0):.3f}\n")
            f.write(f"  Mean phase difference: {results.get('mean_phase_diff', 0):.3f} rad\n")
            f.write(f"  Mean complementary score: {results.get('mean_complementary', 0):.3f}\n\n")
            
            f.write("Correlation Categories:\n")
            f.write(f"  Strong (>0.7): {results.get('strong_correlations', 0)*100:.1f}%\n")
            f.write(f"  Moderate (0.3-0.7): {results.get('moderate_correlations', 0)*100:.1f}%\n")
            f.write(f"  Independent (<0.3): {results.get('independent_groups', 0)*100:.1f}%\n\n")
            
            f.write("Interpretation:\n")
            if results.get('mean_correlation', 0) > 0.5:
                f.write("  - Groups show coordinated behavior\n")
            elif results.get('mean_correlation', 0) < 0.3:
                f.write("  - Groups operate mostly independently\n")
            else:
                f.write("  - Groups show moderate coordination\n")
            
            if results.get('mean_complementary', 0) > 0.3:
                f.write("  - Evidence of complementary dynamics (alternating intensity)\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze inter-group correlations')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with predictions')
    parser.add_argument('--output_dir', type=str,
                       default='outputs_oscillator/inter_group',
                       help='Output directory')
    
    args = parser.parse_args()
    
    pred_path = Path(args.pred_dir)
    output_path = Path(args.output_dir)
    
    if not pred_path.exists():
        print(f"Error: Prediction directory not found: {pred_path}")
        return
    
    analyzer = InterGroupAnalyzer()
    results = analyzer.analyze_directory(pred_path)
    
    if results:
        print("\n" + "="*50)
        print("Inter-Group Analysis Results:")
        print("="*50)
        for key, value in results.items():
            if not key.startswith('all_'):  # Skip raw arrays in console output
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Save results
        analyzer.save_results(results, output_path)
        
        # Create visualizations
        analyzer.create_visualization(results, output_path)
        
        print(f"\nResults saved to {output_path}")
    else:
        print("No results to analyze")

if __name__ == '__main__':
    main()