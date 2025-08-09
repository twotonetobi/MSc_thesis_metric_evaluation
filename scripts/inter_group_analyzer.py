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
        
        for pkl_file in sorted(pred_dir.glob('*.pkl')):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            file_results = self.analyze_file(data)
            file_results['file'] = pkl_file.stem
            all_results.append(file_results)
        
        # Aggregate statistics
        aggregated = self.aggregate_results(all_results)
        
        return aggregated
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate inter-group statistics."""
        
        correlations = []
        phase_diffs = []
        complementary = []
        
        for r in all_results:
            correlations.extend(list(r['correlations'].values()))
            phase_diffs.extend(list(r['phase_relationships'].values()))
            complementary.extend(list(r['complementary_dynamics'].values()))
        
        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'mean_phase_diff': np.mean(phase_diffs),
            'mean_complementary': np.mean(complementary),
            'strong_correlations': sum(1 for c in correlations if abs(c) > 0.7) / len(correlations),
            'independent_groups': sum(1 for c in correlations if abs(c) < 0.3) / len(correlations)
        }
    
    def create_visualization(self, results: Dict, output_path: Path):
        """Create correlation heatmap."""
        
        # This would create various plots showing inter-group relationships
        pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze inter-group correlations')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory with predictions')
    parser.add_argument('--output_dir', type=str,
                       default='outputs_oscillator/inter_group',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = InterGroupAnalyzer()
    results = analyzer.analyze_directory(Path(args.pred_dir))
    
    print("\nInter-Group Analysis Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")

if __name__ == '__main__':
    main()