#!/usr/bin/env python
"""
Visualize the difference between distribution matching and quality achievement paradigms.
This creates a compelling visual that shows why quality-based evaluation is more appropriate.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_paradigm_comparison():
    """Create a visual comparison of the two paradigms."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Old paradigm (distribution matching)
    ax1.set_title('❌ Old Paradigm: Distribution Matching', fontsize=16, fontweight='bold')
    
    # Show mismatched distributions
    x = np.linspace(0, 1, 100)
    gt_dist = np.exp(-((x - 0.6) ** 2) / 0.05)
    gen_dist = np.exp(-((x - 0.4) ** 2) / 0.08)
    
    ax1.fill_between(x, gt_dist, alpha=0.3, color='blue', label='Ground Truth')
    ax1.fill_between(x, gen_dist, alpha=0.3, color='red', label='Generated')
    ax1.plot(x, gt_dist, 'b-', linewidth=2)
    ax1.plot(x, gen_dist, 'r-', linewidth=2)
    
    # Add Wasserstein distance visualization
    ax1.annotate('', xy=(0.4, 1.5), xytext=(0.6, 1.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(0.5, 1.6, 'Large Wasserstein Distance\n→ "Poor" Score', 
            ha='center', fontsize=12, color='red', fontweight='bold')
    
    ax1.set_xlabel('Metric Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.set_ylim(0, 2)
    
    # New paradigm (quality achievement)
    ax2.set_title('✅ New Paradigm: Quality Achievement', fontsize=16, fontweight='bold')
    
    # Show quality thresholds
    thresholds = [0.3, 0.5, 0.7]
    labels = ['Acceptable', 'Good', 'Excellent']
    colors = ['#F39C12', '#3498DB', '#2ECC71']
    
    for i, (thresh, label, color) in enumerate(zip(thresholds, labels, colors)):
        ax2.axvspan(thresh, 1.0 if i == len(thresholds)-1 else thresholds[i+1], 
                   alpha=0.2, color=color)
        ax2.text(thresh + 0.1, 1.8, label, fontsize=11, fontweight='bold')
    
    # Show that both achieve high quality
    ax2.scatter([0.65], [1.0], s=200, c='blue', marker='o', 
               label='Ground Truth', zorder=5)
    ax2.scatter([0.58], [0.8], s=200, c='red', marker='s', 
               label='Generated', zorder=5)
    
    ax2.arrow(0.58, 0.8, 0, -0.5, head_width=0.02, head_length=0.1, 
             fc='red', ec='red', alpha=0.5)
    ax2.arrow(0.65, 1.0, 0, -0.5, head_width=0.02, head_length=0.1,
             fc='blue', ec='blue', alpha=0.5)
    
    ax2.text(0.615, 0.3, 'Both Achieve\n"Good" Quality!', 
            ha='center', fontsize=12, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax2.set_xlabel('Quality Achievement', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 2)
    ax2.legend()
    
    plt.suptitle('Paradigm Shift: From Distribution to Quality', 
                fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = Path('outputs/quality_comparison/paradigm_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Paradigm comparison saved to: {output_path}")


if __name__ == '__main__':
    create_paradigm_comparison()