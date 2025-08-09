from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def create_all_plots(viz_data: dict, file_name: str, output_dir: Path) -> None:
    (output_dir / 'ssm').mkdir(parents=True, exist_ok=True)
    (output_dir / 'novelty').mkdir(parents=True, exist_ok=True)
    create_ssm_plot(
        viz_data['audio_ssm'],
        viz_data['light_ssm'],
        file_name,
        output_dir / 'ssm' / f'{file_name}_ssm.png',
    )
    create_novelty_plot(
        viz_data['audio_novelty'],
        viz_data['light_novelty'],
        viz_data['audio_boundaries'],
        viz_data['light_boundaries'],
        file_name,
        output_dir / 'novelty' / f'{file_name}_novelty.png',
    )


def create_ssm_plot(audio_ssm: np.ndarray, light_ssm: np.ndarray, title: str, output_path: Path) -> None:
    if audio_ssm.size == 0 or light_ssm.size == 0:
        return
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(audio_ssm, cmap='hot', aspect='auto', interpolation='nearest')
    ax1.set_title('Audio SSM', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Time (frames)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(light_ssm, cmap='hot', aspect='auto', interpolation='nearest')
    ax2.set_title('Lighting SSM', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (frames)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(gs[2])
    diff = audio_ssm - light_ssm
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    ax3.set_title('Difference (Audio - Light)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (frames)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(f'Self-Similarity Matrices: {title}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_novelty_plot(audio_nov: np.ndarray, light_nov: np.ndarray, audio_bounds: np.ndarray,
                        light_bounds: np.ndarray, title: str, output_path: Path) -> None:
    if audio_nov.size == 0 or light_nov.size == 0:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    # Time axis assumes downsampling by 10 from 30 fps
    time_axis = np.arange(len(audio_nov)) * (10.0 / 30.0)

    ax1.plot(time_axis, audio_nov, label='Audio Novelty', color='#2E86AB', linewidth=2)
    ax1.fill_between(time_axis, 0, audio_nov, alpha=0.3, color='#2E86AB')
    for b in audio_bounds:
        ax1.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Novelty', fontsize=12)
    ax1.set_title('Audio Novelty Function', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_axis, light_nov, label='Lighting Novelty', color='#A23B72', linewidth=2)
    ax2.fill_between(time_axis, 0, light_nov, alpha=0.3, color='#A23B72')
    for b in light_bounds:
        ax2.axvline(x=b, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Novelty', fontsize=12)
    ax2.set_title('Lighting Novelty Function', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Novelty Functions: {title}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_boxplot(df, metric_columns, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        'ssm_correlation': 'Γ_structure',
        'novelty_correlation': 'Γ_novelty',
        'boundary_f_score': 'Γ_boundary',
        'rms_correlation': 'Γ_loud↔bright',
        'onset_correlation': 'Γ_change',
        'beat_peak_alignment': 'Γ_beat↔peak',
        'beat_valley_alignment': 'Γ_beat↔valley',
        'intensity_variance': 'Ψ_intensity',
        'color_variance': 'Ψ_color',
    }

    n = len(labels)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()

    for i, (col, label) in enumerate(labels.items()):
        if col in df.columns:
            ax = axes[i]
            bp = ax.boxplot(df[col].values, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
            y = df[col].values
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=30, color='#e74c3c')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'{label}\n{col.replace("_", " ").title()}', fontsize=11)
            ax.set_xticklabels([''])
            ax.grid(True, alpha=0.3)
            ax.axhline(y=df[col].mean(), color='green', linestyle='--', alpha=0.7,
                       label=f'Mean: {df[col].mean():.3f}')
            ax.legend(fontsize=8)
    plt.suptitle('Metric Distribution Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(df, output_path: Path) -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'shrink': .8})
    plt.title('Metric Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
