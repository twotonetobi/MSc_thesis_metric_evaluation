import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from structural_evaluator import StructuralEvaluator
from visualizer import create_all_plots


def _derive_light_for_audio(light_dir: Path, base_stem: str) -> Path | None:
    # Look for exact stem match anywhere under light_dir
    cands = list(light_dir.rglob(f"{base_stem}*.pkl"))
    return cands[0] if cands else None


def evaluate_dataset(data_dir: str | Path, output_dir: str | Path) -> Tuple[pd.DataFrame, dict]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    audio_dir = data_dir / 'audio'
    light_dir = data_dir / 'light'

    # Create outputs
    (output_dir / 'plots' / 'ssm').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots' / 'novelty').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots' / 'metrics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    evaluator = StructuralEvaluator()

    all_metrics = []
    all_viz = {}

    audio_files = sorted(audio_dir.glob('*.pkl'))
    for af in audio_files:
        base = af.stem
        lf = _derive_light_for_audio(light_dir, base)
        if lf is None:
            print(f"[WARN] No light file found for {base}")
            continue
        print(f"[INFO] Processing: {base}")
        metrics, viz = evaluator.evaluate_single_file(af, lf)
        metrics['file_name'] = base
        all_metrics.append(metrics)
        all_viz[base] = viz
        create_all_plots(viz, base, output_dir / 'plots')

    df = pd.DataFrame(all_metrics)
    (output_dir / 'reports' / 'metrics.csv').write_text(df.to_csv(index=False))
    (output_dir / 'reports' / 'metrics.json').write_text(json.dumps(all_metrics, indent=2))

    generate_report(df, output_dir / 'reports' / 'evaluation_report.md')
    return df, all_viz


def generate_report(df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append('# Intention-Based Lighting Evaluation Report\n')
    lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Number of Files:** {len(df)}\n\n")
    lines.append('## Summary Statistics\n\n')
    metrics = [
        ('ssm_correlation', 'Γ_structure', 'Structure Correlation'),
        ('novelty_correlation', 'Γ_novelty', 'Novelty Correlation'),
        ('boundary_f_score', 'Γ_boundary', 'Boundary Detection F-Score'),
        ('rms_correlation', 'Γ_loud↔bright', 'RMS-Brightness Correlation'),
        ('onset_correlation', 'Γ_change', 'Onset-Change Correlation'),
        ('beat_peak_alignment', 'Γ_beat↔peak', 'Beat-Peak Alignment'),
        ('beat_valley_alignment', 'Γ_beat↔valley', 'Beat-Valley Alignment'),
        ('intensity_variance', 'Ψ_intensity', 'Intensity Variance'),
        ('color_variance', 'Ψ_color', 'Color Variance'),
    ]
    lines.append('| Metric | Symbol | Mean | Std | Min | Max |\n')
    lines.append('|--------|--------|------|-----|-----|-----|\n')
    for col, sym, name in metrics:
        if col in df.columns and len(df) > 0:
            mean = df[col].mean()
            std = df[col].std()
            mn = df[col].min()
            mx = df[col].max()
            lines.append(f"| {name} | {sym} | {mean:.3f} | {std:.3f} | {mn:.3f} | {mx:.3f} |\n")
    lines.append('\n## Detailed Analysis\n')
    if len(df) > 0:
        lines.append('\n### Structural Coherence\n')
        structural = (df.get('ssm_correlation', pd.Series([0])).mean() +
                      df.get('novelty_correlation', pd.Series([0])).mean() +
                      df.get('boundary_f_score', pd.Series([0])).mean()) / 3.0
        lines.append(f"- **Overall Structural Coherence:** {float(structural):.3f}\n")
        if 'file_name' in df.columns and 'ssm_correlation' in df.columns:
            lines.append(f"- **Best Performing File:** {df.loc[df['ssm_correlation'].idxmax(), 'file_name']}\n")
            lines.append(f"- **Worst Performing File:** {df.loc[df['ssm_correlation'].idxmin(), 'file_name']}\n")
        lines.append('\n### Rhythmic Alignment\n')
        rhythm = (df.get('beat_peak_alignment', pd.Series([0])).mean() +
                  df.get('beat_valley_alignment', pd.Series([0])).mean()) / 2.0
        lines.append(f"- **Overall Rhythmic Alignment:** {float(rhythm):.3f}\n")
        lines.append('\n### Dynamic Response\n')
        dynamics = (df.get('rms_correlation', pd.Series([0])).mean() +
                    df.get('onset_correlation', pd.Series([0])).mean()) / 2.0
        lines.append(f"- **Overall Dynamic Response:** {float(dynamics):.3f}\n")
        lines.append('\n## Per-File Results\n\n')
        lines.append('| File | Structure | Rhythm | Dynamics | Overall |\n')
        lines.append('|------|-----------|--------|----------|---------|\n')
        for _, row in df.iterrows():
            s = (row.get('ssm_correlation', 0) + row.get('novelty_correlation', 0) + row.get('boundary_f_score', 0)) / 3.0
            r = (row.get('beat_peak_alignment', 0) + row.get('beat_valley_alignment', 0)) / 2.0
            d = (row.get('rms_correlation', 0) + row.get('onset_correlation', 0)) / 2.0
            o = (s + r + d) / 3.0
            fname = row.get('file_name', '')
            short = (fname[:30] + '...') if len(fname) > 30 else fname
            lines.append(f"| {short} | {s:.3f} | {r:.3f} | {d:.3f} | {o:.3f} |\n")
    output_path.write_text(''.join(lines))
    print(f"Report saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate intention-based lighting dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to edge_intention directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for results')
    args = parser.parse_args()
    df, _ = evaluate_dataset(args.data_dir, args.output_dir)
    print('\nEvaluation complete!')
    print(f"Results saved to: {args.output_dir}")
