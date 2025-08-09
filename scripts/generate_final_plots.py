from pathlib import Path
import pandas as pd
from visualizer import create_summary_boxplot, create_correlation_heatmap

BASE = Path(__file__).resolve().parents[1]  # .../evaluation
DEFAULT_OUT = BASE / 'outputs'

def generate_final_visualizations(output_dir: str | Path = DEFAULT_OUT) -> None:
    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / 'reports' / 'metrics.csv')
    (output_dir / 'plots' / 'metrics').mkdir(parents=True, exist_ok=True)
    create_summary_boxplot(df, df.columns, output_dir / 'plots' / 'metrics' / 'summary_boxplot.png')
    create_correlation_heatmap(df, output_dir / 'plots' / 'metrics' / 'correlation_heatmap.png')
    print('Final visualizations created!')

if __name__ == '__main__':
    generate_final_visualizations()
