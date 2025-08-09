# Intention-Based Audio–Light Evaluation

This README provides step-by-step instructions to run the intention-based evaluation for audio–lighting correspondence using the scripts in `assets/evaluation/`.

It covers environment setup, dependencies, how to run the evaluation over the dataset, and how to generate the summary plots and report.

## Repository Layout (relevant)

- `assets/evaluation/`
  - `data/edge_intention/`
    - `audio/` — audio feature pickles (`*.pkl`)
    - `light/` — lighting intention pickles (`*.pkl`)
  - `scripts/`
    - `structural_evaluator.py` — core evaluator (metrics + SSM/novelty/boundaries)
    - `evaluate_dataset.py` — runs evaluation over all pairs and saves outputs
    - `visualizer.py` — plotting utilities for SSM and novelty comparisons
    - `generate_final_plots.py` — creates boxplot and correlation heatmap summaries
    - `inspect_pickle_audio.py`, `inspect_pickle_light.py` — pickle inspectors (optional)
  - `outputs/` (created on first run)
    - `plots/ssm/`, `plots/novelty/`, `plots/metrics/`
    - `reports/metrics.csv`, `reports/metrics.json`, `reports/evaluation_report.md`
  - `formulas/intention_eval_formulas.md` — summary of the formulas used
  - `extracted_formulas_intention_based.md` — your original formulas document

## 1) Virtual Environment

A dedicated venv exists at:

- `assets/evaluation/.venv/`

Activate if desired, or call the Python/pip binaries explicitly via `.venv/bin/...` (used below).

## 2) Install Dependencies

Install the required packages into the venv:

```bash
# From: assets/evaluation/
.venv/bin/pip install numpy scipy pandas matplotlib seaborn librosa mir_eval
```

Notes:
- If `mir_eval` fails to install on your system, the scripts will still run; boundary F-score will gracefully fall back to `0.0`.
- `numpy` is already present in the venv from earlier steps.

## 3) Data Expectations

- Audio pickles live in: `assets/evaluation/data/edge_intention/audio/`
- Light pickles live in: `assets/evaluation/data/edge_intention/light/`
- Each audio file is matched to a light file by prefix/stem (the evaluator searches for `light/**/<audio_stem>*.pkl`).

Minimum keys expected in audio pickles (if present they will be used):
- `chroma_stft` — for audio SSM and novelty
- `onset_env` — for onset↔light-change correlation
- `onset_beat` — for beat alignment
- Optional: `rms` or `melspe_db` — used to compute RMS↔brightness correlation

Light pickles are expected to be NumPy arrays shaped `(T, 72)` with 12 groups × 6 parameters; parameter 1 in each group is “intensity peak”.

## 4) Run the Evaluation

This aggregates all metrics and saves plots and a report.

```bash
# Use the saved configuration
python scripts/evaluate_dataset_with_tuned_params.py data/beat_configs/evaluator_config_20250808_185625.json \
    --data_dir data/edge_intention \
    --output_dir outputs
```

Outputs created:
- `outputs/reports/metrics.csv` and `metrics.json` — per-file metrics
- `outputs/reports/evaluation_report.md` — summary report
- `outputs/plots/ssm/*` — side-by-side SSM and difference plots
- `outputs/plots/novelty/*` — novelty functions with boundary markers

## 5) Generate Final Summary Plots

Create a metric distribution boxplot and correlation heatmap from the metrics CSV:

```bash
# From: assets/evaluation/
.venv/bin/python scripts/generate_final_plots.py
```

Outputs created:
- `outputs/plots/metrics/summary_boxplot.png`
- `outputs/plots/metrics/correlation_heatmap.png`

## 6) What the Evaluator Computes

Implemented in `scripts/structural_evaluator.py` (see also `formulas/intention_eval_formulas.md`):

- **SSM correlation (Γ_structure)**: similarity between audio and light SSMs
- **Novelty correlation (Γ_novelty)**: correlation of novelty curves (edge-trimmed)
- **Boundary F-score (Γ_boundary)**: via `mir_eval.segment.detection` (if available)
- **RMS↔brightness (Γ_loud↔bright)**
- **Onset↔change (Γ_change)**
- **Beat↔peak / Beat↔valley alignment**
- **Variance metrics (Ψ_intensity, Ψ_color)**

Defaults (align with formulas):
- `L_kernel=31`, `L_smooth=81`, `H=10`, `beat_align_sigma=0.5`
- `rms_window_size=120`, `onset_window_size=120`
- `peak_distance=15`, `peak_prominence=0.04`
- `boundary_window=2.0`, `fps=30`

## 7) Inspecting Pickle Structure (optional)

If you need to inspect raw pickle contents quickly:

```bash
# Light pickle (LIGHT-only report)
.venv/bin/python scripts/inspect_pickle_light.py \
  "data/edge_intention/light/<your_light_file>.pkl" --out content_light_pickle.txt

# Audio pickle
.venv/bin/python scripts/inspect_pickle_audio.py \
  "data/edge_intention/audio/<your_audio_file>.pkl" --out content_audio_pickle.txt
```

Reports will be written to `assets/evaluation/` with normalized filenames.

## 8) Troubleshooting

- **Missing packages**: run the pip install command in Section 2.
- **No light match found**: ensure the light filename contains the audio stem; evaluator matches using `<stem>*.pkl` under `light/` subfolders.
- **Large memory for SSM**: the evaluator downsamples features (`H=10`) and smooths to reduce matrix size.
- **mir_eval missing**: boundary metrics default to 0; install `mir_eval` if you need them.

## 9) References

- Formulas reference: `assets/evaluation/formulas/intention_eval_formulas.md`
- Original formulas document: `assets/evaluation/extracted_formulas_intention_based.md`
