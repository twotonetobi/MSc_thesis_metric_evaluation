import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# Optional deps (graceful fallback)
try:
    import librosa
except Exception:
    librosa = None

try:
    import mir_eval
except Exception:
    mir_eval = None

try:
    from scipy.ndimage import zoom as nd_zoom
except Exception:
    nd_zoom = None


def _downsample_feature_seq(X: np.ndarray, filt_len: int, down_sampling: int, 
                            expected_time_dim: int = None) -> np.ndarray:
    """Simple smooth + downsample replacement for libfmp.c3.smooth_downsample_feature_sequence.
    
    CRITICAL FIX: Properly handle dimension detection
    - For audio features like chroma: (time, features) -> (2700, 12)
    - For light features: (time, 72)
    
    Returns: (features, downsampled_time) for SSM computation
    """
    if X.ndim != 2:
        return X
    
    # FIXED: Better dimension detection
    # If we have an expected time dimension, use it
    if expected_time_dim is not None:
        if X.shape[0] == expected_time_dim or abs(X.shape[0] - expected_time_dim) <= 1:
            # X is (time, features)
            X_t = X
            T, D = X.shape
        else:
            # X might be transposed
            X_t = X.T
            D, T = X.shape
    else:
        # Heuristic: audio features typically have fewer feature dims than time steps
        # chroma=12, mfcc=20, etc. vs thousands of frames
        if X.shape[0] > 100 and X.shape[1] < 100:
            # Likely (time, features)
            X_t = X
            T, D = X.shape
        elif X.shape[1] > 100 and X.shape[0] < 100:
            # Likely (features, time) - transpose it
            X_t = X.T
            D, T = X.shape
        else:
            # Default assumption for ambiguous cases
            # Assume longer dimension is time
            if X.shape[0] > X.shape[1]:
                X_t = X
                T, D = X.shape
            else:
                X_t = X.T
                D, T = X.shape
    
    print(f"    _downsample: Input shape {X.shape} -> treating as {T} time steps × {D} features")
    
    # Smooth with moving average
    if filt_len and filt_len > 1:
        k = filt_len
        pad = k // 2
        pad_mode = 'edge'
        X_pad = np.pad(X_t, ((pad, pad), (0, 0)), mode=pad_mode)
        cumsum = np.cumsum(X_pad, axis=0)
        smoothed = (cumsum[k:] - cumsum[:-k]) / float(k)
    else:
        smoothed = X_t
    
    # Downsample in time
    if down_sampling and down_sampling > 1:
        smoothed = smoothed[::down_sampling]
    
    print(f"    After downsampling: {smoothed.shape[0]} time steps × {smoothed.shape[1]} features")
    
    # Return (features, time) for SSM computation
    return smoothed.T


def _normalize_columns(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if X.ndim != 2:
        return X
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


class StructuralEvaluator:
    """Evaluates structural correspondence between audio and lighting intentions.
    
    FIXED VERSION with:
    - Proper dimension handling for full song processing
    - Rhythmic intent detection for beat alignment
    - Complete metric computation
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        # Parameters (aligned with formulas doc)
        self.L_kernel = int(self.config.get('L_kernel', 31))
        self.L_smooth = int(self.config.get('L_smooth', 81))
        self.H = int(self.config.get('H', 10))  # Downsampling factor
        self.beat_align_sigma = float(self.config.get('beat_align_sigma', 0.5))
        self.rms_window_size = int(self.config.get('rms_window_size', 120))
        self.onset_window_size = int(self.config.get('onset_window_size', 120))
        self.peak_distance = int(self.config.get('peak_distance', 15))
        self.peak_prominence = float(self.config.get('peak_prominence', 0.04))
        self.boundary_window = float(self.config.get('boundary_window', 2.0))
        self.fps = int(self.config.get('fps', 30))
        self.verbose = bool(self.config.get('verbose', True))  # Default to verbose for debugging
        
        # Rhythmic intent parameters
        self.use_rhythmic_filter = bool(self.config.get('use_rhythmic_filter', True))
        self.rhythmic_window = int(self.config.get('rhythmic_window', 90))
        self.rhythmic_threshold = float(self.config.get('rhythmic_threshold', 0.05))

    # ----------- IO -----------
    def load_data(self, audio_path: str | Path, light_path: str | Path) -> Tuple[dict, np.ndarray]:
        with open(audio_path, 'rb') as f:
            audio_data = pickle.load(f)
        with open(light_path, 'rb') as f:
            light_data = pickle.load(f)
        
        if self.verbose:
            print(f"  Loaded audio data with keys: {list(audio_data.keys())}")
            if 'chroma_stft' in audio_data:
                print(f"    chroma_stft shape: {audio_data['chroma_stft'].shape}")
            print(f"  Loaded light data shape: {light_data.shape}")
        
        return audio_data, light_data

    # ----------- Feature helpers -----------
    def extract_brightness(self, intention_array: np.ndarray) -> np.ndarray:
        """Brightness = sum of intensity peaks across 12 groups (param index 0 of each 6-block)."""
        if intention_array.ndim != 2:
            return np.zeros(0)
        brightness = np.sum(intention_array[:, 0::6], axis=1)
        mx = np.max(brightness) if brightness.size else 0
        return brightness / mx if mx > 0 else brightness
    
    def detect_rhythmic_intent(self, brightness: np.ndarray) -> np.ndarray:
        """Detect rhythmic intent based on brightness variation (rolling STD)."""
        if not self.use_rhythmic_filter:
            return np.ones(len(brightness), dtype=bool)
        
        series = pd.Series(brightness)
        rolling_std = series.rolling(
            window=self.rhythmic_window,
            center=True,
            min_periods=1
        ).std().fillna(0)
        
        rhythmic_mask = rolling_std > self.rhythmic_threshold
        
        if self.verbose:
            rhythmic_pct = rhythmic_mask.mean() * 100
            print(f"    Rhythmic intent: {rhythmic_pct:.1f}% of frames classified as rhythmic")
        
        return rhythmic_mask.to_numpy()

    # ----------- SSM & Novelty -----------
    def compute_ssm(self, features: np.ndarray, feature_type: str = 'light', 
                    expected_frames: int = None) -> np.ndarray:
        """Compute self-similarity using 1 - (L2/sqrt(d)).
        
        FIXED: Properly handle dimensions for full song processing
        """
        if features is None or features.size == 0:
            return np.zeros((0, 0))
        
        if self.verbose:
            print(f"  Computing {feature_type} SSM from shape {features.shape}")
        
        # CRITICAL FIX: Pass expected time dimension for proper detection
        X = _downsample_feature_seq(features, self.L_smooth, self.H, 
                                    expected_time_dim=expected_frames)
        
        if feature_type == 'audio':
            X = _normalize_columns(X)
        
        D, N = X.shape
        if N == 0:
            return np.zeros((0, 0))
        
        if self.verbose:
            print(f"    SSM computation: {D} features × {N} time frames")
        
        # Efficient pairwise distance computation
        XtX = X.T @ X  # (N, N)
        diag = np.sum(X * X, axis=0, keepdims=True)  # (1, N)
        dist2 = diag.T + diag - 2.0 * XtX
        dist2 = np.maximum(dist2, 0.0)
        dist = np.sqrt(dist2)
        S = 1.0 - dist / np.sqrt(max(D, 1))
        
        if self.verbose:
            print(f"    Computed {feature_type} SSM: {S.shape}")
        
        return np.clip(S, 0.0, 1.0)

    def compute_novelty(self, S: np.ndarray) -> np.ndarray:
        """Compute novelty function from SSM using Gaussian checkerboard kernel."""
        L = self.L_kernel
        if S.size == 0 or L <= 0:
            return np.zeros(0)
        
        if S.shape[0] < 2 * L + 1:
            if self.verbose:
                print(f"    Warning: SSM too small ({S.shape[0]}) for kernel size {2*L+1}")
            # Adjust kernel size if SSM is too small
            L = max(1, S.shape[0] // 4)
        
        var = 0.5
        axis = np.arange(-L, L + 1)
        g1 = np.exp(-((axis / (L * var)) ** 2) / 2)
        g2 = np.outer(g1, g1)
        checker = np.outer(np.sign(axis), np.sign(axis))
        kernel = checker * g2
        kernel /= np.sum(np.abs(kernel)) + 1e-9
        
        N = S.shape[0]
        M = 2 * L + 1
        nov = np.zeros(N)
        S_pad = np.pad(S, L, mode='constant')
        
        for n in range(N):
            nov[n] = float(np.sum(S_pad[n:n + M, n:n + M] * kernel))
        
        # Exclude edges
        if L < N:
            nov[:L] = 0
            nov[-L:] = 0
        
        return nov

    def find_peaks_and_boundaries(self, novelty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if novelty.size == 0:
            return np.array([]), np.array([])
        peaks, _ = find_peaks(novelty, distance=self.peak_distance, prominence=self.peak_prominence)
        # Convert to seconds (account for downsampling by H at 30 fps base)
        boundaries_sec = peaks * (self.H / float(self.fps))
        return peaks, boundaries_sec

    # ----------- Alignment helpers -----------
    def align_frames_to_light(self, audio_data: dict, light_data: np.ndarray) -> dict:
        """Truncate audio feature arrays to match light frame count (fix 2701 vs 2700)."""
        try:
            light_frames = int(light_data.shape[0])
        except Exception:
            return audio_data
        
        for key, val in list(audio_data.items()):
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                audio_frames = val.shape[0]
                if audio_frames > light_frames:
                    if self.verbose:
                        print(f"    Aligning {key}: {audio_frames} -> {light_frames} frames")
                    audio_data[key] = val[:light_frames]
                elif audio_frames < light_frames and self.verbose:
                    print(f"    Warning: {key} has fewer frames ({audio_frames}) than light ({light_frames})")
        
        return audio_data

    def align_ssms(self, ssm1: np.ndarray, ssm2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Align two SSMs to the same size via interpolation."""
        if ssm1.size == 0 or ssm2.size == 0:
            return ssm1, ssm2
        
        s1 = int(ssm1.shape[0])
        s2 = int(ssm2.shape[0])
        
        if s1 == s2:
            return ssm1, ssm2
        
        target = max(s1, s2)

        def resize(mat: np.ndarray, src: int, dst: int) -> np.ndarray:
            if src == dst:
                return mat
            if nd_zoom is not None:
                zf = dst / float(src)
                return nd_zoom(mat, zf, order=1)
            # Fallback: crude nearest-neighbor
            idx = (np.linspace(0, src - 1, dst)).round().astype(int)
            idx = np.clip(idx, 0, src - 1)
            return mat[idx][:, idx]

        out1 = resize(ssm1, s1, target)
        out2 = resize(ssm2, s2, target)
        
        # Ensure exact same size
        m = min(out1.shape[0], out2.shape[0])
        out1 = out1[:m, :m]
        out2 = out2[:m, :m]
        
        if self.verbose:
            if s1 != m:
                print(f"    Resized audio SSM from {s1}×{s1} to {m}×{m}")
            if s2 != m:
                print(f"    Resized light SSM from {s2}×{s2} to {m}×{m}")
        
        return out1, out2

    # ----------- Metrics -----------
    def compute_rms_correlation(self, audio_data: dict, light_brightness: np.ndarray) -> float:
        """Compute correlation between audio RMS and lighting brightness."""
        # Prefer provided RMS
        if isinstance(audio_data, dict) and 'rms' in audio_data and audio_data['rms'] is not None:
            rms = np.asarray(audio_data['rms']).flatten()
        else:
            # Fallback: use mel spectrogram
            if isinstance(audio_data, dict) and 'melspe_db' in audio_data and audio_data['melspe_db'] is not None:
                mdb = np.asarray(audio_data['melspe_db'])
                # Convert dB to linear power
                if librosa is not None:
                    mlin = librosa.db_to_power(mdb)
                else:
                    mlin = 10.0 ** (mdb / 10.0)
                rms = np.sqrt(np.mean(np.maximum(mlin, 0.0), axis=1))
            else:
                return 0.0
        
        # Normalize and align
        if rms.size == 0 or light_brightness.size == 0:
            return 0.0
        
        mx = np.max(rms)
        rms = rms / mx if mx > 0 else rms
        L = min(len(rms), len(light_brightness))
        
        if L < 3:
            return 0.0
        
        rms = rms[:L]
        lb = light_brightness[:L]
        
        # Window summary if long enough
        w = self.rms_window_size
        if L // w >= 2:
            r_bins = np.mean(rms[: (L // w) * w].reshape(-1, w), axis=1)
            l_bins = np.mean(lb[: (L // w) * w].reshape(-1, w), axis=1)
            corr = pearsonr(r_bins, l_bins)[0]
            return float(0.0 if np.isnan(corr) else corr)
        
        corr = pearsonr(rms, lb)[0]
        return float(0.0 if np.isnan(corr) else corr)

    def compute_onset_correlation(self, audio_data: dict, light_data: np.ndarray) -> float:
        """Compute correlation between audio onset and lighting changes."""
        if not isinstance(audio_data, dict) or 'onset_env' not in audio_data or audio_data['onset_env'] is None:
            return 0.0
        
        onset = np.asarray(audio_data['onset_env']).flatten()
        if onset.size == 0 or light_data.size == 0:
            return 0.0
        
        # Light change magnitude per frame
        dif = np.zeros(light_data.shape[0], dtype=float)
        if light_data.shape[0] > 1:
            dif[1:] = np.sum(np.abs(light_data[1:] - light_data[:-1]), axis=1)
        
        L = min(len(onset), len(dif))
        if L < 3:
            return 0.0
        
        onset = onset[:L]
        dif = dif[:L]
        
        w = self.onset_window_size
        if L // w >= 2:
            o_bins = np.sum(onset[: (L // w) * w].reshape(-1, w), axis=1)
            d_bins = np.sum(dif[: (L // w) * w].reshape(-1, w), axis=1)
            corr = pearsonr(o_bins, d_bins)[0]
            return float(0.0 if np.isnan(corr) else corr)
        
        corr = pearsonr(onset, dif)[0]
        return float(0.0 if np.isnan(corr) else corr)

    def compute_beat_alignment(self, audio_data: dict, light_brightness: np.ndarray) -> Tuple[float, float]:
        """Compute beat alignment with optional rhythmic intent filtering."""
        if not isinstance(audio_data, dict) or 'onset_beat' not in audio_data or audio_data['onset_beat'] is None:
            return 0.0, 0.0
        
        beats = np.where(np.asarray(audio_data['onset_beat']).flatten() == 1)[0]
        if beats.size == 0 or light_brightness.size == 0:
            return 0.0, 0.0
        
        # Apply rhythmic intent filtering if enabled
        if self.use_rhythmic_filter:
            rhythmic_mask = self.detect_rhythmic_intent(light_brightness)
        else:
            rhythmic_mask = np.ones(len(light_brightness), dtype=bool)
        
        # Find peaks and valleys
        peaks, _ = find_peaks(light_brightness, distance=16, prominence=0.15)
        valleys, _ = find_peaks(1.0 - light_brightness, distance=16, prominence=0.15)
        
        # Filter by rhythmic mask
        rhythmic_peaks = peaks[rhythmic_mask[peaks]] if peaks.size > 0 else np.array([])
        rhythmic_valleys = valleys[rhythmic_mask[valleys]] if valleys.size > 0 else np.array([])
        
        if self.verbose and self.use_rhythmic_filter:
            print(f"    Beat alignment: {len(rhythmic_peaks)}/{len(peaks)} peaks, "
                  f"{len(rhythmic_valleys)}/{len(valleys)} valleys in rhythmic sections")
        
        def score(events: np.ndarray) -> float:
            if events.size == 0:
                return 0.0
            dsum = 0.0
            for e in events:
                d = np.min(np.abs(beats - e)) if beats.size else np.inf
                dsum += np.exp(-(d ** 2) / (2.0 * (self.beat_align_sigma ** 2)))
            return dsum / float(len(events))
        
        return float(score(rhythmic_peaks)), float(score(rhythmic_valleys))

    def compute_variance_metrics(self, intention_array: np.ndarray) -> Tuple[float, float]:
        """Compute variance metrics for intensity and color."""
        if intention_array.size == 0:
            return 0.0, 0.0
        
        intensity = intention_array[:, 0::6]
        psi_intensity = float(np.mean(np.std(intensity, axis=0))) if intensity.size else 0.0
        
        hue = intention_array[:, 4::6]
        sat = intention_array[:, 5::6]
        v_h = float(np.mean(np.std(hue, axis=0))) if hue.size else 0.0
        v_s = float(np.mean(np.std(sat, axis=0))) if sat.size else 0.0
        psi_color = float(np.mean([v_h, v_s]))
        
        return psi_intensity, psi_color

    # ----------- High-level -----------
    def evaluate_single_file(self, audio_path: str | Path, light_path: str | Path) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate a single audio-light pair with all metrics."""
        
        print(f"\nEvaluating: {Path(audio_path).stem}")
        
        # Load data
        audio_data, light_data = self.load_data(audio_path, light_path)
        
        # Fix A: frame alignment (truncate audio arrays to match light frames)
        audio_data = self.align_frames_to_light(audio_data, light_data)
        
        # Extract brightness
        light_brightness = self.extract_brightness(light_data)
        
        # Get the expected frame count for dimension detection
        expected_frames = light_data.shape[0]
        
        # SSMs with proper dimension handling
        print("  Computing SSMs...")
        audio_ssm = self.compute_ssm(
            np.asarray(audio_data.get('chroma_stft', [])), 
            'audio',
            expected_frames=expected_frames
        )
        light_ssm = self.compute_ssm(
            np.asarray(light_data), 
            'light',
            expected_frames=expected_frames
        )
        
        # Fix B: align SSM sizes before downstream ops
        audio_ssm, light_ssm = self.align_ssms(audio_ssm, light_ssm)
        
        # Novelty & boundaries
        print("  Computing novelty functions...")
        audio_nov = self.compute_novelty(audio_ssm)
        light_nov = self.compute_novelty(light_ssm)
        a_peaks, a_bounds = self.find_peaks_and_boundaries(audio_nov)
        l_peaks, l_bounds = self.find_peaks_and_boundaries(light_nov)
        
        # Initialize metrics dictionary
        metrics: Dict[str, float] = {}
        
        # 1. Structure metrics
        print("  Computing structure metrics...")
        if audio_ssm.size and light_ssm.size:
            try:
                ssm_corr = pearsonr(audio_ssm.flatten(), light_ssm.flatten())[0]
            except Exception:
                ssm_corr = 0.0
            metrics['ssm_correlation'] = float(0.0 if np.isnan(ssm_corr) else ssm_corr)
        else:
            metrics['ssm_correlation'] = 0.0
        
        # Novelty correlation
        k = self.L_kernel
        if len(audio_nov) > 2 * k and len(light_nov) > 2 * k:
            try:
                nov_corr = pearsonr(audio_nov[k:-k], light_nov[k:-k])[0]
            except Exception:
                nov_corr = 0.0
            metrics['novelty_correlation'] = float(0.0 if np.isnan(nov_corr) else nov_corr)
        else:
            metrics['novelty_correlation'] = 0.0
        
        # Boundary F-score
        if mir_eval is not None and len(a_bounds) > 0 and len(l_bounds) > 0:
            try:
                p, r, f = mir_eval.segment.detection(a_bounds, l_bounds, window=self.boundary_window)
                metrics['boundary_precision'] = float(p)
                metrics['boundary_recall'] = float(r)
                metrics['boundary_f_score'] = float(f)
            except Exception:
                metrics['boundary_precision'] = 0.0
                metrics['boundary_recall'] = 0.0
                metrics['boundary_f_score'] = 0.0
        else:
            metrics['boundary_precision'] = 0.0
            metrics['boundary_recall'] = 0.0
            metrics['boundary_f_score'] = 0.0
        
        # 2. Dynamic metrics
        print("  Computing dynamic metrics...")
        metrics['rms_correlation'] = self.compute_rms_correlation(audio_data, light_brightness)
        metrics['onset_correlation'] = self.compute_onset_correlation(audio_data, light_data)
        
        # 3. Beat alignment with rhythmic filtering
        print("  Computing beat alignment...")
        bp, bv = self.compute_beat_alignment(audio_data, light_brightness)
        metrics['beat_peak_alignment'] = float(bp)
        metrics['beat_valley_alignment'] = float(bv)
        
        # 4. Variance metrics
        print("  Computing variance metrics...")
        psi_i, psi_c = self.compute_variance_metrics(light_data)
        metrics['intensity_variance'] = float(psi_i)
        metrics['color_variance'] = float(psi_c)
        
        # Print summary
        print("\n  === Metrics Summary ===")
        print(f"  Structure: SSM={metrics['ssm_correlation']:.3f}, Nov={metrics['novelty_correlation']:.3f}, F={metrics['boundary_f_score']:.3f}")
        print(f"  Dynamics: RMS={metrics['rms_correlation']:.3f}, Onset={metrics['onset_correlation']:.3f}")
        print(f"  Beat: Peak={metrics['beat_peak_alignment']:.3f}, Valley={metrics['beat_valley_alignment']:.3f}")
        print(f"  Variance: Int={metrics['intensity_variance']:.3f}, Col={metrics['color_variance']:.3f}")
        
        # Visualization data
        viz = {
            'audio_ssm': audio_ssm,
            'light_ssm': light_ssm,
            'audio_novelty': audio_nov,
            'light_novelty': light_nov,
            'audio_boundaries': a_bounds,
            'light_boundaries': l_bounds,
            'light_brightness': light_brightness,
        }
        
        return metrics, viz


if __name__ == '__main__':
    print('Fixed StructuralEvaluator ready')
    print('Key fixes:')
    print('  1. Proper dimension handling for full song processing')
    print('  2. Rhythmic intent detection for beat alignment')
    print('  3. Verbose output for debugging')
    print('  4. Complete metric computation')