#!/usr/bin/env python
"""
Enhanced Interactive Parameter Tuner V2
- Multiple brightness channels (12 groups + combined)
- Smaller plots (30% size)
- Parameter descriptions
- Audio feature visibility
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.signal import find_peaks
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Make plots smaller globally
plt.rcParams['figure.dpi'] = 60  # Reduced from default 100


class EnhancedRhythmicTunerV2:
    """Interactive tuner V2 with multi-channel support and descriptions."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Beat Alignment & Rhythmic Intent Tuner V2")
        self.root.geometry("1600x1000")
        
        # Data storage
        self.audio_data = None
        self.light_data = None
        self.current_file = None
        
        # Brightness channel selection
        self.brightness_channel = tk.StringVar(value="Combined (All)")
        self.available_channels = ["Combined (All)"] + [f"Group {i+1}" for i in range(12)]
        
        # Parameters with defaults
        self.window_size = tk.IntVar(value=90)
        self.threshold = tk.DoubleVar(value=0.05)
        self.peak_prominence = tk.DoubleVar(value=0.15)
        self.peak_distance = tk.IntVar(value=16)
        self.beat_sigma = tk.DoubleVar(value=0.5)
        
        # Additional parameters
        self.L_kernel = tk.IntVar(value=31)
        self.L_smooth = tk.IntVar(value=81)
        self.H = tk.IntVar(value=10)
        self.rms_window = tk.IntVar(value=120)
        self.onset_window = tk.IntVar(value=120)
        
        # Parameter descriptions
        self.param_descriptions = {
            'window_size': "Rolling window for STD calculation (frames).\nSmaller = more responsive to quick changes.\nLarger = more stable, filters brief variations.",
            'threshold': "Minimum STD to classify as 'rhythmic'.\nLower = more permissive.\nHigher = only strong rhythms.",
            'peak_prominence': "Minimum height of peaks relative to neighbors.\nLower = detect more subtle peaks.\nHigher = only prominent peaks.",
            'peak_distance': "Minimum frames between detected peaks.\nPrevents multiple detections of same peak.",
            'beat_sigma': "Gaussian width for beat alignment scoring.\nSmaller = tighter beat matching required.\nLarger = more forgiving alignment.",
            'L_kernel': "Size of novelty detection kernel.\nAffects sensitivity to structural changes.",
            'L_smooth': "Smoothing filter length for SSM.\nReduces noise in self-similarity matrix.",
            'H': "Downsampling factor.\nReduces computation, 10 = 3fps from 30fps.",
            'rms_window': "Window for RMS correlation (frames).\nFor loudness-brightness correlation.",
            'onset_window': "Window for onset correlation (frames).\nFor change detection correlation."
        }
        
        # Setup UI
        self.setup_ui()
        
        # Initialize empty plot
        self.update_plot()
    
    def setup_ui(self):
        """Create the user interface with improved layout."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # File selection frame
        file_frame = ttk.LabelFrame(left_panel, text="📁 File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="Load Audio PKL", command=self.load_audio).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(file_frame, text="Load Light PKL", command=self.load_light).grid(row=0, column=1, padx=5, pady=2)
        
        self.file_label = ttk.Label(file_frame, text="No files loaded", font=('Arial', 9))
        self.file_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Audio features info
        self.audio_info_label = ttk.Label(file_frame, text="Audio features: Not loaded", 
                                          font=('Arial', 8), foreground='gray')
        self.audio_info_label.grid(row=2, column=0, columnspan=2, pady=2)
        
        # Brightness channel selection
        channel_frame = ttk.LabelFrame(left_panel, text="🔆 Brightness Channel", padding="10")
        channel_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(channel_frame, text="Select Channel:").grid(row=0, column=0, sticky=tk.W)
        channel_combo = ttk.Combobox(channel_frame, textvariable=self.brightness_channel,
                                     values=self.available_channels, state='readonly', width=20)
        channel_combo.grid(row=0, column=1, padx=5)
        channel_combo.bind('<<ComboboxSelected>>', self.on_channel_change)
        
        self.channel_info = ttk.Label(channel_frame, text="", font=('Arial', 8), foreground='blue')
        self.channel_info.grid(row=1, column=0, columnspan=2, pady=2)
        
        # Parameters notebook
        param_notebook = ttk.Notebook(left_panel)
        param_notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Tab 1: Beat Alignment
        beat_frame = ttk.Frame(param_notebook)
        param_notebook.add(beat_frame, text="Beat Alignment")
        self.create_beat_params(beat_frame)
        
        # Tab 2: Advanced
        advanced_frame = ttk.Frame(param_notebook)
        param_notebook.add(advanced_frame, text="Advanced")
        self.create_advanced_params(advanced_frame)
        
        # Tab 3: Help
        help_frame = ttk.Frame(param_notebook)
        param_notebook.add(help_frame, text="Help/Info")
        self.create_help_tab(help_frame)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(left_panel, text="📊 Statistics", padding="10")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=50, font=('Courier', 9))
        self.stats_text.grid(row=0, column=0)
        
        # Save/Export frame
        save_frame = ttk.LabelFrame(left_panel, text="💾 Save & Export", padding="10")
        save_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(save_frame, text="Save for Evaluator", command=self.save_for_evaluator,
                  width=20).grid(row=0, column=0, padx=2)
        ttk.Button(save_frame, text="Load Config", command=self.load_config,
                  width=20).grid(row=0, column=1, padx=2)
        
        self.save_status = ttk.Label(save_frame, text="", foreground="green", font=('Arial', 8))
        self.save_status.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Right panel for plots (smaller)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Create matplotlib figure (30% of original size)
        self.fig, self.axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)  # Reduced from (10, 10)
        self.fig.tight_layout(pad=1.5)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_beat_params(self, parent):
        """Create beat alignment parameter controls with descriptions."""
        
        # Create parameter controls with help buttons
        params = [
            ('window_size', "Rhythmic Window (frames):", 10, 300, self.window_size),
            ('threshold', "STD Threshold:", 0.001, 0.2, self.threshold),
            ('peak_prominence', "Peak Prominence:", 0.01, 0.5, self.peak_prominence),
            ('peak_distance', "Peak Distance (frames):", 5, 60, self.peak_distance),
            ('beat_sigma', "Beat Align Sigma:", 0.1, 2.0, self.beat_sigma)
        ]
        
        for i, (key, label, min_val, max_val, var) in enumerate(params):
            # Label with help button
            label_frame = ttk.Frame(parent)
            label_frame.grid(row=i, column=0, sticky=tk.W, pady=3, padx=5)
            
            ttk.Label(label_frame, text=label).pack(side=tk.LEFT)
            help_btn = ttk.Button(label_frame, text="?", width=2,
                                  command=lambda k=key: self.show_param_help(k))
            help_btn.pack(side=tk.LEFT, padx=2)
            
            # Scale
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var,
                             orient=tk.HORIZONTAL, length=200, command=self.on_param_change)
            scale.grid(row=i, column=1, pady=3)
            
            # Value label
            if key == 'window_size':
                self.window_label = ttk.Label(parent, text=str(var.get()), width=8)
                self.window_label.grid(row=i, column=2, padx=5)
            elif key == 'threshold':
                self.threshold_label = ttk.Label(parent, text=f"{var.get():.3f}", width=8)
                self.threshold_label.grid(row=i, column=2, padx=5)
            elif key == 'peak_prominence':
                self.prom_label = ttk.Label(parent, text=f"{var.get():.3f}", width=8)
                self.prom_label.grid(row=i, column=2, padx=5)
            elif key == 'peak_distance':
                self.dist_label = ttk.Label(parent, text=str(var.get()), width=8)
                self.dist_label.grid(row=i, column=2, padx=5)
            elif key == 'beat_sigma':
                self.sigma_label = ttk.Label(parent, text=f"{var.get():.2f}", width=8)
                self.sigma_label.grid(row=i, column=2, padx=5)
        
        # Presets
        preset_frame = ttk.LabelFrame(parent, text="Quick Presets", padding="5")
        preset_frame.grid(row=6, column=0, columnspan=3, pady=10, padx=5, sticky=(tk.W, tk.E))
        
        presets = [("EDM", 'edm'), ("Pop", 'pop'), ("HipHop", 'hiphop'), ("Ambient", 'ambient')]
        for i, (name, key) in enumerate(presets):
            ttk.Button(preset_frame, text=name, width=8,
                      command=lambda k=key: self.apply_preset(k)).grid(row=0, column=i, padx=2)
    
    def create_advanced_params(self, parent):
        """Create advanced parameter controls."""
        
        # SSM Parameters
        ssm_frame = ttk.LabelFrame(parent, text="SSM Parameters", padding="10")
        ssm_frame.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        ssm_params = [
            ('L_kernel', "Kernel Size:", self.L_kernel, 10, 100),
            ('L_smooth', "Smooth Length:", self.L_smooth, 10, 200),
            ('H', "Downsampling:", self.H, 1, 30)
        ]
        
        for i, (key, label, var, min_val, max_val) in enumerate(ssm_params):
            ttk.Label(ssm_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(ssm_frame, from_=min_val, to=max_val, textvariable=var,
                       width=10, command=self.on_param_change).grid(row=i, column=1, pady=2)
            
            help_btn = ttk.Button(ssm_frame, text="?", width=2,
                                 command=lambda k=key: self.show_param_help(k))
            help_btn.grid(row=i, column=2, padx=5)
        
        # Correlation Windows
        corr_frame = ttk.LabelFrame(parent, text="Correlation Windows", padding="10")
        corr_frame.grid(row=1, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        corr_params = [
            ('rms_window', "RMS Window:", self.rms_window, 30, 300),
            ('onset_window', "Onset Window:", self.onset_window, 30, 300)
        ]
        
        for i, (key, label, var, min_val, max_val) in enumerate(corr_params):
            ttk.Label(corr_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(corr_frame, from_=min_val, to=max_val, textvariable=var,
                       width=10, command=self.on_param_change).grid(row=i, column=1, pady=2)
            
            help_btn = ttk.Button(corr_frame, text="?", width=2,
                                 command=lambda k=key: self.show_param_help(k))
            help_btn.grid(row=i, column=2, padx=5)
    
    def create_help_tab(self, parent):
        """Create help/info tab with descriptions."""
        
        help_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=50, height=20,
                                              font=('Arial', 9))
        help_text.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        help_content = """
BEAT ALIGNMENT TUNER V2 - HELP

=== AUDIO FEATURES USED ===
• onset_beat: Beat positions (1/0 array)
• chroma_stft: For SSM computation
• onset_env: For onset correlation
• rms or melspe_db: For RMS correlation

=== LIGHTING CHANNELS ===
The lighting pickle contains 72 dimensions:
12 groups × 6 parameters each

Each group represents a lighting fixture/zone.
Parameters per group:
1. Intensity Peak (brightness)
2. Slope of Peak Intensity
3. AF Peak Density
4. Intensity of Minima
5. Hue Mean
6. Saturation Mean

"Combined (All)" sums all 12 intensity peaks.
Individual groups let you analyze specific fixtures.

=== WORKFLOW ===
1. Load audio and light pickle files
2. Select brightness channel to analyze
3. Adjust parameters while watching plots
4. Save configuration when satisfied

=== SCORE INTERPRETATION ===
0.8-1.0: Excellent alignment
0.6-0.8: Good alignment
0.4-0.6: Moderate alignment
0.2-0.4: Poor alignment
0.0-0.2: Very poor alignment

=== TIPS ===
• Start with presets for your genre
• Rhythmic % should be 30-70% typically
• Watch the yellow regions in plots
• Red triangles = evaluated peaks
• Gray lines = beat positions
"""
        help_text.insert(1.0, help_content)
        help_text.config(state='disabled')
    
    def show_param_help(self, param_key):
        """Show help popup for a parameter."""
        if param_key in self.param_descriptions:
            messagebox.showinfo(f"Help: {param_key}", self.param_descriptions[param_key])
    
    def apply_preset(self, preset_type):
        """Apply genre-specific presets."""
        presets = {
            'edm': {'window': 60, 'threshold': 0.07, 'sigma': 0.5, 'prominence': 0.15,
                   'info': 'EDM: Tight rhythms, strong beats'},
            'pop': {'window': 90, 'threshold': 0.05, 'sigma': 0.5, 'prominence': 0.15,
                   'info': 'Pop: Mixed rhythmic/melodic sections'},
            'hiphop': {'window': 75, 'threshold': 0.06, 'sigma': 0.7, 'prominence': 0.12,
                      'info': 'Hip-Hop: Strong beat emphasis, some variation'},
            'ambient': {'window': 120, 'threshold': 0.03, 'sigma': 1.0, 'prominence': 0.10,
                       'info': 'Ambient: Subtle variations, loose timing'}
        }
        
        if preset_type in presets:
            p = presets[preset_type]
            self.window_size.set(p['window'])
            self.threshold.set(p['threshold'])
            self.beat_sigma.set(p['sigma'])
            self.peak_prominence.set(p['prominence'])
            self.update_plot()
            self.save_status.config(text=p['info'])
    
    def on_channel_change(self, event=None):
        """Handle brightness channel change."""
        self.update_channel_info()
        self.update_plot()
    
    def update_channel_info(self):
        """Update channel information display."""
        if self.light_data is not None:
            channel = self.brightness_channel.get()
            brightness = self.extract_brightness(self.light_data)
            
            info = f"Mean: {np.mean(brightness):.3f}, Std: {np.std(brightness):.3f}"
            self.channel_info.config(text=info)
    
    def extract_brightness(self, light_data):
        """Extract brightness based on selected channel."""
        channel = self.brightness_channel.get()
        
        if channel == "Combined (All)":
            # Sum all 12 intensity peaks
            brightness = np.sum(light_data[:, 0::6], axis=1)
        else:
            # Extract specific group (0-11)
            group_idx = int(channel.split()[1]) - 1
            brightness = light_data[:, group_idx * 6]  # Get intensity peak for this group
        
        # Normalize
        return brightness / np.max(brightness) if np.max(brightness) > 0 else brightness
    
    def load_audio(self):
        """Load audio pickle file and show features."""
        filename = filedialog.askopenfilename(
            title="Select Audio PKL file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'rb') as f:
                    self.audio_data = pickle.load(f)
                
                self.current_file = Path(filename).stem
                self.update_file_label()
                
                # Show audio features
                features = list(self.audio_data.keys())
                feature_info = f"Features: {', '.join(features[:5])}"
                if len(features) > 5:
                    feature_info += f"... (+{len(features)-5} more)"
                self.audio_info_label.config(text=feature_info)
                
                # Check for required features
                required = ['onset_beat', 'chroma_stft', 'onset_env']
                missing = [f for f in required if f not in self.audio_data]
                if missing:
                    messagebox.showwarning("Missing Features", 
                                          f"Missing: {', '.join(missing)}\nSome metrics may not compute.")
                
                self.update_plot()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {e}")
    
    def load_light(self):
        """Load lighting pickle file."""
        filename = filedialog.askopenfilename(
            title="Select Light PKL file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'rb') as f:
                    self.light_data = pickle.load(f)
                
                # Verify shape
                if self.light_data.shape[1] != 72:
                    messagebox.showwarning("Shape Warning", 
                                          f"Expected 72 dims, got {self.light_data.shape[1]}")
                
                self.update_file_label()
                self.update_channel_info()
                self.update_plot()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load light: {e}")
    
    def update_file_label(self):
        """Update file status label."""
        audio_status = "✓" if self.audio_data is not None else "✗"
        light_status = "✓" if self.light_data is not None else "✗"
        
        if self.current_file:
            name = self.current_file[:25] + "..." if len(self.current_file) > 25 else self.current_file
            self.file_label.config(text=f"Audio: {audio_status} | Light: {light_status}\n{name}")
        else:
            self.file_label.config(text=f"Audio: {audio_status} | Light: {light_status}")
    
    def on_param_change(self, event=None):
        """Handle parameter change."""
        # Update value labels
        self.window_label.config(text=f"{self.window_size.get()}")
        self.threshold_label.config(text=f"{self.threshold.get():.3f}")
        self.prom_label.config(text=f"{self.peak_prominence.get():.3f}")
        self.dist_label.config(text=f"{self.peak_distance.get()}")
        self.sigma_label.config(text=f"{self.beat_sigma.get():.2f}")
        
        self.save_status.config(text="")
        self.update_plot()
    
    def detect_rhythmic_intent(self, brightness):
        """Detect rhythmic intent using rolling STD."""
        series = pd.Series(brightness)
        rolling_std = series.rolling(
            window=self.window_size.get(),
            center=True,
            min_periods=1
        ).std().fillna(0)
        
        rhythmic_mask = rolling_std > self.threshold.get()
        return rhythmic_mask.to_numpy(), rolling_std.to_numpy()
    
    def compute_beat_alignment(self, brightness, beats, rhythmic_mask):
        """Compute beat alignment scores with rhythmic filtering."""
        peaks, _ = find_peaks(brightness, prominence=self.peak_prominence.get(),
                             distance=self.peak_distance.get())
        valleys, _ = find_peaks(1 - brightness, prominence=self.peak_prominence.get(),
                               distance=self.peak_distance.get())
        
        # Filter by rhythmic mask
        rhythmic_peaks = peaks[rhythmic_mask[peaks]] if len(peaks) > 0 else np.array([])
        rhythmic_valleys = valleys[rhythmic_mask[valleys]] if len(valleys) > 0 else np.array([])
        
        # Compute scores
        sigma = self.beat_sigma.get()
        
        peak_score = 0
        for peak in rhythmic_peaks:
            if len(beats) > 0:
                closest_dist = np.min(np.abs(beats - peak))
                peak_score += np.exp(-closest_dist**2 / (2 * sigma**2))
        
        valley_score = 0
        for valley in rhythmic_valleys:
            if len(beats) > 0:
                closest_dist = np.min(np.abs(beats - valley))
                valley_score += np.exp(-closest_dist**2 / (2 * sigma**2))
        
        peak_score = peak_score / len(rhythmic_peaks) if len(rhythmic_peaks) > 0 else 0
        valley_score = valley_score / len(rhythmic_valleys) if len(rhythmic_valleys) > 0 else 0
        
        return peaks, valleys, rhythmic_peaks, rhythmic_valleys, peak_score, valley_score
    
    def update_plot(self):
        """Update visualization with smaller plots."""
        for ax in self.axes:
            ax.clear()
        
        if self.light_data is None:
            self.axes[0].text(0.5, 0.5, 'Load light data to visualize',
                             ha='center', va='center', transform=self.axes[0].transAxes)
            self.canvas.draw()
            return
        
        # Extract brightness for selected channel
        brightness = self.extract_brightness(self.light_data)
        
        # Detect rhythmic intent
        rhythmic_mask, rolling_std = self.detect_rhythmic_intent(brightness)
        
        # Time axis
        time_axis = np.arange(len(brightness)) / 30
        
        # Smaller font for labels
        label_fontsize = 8
        tick_fontsize = 7
        
        # Plot 1: Brightness with rhythmic regions
        ax1 = self.axes[0]
        ax1.plot(time_axis, brightness, 'b-', alpha=0.7, linewidth=0.8)
        ax1.fill_between(time_axis, 0, 1, where=rhythmic_mask,
                         alpha=0.2, color='yellow', label='Rhythmic')
        ax1.set_ylabel('Brightness', fontsize=label_fontsize)
        ax1.set_title(f'Ch: {self.brightness_channel.get()}', fontsize=label_fontsize)
        ax1.legend(loc='upper right', fontsize=tick_fontsize)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=tick_fontsize)
        
        # Plot 2: Rolling STD
        ax2 = self.axes[1]
        ax2.plot(time_axis, rolling_std, 'g-', linewidth=0.8)
        ax2.axhline(y=self.threshold.get(), color='r', linestyle='--', linewidth=0.8,
                   label=f'Thr: {self.threshold.get():.3f}')
        ax2.set_ylabel('STD', fontsize=label_fontsize)
        ax2.set_title('Rolling Standard Deviation', fontsize=label_fontsize)
        ax2.legend(loc='upper right', fontsize=tick_fontsize)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=tick_fontsize)
        
        # Plot 3: Peak/Valley detection
        ax3 = self.axes[2]
        ax3.plot(time_axis, brightness, 'b-', alpha=0.5, linewidth=0.8)
        
        if self.audio_data is not None and 'onset_beat' in self.audio_data:
            beats = np.where(self.audio_data['onset_beat'].flatten() == 1)[0]
            peaks, valleys, r_peaks, r_valleys, p_score, v_score = self.compute_beat_alignment(
                brightness, beats, rhythmic_mask
            )
            
            # Plot peaks/valleys with smaller markers
            ax3.plot(time_axis[peaks], brightness[peaks], 'ro', alpha=0.3, markersize=3)
            ax3.plot(time_axis[valleys], brightness[valleys], 'go', alpha=0.3, markersize=3)
            ax3.plot(time_axis[r_peaks], brightness[r_peaks], 'r^', markersize=5)
            ax3.plot(time_axis[r_valleys], brightness[r_valleys], 'gv', markersize=5)
            
            self.update_statistics(peaks, valleys, r_peaks, r_valleys, 
                                  p_score, v_score, rhythmic_mask)
        else:
            peaks, _ = find_peaks(brightness, prominence=self.peak_prominence.get(),
                                 distance=self.peak_distance.get())
            valleys, _ = find_peaks(1 - brightness, prominence=self.peak_prominence.get(),
                                   distance=self.peak_distance.get())
            
            r_peaks = peaks[rhythmic_mask[peaks]] if len(peaks) > 0 else np.array([])
            r_valleys = valleys[rhythmic_mask[valleys]] if len(valleys) > 0 else np.array([])
            
            ax3.plot(time_axis[peaks], brightness[peaks], 'ro', alpha=0.3, markersize=3)
            ax3.plot(time_axis[valleys], brightness[valleys], 'go', alpha=0.3, markersize=3)
            ax3.plot(time_axis[r_peaks], brightness[r_peaks], 'r^', markersize=5)
            ax3.plot(time_axis[r_valleys], brightness[r_valleys], 'gv', markersize=5)
            
            self.update_statistics(peaks, valleys, r_peaks, r_valleys, 0, 0, rhythmic_mask)
        
        ax3.set_ylabel('Brightness', fontsize=label_fontsize)
        ax3.set_title('Peak/Valley Detection', fontsize=label_fontsize)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=tick_fontsize)
        
        # Plot 4: Beat alignment
        ax4 = self.axes[3]
        if self.audio_data is not None and 'onset_beat' in self.audio_data:
            ax4.plot(time_axis, brightness, 'b-', alpha=0.3, linewidth=0.8)
            
            # Plot beats
            for beat in beats:
                ax4.axvline(x=time_axis[beat], color='gray', alpha=0.3, linestyle=':', linewidth=0.5)
            
            ax4.fill_between(time_axis, 0, 1, where=rhythmic_mask, alpha=0.2, color='yellow')
            
            ax4.set_title(f'Scores: P={p_score:.3f}, V={v_score:.3f}', fontsize=label_fontsize)
        else:
            ax4.text(0.5, 0.5, 'Load audio with onset_beat', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=label_fontsize)
        
        ax4.set_xlabel('Time (s)', fontsize=label_fontsize)
        ax4.set_ylabel('Brightness', fontsize=label_fontsize)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=tick_fontsize)
        
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()
    
    def update_statistics(self, peaks, valleys, r_peaks, r_valleys, 
                         p_score, v_score, rhythmic_mask):
        """Update statistics display."""
        self.stats_text.delete(1.0, tk.END)
        
        stats = []
        stats.append(f"Channel: {self.brightness_channel.get()}\n")
        stats.append(f"{'='*40}\n")
        stats.append(f"Frames: {len(rhythmic_mask)}\n")
        stats.append(f"Rhythmic: {np.sum(rhythmic_mask)} ({100*np.mean(rhythmic_mask):.1f}%)\n")
        stats.append(f"\nAll peaks: {len(peaks)}\n")
        stats.append(f"Rhythmic peaks: {len(r_peaks)} ({100*len(r_peaks)/max(len(peaks),1):.1f}%)\n")
        stats.append(f"All valleys: {len(valleys)}\n")
        stats.append(f"Rhythmic valleys: {len(r_valleys)} ({100*len(r_valleys)/max(len(valleys),1):.1f}%)\n")
        
        if p_score > 0 or v_score > 0:
            stats.append(f"\n{'='*40}\n")
            stats.append(f"BEAT ALIGNMENT SCORES:\n")
            stats.append(f"Peak alignment: {p_score:.4f}\n")
            stats.append(f"Valley alignment: {v_score:.4f}\n")
            stats.append(f"Combined: {(p_score + v_score)/2:.4f}\n")
            
            combined = (p_score + v_score) / 2
            if combined > 0.7:
                rating = "⭐⭐⭐⭐⭐ Excellent"
            elif combined > 0.5:
                rating = "⭐⭐⭐⭐ Good"
            elif combined > 0.3:
                rating = "⭐⭐⭐ Moderate"
            else:
                rating = "⭐⭐ Poor"
            stats.append(f"\nRating: {rating}\n")
        
        self.stats_text.insert(1.0, ''.join(stats))
    
    def save_for_evaluator(self):
        """Save configuration for StructuralEvaluator."""
        
        config = {
            'use_rhythmic_filter': True,
            'rhythmic_window': self.window_size.get(),
            'rhythmic_threshold': self.threshold.get(),
            'beat_align_sigma': self.beat_sigma.get(),
            'peak_distance': self.peak_distance.get(),
            'peak_prominence': self.peak_prominence.get(),
            'L_kernel': self.L_kernel.get(),
            'L_smooth': self.L_smooth.get(),
            'H': self.H.get(),
            'rms_window_size': self.rms_window.get(),
            'onset_window_size': self.onset_window.get(),
            'boundary_window': 2.0,
            'fps': 30,
            'verbose': True,
            'brightness_channel': self.brightness_channel.get(),
            'tuned_on': datetime.now().isoformat(),
            'tuned_with_file': self.current_file if self.current_file else 'unknown'
        }
        
        default_name = f"evaluator_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.save_status.config(text=f"✓ Saved: {Path(filename).name}")
            messagebox.showinfo("Saved", f"Configuration saved!\nChannel: {config['brightness_channel']}")
    
    def load_config(self):
        """Load saved configuration."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Load parameters
                if 'rhythmic_window' in config:
                    self.window_size.set(config['rhythmic_window'])
                if 'rhythmic_threshold' in config:
                    self.threshold.set(config['rhythmic_threshold'])
                if 'beat_align_sigma' in config:
                    self.beat_sigma.set(config['beat_align_sigma'])
                if 'brightness_channel' in config:
                    self.brightness_channel.set(config['brightness_channel'])
                
                self.update_plot()
                self.save_status.config(text=f"✓ Loaded: {Path(filename).name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")


def main():
    """Run the enhanced interactive tuner V2."""
    root = tk.Tk()
    app = EnhancedRhythmicTunerV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()