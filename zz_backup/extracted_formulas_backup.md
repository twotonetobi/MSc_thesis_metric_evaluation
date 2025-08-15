# Matrix for Intention-Based Evaluation - Extracted Formulas

## 1. Self-Similarity Matrix (SSM) Computation

### 1.1 Audio SSM from Chroma Features
For audio features using chroma_stft:

$$S_{audio}(i,j) = 1 - \frac{||C_i - C_j||_2}{\sqrt{d}}$$

Where:
- $C_i$ is the chroma vector at frame $i$
- $d$ is the dimensionality (12 for chroma)
- Features are smoothed with filter length $L_{smooth} = 81$
- Downsampled by factor $H = 10$

### 1.2 Lighting SSM from Intention Features
For lighting intention array (72 dimensions):

$$S_{light}(i,j) = 1 - \frac{||I_i - I_j||_2}{\sqrt{d}}$$

Where:
- $I_i$ is the intention vector at frame $i$ (72 dimensions)
- Apply same smoothing and downsampling

## 2. Novelty Function

### 2.1 Gaussian Checkerboard Kernel
$$K(i,j) = \text{sign}(i) \cdot \text{sign}(j) \cdot \exp\left(-\frac{i^2 + j^2}{2(L \cdot \sigma)^2}\right)$$

Where:
- $L = 31$ (kernel size parameter, resulting in $(2L+1) \times (2L+1)$ kernel)
- $\sigma = 0.5$ (variance parameter)

### 2.2 Novelty Computation
$$\text{nov}(n) = \sum_{i,j} S_{padded}[n-L:n+L+1, n-L:n+L+1] \odot K$$

Where $\odot$ denotes element-wise multiplication.

### 2.3 Peak Detection
Peaks are detected with:
- `distance = 15` frames minimum between peaks
- `prominence = 0.04` minimum prominence value

## 3. RMS Correlation ($\Gamma_{loud \leftrightarrow bright}$)

### 3.1 Audio RMS Computation
$$\text{RMS}_{audio}(n) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$

Computed over windows and normalized.

### 3.2 Lighting Brightness
$$B_{light} = \sum_{g=1}^{12} I_{g,1}$$

Where $I_{g,1}$ is the intensity peak (parameter 1) for group $g$.

### 3.3 Correlation
$$\Gamma_{RMS} = \text{Pearson}(\text{RMS}_{audio}, B_{light})$$

With window size = 120 frames (4 seconds at 30fps)

## 4. Onset Correlation ($\Gamma_{change}$)

### 4.1 Lighting Change Detection
$$\Delta L(t) = ||L(t) - L(t-1)||$$

### 4.2 Correlation with Onset Envelope
$$\Gamma_{onset} = \text{Pearson}(\text{onset\_env}, \Delta L)$$

With window size = 120 frames

## 5. Context-Aware Beat Alignment Score

### 5.1 Rhythmic Intent Detection
Before computing beat alignment, identify rhythmic sections using brightness variation:

$\text{STD}_{rolling}(t) = \sqrt{\frac{1}{w} \sum_{i=t-w/2}^{t+w/2} (B_i - \bar{B}_w)^2}$

Where:
- $B_i$ is the brightness at frame $i$
- $\bar{B}_w$ is the mean brightness in window $w$
- $w = 90$ frames (3 seconds at 30fps)

Rhythmic mask:
$M_{rhythmic}(t) = \begin{cases}
1 & \text{if } \text{STD}_{rolling}(t) > \tau \\
0 & \text{otherwise}
\end{cases}$

Where $\tau$ is the threshold (typically 0.03-0.08, requires tuning).

### 5.2 Filtered Beat-to-Peak Alignment ($\Gamma_{beat \leftrightarrow peak}$)
For each detected peak $p$ in lighting brightness WHERE $M_{rhythmic}(p) = 1$:

$\text{score}_{peak} = \sum_{p \in P_{rhythmic}} \exp\left(-\frac{d(p, \text{nearest\_beat})^2}{2\sigma^2}\right)$

Where:
- $P_{rhythmic} = \{p \in P : M_{rhythmic}(p) = 1\}$ (filtered peaks)
- $d(p, b)$ is the distance in frames from peak to nearest beat
- $\sigma = 0.5$ (beat alignment sigma)

### 5.3 Filtered Beat-to-Valley Alignment ($\Gamma_{beat \leftrightarrow valley}$)
Same formula applied to valleys in inverted signal, filtered by rhythmic mask:

$\text{score}_{valley} = \sum_{v \in V_{rhythmic}} \exp\left(-\frac{d(v, \text{nearest\_beat})^2}{2\sigma^2}\right)$

## 6. Structure Metrics

### 6.1 SSM Correlation ($\Gamma_{structure}$)
$$\Gamma_{structure} = \text{Pearson}(S_{audio}.flatten(), S_{light}.flatten())$$

### 6.2 Novelty Correlation ($\Gamma_{novelty}$)
$$\Gamma_{novelty} = \text{Pearson}(\text{nov}_{audio}[k:-k], \text{nov}_{light}[k:-k])$$

Where $k = \max(L_{audio}, L_{light})$ to exclude edge effects.

### 6.3 Boundary Detection F-Score ($\Gamma_{boundary}$)
Using mir_eval with window = 2 seconds:
$$F = \frac{2 \cdot P \cdot R}{P + R}$$

Where $P$ is precision and $R$ is recall for boundary detection.

## 7. Variance Metrics ($\Psi$)

### 7.1 Intensity Variance
$$\Psi_{intensity} = \frac{1}{G} \sum_{g=1}^{G} \text{std}(I_{g,1})$$

### 7.2 Color Variance
$$\Psi_{color} = \text{mean}(\max(\text{std}(H), \text{std}(S)))$$

Where $H$ and $S$ are hue and saturation from parameters 5 and 6.

### 7.3 Position Variance (if applicable)
$$\Psi_{pan}, \Psi_{tilt} = \text{std}(\text{pan}), \text{std}(\text{tilt})$$

## 8. Aggregate Metrics

### 8.1 Overall Structural Coherence
$$\text{Coherence} = w_1 \cdot \Gamma_{structure} + w_2 \cdot \Gamma_{novelty} + w_3 \cdot \Gamma_{boundary}$$

### 8.2 Rhythmic Alignment
$$\text{Rhythm} = w_4 \cdot \Gamma_{beat \leftrightarrow peak} + w_5 \cdot \Gamma_{beat \leftrightarrow valley}$$

### 8.3 Dynamic Response
$$\text{Dynamics} = w_6 \cdot \Gamma_{loud \leftrightarrow bright} + w_7 \cdot \Gamma_{change}$$

## Parameter Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| L_kernel | 31 | Novelty kernel size |
| L_smooth | 81 | SSM smoothing filter length |
| H | 10 | Downsampling factor |
| σ_beat | 0.5 | Beat alignment sigma |
| window_rms | 120 | RMS correlation window (frames) |
| window_onset | 120 | Onset correlation window (frames) |
| peak_distance | 15 | Minimum frames between peaks |
| peak_prominence | 0.04 | Minimum peak prominence |
| boundary_window | 2.0 | Boundary detection window (seconds) |