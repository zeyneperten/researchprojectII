# A Parametric Statistical Model for Neural Sequence Simulation & Benchmarking

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Dependency Manager: Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Research Project II**  
> **Author:** Zeynep Erten  
> **Supervisor:** Prof. Dr. Christian Leibold  
> **Affiliation:** Albert-Ludwigs-Universität Freiburg  

---

## 📌 Overview

During active navigation and subsequent slow-wave sleep (SWS), the mammalian hippocampus coordinates population-level spike sequences (e.g., place cell trajectories and sharp-wave ripple [SWR] replay) that are crucial for memory consolidation. While unsupervised sequence detection algorithms (such as graph community detection on rank-order correlation graphs) are increasingly employed to detect these patterns without behavioral templates, **evaluating their sensitivity and false-positive rates remains difficult due to the absence of ground-truth labels in experimental recordings**.

This repository provides:
1. **A Parametric Generative Sequence Model:** Synthesizes artificial neural spike trains with ground-truth motifs governed by interpretable biophysical parameters (neuronal latency $\mu$, temporal dispersion $\sigma$, and participation probability $v$).
2. **A Downsampling Evaluation Pipeline:** Models the biological reality of multi-electrode array recordings by randomly subsampling small neuronal subsets ($R = 100$) from dense populations ($N = 100,000$) to evaluate sequence retention and structural invariance.
3. **Hierarchical Cell Assembly Sequence Generation:** Generates multi-level temporal structures (assembly-level order and neuron-level spike times) across three distinct temporal integration architectures.
4. **Unsupervised Sequence Clustering & Validation:** Implements pairwise Spearman rank correlation $Z$-score matrices, null-shuffling controls against chance co-firings, and Leiden community detection with rigorous intra- and inter-cluster validation criteria.

---

## 🔬 Generative Model Architecture

For each neuron $i \in \{1, \dots, N\}$ and sequence motif $k \in \{1, \dots, K\}$, the firing behavior is controlled by three physiological parameters:

$$\tilde{f}_{ik}(t) = \exp\left(-\frac{(t - \mu_{ik})^2}{2\sigma_{ik}^2}\right), \quad t \in [0, 1]$$

$$f_{ik}(t) = v_{ik} \frac{\tilde{f}_{ik}(t)}{\int_0^1 \tilde{f}_{ik}(u)\,du}, \quad \int_0^1 f_{ik}(t)\,dt = v_{ik} \in (0, 1)$$

- **$\mu_{ik}$ (Latency / Preferred Firing Time):** Sampled from $\mathcal{U}(0, 1)$, dictating the relative temporal recruitment position of the neuron within the motif.
- **$\sigma_{ik}$ (Temporal Dispersion / Jitter):** Sampled uniformly from $[\sigma_{\min}, \sigma_{\max}]$, controlling the temporal spread and spike jitter. When $\sigma = (1, 1)$, temporal order is completely abolished (pure noise regime).
- **$v_{ik}$ (Participation Probability / Volume):** Sampled from a Beta distribution $\text{Beta}(a, b)$, governing sequence length and network sparsity. An optional correlation coefficient ($\rho$) allows neurons to maintain consistent activity across motifs.

Spike times are discretized across $B = 100$ normalized time bins using a two-step sampling process: participation is determined via $u < v_{ik}$, and spike latency is sampled from the conditional distribution $t_{ik} \sim f_{ik}(t) / v_{ik}$.

---

## 📊 Key Findings

### 1. Invariance to Extreme Downsampling
- **Retention Proxy:** When subsampling $R = 100$ neurons from $N = 100,000$ across 20 motifs (over 50 independent trials), motif sequence retention is dictated by the volume parameter ($v_{ik}$). Mean sequence participation per neuron serves as a strong structural proxy predicting the $50\%$ sequence retention threshold ($R_{50\%}$).
- **No Spurious Motifs:** In the unstructured regime ($\sigma = (1, 1)$), the clustering pipeline extracts zero false-positive clusters. Severe downsampling reduces the sequence yield but does **not** introduce artificial patterns or alter the underlying geometry.

### 2. Hierarchical Cell Assembly Sequences
The model evaluates three distinct integration methods for multi-layered assembly dynamics:
- **Method 1 (Append):** Subsequences of distinct neurons appended sequentially.
- **Method 2 (Scaled Offset):** Subsequences of distinct neurons mapped onto the global interval with temporal overlaps.
- **Method 3 (Shared/Overlap):** Subsequences of shared neuron IDs with temporal overlaps.

Evaluating cluster homogeneity (via Adjusted Rand Index, ARI) across noise regimes revealed that abolishing assembly-level sequence order has a more drastic effect on correlation matrices than removing order within individual assemblies.

---

## 📁 Repository Structure

```text
├── analysis.ipynb                  # Experimental analysis & validation on rodent sleep replay data
├── analysis_correlation.ipynb      # Correlation matrix sweeps across varying noise regimes
├── downsample.ipynb                # Large-scale downsampling benchmarks (100k -> 100 neurons)
├── data/
│   ├── rat944_sleep_center.pkl     # Preprocessed hippocampal recording (Rat 944)
│   └── rat987_sleep_center.pkl     # Preprocessed hippocampal recording (Rat 987)
├── pyproject.toml                  # Project metadata and dependencies (Poetry)
├── poetry.lock                     # Lockfile for reproducible environment
└── scripts/
    ├── config.py                   # Central simulation & clustering configurations
    ├── data_utils.py               # Data loading, filtering, and serialization utilities
    ├── simulation/
    │   ├── sequence.py             # Core parametric generative model & assembly generator
    │   ├── parameter_tuning.py     # Parameter sweeps across alpha, beta, sigma, and rho
    │   ├── rank_correlation.py     # Pairwise Spearman rank correlation matrices & Z-scores
    │   └── correlation_mean.py     # Mean correlation calculations across parameter sets
    ├── clustering/
    │   ├── core.py                 # Clustering coordination & pipeline entrypoints
    │   ├── leiden.py               # Graph-based Leiden community detection
    │   ├── distances.py            # Pairwise distance metrics (Jaccard, rank correlation)
    │   ├── shuffling.py            # Null model shuffling controls against chance co-activity
    │   ├── evaluation_helpers.py   # Intra-cluster and inter-cluster separation metrics
    │   └── parameter_tuning.py     # Cluster grid search and threshold optimization
    ├── analysis/
    │   └── analysis.py             # Downsampling routines and empirical data processing
    └── visualization/
        ├── plots_simulation.py     # Raster plots, Beta distributions, and parameter sweeps
        ├── plots_clustering.py     # Z-score correlation heatmaps and cluster diagnostics
        ├── plots_analysis.py       # Sequence retention curves and downsample stability plots
        ├── plots_raw.py            # Raw spike raster and sequence trajectory plots
        ├── plots_helpers.py        # Shared formatting & plotting helper functions
        └── style.py                # Publication-quality Matplotlib styling
```

---

## 🚀 Getting Started

### Prerequisites
- Python $\ge$ 3.12
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

Clone the repository and install dependencies with Poetry:

```zsh
git clone git@github.com:zeyneperten/researchprojectII.git
cd researchprojectII
poetry install
```

To activate the virtual environment:

```zsh
poetry shell
```

Or run commands directly via `poetry run`:

```zsh
poetry run jupyter lab
```

---

## 💻 Usage Example

```python
from scripts.simulation.sequence import generate_synthetic_data
from scripts.clustering.core import run_sequence_clustering

# 1. Generate synthetic sequences with ground-truth motifs
data = generate_synthetic_data(
    n_neurons=100,
    n_motifs=20,
    n_sequences_per_motif=100,
    sigma_range=(0.02, 0.4),   # Temporal dispersion
    volume_params=(0.07, 0.9), # Beta(a, b) participation
    volume_correlation=0.4     # Cross-motif consistency (rho)
)

# 2. Run the unsupervised clustering & validation pipeline
clusters, valid_motifs = run_sequence_clustering(
    sequences=data["sequences"],
    intra_cluster_thresh=0.4,
    inter_cluster_ratio=7.0,
    n_shuffles=30
)

print(f"Identified {len(valid_motifs)} statistically valid sequence motifs.")
```

---

## 📖 Citation & References

1. **Ackermann, E., Kemere, C., Maboudi, K., & Diba, K.** (2017). *Latent Variable Models for Hippocampal Sequence Analysis.* IEEE.
2. **Diba, K., & Buzsáki, G.** (2007). *Forward and reverse hippocampal place-cell sequences during ripples.* Nature Neuroscience, 10, 1241–1242.
3. **Foster, D. J., & Wilson, M. A.** (2006). *Reverse replay of behavioural sequences in hippocampal place cells during the awake state.* Nature, 440, 680–683.
4. **Harris, K. D., Csicsvari, J., Hirase, H., Dragoi, G., & Buzsáki, G.** (2003). *Organization of cell assemblies in the hippocampus.* Nature, 424, 552–556.
5. **Lee, A. K., & Wilson, M. A.** (2002). *Memory of sequential experience in the hippocampus during slow wave sleep.* Neuron, 36, 1183–1194.
