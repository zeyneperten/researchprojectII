# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize


def plot_sequence_statistics(sequences, n_neurons=75):
    n_clusters = len(sequences)
    
    # 1. Compute sequence lengths and activation frequencies
    length_distributions = []
    activation_freq = np.zeros((n_clusters, n_neurons))
    
    for k, seqs in sequences.items():
        # collect lengths
        lengths = [len(seq) for seq in seqs]
        length_distributions.append(lengths)
        
        # count activations
        counts = np.zeros(n_neurons)
        for s in seqs:
            unique_cells = np.unique(s)  # avoid counting duplicates within a sequence
            for cell in unique_cells:
                counts[cell] += 1
        activation_freq[k, :] = counts 
    
    # 2. Plot histograms of sequence lengths per cluster
    fig1, axes = plt.subplots(5, 4, figsize=(7, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    max_len = max(max(lens) for lens in length_distributions)
    for k, ax in enumerate(axes[:n_clusters]):
        ax.hist(length_distributions[k], bins=range(1, max_len + 2), edgecolor='black')
        ax.set_title(f'Motif {k}')
        # Add mean in the upper right corner
        mean_val = np.mean(length_distributions[k])
        ax.text(
            0.95, 0.95, f"mean={mean_val:.2f}", 
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    fig1.suptitle('Distribution of Sequence Lengths per Cluster')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3. Plot heatmap of neuron activation frequencies
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    im = ax2.imshow(activation_freq, aspect='auto', origin='lower')
    ax2.set_xlabel('Neuron Index')
    ax2.set_ylabel('Motifs ID')
    ax2.set_title('Neuron Activation Frequency per Cluster')
    fig2.colorbar(im, ax=ax2, label='Activation Frequency')
    
    plt.show()
    

def plot_cluster_composition(
    y_true,
    y_pred,
    normalize=True,       # True → fractions per predicted cluster; False → raw counts
    sort="size",          # "size" (desc) or "label"
    include_noise=True,   # set False to drop y_pred == -1
    return_table=False,   # return a DataFrame with the composition
    figsize=(5, 3),
    palette="hsv",      # palette for true labels
    n_colorbar_ticks=6,    # how many labels to show on the colorbar (approx.)
    ratios=None
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if not include_noise:
        mask = (y_pred != -1)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # unique labels in deterministic order
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # contingency: rows=true labels, cols=predicted cluster labels
    cm = pd.crosstab(
        pd.Categorical(y_true, categories=true_labels),
        pd.Categorical(y_pred, categories=pred_labels),
        dropna=False
    ).to_numpy()

    # per-cluster sizes (columns)
    col_sizes = cm.sum(axis=0)
    if normalize:
        comp = np.divide(cm, col_sizes, where=col_sizes > 0, out=np.zeros_like(cm, dtype=float))
        ylab = "Fraction of cluster"
    else:
        comp = cm.astype(float)
        ylab = "# Sequences"

    # sort clusters
    if sort == "size":
        order = np.argsort(col_sizes)[::-1]  # largest first
    elif sort == "label":
        order = np.arange(len(pred_labels))
    else:
        raise ValueError("sort must be 'size' or 'label'.")

    comp = comp[:, order]
    col_sizes = col_sizes[order]
    pred_labels_sorted = pred_labels[order]

    # --- colors for TRUE labels (rows) ---
    n_true = len(true_labels)
    # distinct palette sized to number of true labels
    colors = np.array(sns.color_palette(palette, n_colors=n_true))
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=-0.5, vmax=n_true - 0.5)

    # stacked bars
    x = np.arange(len(pred_labels_sorted))
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_prop_cycle(color=sns.color_palette("hsv", 20))
    bottom = np.zeros(len(pred_labels_sorted), dtype=float)
    for i, t_lab in enumerate(true_labels):
        ax.bar(x, comp[i, :], bottom=bottom, color=colors[i], edgecolor="none")
        bottom += comp[i, :]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{pl}" for pl, n in zip(pred_labels_sorted, col_sizes)], #(n={n})
                       rotation=45, ha="right")
    ax.set_ylabel(ylab)
    ax.set_xlabel("Predicted clusters")
    #ax.grid(axis="y", linestyle="--", alpha=0.5)

    # --- concise discrete colorbar instead of legend ---
    # choose a subset of ticks to label
    if n_true <= n_colorbar_ticks:
        tick_idx = np.arange(n_true, dtype=int)
    else:
        # spread ~n_colorbar_ticks indices across the range, unique & sorted
        tick_idx = np.unique(np.linspace(0, n_true - 1, n_colorbar_ticks, dtype=int))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required by mpl for colorbar
    cbar = fig.colorbar(sm, ax=ax, pad=0.15, ticks=tick_idx)
    cbar.ax.set_yticklabels([str(true_labels[i]) for i in tick_idx])
    cbar.set_label("Motifs")

    ratios_aligned = None
    if ratios is not None and len(ratios) > 0:
        ratios = np.asarray(ratios).ravel()
        if ratios.size == pred_labels.size:
            ratios_aligned = ratios[order]
        elif ratios.size == pred_labels.size - int(np.any(pred_labels == -1)):
            ratios_full = np.full(pred_labels.size, np.nan, dtype=float)
            ratios_full[pred_labels != -1] = ratios
            ratios_aligned = ratios_full[order]

    if ratios_aligned is not None:
        ax2 = ax.twinx()
        ax2.plot(range(len(ratios_aligned)), ratios_aligned, color='k', marker='.', linestyle='-', linewidth=1)
        ax2.set_ylabel('Ratio score',color='k')
        ax2.tick_params(axis='y', labelcolor='k')

    fig.tight_layout()

    if return_table:
        comp_df = pd.DataFrame(comp.T, index=pred_labels_sorted, columns=true_labels)
        comp_df.insert(0, "cluster_size", col_sizes)
        return comp_df