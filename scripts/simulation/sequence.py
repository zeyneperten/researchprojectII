# Standard library imports
import itertools
import pickle

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm

# =============================================================================
# Main function
# =============================================================================

def simulate_sequences(n_neurons, n_motifs, n_bins, n_sequences,
                     sigma_range, vol_param,
                     corr_mu=False, rho_mu=0.0,
                     corr_sigma=False, rho_sig=0.0,
                     corr_volume=False, rho_vol=0.0,
                     shuffle_order=False, 
                     random_state=0, plot=False, savepath=None):
    """
    Simulate sequences of neuronal activity with given parameters.
    Input: 
        n_neurons: Number of neurons
        n_motifs: Number of clusters
        n_bins: Number of bins for the time grid
        n_sequences: Number of sequences to generate for each cluster
        sigma_range: Range for the standard deviation of the Gaussian PDF.
                     A smaller range makes neurons fire at similar times,
                     while a larger range makes them fire at different times.
                     Increasing the range makes sequences less similar.
        vol_param: Parameters for the Beta distribution for volume
                    a inactive neurons (lower more neurons inactive), if higher seq get longer, mean len shifts  
                    b active (lower more neurons active, if <1, skewed to 1), if lower seqs get longer, mean len shifts
        corr_mu, rho_mu: Correlation and correlation coefficient for mu, 
                    individual neurons fire at similar times across clusters
        corr_sigma, rho_sig: Correlation and correlation coefficient for sigma, 
                    individual neurons have similar variability across clusters
        corr_volume, rho_vol: Correlation and correlation coefficient for volume (activity level), 
                    individual neurons have similar activity level across clusters
        plot: If True, plot the PDFs and CDFs of the neurons in each cluster.
        savepath: If provided, save the simulation data to this path.
    """
    root_rng = np.random.default_rng(random_state)
    rng_params, rng_samples = root_rng.spawn(2)

    # Parameters for the simulation
    mu, sigma, volume = get_mu_sigma_volume(
        n_neurons, n_motifs, rng_params,
        corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sigma_range, vol_param)
    
    # Calculate the PDFs and CDFs for each neuron in each cluster
    densities, cdfs, t = build_pdfs_and_cdfs(n_neurons, n_motifs, n_bins,
                     mu, sigma, volume, plot)
    # Sample spike times and generate sequences
    sequences, spike_times = generate_sequences(n_neurons, n_motifs, n_sequences, cdfs, t, rng=rng_samples, shuffle_order=shuffle_order)

    # Restructure the sequences into a flat list of sequences and their labels
    seqs, seqs_labels, spk_times = restructure(sequences, spike_times)

    # Get the true templates for each cluster based on the mu values
    true_templates = get_true_template(seqs, seqs_labels, mu)

    if savepath:
        save_simulation(seqs, seqs_labels, spk_times, sequences, spike_times, true_templates, 
                     mu, sigma, volume, densities, cdfs,
                     n_neurons, n_motifs, n_bins, n_sequences,
                     sigma_range, vol_param,
                     corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol,
                     shuffle_order, random_state, savepath)

    return seqs, seqs_labels, spk_times, sequences, true_templates, mu, sigma, volume, densities, cdfs


def save_simulation(seqs, seqs_labels, spk_times, sequences, spike_times, true_templates,
                    mu, sigma, volume, densities, cdfs, 
                    n_neurons, n_motifs, n_bins, n_sequences,
                    sigma_range, vol_param,
                    corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol,
                    shuffle_order, random_state, savepath):
    """
    Save the simulation data to a file.
    """
        
    simulation = {}

    simulation["seqs"] = seqs
    simulation["seqs_labels"] = seqs_labels
    simulation["spike_times"] = spk_times
    simulation["seqs_unfiltered"] = sequences
    simulation["true_templates"] = true_templates
    simulation["mu"] = mu
    simulation["sigma"] = sigma
    simulation["volume"] = volume
    simulation["cdfs"] = cdfs
    simulation["densities"] = densities
    simulation["parameters"] = {
                "random_state": random_state,
                "shuffle_order": shuffle_order,
                "n_neurons": n_neurons,
                "n_motifs": n_motifs,
                "n_bins": n_bins,
                "n_sequences": n_sequences,
                "sigma_range": sigma_range,
                "vol_param": vol_param,
                "corr_mu": corr_mu,
                "rho_mu": rho_mu,
                "corr_sigma": corr_sigma,
                "rho_sig": rho_sig,
                "corr_volume": corr_volume,
                "rho_vol": rho_vol
                }
    
    with open(savepath, 'wb') as f:
        pickle.dump(simulation, f)
    print("saved", savepath)


def downsample_sequences(sequences, spike_times, n_neurons_keep):
    """
    Downsample sequences to a fixed number of neurons.
    """
    seqs_downsampled = {
    k: [s for s in ([x for x in seq if x <= n_neurons_keep] for seq in seqs) if s]
    for k, seqs in sequences.items()
    }
    spk_times_downsampled = {
        k: [spk[:n_neurons_keep+1] for spk in seqs]
        for k, seqs in spike_times.items()
    }
    return seqs_downsampled, spk_times_downsampled


def restructure(sequences, spike_times, min_len=5):
    """
    Flatten the sequences dictionary into a list of sequences and their corresponding labels.
    """
    filtered_sequences = {}
    filtered_spiketimes = {}

    for k in sequences.keys():
        filtered_sequences[k] = []
        filtered_spiketimes[k] = []
        for seq, spk in zip(sequences[k], spike_times[k]):
            if len(seq) >= min_len:
                filtered_sequences[k].append(seq)
                filtered_spiketimes[k].append(spk)

    # Flatten
    seqs = list(itertools.chain(*filtered_sequences.values()))
    spk_times = list(itertools.chain(*filtered_spiketimes.values()))
    seq_labels = [cid for cid, seqs_k in filtered_sequences.items() for _ in seqs_k]

        
    return seqs, seq_labels, spk_times


# =============================================================================
# Helper functions
# =============================================================================

# GENERATE PARAMETERS MU, SIGMA AND VOLUME
def get_mu_sigma_volume(n_neurons, n_motifs, rng,
                        corr_mu=False, rho_mu=0.999,
                        corr_sigma=False, rho_sig=0.999,
                        corr_volume=False, rho_vol=0.999,
                        sigma_range=(0.02, 0.5), vol_param=(0.07, 0.9)):
    """
    Generate mu, sigma and volume for each neuron in each cluster.
    """
    # Generate mus, sigmas and volumes
    if corr_mu:
        # CORRELATE NEURONS (across clusters)
        # cluster–to–cluster correlation matrix (n_motifs×n_motifs)
        cov = build_cov(n_motifs, rho=rho_mu)
        L = np.linalg.cholesky(cov)
        # Generate correlated mus
        a, b = 0, 1
        mu = np.empty((n_neurons, n_motifs))
        for i in range(n_neurons):
            z = L @ rng.standard_normal(n_motifs)  # multivariate normal with covariance cov
            u = norm.cdf(z)  # resulting in uniform vector with same pairwise correlations
            mu[i] = a + (b - a) * u  # scaling to [a, b]
    
    else:
        mu = rng.uniform(0, 1, [n_neurons,n_motifs])
    
    if corr_sigma:
        cov = build_cov(n_motifs, rho=rho_sig)
        L = np.linalg.cholesky(cov)
        # Generate correlated mus
        a, b = sigma_range
        sigma = np.empty((n_neurons, n_motifs))
        for i in range(n_neurons):
            z = L @ rng.standard_normal(n_motifs)
            u = norm.cdf(z)
            sigma[i] = a + (b - a) * u
    else:
        sigma = rng.uniform(*sigma_range, [n_neurons,n_motifs])
    
    if corr_volume:
        cov = build_cov(n_motifs, rho=rho_vol)
        # Cholesky factor
        L = np.linalg.cholesky(cov)
        # Generate correlated volumes
        volume = np.empty((n_neurons, n_motifs))
        for i in range(n_neurons):
            z = L @ rng.standard_normal(n_motifs)  # correlated normals
            u = norm.cdf(z)                          # uniform marginals
            volume[i] = beta.ppf(u, *vol_param)      # Beta marginals, remapping unifrom to beta
    else:
        volume = rng.beta(*vol_param, [n_neurons,n_motifs])

    return mu, sigma, volume


def build_cov(n, rho=0.7):
    """
    Build an n×n covariance matrix with:
      cov[i,i] = 1
      cov[i,j] = rho  for i != j
    """
    cov = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    return cov


# BUILD PDFs AND CDFs
def build_pdfs_and_cdfs(n_neurons, n_motifs, n_bins,
                     mu, sigma, volume, plot):
    """
    Build PDFs and CDFs for each neuron in each cluster.
    """
    # Initialize arrays
    densities = np.zeros((n_neurons, n_motifs, n_bins))
    cdfs      = np.zeros_like(densities)
    # time grid
    t = np.linspace(0, 1, n_bins)
    
    for i in range(n_neurons):
        for k in range(n_motifs):
            # Gaussian-shaped PDF
            g = np.exp(-0.5 * ((t - mu[i, k]) / sigma[i, k])**2)
            g /= g.sum()                  # volume = 1
            pdf = g * volume[i, k]        # rescale volume
    
            densities[i, k] = pdf
            cdfs[i, k]      = np.cumsum(pdf)
    
    # plot one cluster's PDFs
    if plot:
        fig, axes = plt.subplots(n_motifs, 2, figsize=(10, 2 * n_motifs), sharex=True)
        for c in range(n_motifs):
            for i in range(n_neurons):
                axes[c,0].plot(t, densities[i,c], alpha=0.3)
                axes[c,1].plot(t, cdfs[i,c], alpha=0.3)
                axes[c,1].set_ylim(0,1)
    return densities, cdfs, t


# GENERATE SEQUENCES
def generate_sequences(n_neurons, n_motifs, n_sequences, cdfs, t, rng=None, shuffle_order=False):
    """
    Generate sequences of neuronal activity based on the given parameters.

    Returns
    -------
    sequences : dict
        For each cluster, a list of sequences (list of neuron indices in firing order).
    spike_times_all : dict
        For each cluster, a list of sequences, where each sequence is a list of length n_neurons,
        and each element is an array of spike times (empty if neuron did not fire).
    """
    # Initialize a dictionary to hold sequences for each cluster
    rng = np.random.default_rng() if rng is None else rng
    sequences = {}
    spike_times_all = {}
    
    for k in range(n_motifs):
        seqs = [] 
        seqs_spiketimes = []
        for _ in range(n_sequences):
        
            # threshold each neuron via CDF, collect candidates
            spike_times = np.full(n_neurons, np.inf)
            spike_times_list = [np.array([], dtype=float) for _ in range(n_neurons)]
            for i in range(n_neurons):
                u = rng.random()
                if u <= cdfs[i, k, -1]:
                    idx = np.searchsorted(cdfs[i, k], u)
                    st = t[idx]
                    spike_times[i] = st
                    spike_times_list[i] = np.array([
                        
                        
                        st])
    
            # get order of neurons
            fired = np.where(np.isfinite(spike_times))[0]
            order = list(fired[np.argsort(spike_times[fired])])
            if shuffle_order and len(order) > 1:
                rng.shuffle(order)
                
            seq = [int(s) for s in order]
            seqs.append(seq)
            seqs_spiketimes.append(spike_times_list)
    
        sequences[k] = seqs
        spike_times_all[k] = seqs_spiketimes
    return sequences, spike_times_all


# =============================================================================
# Templates
# =============================================================================

# CALCULATE TEMPALTE
def get_true_template(seqs, seqs_labels, mu):
    true_templates = []
    # get mu from all neurons that occur in s cluster and make the tempalte sequence
    for c in np.unique(seqs_labels):
        # indices of sequences with in cluster c
        idxs = [i for i, x in enumerate(seqs_labels) if x == c]
        # collect unique active neurons from those sequences
        active_idxs = np.unique([x for i in idxs for x in seqs[i]])
        # get their mu values
        active_mu = mu[active_idxs,c]
        # sort by mu to get template
        order = np.argsort(active_mu)
        temp = active_idxs[order]
        true_templates.append(temp)
    return true_templates


