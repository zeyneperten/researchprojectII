# Standard library imports
import itertools
import pickle
from functools import partial

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm
from collections import defaultdict

# =============================================================================
# Main function
# =============================================================================

def simulate_sequences(n_neurons, n_motifs, n_bins, n_sequences, n_assemblies, n_a_motifs, n_a_seqs,
                     sigma_range, vol_param, sig_a, vol_a,
                     corr_mu=False, rho_mu=0.0,
                     corr_sigma=False, rho_sig=0.0,
                     corr_volume=False, rho_vol=0.0,
                     shuffle_order=False, 
                     random_state=0, plot=False, savepath=None, batch_size=None, return_subsequences=False):
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
        batch_size: If provided, process neurons in batches to reduce memory usage.
                    Recommended: 1000-5000 for large n_neurons.
    """
    root_rng = np.random.default_rng(random_state)
    rng_params, rng_samples = root_rng.spawn(2)

    # Parameters for the simulation
    mu, sigma, volume = get_mu_sigma_volume(
        n_neurons, n_motifs, rng_params,
        corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sigma_range, vol_param)
    
    # Calculate the PDFs and CDFs for each neuron in each cluster
    if batch_size is not None and n_neurons > batch_size:
        densities, cdfs, t = build_pdfs_and_cdfs_batched(n_neurons, n_motifs, n_bins,
                         mu, sigma, volume, batch_size, plot)
    else:
        densities, cdfs, t = build_pdfs_and_cdfs_vectorized(n_neurons, n_motifs, n_bins,
                         mu, sigma, volume, plot)
    # Sample spike times and generate sequences

    
    if batch_size is not None and n_neurons > batch_size:
        sequences, spike_times = generate_sequences_batched(n_neurons, n_motifs, n_sequences, 
                                                                mu, sigma, volume, t, batch_size, rng=rng_samples, shuffle_order=shuffle_order)
    else:
        sequences, spike_times = generate_sequences(n_neurons, n_motifs, n_sequences, cdfs, t, rng=rng_samples, shuffle_order=shuffle_order)

    # Restructure the sequences into a flat list of sequences and their labels
    seqs, seqs_labels, spk_times = restructure(sequences, spike_times, min_len=1)

    # Get the true templates for each cluster based on the mu values
    true_templates = get_true_template(seqs, seqs_labels, mu)

    if savepath:
        save_simulation(seqs, seqs_labels, spk_times, sequences, spike_times, true_templates, 
                     mu, sigma, volume, densities, cdfs,
                     n_neurons, n_motifs, n_bins, n_sequences,
                     sigma_range, vol_param,
                     corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol,
                     shuffle_order, random_state, savepath)

    if return_subsequences:
       
        offset_seqs = []

        for i, motif_id in enumerate(seqs_labels):
            id_offset = motif_id * 100  # Motif 0 -> 0, Motif 1 -> 100, Motif 2 -> 200, etc.
            
            offsets = []
            for n in seqs[i]:
                n += id_offset
                offsets.append(n)
            offset_seqs.append(offsets)
            
        offset_true_templates = []
        for i, c in enumerate(np.unique(seqs_labels)):
            id_offset = c * 100
            # Apply the exact same offset to the pre-calculated base templates
            offset_temp = [n + id_offset for n in true_templates[i]]
            offset_true_templates.append(offset_temp)
    
        assembly_n_seq, assembly_n_spk, assembly_n_labels, subsequences_dict, subspike_times, raw_assembly_neurons_seq = subsequences(n_neurons, n_motifs, n_bins, n_assemblies, n_sequences, n_a_motifs, n_a_seqs, corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sig_a, vol_a, sigma_range, vol_param, random_state, plot, shuffle_order, mu, sigma, volume, densities, cdfs, t)
        
        return offset_seqs, seqs_labels, spk_times, sequences, offset_true_templates, mu, sigma, volume, densities, cdfs, assembly_n_seq, assembly_n_spk, assembly_n_labels, subsequences_dict, subspike_times, raw_assembly_neurons_seq
        
    else:
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
    simulation["sequences"] = sequences
    simulation["spike_times"] = spike_times
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


def downsample_sequences(sequences, spike_times, volume, n_neurons, n_neurons_keep, min_length=0, random_state=None):
    """
    Downsample sequences to a fixed number of neurons by randomly selecting neuron indices.
    
    Parameters:
    -----------
    sequences : dict
        Dictionary mapping cluster ID to list of sequences
    spike_times : list
        Flat list of spike times arrays
    volume : ndarray
        Shape (n_neurons, n_motifs) array
    n_neurons_keep : int
        Number of neurons to keep (randomly selected)
    min_length : int, optional
        Minimum length for a sequence to be kept. Sequences shorter than this are filtered out.
        Default is 0 (no filtering).
    random_state : int, np.random.RandomState, or None, optional
        Random seed for reproducibility. If int, creates a new RandomState with that seed.
        If None, uses unseeded random state (different results each time). Default is None.
    
    Returns:
    --------
    seqs_downsampled : dict
        Downsampled sequences (filtered by min_length)
    spk_times_downsampled : list
        Downsampled spike times
    volume_downsampled : ndarray
        Downsampled volume
    """
    # Create random number generator
    if isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState()  # Unseeded, new random state each call
    
    # Randomly select n_neurons_keep indices from all neurons
    neurons_to_keep = rng.choice(n_neurons, size=n_neurons_keep, replace=False)
    neurons_to_keep_sorted = np.sort(neurons_to_keep)  # Sort for proper indexing
    neurons_to_keep_set = set(neurons_to_keep)

    seqs_downsampled = {
        k: [s for s in ([x for x in seq if x in neurons_to_keep_set] for seq in seqs) 
            if s and len(s) >= min_length]
        for k, seqs in sequences.items()
    }

    # Downsample spike times and volume using selected indices
    # Build spike times list by indexing each element separately (handles ragged arrays)
    
    volume_downsampled = None if volume is None or volume.size == 0 else volume[neurons_to_keep_sorted, :]

    spk_times_downsampled = None if spike_times is None or len(spike_times) == 0 else [
        [spk[i] for i in neurons_to_keep_sorted]
        for spk in spike_times
        ]

    # Downsample spike times (list -> slice each array)
    #spk_times_downsampled = [spk[:n_neurons_keep+1] for spk in spike_times]
    
    # Downsample volume (ndarray -> slice rows)
    #volume_downsampled = volume[:n_neurons_keep, :]
    
    return seqs_downsampled, spk_times_downsampled, volume_downsampled



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

# SUBSEQUENCES
def subsequences(n_neurons, n_motifs, n_bins, n_assemblies, n_sequences, n_a_motifs, n_a_seqs,
                 corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, 
                 sig_a, vol_a, sig_n, vol_n, 
                 random_state, plot, shuffle_order, n_mu, n_sigma, n_volume, n_densities, n_cdfs, n_t):
    root_rng = np.random.default_rng(random_state)
    rng_params, rng_samples, rng_neural_params, rng_neural_draws = root_rng.spawn(4)

    # --- STEP 1: ASSEMBLY MOTIF LEVEL  ---
    a_mu, a_sigma, a_volume = get_mu_sigma_volume(
        n_assemblies, n_a_motifs, rng_params,
        corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sig_a, vol_a)
    
    a_densities, a_cdfs, a_t = build_pdfs_and_cdfs(n_assemblies, n_a_motifs, n_bins,
                         a_mu, a_sigma, a_volume, plot)
    
    subsequences_dict, subspike_times = generate_sequences(n_assemblies, n_a_motifs, n_a_seqs, a_cdfs, a_t, rng=rng_samples, shuffle_order=shuffle_order)

    # --- STEP 2: NEURAL SEQUENCE LEVEL ---
    flat_occ_n_seq = []
    flat_occ_n_spk = []
    flat_occ_labels = [] 
    raw_assembly_neurons_seq = defaultdict(lambda: defaultdict(list)) # keys are assembly motif ids
    
    # Sort motifs by ID
    for a_id, motifs in sorted(subsequences_dict.items()):
        for seq_idx, m in enumerate(motifs):
            current_spike_times = subspike_times[a_id][seq_idx]
            
            # Sort the assembly IDs in this motif instance by their start time
            sorted_m = sorted(m, key=lambda a: current_spike_times[a][0] if len(current_spike_times[a]) > 0 else float('inf'))
            
            L = len(sorted_m)
            if L == 0:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
                continue
                
            window = 1.0 / L  
            occurrence_dict = {}  # firing time of each neuron within that assembly motif. firing times of the neuron across different assemblies are appended
            captured_local_data = [] # A list of tuples: (assembly_id, spikes, start_time)
            
            for a in sorted_m:
                if a < n_assemblies:
                    # Draw EXACTLY ONE unique neural sequence for this specific sequence of assembly 'a'
                    # Pass the pre-computed CDF for this specific assembly 'a'
                    neuron_seq_dict, neuron_spk_dict = generate_sequences(
                        n_neurons, 1, 1, 
                        cdfs=n_cdfs[:, [a], :], t=n_t, 
                        rng=rng_neural_draws, shuffle_order=shuffle_order
                    )
                    
                    # Because n_seqs=1, the data is stored at key [0]
                    # {motif_id: [ [neuron_ids], ... ]}
                    # Since we only have 1 motif (the assembly itself), we grab the first item
                    current_neuron_ids = neuron_seq_dict[0][0]
                    current_instance_spikes = neuron_spk_dict[0][0]

                    raw_assembly_neurons_seq[a_id][a].append(current_neuron_ids)
                    
                    # --- TEMPORAL ORDERING OF NEURAL SEQ WITHIN ASSEMBLY SEQ ---
                    assembly_time = current_spike_times[a][0] if len(current_spike_times[a]) > 0 else 0.0
                    
                    if L == 1:
                        start_time = 0.0
                    else:
                        ideal_start = assembly_time - (window / 2) # the window is positioned around assebmly time
                        start_time = np.clip(ideal_start, 0.0, 1.0 - window) # if ideal_start < 0, start from 0 | if ideal time is above the max possible time, the sequence will go beyond 1, 
                                                                             # in that case start from the max possible time
                    captured_local_data.append((a, current_neuron_ids.copy(), current_instance_spikes.copy(), start_time))

                    for neuron_id in current_neuron_ids:
                        neuron_spikes = current_instance_spikes[neuron_id]
                        # Scale the 0-1 neural spikes into the specific window inside the motif
                        scaled_spikes = [start_time + (t * window) for t in neuron_spikes]
                        
                        if neuron_id not in occurrence_dict:
                            occurrence_dict[neuron_id] = []
                        occurrence_dict[neuron_id].extend(scaled_spikes)
                        
            # Optional: Visual check of the first motif instance (you can comment it out)
            if a_id == 0 and seq_idx == 0:
                plot_motif_nesting(a_id, captured_local_data, window)

            # --- DEDUPLICATION & CHRONOLOGICAL SORTING ---
            unique_neurons = []
            mean_spikes = []
            
            for neuron, all_spikes in occurrence_dict.items():
                if len(all_spikes) > 0:
                    unique_neurons.append(neuron)
                    mean_spikes.append([np.mean(all_spikes)]) 
            
            if len(unique_neurons) > 0:
                sorted_pairs = sorted(zip(unique_neurons, mean_spikes), key=lambda x: x[1][0])
                sorted_neurons, sorted_mean_spikes = zip(*sorted_pairs)
                
                flat_occ_n_seq.append(list(sorted_neurons))
                flat_occ_n_spk.append(list(sorted_mean_spikes))
                flat_occ_labels.append(a_id)
            else:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
    
    return flat_occ_n_seq, flat_occ_n_spk, flat_occ_labels, subsequences_dict, subspike_times, raw_assembly_neurons_seq

import numpy as np
from collections import defaultdict

def subsequences_offset(n_neurons, n_motifs, n_bins, n_assemblies, n_sequences, n_a_motifs, n_a_seqs,
                  corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, 
                  sig_a, vol_a, sig_n, vol_n, 
                  random_state, plot, shuffle_order, n_mu, n_sigma, n_volume, n_densities, n_cdfs, n_t):
    
    root_rng = np.random.default_rng(random_state)
    rng_params, rng_samples, rng_neural_params, rng_neural_draws = root_rng.spawn(4)

    # --- STEP 1: ASSEMBLY MOTIF LEVEL ---
    # Generate the higher-level "blueprints" for which assemblies fire in which motifs
    a_mu, a_sigma, a_volume = get_mu_sigma_volume(
        n_assemblies, n_a_motifs, rng_params,
        corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sig_a, vol_a)
    
    a_densities, a_cdfs, a_t = build_pdfs_and_cdfs(n_assemblies, n_a_motifs, n_bins,
                                         a_mu, a_sigma, a_volume, plot)
    
    # Generate assembly sequences (the motif instances)
    subsequences_dict, subspike_times = generate_sequences(
        n_assemblies, n_a_motifs, n_a_seqs, a_cdfs, a_t, 
        rng=rng_samples, shuffle_order=shuffle_order)

    # --- STEP 2: NEURAL SEQUENCE RECONSTRUCTION ---
    flat_occ_n_seq = []
    flat_occ_n_spk = []
    flat_occ_labels = [] 
    raw_assembly_neurons_seq = defaultdict(lambda: defaultdict(list))
    
    # Iterate through each assembly motif (a_id) and its instances
    for a_id, motifs in sorted(subsequences_dict.items()):
        for seq_idx, m in enumerate(motifs):
            current_spike_times = subspike_times[a_id][seq_idx]
            
            sorted_m = sorted(m, key=lambda a: current_spike_times[a][0] if len(current_spike_times[a]) > 0 else float('inf'))
            
            L = len(sorted_m)
            if L == 0:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
                continue
                
            window = 1.0 / L  
            occurrence_dict = {}  # Stores shifted_id: [global_spike_times]
            captured_local_data = [] 
            
            for a in sorted_m:
                neuron_seq_dict, neuron_spk_dict = generate_sequences(
                    n_neurons, 1, 1, 
                    cdfs=n_cdfs[:, [a], :], t=n_t, 
                    rng=rng_neural_draws, shuffle_order=shuffle_order
                )
                
                # neuron_seq_dict[0][0] contains list of neuron IDs
                # neuron_spk_dict[0][0] contains {neuron_id: [local_spikes]}
                current_neuron_ids = neuron_seq_dict[0][0]
                current_instance_spikes = neuron_spk_dict[0][0]

                assembly_anchor_time = current_spike_times[a][0] if len(current_spike_times[a]) > 0 else 0.0
                
                if L == 1:
                    start_time = 0.0
                else:
                    ideal_start = assembly_anchor_time - (window / 2)
                    start_time = np.clip(ideal_start, 0.0, 1.0 - window)
                
                id_offset = a * 100 # Assembly 18 -> offset 1800
                
                shifted_neuron_ids = []
                shifted_instance_spikes = {}
                for neuron_id in current_neuron_ids:
                    shifted_id = neuron_id + id_offset
                    shifted_neuron_ids.append(shifted_id)
                    
                    neuron_spikes = current_instance_spikes[neuron_id]
                    scaled_spikes = [start_time + (t * window) for t in neuron_spikes]

                    shifted_instance_spikes[shifted_id] = neuron_spikes.copy()
                    
                    if shifted_id not in occurrence_dict:
                        occurrence_dict[shifted_id] = []
                    occurrence_dict[shifted_id].extend(scaled_spikes)

                captured_local_data.append((a, shifted_neuron_ids, shifted_instance_spikes, start_time))

                raw_assembly_neurons_seq[a_id][a].append(shifted_neuron_ids)

            # Optional: Visual check of the first motif instance
            if a_id == 0 and seq_idx == 0:
                plot_motif_nesting(a_id, captured_local_data, window)

            # --- DEDUPLICATION & CHRONOLOGICAL SORTING ---
            unique_neurons = []
            mean_spikes = []
            
            for neuron, all_spikes in occurrence_dict.items():
                if len(all_spikes) > 0:
                    unique_neurons.append(neuron)
                    mean_spikes.append([np.mean(all_spikes)]) 
            
            if len(unique_neurons) > 0:
                sorted_pairs = sorted(zip(unique_neurons, mean_spikes), key=lambda x: x[1][0])
                sorted_neurons, sorted_mean_spikes = zip(*sorted_pairs)
                
                flat_occ_n_seq.append(list(sorted_neurons))
                flat_occ_n_spk.append(list(sorted_mean_spikes))
                flat_occ_labels.append(a_id)
            else:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
    
    return flat_occ_n_seq, flat_occ_n_spk, flat_occ_labels, subsequences_dict, subspike_times, raw_assembly_neurons_seq

import numpy as np
from collections import defaultdict

def subsequences_append(n_neurons, n_motifs, n_bins, n_assemblies, n_sequences, n_a_motifs, n_a_seqs,
                                      corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, 
                                      sig_a, vol_a, sig_n, vol_n, 
                                      random_state, plot, shuffle_order, n_mu, n_sigma, n_volume, n_densities, n_cdfs, n_t):
    
    root_rng = np.random.default_rng(random_state)
    rng_params, rng_samples, rng_neural_params, rng_neural_draws = root_rng.spawn(4)

    # --- STEP 1: ASSEMBLY MOTIF LEVEL ---
    a_mu, a_sigma, a_volume = get_mu_sigma_volume(
        n_assemblies, n_a_motifs, rng_params,
        corr_mu, rho_mu, corr_sigma, rho_sig, corr_volume, rho_vol, sig_a, vol_a)
    
    a_densities, a_cdfs, a_t = build_pdfs_and_cdfs(n_assemblies, n_a_motifs, n_bins,
                                                 a_mu, a_sigma, a_volume, plot)
    
    subsequences_dict, subspike_times = generate_sequences(
        n_assemblies, n_a_motifs, n_a_seqs, a_cdfs, a_t, 
        rng=rng_samples, shuffle_order=shuffle_order)

    # --- STEP 2: CONSECUTIVE NEURAL SEQUENCING ---
    flat_occ_n_seq = []
    flat_occ_n_spk = []
    flat_occ_labels = [] 
    raw_assembly_neurons_seq = defaultdict(lambda: defaultdict(list))
    
    for a_id, motifs in sorted(subsequences_dict.items()):
        for seq_idx, m in enumerate(motifs):
            current_spike_times = subspike_times[a_id][seq_idx]
            
            # Sort the sampled assemblies chronologically based on their original activation order
            sorted_m = sorted(m, key=lambda a: current_spike_times[a][0] if len(current_spike_times[a]) > 0 else float('inf'))
            
            L = len(sorted_m)
            if L == 0:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
                continue
                
            occurrence_dict = {}  
            captured_local_data = [] 
            
            # --- BASELINE CHAINING LOGIC ---
            # Start a timeline at 0.0 and stack each uncompressed assembly one after another
            current_offset = 0.0
            block_duration = 1.0  # Each assembly keeps its full unscaled 1-second interval
            
            for a in sorted_m:
                # Sample the neural sequence for this instance
                neuron_seq_dict, neuron_spk_dict = generate_sequences(
                    n_neurons, 1, 1, 
                    cdfs=n_cdfs[:, [a], :], t=n_t, 
                    rng=rng_neural_draws, shuffle_order=shuffle_order
                )
                
                current_neuron_ids = neuron_seq_dict[0][0]
                current_instance_spikes = neuron_spk_dict[0][0]
                
                id_offset = a * 100 
                
                shifted_neuron_ids = []
                shifted_instance_spikes = {}
                for neuron_id in current_neuron_ids:
                    shifted_id = neuron_id + id_offset
                    shifted_neuron_ids.append(shifted_id)
                    
                    # Grab the raw local 0-1s spikes
                    neuron_spikes = current_instance_spikes[neuron_id]
                    
                    # 2. Store the raw 0-1s spikes using the shifted_id as the key
                    # This keeps the x-axis unscaled (0-1s) but offsets the y-axis label!
                    shifted_instance_spikes[shifted_id] = neuron_spikes.copy()
                    
                    # Accumulate globally: Shift spikes linearly forward without time window scaling
                    chained_spikes = [current_offset + t for t in neuron_spikes]
                    
                    if shifted_id not in occurrence_dict:
                        occurrence_dict[shifted_id] = []
                    occurrence_dict[shifted_id].extend(chained_spikes)

                captured_local_data.append((a, shifted_neuron_ids, shifted_instance_spikes, current_offset))
    
                raw_assembly_neurons_seq[a_id][a].append(shifted_neuron_ids)
                
                # Move timeline directly to the end of this assembly block
                current_offset += block_duration

            # Raster visualization compatibility (width is a full 1s block now)
            if a_id == 0 and seq_idx == 0:
                plot_motif_nesting(a_id, captured_local_data, window=block_duration)

            # --- DEDUPLICATION & TEMPORAL TEMPLATE GENERATION ---
            unique_neurons = []
            mean_spikes = []
            
            for neuron, all_spikes in occurrence_dict.items():
                if len(all_spikes) > 0:
                    unique_neurons.append(neuron)
                    mean_spikes.append([np.mean(all_spikes)]) 
            
            if len(unique_neurons) > 0:
                # Sort everything globally across the full accumulated timeline
                sorted_pairs = sorted(zip(unique_neurons, mean_spikes), key=lambda x: x[1][0])
                sorted_neurons, sorted_mean_spikes = zip(*sorted_pairs)
                
                flat_occ_n_seq.append(list(sorted_neurons))
                flat_occ_n_spk.append(list(sorted_mean_spikes))
                flat_occ_labels.append(a_id)
            else:
                flat_occ_n_seq.append([]); flat_occ_n_spk.append([]); flat_occ_labels.append(a_id)
    
    return flat_occ_n_seq, flat_occ_n_spk, flat_occ_labels, subsequences_dict, subspike_times, raw_assembly_neurons_seq


import math

def plot_motif_nesting(a_id, captured_data, window):
    """
    captured_data: list of (asmb_id, spikes, t_start)
    window: time window size for each assembly
    """
    n_items = len(captured_data)
    is_even = (n_items % 2 == 0)
    
    # Calculate rows needed
    # If even: rows for local (n/2) + 1 for global spanning
    # If odd: (n+1)/2 rows, everything fits in 2 columns
    if is_even:
        n_rows = (n_items // 2) + 1
    else:
        n_rows = (n_items + 1) // 2
    
    fig = plt.figure(figsize=(5, 1.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 2)
    
    # Generate unique colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_items))
    
    # --- 1. LOCAL PLOTS ---
    for i, (asmb_id, neuron_id, spikes_list, t_start) in enumerate(captured_data):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        ax.set_title(f"Assembly {asmb_id} (Local 0-1s)", color=colors[i], fontsize=10)
        
        # Sort neurons by mean time
        firing_info = []
        for idx, n_id in enumerate(neuron_id):
            s_times = spikes_list[n_id]
            if len(s_times) > 0:
                firing_info.append((n_id, s_times, np.mean(s_times)))
                
        firing_info.sort(key=lambda x: x[2])
        
        for rank, (n_id, s_times,_) in enumerate(firing_info):
            ax.scatter(s_times, [rank] * len(s_times), 
                       marker='|', color=colors[i], s=50)

        sorted_neuron_ids = [item[0] for item in firing_info]
        ax.set_yticks(range(len(sorted_neuron_ids)))
        ax.set_yticklabels(sorted_neuron_ids, fontsize=5)
        ax.set_ylabel("Neuron ID", fontsize=9)
        
        ax.axvspan(0, 1, color=colors[i], alpha=0.05)
        ax.set_xlim(0, 1)

    # --- 2. GLOBAL MOTIF PLOT ---
    if is_even:
        ax_global = fig.add_subplot(gs[n_rows-1, :])
    else:
        ax_global = fig.add_subplot(gs[n_rows-1, 1])

    ax_global.set_title(f"Global Motif {a_id}", fontsize=10)
    
    # Dynamic timeframe window supports consecutive chaining natively
    total_motif_duration = captured_data[-1][3] + window
    
    # Dictionaries to aggregate spikes across assemblies for shared-neuron safety
    all_global_spikes_per_neuron = defaultdict(list)
    spikes_by_assembly_contribution = defaultdict(list) # n_id -> list of (g_times, color_idx)

    for i, (asmb_id, neuron_ids, spikes_list, t_start) in enumerate(captured_data):
        for n_id in neuron_ids:
            s_times = spikes_list[n_id]
            if len(s_times) > 0:
                # Calculate true chronological positions on the global timeline
                g_times = [t_start + (s * window) for s in s_times]
                
                # Append to master lists for sorting and plotting
                all_global_spikes_per_neuron[n_id].extend(g_times)
                spikes_by_assembly_contribution[n_id].append((g_times, i))

    # Calculate overall mean time for each UNIQUE neuron across all its appearances
    neuron_global_means = {n_id: np.mean(spikes) for n_id, spikes in all_global_spikes_per_neuron.items()}
    
    # Sort unique neuron IDs by their absolute chronological order
    sorted_unique_neurons = sorted(neuron_global_means.keys(), key=lambda x: neuron_global_means[x])

    # Plot unique rows sequentially
    for rank, n_id in enumerate(sorted_unique_neurons):
        # Unpack every assembly segment that contributed spikes to this specific neuron ID
        for g_times, color_idx in spikes_by_assembly_contribution[n_id]:
            # Each spike marker '|' retains the exact color of the assembly that fired it!
            ax_global.scatter(g_times, [rank] * len(g_times), 
                             marker='|', color=colors[color_idx], s=50)

    # Label the y-axis with the exact unique neuron IDs
    ax_global.set_yticks([])
    ax_global.set_yticklabels([])
    
    ax_global.set_xlim(0, total_motif_duration)
    ax_global.set_xlabel(f"Global Motif Time (0-{int(np.ceil(total_motif_duration))}s)")
    ax_global.set_ylabel("Neuron ID", fontsize=9)
    
    plt.tight_layout()
    #plt.savefig("raster_append.jpg", dpi=300)
    #plt.savefig("raster_append.pdf", dpi=300)
    plt.show()

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
        plt.tight_layout()
        plt.show()
    return densities, cdfs, t

def build_pdfs_and_cdfs_vectorized(n_neurons, n_motifs, n_bins, mu, sigma, volume, plot):
    t = np.linspace(0, 1, n_bins)
    
    # Vectorize: shape (n_neurons, n_motifs, n_bins)
    mu_exp = mu[:, :, np.newaxis]      # (n_neurons, n_motifs, 1)
    sigma_exp = sigma[:, :, np.newaxis]  # (n_neurons, n_motifs, 1)
    t_grid = t[np.newaxis, np.newaxis, :]  # (1, 1, n_bins)
    
    g = np.exp(-0.5 * ((t_grid - mu_exp) / sigma_exp)**2)
    g /= g.sum(axis=2, keepdims=True)  # Normalize
    
    volume_exp = volume[:, :, np.newaxis]
    densities = g * volume_exp
    cdfs = np.cumsum(densities, axis=2)
    
    # plot PDFs and CDFs
    if plot:
        fig, axes = plt.subplots(n_motifs, 2, figsize=(10, 2 * n_motifs), sharex=True)
        for c in range(n_motifs):
            for i in range(n_neurons):
                axes[c,0].plot(t, densities[i,c], alpha=0.3)
                axes[c,1].plot(t, cdfs[i,c], alpha=0.3)
                axes[c,1].set_ylim(0,1)
        plt.tight_layout()
        plt.show()
    
    return densities, cdfs, t


def build_pdfs_and_cdfs_batched(n_neurons, n_motifs, n_bins, mu, sigma, volume, batch_size, plot):
    """
    Build PDFs and CDFs for each neuron in batches to reduce memory usage.
    """
    t = np.linspace(0, 1, n_bins)
    densities = np.zeros((n_neurons, n_motifs, n_bins))
    cdfs = np.zeros_like(densities)
    
    # Process neurons in batches
    for batch_start in range(0, n_neurons, batch_size):
        batch_end = min(batch_start + batch_size, n_neurons)
        batch_size_actual = batch_end - batch_start
        
        # Extract batch
        mu_batch = mu[batch_start:batch_end, :]      # (batch_size, n_motifs)
        sigma_batch = sigma[batch_start:batch_end, :] # (batch_size, n_motifs)
        volume_batch = volume[batch_start:batch_end, :] # (batch_size, n_motifs)
        
        # Vectorize for this batch
        mu_exp = mu_batch[:, :, np.newaxis]
        sigma_exp = sigma_batch[:, :, np.newaxis]
        t_grid = t[np.newaxis, np.newaxis, :]
        
        g = np.exp(-0.5 * ((t_grid - mu_exp) / sigma_exp)**2)
        g /= g.sum(axis=2, keepdims=True)
        
        volume_exp = volume_batch[:, :, np.newaxis]
        densities_batch = g * volume_exp
        cdfs_batch = np.cumsum(densities_batch, axis=2)
        
        # Store results
        densities[batch_start:batch_end, :, :] = densities_batch
        cdfs[batch_start:batch_end, :, :] = cdfs_batch
        
        print(f"Processed batch: neurons {batch_start}-{batch_end}/{n_neurons}")
    
    # plot PDFs and CDFs (sample from first batch for visualization)
    if plot:
        sample_neurons = min(n_neurons, 100)  # Sample first 100 neurons for clarity
        fig, axes = plt.subplots(n_motifs, 2, figsize=(10, 2 * n_motifs), sharex=True)
        for c in range(n_motifs):
            for i in range(sample_neurons):
                axes[c,0].plot(t, densities[i,c], alpha=0.3)
                axes[c,1].plot(t, cdfs[i,c], alpha=0.3)
                axes[c,1].set_ylim(0,1)
        fig.suptitle(f'PDFs and CDFs (showing first {sample_neurons} neurons per motif)')
        plt.tight_layout()
        plt.show()
    
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


def generate_sequences_batched(n_neurons, n_motifs, n_sequences, mu, sigma, volume, t, batch_size, rng=None, shuffle_order=False):
    """
    Generate sequences by processing neurons in batches to minimize memory usage.
    Only keeps one batch's CDFs in memory at a time.
    """
    rng = np.random.default_rng() if rng is None else rng
    sequences = {}
    spike_times_all = {}
    n_bins = len(t)
    
    for k in range(n_motifs):
        seqs = []
        seqs_spiketimes = []
        for seq_idx in range(n_sequences):
            spike_times_full = np.full(n_neurons, np.inf)
            spike_times_list_full = [np.array([], dtype=float) for _ in range(n_neurons)]
            
            # Process neurons in batches
            for batch_start in range(0, n_neurons, batch_size):
                batch_end = min(batch_start + batch_size, n_neurons)
                
                # Extract batch and compute CDFs
                mu_batch = mu[batch_start:batch_end, k]
                sigma_batch = sigma[batch_start:batch_end, k]
                volume_batch = volume[batch_start:batch_end, k]
                
                # Compute Gaussian PDF for batch (vectorized)
                t_expanded = t[np.newaxis, :]
                mu_expanded = mu_batch[:, np.newaxis]
                sigma_expanded = sigma_batch[:, np.newaxis]
                
                g = np.exp(-0.5 * ((t_expanded - mu_expanded) / sigma_expanded)**2)
                g /= g.sum(axis=1, keepdims=True)
                
                pdf = g * volume_batch[:, np.newaxis]
                cdf_batch = np.cumsum(pdf, axis=1)
                
                # Generate spike times for this batch
                for batch_idx in range(batch_end - batch_start):
                    neuron_idx = batch_start + batch_idx
                    u = rng.random()
                    if u <= cdf_batch[batch_idx, -1]:
                        idx = np.searchsorted(cdf_batch[batch_idx], u)
                        st = t[min(idx, n_bins - 1)]
                        spike_times_full[neuron_idx] = st
                        spike_times_list_full[neuron_idx] = np.array([st])
            
            # Get order of neurons
            fired = np.where(np.isfinite(spike_times_full))[0]
            order = list(fired[np.argsort(spike_times_full[fired])])
            if shuffle_order and len(order) > 1:
                rng.shuffle(order)
            
            seq = [int(s) for s in order]
            seqs.append(seq)
            seqs_spiketimes.append(spike_times_list_full)
        
        sequences[k] = seqs
        spike_times_all[k] = seqs_spiketimes
        print(f"Generated sequences for motif {k}/{n_motifs}")
    
    return sequences, spike_times_all


# =============================================================================
# Subseqeunces 
# =============================================================================

# With generate_sequences, we get ordered sequences of neuron indices for each cluster.
# Implement such a function so that while generating sequences, we get subsequences of neurons that fire together i.e share mu. 
# In original sequences they might not be together because of the noise (sigma) but they share the same mu, so they fire around the same time.
# Choose the neurons to be fired simultaneously randomly. A 25 ms time window is a reasonable choice for "together" (based on typical neural firing patterns).
# Since we have 100 time bins between 0 and 1, each time bin corresponds to 10 ms. So we can consider neurons that fire within 3 time bins (30 ms) as firing together.
# Only keep subsequences that have at least 2 neurons, and filter out the rest. This way we can analyze the subsequences of neurons that tend to fire together, which might be more robust to noise and more reflective of the underlying motifs.
# In generate_sequences, choose random neurons to fire in a way that they will have similar spike timing (30 ms window) but are not in the same sequence, and add them to a subsequence list.

def sub_generate_sequences(n_neurons, n_motifs, n_sequences, cdfs, t, subsequences=False, rng=None, shuffle_order=False):
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
    subs = {}
    
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

    return sequences, spike_times_all, subs



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
