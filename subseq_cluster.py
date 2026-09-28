import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
from pathlib import Path

# Get the workspace root
workspace_root = Path.cwd().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import copy
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from scripts.data_utils import load_nrm
from scripts.clustering import core as sc
from scripts.clustering import shuffling as sh
from scripts.analysis import analysis as an

# 1. Worker function for a single parameter set
def process_params(vol_a, vol_b, sn, sa):
    # Construct filename using the looping 'a' and constant 'b'
    file_name = f'append_subseqdata_vol_{vol_a}_{vol_b}_sig_{sn}N_{sa}A.pkl'
    
    try:
        with open(file_name, 'rb') as f:
            simsub = pickle.load(f)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None

    # Prepare data dictionary 
    data = {
        "bursts": simsub['bursts'],
        "seqs": simsub['seqs'],
        "seq_method": 'center_of_mass',
    }

    # Generate Survival Scores
    scores = sh.survival_scores(data, k=50, thr_c=0.4, seeds=range(30))

    # Masking logic
    mask = (
        (scores["survival_freq"] > 0.1) & 
        (scores["pairwise_jaccard_cond_survival"] > 0.1)
    )
    
    mat_dict = sc.allmot(data["seqs"])
    red, rd = sh.run_one(data["seqs"], data, mat_dict, k=30, thr_c=0.4)
    
    ids_order = rd["ids_clust"].copy()
    ids_order[mask] = -2  # Mark chance clusters
    
    # Information clustering and scoring
    rd_ = sc.info_cluster(data["bursts"], data["seqs"], ids_order, method='center_of_mass')
    sc.add_within_clust_score(rd_, mat_dict)
    
    red_ = an.sort_and_filter_labels(
        ids_clust=rd_["ids_clust"],
        clust_scores=rd_["clust_scores"],
        sort_by="within_clust",
        ascending=True,
        min_score={"within_clust": 0.4, "within_clust_ratio": 0},
        min_size=1,
        replace_with=-1,
    )

    # Calculate final counts
    s = np.unique(red_["ids_clust_replaced"], return_counts=True)
    sf_clust = s[1][np.where(s[0] > 0)[0]]
    
    return {
        "params": (vol_a, vol_b, sn, sa),
        "scores": scores,
        "rd_": rd_,
        "rd": rd,
        "count_seqs": sf_clust.sum(),
        "count_clust": len(sf_clust),
    }

if __name__ == "__main__":
    # Define parameters
    vol_a_list = [(0.07, 0.9), (0.07, 0.5)] # The two different vol_n_a[0] values to loop
    vol_b_fix = (0.1, 0.3)         # Keeping vol_n_a[1] same throughout
    sig_n = ['low', 'high']
    sig_a = ['low', 'high']

    # Flatten the parameter space into 8 tasks
    tasks = list(itertools.product(vol_a_list, sig_n, sig_a))

    print(f"Starting parallel processing for {len(tasks)} combinations...")
    
    # Distribute tasks across 8 cores [cite: 12, 13]
    results_list = Parallel(n_jobs=8)(
        delayed(process_params)(va, vol_b_fix, sn, sa) for va, sn, sa in tasks
    )

    # Consolidate and save 
    all_results = {res["params"]: res for res in results_list if res is not None}
    with open(f"{vol_b_fix}_append_subseq_clustering_data_labelsfixed.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("Analysis complete. Data saved.")