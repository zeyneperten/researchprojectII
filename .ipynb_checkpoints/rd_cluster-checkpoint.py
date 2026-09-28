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

    mat_dict = sc.allmot(data["seqs"])
    red, rd = sh.run_one(data["seqs"], data, mat_dict, k=30, thr_c=0.4)

    with open(f"{vol_b}_append_subseq_clustering_data.pkl", "rb") as f:
        to_edit = pickle.load(f)

    for va, sn, sa in tasks:
        to_edit[(vol_a, vol_b, sn, sa)]['rd'] = rd
        
    return to_edit
        

if __name__ == "__main__":
    # Define parameters
    vol_a_list = [(0.07, 0.9), (0.07, 0.5)] # Rows to loop
    vol_b_fix = (0.1, 0.3)                  # Constant volume combo
    sig_n = ['low', 'high']
    sig_a = ['low', 'high']

    # Flatten parameter space into 8 structural tasks
    tasks = list(itertools.product(vol_a_list, sig_n, sig_a))

    print(f"Starting parallel processing for {len(tasks)} combinations...")
    
    # Distribute execution across 8 CPU cores safely
    results = Parallel(n_jobs=4)(
        delayed(process_params)(va, vol_b_fix, sn, sa) for va, sn, sa in tasks
    )
    
    # Dump out the final parameter cache pkl file
    output_filename = f"try_{vol_b_fix}_append_subseq_clustering_data.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Analysis complete. Data successfully saved to {output_filename}")