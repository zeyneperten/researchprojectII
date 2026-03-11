# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, v_measure_score,
                             fowlkes_mallows_score)
from sklearn.metrics.cluster import contingency_matrix

# Local imports
from ..clustering import core as sc
from ..clustering.evaluation_helpers import silhouette_ignore_singletons

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)  # rows=true, cols=pred
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

def overall_scores(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),  # also returns (h,c) if needed
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "Purity": purity_score(y_true, y_pred),
    }


def _eval_matrix(name, distmat, ks, spike_times, seqs, seqs_labels, method, mat_dict):
    res = {"ks": ks, "scores": {}}
    D_sq = squareform(distmat)  # compute once per matrix

    for k in ks:
        ids_clust = sc.seq_cluster(distmat, k=k)

        rd = sc.info_cluster(spike_times, seqs, ids_clust, method)
        sc.add_within_across_score(rd, spike_times, seqs, permutation=True)
        sc.add_within_clust_score(rd, mat_dict)

        groundtruth = overall_scores(seqs_labels, ids_clust)
        
        cs = rd["clust_scores"]
        within_cl = np.nanmean(cs["within_clust"])
        valid = ~np.isnan(cs["within_clust"])
        over05 = (cs["within_clust"] > 0.5) & valid
        
        within_cl_05 = np.nanmean(cs["within_clust"][over05]) if np.any(over05) else np.nan
        #if np.any(over05):
        #    within_cl_05_w = np.average(cs["within_clust"][over05], weights=counts[over05])
        #else:
        #    within_cl_05_w = np.nan

        sil = silhouette_ignore_singletons(D_sq, ids_clust)
        #auc = np.nanmean(np.asarray(cs["auc"]) - 0.5)
        ratio = np.asarray(cs["ratio"])
        sig = np.where(np.array(cs["pval"]) < 0.05)[0] 
        if any(sig):
            ratio_mean=np.nanmean(np.array(cs["ratio"])[sig])
            within=np.nanmean(np.array(cs["within"])[sig])
        else:
            ratio_mean=np.nan
            within=np.nan

        res["scores"][k] = {
            "sil": sil,
            #"auc": auc,
            "within_temp": within,
            "within_cl": within_cl,
            "within_cl_05": within_cl_05,
            #"within_cl_05_w": within_cl_05_w,
            "ratio": ratio_mean, 
            "ARI": groundtruth["ARI"],
            "AMI": groundtruth["AMI"],
            "labels": ids_clust,
        }
    return name, res

def clust_parameters(distmat_dict, spike_times, seqs, seqs_labels, method, mat_dict,
                     plot=True, ks=None, n_jobs=-1, prefer="processes", verbose=10):
    if ks is None:
        ks = [2,4,8,10,15,18,20,25,30]

    # Parallelize over matrices
    jobs = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose, batch_size="auto")(
        delayed(_eval_matrix)(name, distmat, ks, spike_times, seqs, seqs_labels, method, mat_dict)
        for name, distmat in distmat_dict.items()
    )
    results = {name: res for name, res in jobs}

    if plot:
        metrics = [
            ("sil",        "Silhouette score", "Silhouette"),
            ("ratio",      "Ratio score",      "Ratio"),
            ("within_temp","Within score",     "Within Temp"),
            ("within_cl",  "Within score",     "Within Clust"),
            ("within_cl_05",   "Within score",     "Within Clust > 0.5"),
            #("within_cl_05_w", "Within score",     "Within Clust > 0.5 weighted"),
            #("auc",        "AUC score",        "AUC"),
            ("ARI",        "Groundtruth ARI",  "ARI"),
            ("AMI",        "Groundtruth AMI",  "AMI"),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3*len(metrics), 3), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        names = list(results.keys())

        for ax_idx, (mkey, ylabel, title) in enumerate(metrics):
            ax = axes[ax_idx]
            for color, name in zip(colors, names):
                ks_list = results[name]["ks"]
                yvals = [results[name]["scores"][k][mkey] for k in ks_list]
                ax.plot(ks_list, yvals, marker="o", alpha=0.7, label=name, color=color)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlabel("k")
            ax.grid(alpha=0.2)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    return results