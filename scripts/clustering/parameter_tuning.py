# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

# Local imports
from .core import seq_cluster_leiden, info_cluster, add_within_across_score, add_within_clust_score, seq_cluster
from .evaluation_helpers import silhouette_ignore_singletons

# =============================================================================
# Parameter tuning
# =============================================================================
    
def _eval_matrix(name, distmat, ks, bursts, seqs, data_dict, mat_dict, method):
    res = {"ks": ks, "scores": {}}
    D_sq = squareform(distmat)  # compute once per matrix

    for k in ks:
        ids_clust = seq_cluster(distmat, k=k, method=method)
        uniq_labels, counts = np.unique(ids_clust, return_counts=True)

        rd = info_cluster(bursts, seqs, ids_clust, data_dict['seq_method'])
        if "ids_clust" not in rd:
            rd["ids_clust"] = ids_clust
        # mutate, add scores
        add_within_across_score(rd, bursts, seqs, permutation=False)
        add_within_clust_score(rd, mat_dict)

        cs = rd["clust_scores"]
        sil = silhouette_ignore_singletons(D_sq, ids_clust)

        within_cl = np.nanmean(cs["within_clust"])
        valid = ~np.isnan(cs["within_clust"])
        over05 = (cs["within_clust"] > 0.5) & valid
        over10 = (counts > 10) & valid
        within_cl_05 = np.nanmean(cs["within_clust"][over05]) if np.any(over05) else np.nan
        if np.any(over05):
            within_cl_05_w = np.average(cs["within_clust"][over05], weights=counts[over05])
        else:
            within_cl_05_w = np.nan
        within_cl_min10 = np.nanmean(cs["within_clust"][over10]) if np.any(over10) else np.nan

        ratio = np.asarray(cs["ratio"])
        ratio_mean = np.nanmean(ratio)
        ratio_pos = np.nanmean(ratio[ratio > 0]) if np.any(ratio > 0) else np.nan
        within = np.nanmean(np.asarray(cs["within"]))        
        #auc = np.nanmean(np.asarray(cs["auc"]) - 0.5)
        

        res["scores"][k] = {
            "sil": sil,
            #"auc": auc,
            "within": within,
            "within_cl": within_cl,
            "within_cl_05": within_cl_05,
            "within_cl_min10": within_cl_min10,
            "ratio": ratio_mean,
            "ratio_pos": ratio_pos,
            "labels": ids_clust,
        }
    return name, res

def clust_parameters(distmat_dict, bursts, seqs, data_dict, mat_dict,
                     plot=True, ks=None, method="ward", n_jobs=-1, prefer="processes", verbose=10):
    if ks is None:
        ks = [2,10,20,30,40,50,80,100,130,160,200,300,400,500]

    # Parallelize over matrices
    jobs = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose, batch_size="auto")(
        delayed(_eval_matrix)(name, distmat, ks, bursts, seqs, data_dict, mat_dict, method)
        for name, distmat in distmat_dict.items()
    )
    results = {name: res for name, res in jobs}

    if plot:
        metrics = [
            ("sil",            "Silhouette score", "Silhouette"),
            ("ratio",          "Ratio score",      "Ratio"),
            ("within",         "Within score",     "Within Temp"),
            ("within_cl",      "Within score",     "Within Clust"),
            ("within_cl_05",   "Within score",     "Within Clust > 0.5"),
            ("within_cl_min10", "Within score",     "Within Clust min 10"),
            #("auc",            "AUC score",        "AUC"),
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


def clust_parameters_leiden(excess_dict, mat_dict, bursts, seqs, data_dict, w_keep=2.,min_comp_size=0, rs=None, plot=True):
    if rs is None:
        rs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    name = "laiden"
    res = {"rs": rs, "scores": {}}
    results = {name: res} #extend with w_keeps
    
    for r in rs:
        ids_clust = seq_cluster_leiden(mat_dict, excess_dict, excess_dict["excess"], r, w_keep=w_keep, min_comp_size=min_comp_size, verbose=False)
        uniq_labels, counts = np.unique(ids_clust, return_counts=True)

        rd = info_cluster(bursts, seqs, ids_clust, data_dict['seq_method'])
        if "ids_clust" not in rd:
            rd["ids_clust"] = ids_clust
        # mutate, add scores
        add_within_across_score(rd, bursts, seqs, permutation=False)
        add_within_clust_score(rd, mat_dict)

        cs = rd["clust_scores"]
        #sil = sc_helpers.silhouette_ignore_singletons(D_sq, ids_clust)

        within_cl = np.nanmean(cs["within_clust"])
        valid = ~np.isnan(cs["within_clust"])
        over05 = (cs["within_clust"] > 0.5) & valid
        over10 = (counts > 10) & valid
        within_cl_05 = np.nanmean(cs["within_clust"][over05]) if np.any(over05) else np.nan
        if np.any(over05):
            within_cl_05_w = np.average(cs["within_clust"][over05], weights=counts[over05])
        else:
            within_cl_05_w = np.nan
        within_cl_min10 = np.nanmean(cs["within_clust"][over10]) if np.any(over10) else np.nan

        ratio = np.asarray(cs["ratio"])
        ratio_mean = np.nanmean(ratio)
        ratio_pos = np.nanmean(ratio[ratio > 0]) if np.any(ratio > 0) else np.nan
        within = np.nanmean(np.asarray(cs["within"]))        
        #auc = np.nanmean(np.asarray(cs["auc"]) - 0.5)
        

        res["scores"][r] = {
            #"sil": sil,
            #"auc": auc,
            "within": within,
            "within_cl": within_cl,
            "within_cl_05": within_cl_05,
            "within_cl_min10": within_cl_min10,
            "ratio": ratio_mean,
            "ratio_pos": ratio_pos,
            "labels": ids_clust,
        }

    results[name] = res
    
    if plot:
        metrics = [
            #("sil",            "Silhouette score", "Silhouette"),
            ("ratio",          "Ratio score",      "Ratio"),
            ("within",         "Within score",     "Within Temp"),
            ("within_cl",      "Within score",     "Within Clust"),
            ("within_cl_05",   "Within score",     "Within Clust > 0.5"),
            ("within_cl_min10", "Within score",     "Within Clust min 10"),
            #("auc",            "AUC score",        "AUC"),
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3*len(metrics), 3), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        names = list(results.keys())

        for ax_idx, (mkey, ylabel, title) in enumerate(metrics):
            ax = axes[ax_idx]
            for color, name in zip(colors, names):
                rs_list = results[name]["rs"]
                yvals = [results[name]["scores"][r][mkey] for r in rs_list]
                ax.plot(rs_list, yvals, marker="o", alpha=0.7, label=name, color=color)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlabel("res")
            ax.grid(alpha=0.2)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    return res
    