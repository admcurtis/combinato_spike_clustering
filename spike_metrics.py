# %% Set interactive backend to open plots in a separate window
# type: ignore
%matplotlib qt

# %% Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ptt_fwhm as metric
from itertools import combinations

# %% Paths
out_path = Path("metric_plots")
out_path.mkdir(parents=True, exist_ok=True)

# %% Constants
WF_COLS = [f"s{i}" for i in range(64)]

# %%
data = pd.read_csv("all_spikes.csv")
selected_clusters = pd.read_csv("cluster_selections_long.csv")
data = metric.add_waveform_stats(data, WF_COLS)

# %% 
grouped_clusters = selected_clusters.groupby(["patient", "visit", "sensor"])

# %%
for id, df in grouped_clusters: 

    patient, visit, sensor = id
    patient = f"Patient{patient}"
    visit = f"Visit{visit}"
    sensor = f"sensor{sensor}"

    combos = list(combinations(df["cluster"], 2))

    if not combos:
        continue

    groups = metric.get_groups(data, patient, visit, sensor)

    for cluster1, cluster2 in combos:

        try:
            g1, g2 = groups[cluster1], groups[cluster2]
        except KeyError as e:
            print(f"{patient} {visit} {sensor}: cluster {e.args[0]} not found in groups")
            continue

        fig, ax = metric.plot_groups_3d(g1, g2, standardise=True, show=False)

        fig.savefig(
            out_path / f"{patient}_{visit}_{sensor}_c{cluster1}_vs_c{cluster2}.png",
            dpi=150,
            bbox_inches="tight"
        )
        
        plt.close(fig)

    for label, cluster in groups.items():

        fig, ax = metric.plot_fwhm(cluster[WF_COLS].mean())

        fig.savefig(
            out_path / f"{patient}_{visit}_{sensor}_c{label}_avg_metrics.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)



# %%
data_mat = np.array(data[WF_COLS])
wf = data_mat[200,:]
metric.plot_fwhm(wf)

# %%

# %%
groups = metric.get_groups(data, patient, visit, sensor)

# %%
metric.plot_groups_3d(groups[1], groups[2], standardise=True)



