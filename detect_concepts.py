#%% DEPENDENCIES
import numpy as np
import pandas as pd
from scipy.stats import norm

#%% Multiple comparisons function
def sd_threshold(num_comparisons, fwer=0.05):
    """
    Calculate the number of SDs above the mean needed to control 
    the family-wise error rate (FWER) for multiple comparisons.
    """
    alpha_per_comparison = fwer / num_comparisons
    z_threshold = norm.ppf(1 - alpha_per_comparison)
    return z_threshold


def detect_concepts(spike_data=None):

    if spike_data is None:
        print("Loading data from CSV...")
        spike_data = pd.read_csv("./spike_counts.csv")

    spike_stats = (
        spike_data
        .groupby(by=["ppt", "sensor", "unit", "stimulus"])["n_spikes"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    baselines = spike_stats[spike_stats["stimulus"] == "BASELINE"].copy()
    stimuli = spike_stats[spike_stats["stimulus"] != "BASELINE"].copy()

    z_threshold = sd_threshold(stimuli.shape[0])

    baselines["threshold"] = baselines["mean"] + (baselines["std"] * z_threshold)

    stimuli = pd.merge(
        stimuli,
        baselines[["ppt", "sensor", "unit", "mean", "std", "threshold"]],
        on=["ppt", "sensor", "unit"],
        suffixes=("", "_baseline")
        )

    stimuli["concept_cell"] = np.where(
        stimuli["median"] > stimuli["threshold"],
        1,
        0
    )

    concepts = stimuli[
        (stimuli["concept_cell"] == 1) &
        (stimuli["median"] >= 2)
    ]

    return concepts

#%% Save
if __name__ == "__main__":
    concepts = detect_concepts()
    concepts.to_csv("./detected_concepts.csv", index=False)