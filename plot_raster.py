#%% Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% Load data
concept_cells = pd.read_csv("./detected_concepts.csv")
spike_data = pd.read_csv("./all_spike_waveforms.csv")
event_times = pd.read_csv("./spike_counts.csv")

#%%

concept_spikes = spike_data.merge(
    concept_cells,
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)

concept_spikes = concept_spikes[["ppt", "sensor", "unit", "stimulus", "spike_time"]]

concept_intervals = event_times.merge(
    concept_spikes[["ppt", "sensor", "unit", "stimulus"]].drop_duplicates(),
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)

full_data = concept_spikes.merge(
    concept_intervals,
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)

# NOTE: the n_spikes column may differ from the number of rows. This is expected.
# n_spikes only inlcudes trials +0.3ms after stim onset. 
# Plotting requires all spikes. 
full_data = full_data[
    (full_data["spike_time"] >= full_data["start"]) &
    (full_data["spike_time"] <= full_data["end"])
]

#%% Get baseline spikes

# for filtering
x = full_data[["ppt", "sensor", "unit", "stimulus", "start", "end"]].drop_duplicates()


#%% NOTE
# Then you can filter waveforms to get the spikes that are +- around the stimulus of 
# interest for each trial.

# The data need to be in following format
spike_times = [
    np.array([0.12, 0.45, 0.78]),     # trial 1
    np.array([0.05, 0.33]),           # trial 2
    np.array([0.22, 0.61, 0.89]),     # trial 3
    np.array([0.10, 0.40, 0.70]),     # trial 4
    np.array([0.15, 0.55]),           # trial 5
    np.array([0.30, 0.80])            # trial 6
]

#%% Plotting

plt.figure(figsize=(6, 4))

for trial_idx, trial_spikes in enumerate(spike_times):
    plt.vlines(
        trial_spikes,
        trial_idx + 0.5,
        trial_idx + 1.5
    )

plt.xlabel("Time (s)")
plt.ylabel("Trial")
plt.yticks(range(1, len(spike_times) + 1))
plt.title("Raster plot")

plt.show()
