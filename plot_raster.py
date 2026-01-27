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

concept_spikes["unit"] = concept_spikes["unit"].astype(int)

concept_spikes = concept_spikes[["ppt", "sensor", "unit", "stimulus", "spike_time"]]

concept_intervals = event_times.merge(
    concept_spikes[["ppt", "sensor", "unit", "stimulus"]].drop_duplicates(),
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)

#%%
spikes_list = []
for row in concept_intervals.itertuples(index=False):

    print(row.ppt, row.sensor, row.unit, row.stimulus)

    spikes = concept_spikes[
        (concept_spikes["ppt"] == row.ppt) &
        (concept_spikes["sensor"] == row.sensor) &
        (concept_spikes["unit"] == row.unit) &
        (concept_spikes["stimulus"] == row.stimulus)
    ]

    spikes = spikes["spike_time"][
        (spikes["spike_time"] >= row.start + 0.3) &
        (spikes["spike_time"] <= row.end)
    ]

    if spikes.empty:
        spikes = []
    else:
        spikes = list(spikes)

    spikes_list.append(spikes)

concept_intervals["spikes"] = spikes_list


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
