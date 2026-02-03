#%% Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% Load data
concept_cells = pd.read_csv("./detected_concepts.csv")
spike_data = pd.read_csv("./all_spike_waveforms.csv")
event_times = pd.read_csv("./spike_counts.csv")


#%% Process data
# Get a dataframe of spikes to each concept cell
concept_spikes = spike_data.merge(
    concept_cells,
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)
concept_spikes["unit"] = concept_spikes["unit"].astype(int)

concept_spikes = concept_spikes[["ppt", "sensor", "unit", "stimulus", "spike_time"]]

# Get dataframe of spikes at baseline
baseline_spikes = spike_data[spike_data["stimulus"] == "BASELINE"]

# Get a dataframe of trial times for each concept cell.
concept_intervals = event_times.merge(
    concept_spikes[["ppt", "sensor", "unit", "stimulus"]].drop_duplicates(),
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)


#%% Get list of spike times for each concept trial
spikes_list = []
bl_spike_list = []

for row in concept_intervals.itertuples(index=False):

    stim_spikes = concept_spikes[
        (concept_spikes["ppt"] == row.ppt) &
        (concept_spikes["sensor"] == row.sensor) &
        (concept_spikes["unit"] == row.unit) &
        (concept_spikes["stimulus"] == row.stimulus)
    ]

    stim_spikes = stim_spikes["spike_time"][
        (stim_spikes["spike_time"] >= row.start) &
        (stim_spikes["spike_time"] <= row.end)
    ]

    bl_spikes = baseline_spikes[
        (baseline_spikes["ppt"] == row.ppt) &
        (baseline_spikes["sensor"] == row.sensor) &
        (baseline_spikes["unit"] == row.unit) 
    ]

    bl_spikes = bl_spikes["spike_time"][
        (bl_spikes["spike_time"] >= row.start - 0.3) &
        (bl_spikes["spike_time"] <= row.end + 0.3)
    ]

    stim_spikes = [] if stim_spikes.empty else list(stim_spikes)
    bl_spikes = [] if bl_spikes.empty else list(bl_spikes)

    spikes_list.append(stim_spikes)
    bl_spike_list.append(bl_spikes)


# Place spikes during stimulus and spikes at baseline into a single list
concept_intervals["all_spikes"] = [
    stim_spike + bl_spike for stim_spike, bl_spike in zip(spikes_list, bl_spike_list)
]

# Normalise spike times so they are relative to the stimulus onset
concept_intervals["rel_spikes"] = concept_intervals.apply(
    lambda row: np.array(row["all_spikes"]) - row["start"],
    axis=1
)


#%% Plotting
stim = "Danny Dyer"

cell_to_plot = concept_intervals[concept_intervals["stimulus"] == stim]
cell_to_plot = [sorted(np.array(i)) for i in cell_to_plot["rel_spikes"]]

plt.figure(figsize=(6, 4))

for trial_idx, trial_spikes in enumerate(cell_to_plot):
    plt.vlines(
        trial_spikes,
        trial_idx + 0.5,
        trial_idx + 1.5
    )

plt.xlabel("Time (s)")
plt.ylabel("Trial")
plt.yticks(range(1, len(cell_to_plot) + 1))
plt.title("Raster plot")
plt.axvline(0, color="red")
plt.axvline(1, color="red")

plt.show()
