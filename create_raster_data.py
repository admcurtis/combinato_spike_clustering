#%% Dependencies
import pandas as pd
import numpy as np


#%% Process data
def get_concept_trials(concept_cells=None, spike_data=None, event_times=None):

    if concept_cells is None or spike_data is None or event_times is None:
        print("Loading data from CSV...")
        concept_cells = pd.read_csv("./detected_concepts.csv")
        spike_data = pd.read_csv("./all_spike_waveforms.csv")
        event_times = pd.read_csv("./spike_counts.csv")

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

    return concept_spikes, concept_intervals, baseline_spikes


def list_spikes_per_concept_trial(concept_spikes, concept_intervals, baseline_spikes):

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

    return spikes_list, bl_spike_list


def combine_stim_and_bl_spikes(concept_intervals, spikes_list, bl_spike_list):

    combined_spikes = concept_intervals.copy()

    # Place spikes during stimulus and spikes at baseline into a single list
    combined_spikes["all_spikes"] = [
        stim_spike + bl_spike for stim_spike, bl_spike in zip(spikes_list, bl_spike_list)
    ]

    # Normalise spike times so they are relative to the stimulus onset
    combined_spikes["rel_spikes"] = combined_spikes.apply(
        lambda row: np.array(row["all_spikes"]) - row["start"],
        axis=1
    )

    return combined_spikes 


def create_raster_data(concept_cells=None, spike_data=None, event_times=None):

    concept_spikes, concept_intervals, baseline_spikes = get_concept_trials(
        concept_cells, spike_data, event_times
    )

    spikes_list, bl_spike_list = list_spikes_per_concept_trial(
         concept_spikes, concept_intervals, baseline_spikes
    )

    combined_spikes = combine_stim_and_bl_spikes(
        concept_intervals, spikes_list, bl_spike_list
    )

    return combined_spikes

#%% Run
if __name__ == "__main__":
    concept_intervals = create_raster_data()
    concept_intervals.to_pickle("raster_data.pkl")

