#%% DEPENDENCIES
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # prevent errors loading H5 in WSL

import pandas as pd
import numpy as np
from glob import glob
import spike_utils
import extract_spikes
import tabularise

#%% I/O PATHS
data_path = "./processed_data/"
sensor_paths = glob(f"{data_path}*/ppt*/", recursive = True)
sensor_paths = [os.path.normpath(path) for path in sensor_paths]

#sensor_paths = [sensor_paths[0]]


#%% MAIN SCRIPT
waveform_dfs = []
events_dfs = []
for sensor_path in sensor_paths:

    ppt_num, sensor = spike_utils.get_ppt_sensor_nums(sensor_path)

    waveforms, spike_times = spike_utils.load_spike_data(sensor_path, ppt_num, sensor)
    cluster_labels = spike_utils.load_cluster_labels(sensor_path)

    # Continue if no cluster labels (sensor detected no spikes)
    if cluster_labels is None:
        print(f"No units detected for ppt{ppt_num} sensor{sensor}")
        continue

    spikes_per_cluster = spike_utils.sort_spikes_by_cluster(cluster_labels, spike_times)

    # Continue if all spikes were unassigned
    if not spikes_per_cluster:
        continue

    behave_output = spike_utils.load_behave_data(ppt_num)
    stim = behave_output.stimulus 
    stim_times = behave_output.presTime

    stim_start_end = dict(zip(stim, stim_times))

    # Extract the spikes that occured for each stimulus and the baseline 
    spikes_per_stimulus = {key: None for key, _ in spikes_per_cluster.items()}

    for cluster, spikes in spikes_per_cluster.items():

        stimulus_spikes = extract_spikes.spikes_to_stimuli(
            spikes, stim_start_end, t_min=0
        )

        baseline_spikes = extract_spikes.spikes_at_baseline(
            spikes, stim_start_end, t_min=0
        )

        stimulus_spikes["BASELINE"] = baseline_spikes
        spikes_per_stimulus[cluster] = stimulus_spikes

    # Create tabular data for each spike detected
    spike_time_df = tabularise.create_waveform_df(spikes_per_stimulus, sensor, ppt_num)

    waveforms = pd.DataFrame(waveforms)
    waveforms.insert(0, "spike_time", spike_times)

    spike_times_with_waveforms = pd.merge(
        spike_time_df,
        waveforms,
        on="spike_time",
        how="outer"
    ).dropna()

    waveform_dfs.append(spike_times_with_waveforms)

    # Create tabular data for each event (spike count to each event added later)
    events_df = tabularise.create_event_df(
        stim_start_end,
        cluster_labels,
        sensor,
        ppt_num,
        tmin=0.3
    )
    
    events_dfs.append(events_df)
        
waveform_df = pd.concat(waveform_dfs, ignore_index=True)
waveform_df = waveform_df.sort_values(by=["ppt", "sensor", "unit", "stimulus"])

events_df = pd.concat(events_dfs, ignore_index=True)
spike_count_df = tabularise.add_spike_counts(events_df, waveform_df)

#%% Save
if sum(spike_count_df["n_spikes"]) != waveform_df.shape[0]:

    print(f"Error: sum(n_spikes)={sum(spike_count_df['n_spikes'])}, "
          f"waveform_df rows={waveform_df.shape[0]}")
    
    raise ValueError("Spike counts do not match waveform rows!")

#%%
waveform_df.to_csv("./spike_waveforms.csv", index=False)
spike_count_df.to_csv("spike_counts.csv", index=False)

