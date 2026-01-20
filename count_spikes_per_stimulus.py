#%% DEPENDENCIES
import pandas as pd
import numpy as np
from glob import glob
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # prevent errors loading H5 in WSL
import spike_utils
import count_spikes

#%% I/O PATHS
data_path = "./processed_data/"
sensor_paths = glob(f"{data_path}*/ppt*/", recursive = True)
sensor_paths = [os.path.normpath(path) for path in sensor_paths]

#sensor_paths = [sensor_paths[0]]

#%% MAIN SCRIPT
waveform_dfs = []
spike_count_dfs = []
for sensor_path in sensor_paths:

    ppt_num, sensor = spike_utils.get_ppt_sensor_nums(sensor_path)

    waveforms, spike_times = spike_utils.load_spike_data(sensor_path, ppt_num, sensor)
    cluster_labels = spike_utils.load_cluster_labels(ppt_num, sensor, sensor_path)

    if cluster_labels is None:
        continue

    spikes_per_cluster = spike_utils.sort_spikes_by_cluster(cluster_labels, spike_times)

    # Continue if all spikes were unassigned
    if not spikes_per_cluster:
        continue

    behave_output = spike_utils.load_behave_data(ppt_num)
    stim = behave_output.stimulus 
    stim_times = behave_output.presTime

    stim_start_end = dict(zip(stim, stim_times))

    # Count the spikes that occured for each stimulus and the baseline 
    spikes_per_stimulus = {key: None for key, _ in spikes_per_cluster.items()}

    for cluster, spikes in spikes_per_cluster.items():

        stimulus_spikes = count_spikes.spikes_to_stimuli(spikes, stim_start_end)

        baseline_spikes = count_spikes.spikes_at_baseline(spikes, stim_start_end)

        stimulus_spikes["BASELINE"] = baseline_spikes
        spikes_per_stimulus[cluster] = stimulus_spikes

    # Create waveform df
    rows = [
        {"unit": unit, "stimulus": stim, "spike_times": spikes}
        for unit, stim_dict in spikes_per_stimulus.items()
        for stim, spikes in stim_dict.items()
    ]

    df = pd.DataFrame(rows)
    df.insert(0, "sensor", sensor)
    df.insert(0, "ppt", ppt_num)

    df_long= (
        df
        .explode("spike_times")
        .dropna(subset=["spike_times"])
        .rename(columns={"spike_times": "spike_time"})
    )

    df_long["spike_time"] = pd.to_numeric(df_long["spike_time"], errors="coerce")

    waveforms_df = pd.DataFrame(waveforms)
    waveforms_df.insert(0, "spike_time", spike_times)

    data_with_waveforms = pd.merge(df_long, waveforms_df, on="spike_time", how="outer")
    data_with_waveforms = data_with_waveforms.dropna()

    waveform_dfs.append(data_with_waveforms)

    # Create spike count df
    rows = []
    for stim, intervals in stim_start_end.items():
        for i, (start, end) in enumerate(intervals):
            rows.append(
                 {'stimulus': stim, 'trial_num': i+1, 'start': start, 'end': end}
            )

    temp = []
    for unit in np.unique(cluster_labels):
        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        temp.append(df)

        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        df = df.assign(stimulus="BASELINE")
        df["end"] = df["start"]
        df["start"] = df["start"] - 0.3
        df = df.sort_values(by=["start"])
        df["trial_num"] = list(range(1, 301))
        temp.append(df)

    df = pd.concat(temp, ignore_index=True)
    
    spike_count_dfs.append(df)
        

waveform_df = pd.concat(waveform_dfs, ignore_index=True)
waveform_df = waveform_df.sort_values(by=["ppt", "sensor", "unit", "stimulus"])

spike_count_df = pd.concat(spike_count_dfs, ignore_index=True)

#%% Merge
# Merge spikes onto stimulus presentations
merged = spike_count_df.merge(
    waveform_df,
    on=['ppt', 'sensor', 'unit', 'stimulus'],
    how='left'
)

# Keep only spikes that fall within the stimulus window
in_window = merged[
    (merged['spike_time'] >= merged['start']) &
    (merged['spike_time'] <= merged['end'])
]

# Count spikes per stimulus presentation
spike_counts = (
    in_window
    .groupby(spike_count_df.columns.tolist(), dropna=False)
    .size()
    .reset_index(name='n_spikes')
)

stim_df = spike_count_df.merge(
    spike_counts,
    on=spike_count_df.columns.tolist(),
    how='left'
)

stim_df['n_spikes'] = stim_df['n_spikes'].fillna(0).astype(int)



#%% Save
waveform_df.to_csv("./spike_waveforms.csv", index=False)
spike_count_df.to_csv("spike_counts.csv", index=False)

