#%% DEPENDENCIES
import pandas as pd
from glob import glob
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # prevent errors loading H5 in WSL
import spike_utils
import count_spikes

#%% I/O PATHS
data_path = "./processed_data/"
sensor_paths = glob(f"{data_path}*/ppt*/", recursive = True)
sensor_paths = [os.path.normpath(path) for path in sensor_paths]

# sensor_paths = [sensor_paths[0]]

#%% MAIN SCRIPT
all_dfs = []
for sensor_path in sensor_paths:

    ppt_num, sensor = spike_utils.get_ppt_sensor_nums(sensor_path)

    neg_spikes, neg_times = spike_utils.load_spike_data(sensor_path, ppt_num, sensor)
    cluster_labels = spike_utils.load_cluster_labels(ppt_num, sensor, sensor_path)

    if cluster_labels is None:
        continue

    spikes_per_cluster = spike_utils.sort_spikes_by_cluster(cluster_labels, neg_times)

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
        spikes_per_trial = {
            stim: count_spikes.group_spikes_per_trial(stim_spikes)
            for stim, stim_spikes in stimulus_spikes.items()
        }

        baseline_spikes = count_spikes.spikes_at_baseline(spikes, stim_start_end)
        spikes_per_baseline = count_spikes.group_spikes_per_trial(baseline_spikes)

        spikes_per_trial["BASELINE"] = spikes_per_baseline
        spikes_per_stimulus[cluster] = spikes_per_trial

    # Create dataframe of the counted spikes
    rows = [
        {"unit": unit, "stimulus": stim, "spike_times": spikes}
        for unit, stim_dict in spikes_per_stimulus.items()
        for stim, spikes in stim_dict.items()
    ]

    df = pd.DataFrame(rows)
    # df.insert(0, "sensor", sensor)
    # df.insert(0, "ppt", ppt_num)

    # # Descriptive statistics
    # df["mean_spikes"] = df["total_spikes"].apply(np.mean)
    # df["median_spikes"] = df["total_spikes"].apply(np.median)
    # df["std_spikes"] = df["total_spikes"].apply(np.std)

    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)

#final_df.to_csv("./spike_counts.csv", index=False)

