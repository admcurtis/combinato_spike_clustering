#%% DEPENDENCIES
from scipy.io import loadmat
import numpy as np
import pandas as pd
import count_spikes
from glob import glob
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # prevent errors loading H5 in WSL
import h5py
import spike_utils

#%% I/O PATHS
data_path = "./processed_data/"
sensor_paths = glob(f"{data_path}*/ppt*/", recursive = True)
sensor_paths = [os.path.normpath(path) for path in sensor_paths]

sensor_paths = [sensor_paths[0]]

#%% MAIN SCRIPT
all_dfs = []
for sensor_path in sensor_paths:

    ppt_num, sensor = spike_utils.get_ppt_sensor_nums(sensor_path)

    neg_spikes, neg_times = spike_utils.load_spike_data(sensor_path, ppt_num, sensor)

    neg_cluster_labs = spike_utils.load_cluster_labels(ppt_num, sensor, sensor_path)

    if neg_cluster_labs is None:
        continue

    # Sort the spikes into thier clusters
    neg_clusters = spike_utils.sort_cluster_times(neg_cluster_labs, neg_times)

    # Continue if all spikes were unassigned
    if not neg_clusters:
        continue

    behave_output = spike_utils.load_behave_data(ppt_num)
    
    # Stimulus labels | shape: 50; stimulus names
    stim = behave_output.stimulus 

    # Presentation times | shape: 50, 6, 2; stim x presentation x start-end
    pres_time = behave_output.presTime

    # Count the spikes that occured for each stimulus and the baseline 
    stim_start_end = dict(zip(stim, pres_time))
    spikes_per_cluster = {key: None for key, _ in neg_clusters.items()}

    for cluster, spikes in neg_clusters.items():
        my_spikes = count_spikes.spikes_to_stimuli(spikes, stim_start_end)
        spikes_per_cluster[cluster] = count_spikes.count_spikes(spikes, stim_start_end)

    # Create dataframe of the counted spikes
    rows = [
        {"unit": unit, "stimulus": stim, "spike_times": spikes}
        for unit, stim_dict in spikes_per_cluster.items()
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

#%% Save
final_df.to_csv("./spike_counts.csv", index=False)

