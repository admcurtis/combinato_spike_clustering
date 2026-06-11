# %% Dependencies
import numpy as np
from scipy.io import loadmat
from spike_utils import load_spike_data

# %% Load data 
spikes, times = load_spike_data(
    "/mnt/d/adam_curtis/spike_clustering/processed_data/Patient5/Patient5_Visit2_sensor1",
    ppt_num=5,
    sensor=1,
    visit=2
)

# Times will be the last column in the array
spikes_times = np.column_stack([spikes, times])

# %% Load unclustered data to get samples, sample rate and number of samples in each run
concat_data = loadmat("./processed_data/Patient5/Patient5_Visit2_sensor1.mat")
sr = int(concat_data["sr"].squeeze())
runs = concat_data["runs"]

samps_per_run = concat_data["samps_per_run"].squeeze()

# Times to slice at. Spikes are in seconds, so use samples divided by sampling rate. 
slice_times = np.cumsum(samps_per_run)  / sr

# %% Slice the spikes and times.
new_slices = []
for i in range(len(slice_times)):
    
    start = 0 if i == 0 else slice_times[i - 1]
    end = slice_times[i]

    time_slice = spikes_times[
        (spikes_times[:,-1] >= start) & (spikes_times[:,-1] < end)
    ]
    new_slices.append(time_slice)

# %% Create .mat strcuture and save
mat_strcut = {
    "spikes": [s[:, :-1] for s in new_slices], # All columns except last
    "times": [s[:, -1] for s in new_slices], # Last column
    "runs": runs
}



