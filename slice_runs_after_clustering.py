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

# %%
unclustered = loadmat("./processed_data/Patient5/Patient5_Visit2_sensor1.mat")
sr = int(unclustered["sr"].squeeze())
runs = unclustered["runs"]

samps_per_run = unclustered["samps_per_run"].squeeze()

slice_times = np.cumsum(samps_per_run)  / sr

new_slices = []
for i in range(len(slice_times)):
    
    start = 0 if i == 0 else slice_times[i - 1]
    end = slice_times[i]

    time_slice = times[(times >= start) & (times < end)]
    new_slices.append(time_slice)


mat_strcut = {
    "times": new_slices,
    "runs": runs
    
}




