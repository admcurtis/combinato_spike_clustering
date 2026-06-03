# %% Dependencies
import h5py
import numpy as np
from scipy.io import loadmat
from spike_utils import load_spike_data

# %% Load data 
data = load_spike_data(
    "/mnt/d/adam_curtis/spike_clustering/processed_data/Patient5/Patient5_Visit1_sensor1",
    ppt_num=5,
    sensor=1
)