#%% Modules
import h5py
import numpy as np
from scipy.io import loadmat
import os

#%% Functions 

def get_ppt_sensor_nums(sensor_path): 
    ppt_num = sensor_path.split(os.sep)[-2][3:]
    sensor = sensor_path.split(os.sep)[-1][13:]
    print(ppt_num, sensor)
    return ppt_num, sensor


def load_spike_data(sensor_path, ppt_num, sensor):
    # load h5 data containing spikes
    ppt_data = h5py.File(f"{sensor_path}/data_ppt{ppt_num}_sensor{sensor}.h5", "r") 

    # spikes with negative deflections
    neg_spikes = ppt_data["neg"]["spikes"][:]
    neg_times = ppt_data["neg"]["times"][:] / 1000 # convert ms to seconds

    # remove artifacts if necessary 
    if "artifacts" in ppt_data["neg"]:
        neg_artifacts = ppt_data["neg"]["artifacts"][:]
        neg_spikes = neg_spikes[neg_artifacts == 0, :]
        neg_times = neg_times[neg_artifacts == 0]

    return neg_spikes, neg_times


