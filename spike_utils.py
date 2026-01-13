#%% Modules
import h5py
import numpy as np
from scipy.io import loadmat
import os

#%% Functions 

def get_ppt_sensor_nums(sensor_path): 
    """
    Parses the sensor_path to get the participant and sensor numbers
    """
    ppt_num = sensor_path.split(os.sep)[-2][3:]
    sensor = sensor_path.split(os.sep)[-1][13:]
    print(ppt_num, sensor)
    return ppt_num, sensor


def load_spike_data(sensor_path, ppt_num, sensor):
    """
    Return waveforms and onset times for a given participant and sensor. 
    Removes artifact spikes if neccessary
    """

    path_to_data = f"{sensor_path}/data_ppt{ppt_num}_sensor{sensor}.h5"

    # load h5 data containing spikes
    with h5py.File(path_to_data, "r") as ppt_data:
        neg_spikes = ppt_data["neg"]["spikes"][:]
        neg_times = ppt_data["neg"]["times"][:] / 1000  # convert ms to seconds

        # remove artifacts if necessary 
        if "artifacts" in ppt_data["neg"]:
            neg_artifacts = ppt_data["neg"]["artifacts"][:]
            neg_spikes = neg_spikes[neg_artifacts == 0, :]
            neg_times = neg_times[neg_artifacts == 0]

    return neg_spikes, neg_times


def load_cluster_labels(ppt_num, sensor, sensor_path):
    """
    Loads the labels of the clusters. Returns an array of indices with which to filter
    the spike data.
    Returns none if file not found. 
    File not found indicates that Combinato did not detect any clusters on this sensor.
    """
    try:
        neg_sort_file = f"{sensor_path}/sort_neg_ada/sort_cat.h5"
        with h5py.File(neg_sort_file, "r") as neg_sort_data:
            neg_cluster_idx = np.array(neg_sort_data["classes"])
            return neg_cluster_idx
    except FileNotFoundError:
        print(f"No units detected for ppt{ppt_num} sensor{sensor}")
        return None
    







