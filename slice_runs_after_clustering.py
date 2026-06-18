# %% Dependencies
import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path
import os
import re
from spike_utils import load_spike_data, load_cluster_labels

# %%
processed_data = Path(os.getcwd()) / "processed_data"
patients = list(processed_data.glob("Patient*/"))

# %%
for patient in patients:

    concat_mat_files = list(patient.glob("*.mat"))

    for file in concat_mat_files:

        ppt, visit, sensor = re.findall(r"\d+", file.stem)
        print(f"Processing: Patient{ppt}, visit{visit}, Sensor{sensor}")

        clustered_path = patient / f"Patient{ppt}_Visit{visit}_sensor{sensor}"

        # The is the .h5 containing the clustered spikes and their times
        spikes, times = load_spike_data(
            clustered_path,
            ppt_num=ppt,
            sensor=sensor,
            visit=visit
        )
        
        # Load cluster labels
        labels = load_cluster_labels(clustered_path)

        # Times will be the last column in the array
        spikes_labs_times = np.column_stack([spikes, labels, times])

        # Load unclustered data to get sample rate and number of samples per run metadata
        concat_data = loadmat(patient / f"Patient{ppt}_Visit{visit}_sensor{sensor}.mat")
        sr = int(concat_data["sr"].squeeze())
        runs = concat_data["runs"]
        samps_per_run = concat_data["samps_per_run"].squeeze()

        # Times to slice at. Spikes are in seconds, so use samples divided by sample rate
        slice_times = np.cumsum(samps_per_run)  / sr

        # Slice the spikes and times.
        new_slices = []
        for i in range(len(slice_times)):
            
            start = 0 if i == 0 else slice_times[i - 1]
            end = slice_times[i]

            time_slice = spikes_labs_times[
                (spikes_labs_times[:,-1] >= start) & (spikes_labs_times[:,-1] < end)
            ]
            new_slices.append(time_slice)

        # Create .mat strcuture and save
        mat_struct = {
            "spikes": [s[:, :-2] for s in new_slices], # All columns except last
            "labels": [s[:, -2] for s in new_slices], # Second to last column
            "times": [s[:, -1] for s in new_slices], # Last column
            "runs": runs,
            "samps_per_run": samps_per_run
        }

        # Save .mat
        save_name = f"Patient{ppt}_Visit{visit}_sensor{sensor}_sliced.mat"
        savemat(patient / save_name, mat_struct)