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

    os.makedirs(patient / "sliced_after_clustering", exist_ok=True)
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
        if labels is None:
            print("No labels found! Assuming no neurons for this sensor")
            continue

        # Attach labels and times to spike waveform array as the final two columns
        spikes_labs_times = np.column_stack([spikes, labels, times])

        # Load unclustered data to get sample rate and number of samples per run data
        concat_data = loadmat(patient / f"Patient{ppt}_Visit{visit}_sensor{sensor}.mat")
        sr = int(concat_data["sr"].squeeze())
        runs = concat_data["runs"]
        samps_per_run = concat_data["samps_per_run"].squeeze()

        # Times to slice at. Spikes are in seconds, so use samples over sample rate
        slice_times = np.cumsum(samps_per_run)  / sr

        # Slice the spikes, labels and, times.
        new_slices = []

        for i in range(len(slice_times)):
            
            start = 0 if i == 0 else slice_times[i - 1]
            end = slice_times[i]

            time_slice = spikes_labs_times[
                (spikes_labs_times[:,-1] >= start) & (spikes_labs_times[:,-1] < end)
            ]

            # normalise times
            time_slice[:, -1] = time_slice[:, -1] - start

            new_slices.append(time_slice)
            

        for run, data, samples in zip(runs, new_slices, samps_per_run):
            
            print(f"Processing: {run} containing {data.shape[0]} spikes")


            # Create .mat strcuture and save
            mat_struct = {
                "spikes": data[:, :-2], # All columns except last
                "labels": data[:, -2], # Second to last column
                "times": data[:, -1] , # Last column
                "run": run,
                "samples": samples,
                "sr": sr
            }

            task = Path(run).parent.stem

            # Save .mat
            save_name = f"Patient{ppt}_Visit{visit}_sensor{sensor}_{task}.mat"
            savemat(patient / "sliced_after_clustering" / save_name, mat_struct)