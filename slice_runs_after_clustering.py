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

        # Visit can contain letters (e.g. "Visit2b"), so don't just grab the digits
        name_match = re.fullmatch(r"Patient(\d+)_Visit([^_]+)_sensor(\d+)", file.stem)
        if name_match is None:
            print(f"Skipping {file.name}: name doesn't match PatientN_VisitN_sensorN")
            continue
        ppt, visit, sensor = name_match.groups()
        print(f"Processing: Patient{ppt}, visit{visit}, Sensor{sensor}")

        clustered_path = patient / f"Patient{ppt}_Visit{visit}_sensor{sensor}"

        # The is the .h5 containing the clustered spikes and their times
        spikes, times = load_spike_data(
            clustered_path,
            ppt_num=ppt,
            sensor=sensor,
            visit=visit,
            remove_artifacts=False
        )
        
        # Load cluster labels, plus the rows of the spike arrays they belong to
        labels, sorted_idx = load_cluster_labels(clustered_path, return_index=True)
        if labels is None:
            print("No labels found! Assuming no neurons for this sensor")
            continue

        # Combinato only sorts spikes that weren't masked as artifacts, so the labels
        # line up with spikes[sorted_idx], not with every spike in the .h5
        n_unsorted = len(times) - len(sorted_idx)
        if n_unsorted:
            print(f"Dropping {n_unsorted} spikes Combinato didn't sort (masked artifacts)")
        spikes, times = spikes[sorted_idx], times[sorted_idx]

        # Attach labels and times to spike waveform array as the final two columns
        spikes_labs_times = np.column_stack([spikes, labels, times])

        # Load sample rate, task names, paths and samples per run from the unclustered
        concat_data = loadmat(
            patient / f"Patient{ppt}_Visit{visit}_sensor{sensor}.mat",
            variable_names=["sr", "tasks", "paths", "samples"]
        )
        sr = int(concat_data["sr"].squeeze())
        
        # savemat stores lists as 1xN arrays and pads strings, so flatten and strip
        tasks = [str(t).strip() for t in np.ravel(concat_data["tasks"])]
        paths = [str(p).strip() for p in np.ravel(concat_data["paths"])]
        samps_per_run = np.ravel(concat_data["samples"])

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

            # Centre times so that they are relative to the start of the run
            time_slice[:, -1] = time_slice[:, -1] - start

            new_slices.append(time_slice)
            

        for task, data, samples, path in zip(tasks, new_slices, samps_per_run, paths):
            
            print(f"Processing: {task} containing {data.shape[0]} spikes")


            # Create .mat strcuture and save
            mat_struct = {
                "spikes": data[:, :-3], # All columns except last 3
                "labels": data[:, -3:-1], # Second and third to last columns
                "times": data[:, -1] , # Last column
                "task": task,
                "path": path,
                "samples": samples,
                "sr": sr
            }

            # Save .mat
            save_name = f"Patient{ppt}_Visit{visit}_sensor{sensor}_{task}.mat"
            savemat(patient / "sliced_after_clustering" / save_name, mat_struct)

print("DONE!")
