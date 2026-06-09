#%% DEPENDENCIES
# type: ignore
from scipy.io import savemat
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile
import os
from convert_ns6_utils import (
    sort_data_chronologically, get_selected_chans, get_selected_signals
)
import gc

#%%
sensor_selections = pd.read_csv("./practice_sensor_selection.csv")

#%%
os.makedirs("processed_data", exist_ok=True)
log_file = "run_errors.txt"

# Glob paths to all .ns6 files
root = Path("../ieeg_data")
ns6_files = [
    p for p in root.rglob("*.ns6")
    if "Visit" in str(p)
    and "Baseline" not in str(p)
    and "Closed Loop" not in str(p)
]

# Patient x visit x path dictionary
groups = defaultdict(lambda: defaultdict(list))
for path in ns6_files:
    try:
        patient = next(part for part in path.parent.parts if "Patient" in part)
        visit = next(part for part in path.parent.parts if "Visit" in part)
    except StopIteration:
        continue
    groups[patient][visit].append(str(path))

#%% Process
for patient, visits in groups.items():

    for visit, paths in visits.items():

        patient = patient.replace(" ", "")
        visit = visit.replace(" ", "")
        error_msg = ""

        selected_sensors = sensor_selections[
            (sensor_selections["patient"] == patient) &
            (sensor_selections["visit"] == visit)
        ]
        
        if selected_sensors.empty:
            print(f"{patient}, {visit} not found in DataFrame")
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: Not found in selections dataframe\n\n")
            continue

        chans_per_run = [
            set(df["chan_id"]) for _, df in selected_sensors.groupby("run")
        ]

        if len(chans_per_run) != len(paths):
            msg = "Selection does not contain channels from all runs"
            print(f"{patient}, {visit}: {msg}")
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: {msg}\n\n")
            continue

        try:
            chans_in_all_runs = set.intersection(*chans_per_run)
        except TypeError:
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: No channels common in all runs\n\n")
            continue

        visit_data = [NsxFile(f) for f in paths]

        sorted_paths, sorted_data = sort_data_chronologically(paths, visit_data)
        del visit_data # save memory
        gc.collect()

        full_data = [f.getdata() for f in sorted_data]

        signals = [np.array(f["data"]).squeeze() for f in full_data]
        chan_ids = [f["elec_ids"] for f in full_data]
        samp_rates = [float(f["samp_per_s"]) for f in full_data]
        samples = [data.shape[-1] for data in signals]
        del full_data # save memory
        gc.collect()

        chan_id, chan_indx = get_selected_chans(chan_ids, chans_in_all_runs)
        selected_signals = get_selected_signals(signals, chan_indx)

        # Sanity checks
        if not all(sig.shape[0] == selected_signals[0].shape[0]
                   for sig in selected_signals
                ):
            error_msg = "Num chans differs across runs "

        if not all(chans == chan_id[0] for chans in chan_id):
            error_msg += "Chan ids do not match across runs "

        if not all(rate == samp_rates[0] for rate in samp_rates):
            error_msg += "Sampling rate differs across runs "

        if error_msg:
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}, {error_msg}\n")
                f.write("\n".join(paths))
                f.write("\n\n")  
            continue

        # Sample rate
        sr = samp_rates[0]

        # Save
        print(f"Saving .mat data for {patient} {visit}")
        for i, chan in enumerate(chan_id[0]):

            print(f"concatenating {patient}, {visit}, channel {chan}")
            combined_signals = np.concatenate(
                [sig[i, :] for sig in selected_signals]
            )

            # Create .mat structure
            mat_struct = {
                "data": combined_signals, 
                "sr": sr,
                "runs": sorted_paths,
                "samps_per_run": samples
            }

            save_path = f"processed_data/{patient}/"
            os.makedirs(save_path, exist_ok=True)

            save_name = f"{patient}_{visit}_sensor{chan}.mat"

            # Save .mat
            savemat(save_path + save_name, mat_struct)
        
        del signals
        del selected_signals
        del sorted_data
        gc.collect()











