#%% DEPENDENCIES
# type: ignore
from scipy.io import savemat
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile
import os
from convert_ns6_utils import sort_data_chronologically
import gc

#%%
sensor_selections = pd.read_csv("./selected_sensors.csv")

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
    and "Macro LFP Reference" not in str(p)
    and len(p.parts) == 6 # ensures only ns6 files with path patientX/visitX/task/*.ns6
]

# %% Patient x visit x path dictionary
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

        # Get the channel names in each run
        chans_per_run = [
            (run, list(df["chan_id"])) for run, df in selected_sensors.groupby("run")
        ]

        # Log an error if there are runs with no sensors
        if len(chans_per_run) != len(paths):
            msg = "Selection does not contain channels from all runs"
            print(f"{patient}, {visit}: {msg}")
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: {msg}\n\n")
            continue

        # Check to see whether any runs have repeating channel names
        repeating = [
            (run, arr) for run, arr in chans_per_run if len(set(arr)) < len(arr)
        ]

        if repeating:
            msg = "Some runs have channel names that repeat"
            print(f"{patient}, {visit}: {msg}")
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: {msg}\n\n")
            continue

        # Get channels that have been selected and occur in all runs. 
        try:
            chan_lists = [set(chan_list) for _, chan_list in chans_per_run]
            chan_set = set.intersection(*chan_lists)
            # Convert set to list to ensure order preservation
            chans_in_all_runs = [chan for chan in chan_lists[0] if chan in chan_set] 
        except TypeError:
            with open(f"processed_data/{log_file}", "a") as f:
                f.write(f"{patient}, {visit}: No channels shared in all runs\n\n")
            continue

        # Load the .ns6 file for this visit
        visit_data = [(f, NsxFile(f)) for f in paths]

        sorted_data = sort_data_chronologically(visit_data)

        del visit_data # save memory
        gc.collect()

        full_data = {}
        sample_rates = []
        for path, data in sorted_data:
            temp_data = data.getdata()
            task_name = path.split("/")[-2]
            signal = np.array(temp_data["data"]).squeeze() 
            sr = float(temp_data["samp_per_s"])

            sample_rates.append(sr)

            full_data[task_name] = {
                "path":      path,
                "signal":    signal,
                "chan_ids":  temp_data["elec_ids"],
                "sr":        sr,
                "samples":   signal.shape[-1]
            }

            del temp_data
            gc.collect()

        # Add keys to the selected channels and their respective signals
        for task, data in full_data.items():

            chan_ids = data["chan_ids"]
            signal   = data["signal"]
            selected_idx = [
                i for i, chan in enumerate(chan_ids) if chan in chans_in_all_runs
            ]

            selected_chans   = [chan_ids[i] for i in selected_idx]
            selected_signals = signal[selected_idx, :]

            full_data[task]["selected_chans"]  = selected_chans
            full_data[task]["selected_signal"] = selected_signals

        # # Sanity checks
        # if not all(sig.shape[0] == selected_signals[0].shape[0]
        #            for sig in selected_signals
        #         ):
        #     error_msg = "Num chans differs across runs "

        # if not all(chans == chan_id[0] for chans in chan_id):
        #     error_msg += "Chan ids do not match across runs "

        # if not all(rate == samp_rates[0] for rate in samp_rates):
        #     error_msg += "Sampling rate differs across runs "

        # if error_msg:
        #     with open(f"processed_data/{log_file}", "a") as f:
        #         f.write(f"{patient}, {visit}, {error_msg}\n")
        #         f.write("\n".join(paths))
        #         f.write("\n\n")  
        #     continue

        # Sample rate
        sr = sample_rates[0]

        # Save
        print(f"Saving .mat data for {patient} {visit}")
        for i, chan in enumerate(chans_in_all_runs):

            print(f"concatenating {patient}, {visit}, channel {chan}")

            mat_struct = {
                "data": np.concatenate([
                    data["selected_signal"][i, :]
                    for data in full_data.values()
                ]),
                "sr":       sr,
                "tasks":    list(full_data.keys()),
                "paths":    [data["path"] for data in full_data.values()],
                "samples":  [data["samples"] for data in full_data.values()]
            }

            save_path = f"processed_data/{patient}/"
            os.makedirs(save_path, exist_ok=True)

            save_name = f"{patient}_{visit}_sensor{chan}.mat"

            # Save .mat
            savemat(save_path + save_name, mat_struct)
        
        del mat_struct
        del selected_signals
        del sorted_data
        gc.collect()












# %%
