#%% DEPENDENCIES
# type: ignore
from scipy.io import savemat
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile

from convert_ns6_utils import (
    get_odd_chans, remove_stimulus_chan, get_odd_signals,
    sort_data_chronologically
    )

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

        # Load all runs for this visit
        visit_data = [NsxFile(f) for f in paths]

        sorted_paths, sorted_data = sort_data_chronologically(paths, visit_data)

        full_data = [f.getdata() for f in sorted_data]

        signals = [np.array(f["data"]).squeeze() for f in full_data]
        chan_ids = [f["elec_ids"] for f in full_data]
        samp_rates = [float(f["samp_per_s"]) for f in full_data]
        samples = [data.shape[-1] for data in signals]

        # Extract only odd numbered chans (even numbered chans are not brain data)
        odd_chans, odd_idxs = get_odd_chans(chan_ids)

        # Get signals only corresponding to odd-numbered channels
        odd_signals = get_odd_signals(signals, odd_idxs)

        # Remove stimulus channels
        final_chans, final_signals = remove_stimulus_chan(odd_chans, odd_signals)

        # Sanity checks
        if not all(sig.shape[0] == final_signals[0].shape[0] for sig in final_signals):
            error_msg = "Num chans differs across runs "

        if not all(chans == final_chans[0] for chans in final_chans):
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
        for i, chan in enumerate(odd_chans):

            print(f"concatenating {patient}, {visit}, channel {chan}")
            combined_signals = np.concatenate(
                [sig[i, :] for sig in odd_signals]
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