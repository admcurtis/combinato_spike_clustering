#%% DEPENDENCIES
# type: ignore
from scipy.io import savemat
import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile

#%%
os.makedirs("processed_data", exist_ok=True)

# Glob paths to all .ns6 files
root = Path("patient_data")
ns6_files = [str(p) for p in root.rglob("*.ns6")]
ns6_files = [p for p in ns6_files if "Visit" in p]

# Patient x visit x path dictionary
groups = defaultdict(lambda: defaultdict(list))
for f in ns6_files:
    path = Path(f)
    patient = next(part for part in path.parts if "Patient" in part)
    visit = next(part for part in path.parts if "Visit" in part)
    groups[patient][visit].append(f)

#%% Process
for patient, visits in groups.items():

    for visit, paths in visits.items():

        patient = patient.replace(" ", "")
        visit = visit.replace(" ", "")

        # Load all runs for this visit
        visit_data = [NsxFile(f) for f in paths]

        # Sort chronologically
        sorted_data = sorted(
            visit_data,
            key=lambda x: x.basic_header["TimeOrigin"]
        )

        full_data = [f.getdata() for f in sorted_data]

        signals = [np.array(f["data"]).squeeze() for f in full_data]
        chan_ids = [f["elec_ids"] for f in full_data]
        samp_rates = [float(f["samp_per_s"]) for f in full_data]
        samples = [data.shape[1] for data in signals]

        # Sanity checks
        assert all(sig.shape[0] == signals[0].shape[0] for sig in signals), (
            f"Data shape differs across runs {patient} {visit}"
        )

        assert all(chans == chan_ids[0] for chans in chan_ids),  (
            f"Chan ids do not match across runs {patient} {visit}"
        )

        assert all(rate == samp_rates[0] for rate in samp_rates), (
            f"Sampling rate differs across runs {patient} {visit}"
        )

        # Sample rate
        sr = samp_rates[0]

        # Extract only even numbered chans (odd numbered chans are not brain data)
        signals_df = [
            pd.DataFrame(data.T, columns=ids) for data, ids in zip(signals, chan_ids)
        ]
        even_chans = [col for col in signals_df[0].columns if int(col) % 2 == 1]
        even_signals = [np.array(signal[even_chans]).T for signal in signals_df]

        # Concatenate signals for clustering
        print(f"concatenating {patient}, {visit}")
        combined_signals = np.concatenate(even_signals, axis=1)

        # Create .mat structure
        mat_struct = {
            "data": combined_signals, 
            "sr": sr,
            "runs": paths,
            "samps_per_run": samples
        }

        # Save
        print(f"Saving .mat data for {patient} {visit}")
        for i, chan in enumerate(even_chans):

            save_path = f"processed_data/{patient}/"
            os.makedirs(save_path, exist_ok=True)

            save_name = f"{patient}_{visit}_sensor{chan}.mat"

            temp_data = mat_struct.copy()
            this_sensor = temp_data["data"][i,:]
            temp_data["data"] = this_sensor

            # Save .mat
            savemat(save_path + save_name, temp_data)

