#%% DEPENDENCIES
# type: ignore
from scipy.io import savemat
import numpy as np
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

        visit = visit.replace(" ", "")
        patient = patient.replace(" ", "")

        visit_data = [NsxFile(f) for f in paths]

        # Sort chronologically
        sorted_data = sorted(
            visit_data,
            key=lambda x: x.basic_header["TimeOrigin"]
        )

        full_data = [f.getdata() for f in sorted_data]
        signals = [np.array(f["data"]).squeeze() for f in full_data]

        samp_rates = [float(f["samp_per_s"]) for f in full_data]
        sr = samp_rates[0]

        print(f"concatenating {patient}, {visit}")
        combined_signals = np.concatenate(signals, axis=1)

        # Create .mat structure
        mat_struct = {"data": combined_signals, "sr": sr}

        print(f"Saving .mat data for {patient} {visit}")
        for sensor in range(0, mat_struct["data"].shape[0]):

            save_path = f"processed_data/{patient}/"
            os.makedirs(save_path, exist_ok=True)

            save_name = f"{patient}_{visit}_sensor{sensor}.mat"

            temp_data = mat_struct.copy()
            this_sensor = temp_data["data"][sensor,:]
            temp_data["data"] = this_sensor

            # Save .mat
            savemat(save_path + save_name, temp_data)
