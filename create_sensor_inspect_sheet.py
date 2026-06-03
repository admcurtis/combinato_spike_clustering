#%% DEPENDENCIES
# type: ignore
import numpy as np
import pandas as pd
import openpyxl
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile
from convert_ns6_utils import sort_data_chronologically

#%% Constants
n = 20 # number of samples to include in xlsx
samp_cols = [f"samp{j+1}" for j in range(n)]

#%% Glob paths to all .ns6 files
root = Path("../ieeg_data")
ns6_files = [
    p for p in root.rglob("*.ns6")
    if "Visit" in str(p)
    and "Baseline" not in str(p)
    and "Closed Loop" not in str(p)
]

#%% Patient x visit x path dictionary
groups = defaultdict(lambda: defaultdict(list))
for path in ns6_files:
    try:
        patient = next(part for part in path.parent.parts if "Patient" in part)
        visit = next(part for part in path.parent.parts if "Visit" in part)
    except StopIteration:
        continue
    groups[patient][visit].append(str(path))

#%% Process
with pd.ExcelWriter("visually_inspect_sensors.xlsx", engine="openpyxl") as writer:

    for patient, visits in groups.items():

        patient_clean = patient.replace(" ", "")

        patient_dfs = [] 

        for visit, paths in visits.items():

            visit_clean = visit.replace(" ", "")

            visit_data = [NsxFile(f) for f in paths]
            sorted_paths, sorted_data = sort_data_chronologically(paths, visit_data)

            full_data = [f.getdata() for f in sorted_data]

            signals = [np.array(f["data"]).squeeze() for f in full_data]
            chan_ids = [f["elec_ids"] for f in full_data]

            for i, run in enumerate(signals):
                mins  = np.min(run, axis=1)
                maxs  = np.max(run, axis=1)
                means = np.mean(run, axis=1)
                meds  = np.median(run, axis=1)
                stds  = np.std(run, axis=1)

                qrmed = np.abs(meds) / 0.6745

                first_n_samps = run[:, :n]
                samp_df = pd.DataFrame(first_n_samps, columns=samp_cols)

                df = pd.DataFrame({
                    "patient": patient_clean,
                    "visit": visit_clean,
                    "run": paths[i],
                    "chan_id": chan_ids[i],
                    "min": mins,
                    "max": maxs,
                    "mean": means,
                    "median": meds,
                    "qrmed": qrmed,
                    "std": stds
                })

                df = pd.concat([df, samp_df], axis=1)

                patient_dfs.append(df)

        # concatenate runs for this patient
        patient_df = pd.concat(patient_dfs, ignore_index=True)

        # write patient
        patient_df.to_excel(writer, sheet_name=patient_clean, index=False)
