# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Paths
root = Path("../ieeg_data")

# Processed screening files (contain the "out" struct). Other *Screening*.mat
# files in these folders are raw PsychToolbox output or spike analyses.
# "_old" files are superseded by a sibling file in the same folder.
behave_files = [
    p for p in root.rglob("*_lfpScreeningData*.mat")
    if len(p.parts) == 6 and not p.stem.endswith("_old")
]

# %% Process screening data
screening_dfs = []
for behave_path in behave_files:

    patient = next(
        part for part in behave_path.parts if "Patient" in part
    ).replace(" ", "")

    visit = next(
        part for part in behave_path.parts if "Visit" in part
    ).replace(" ", "")

    block = next(
        part for part in behave_path.parts if "Screening" in part
    ).replace(" ", "")

    behave_data = loadmat(
        behave_path,
        squeeze_me=True,
        struct_as_record=False
    )["out"]

    # stimIDs: column 0 = category, column 1 = 1-based index into stimNames
    stim_ids = np.atleast_2d(behave_data.stimIDs)
    stim_names = np.atleast_1d(behave_data.stimNames)
    n_trials = stim_ids.shape[0]

    stim_times = np.atleast_1d(behave_data.stimTimes).astype(float)
    if len(stim_times) != n_trials:
        print(
            f"WARNING: {behave_path} has {len(stim_times)} stimTimes but "
            f"{n_trials} trials; stim_onset set to NaN"
        )
        stim_times = np.full(n_trials, np.nan)

    df = pd.DataFrame(
        {
            "patient":       patient,
            "visit":         visit,
            "block":         block,
            "file_path":     str(behave_path),
            "trial":         np.arange(1, n_trials + 1),
            "stim_onset":    stim_times,
            "stim_category": stim_ids[:, 0],
            "stim_id":       stim_ids[:, 1],
            "stim_name":     stim_names[stim_ids[:, 1] - 1],
            "respKey":       np.atleast_1d(behave_data.respKey),
            "respRT":        np.atleast_1d(behave_data.respRT)
        }
    )

    screening_dfs.append(df)

full_screening_data = pd.concat(
    screening_dfs,
    axis=0
)

# %% Save
full_screening_data.to_csv("screening_behaviour.csv", index=False)
