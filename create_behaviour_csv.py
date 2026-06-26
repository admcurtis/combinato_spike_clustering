# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Constants
ntrials = 15

# %% Paths
root = Path("../ieeg_data")

# Patient x visit x path dictionary
behave_files = [
    p for p in root.rglob("*assMemData.mat")
    if "Visit" in str(p)
    and "Baseline" not in str(p)
    and "Closed Loop" not in str(p)
]

behave_path = behave_files[0]

patient = next(
    part for part in behave_path.parts if "Patient" in part
).replace(" ", "")

visit = next(
    part for part in behave_path.parts if "Visit" in part
).replace(" ", "")

# %% Load
behave_data = loadmat(
    behave_path,
    squeeze_me=True,
    struct_as_record=False
)

behave_data = behave_data["eegData"]


study_df = pd.DataFrame(
    {
        "patient": [patient for _ in range(ntrials)],
        "visit":   [visit for _ in range(ntrials)],
        "event":   list(behave_data.Study.Event),
        "pair":    list(behave_data.Study.Pair)
    }
)

stim_df = pd.DataFrame(
    behave_data.Study.Stimuli,
    columns = ["stim1", "stim2"]
)

time_df = pd.DataFrame(
    behave_data.Study.Times,
    columns = ["fixation_onset", "stim_onset"]
)

df = pd.concat(
    [study_df, stim_df, time_df],
    axis=1
)


# %%
last_trial = np.nanmax(behave_data.Test.Times)