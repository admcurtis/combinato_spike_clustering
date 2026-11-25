# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Paths
root = Path("../ieeg_data")

# Patient x visit x path dictionary
behave_files = [
    p for p in root.rglob("*Screening*.mat")
    if len(p.parts) == 6
]


# %%
behave_path = behave_files[13]

# %%
behave_data = loadmat(
    behave_path,
    squeeze_me=True,
    struct_as_record=False
)["out"]

# %%
# Response key 300x1
behave_data.respKey
# Response time 300x1
behave_data.respRT
# Stimulus IDs 300x2
behave_data.stimIDs
# stimulus names 50x1
behave_data.stimNames
# stimulus times 300x1
behave_data.stimTimes



