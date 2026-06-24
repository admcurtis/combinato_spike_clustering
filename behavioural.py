# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# %%
behave_path = Path(
    "../ieeg_data/Patient 5/Visit 1/Memory Task/Patient5_Visit1_assMemData.mat"
)

spike_path = Path(
    "processed_data/Patient5/sliced_after_clustering/Patient5_Visit1_sensor1_Memory Task.mat"
)

# %% Load

behave_data = loadmat(
    behave_path,
    squeeze_me=True,
    struct_as_record=False
)
behave_data = behave_data["eegData"]

spike_data = loadmat(spike_path)

last_spike = np.max(spike_data["times"])

last_trial = np.nanmax(behave_data.Test.Times)
