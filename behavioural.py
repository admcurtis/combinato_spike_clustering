# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %%
root = Path("./processed_data")
spike_path = root / "Patient5/sliced_after_clustering/Patient5_Visit1_sensor1_Memory Task.mat"

# %% Load

# behave_data = loadmat(
#     behave_path,
#     squeeze_me=True,
#     struct_as_record=False
# )
# behave_data = behave_data["eegData"]
# last_trial = np.nanmax(behave_data.Test.Times)

patient, visit, sensor, task = spike_path.stem.split("_")

spike_data = loadmat(spike_path)

waveforms = spike_data["spikes"]
spike_times = spike_data["times"].squeeze()
neurons = spike_data["labels"].squeeze()

if not waveforms.shape[0] == spike_times.shape[0] == neurons.shape[0]:
    raise RuntimeError(
        f"{patient} {visit} {sensor} {task}: waveform, sensor, neuron shape mismatch")

rows = waveforms.shape[0]

waveform_df = pd.DataFrame(
    spike_data["spikes"],
    columns=[f"s{i}" for i in range(64)]
)

df = pd.DataFrame(
    {"pateint":    [patient for _ in range(rows)],
     "visit":      [visit for _ in range(rows)],
     "sensor":     [sensor for _ in range(rows)],
     "task":       [task for _ in range(rows)],
     "neuron":     list(neurons),
     "spike_time": list(spike_times)
     }
)

df = pd.concat(
    [df, waveform_df],
    axis=1
)








