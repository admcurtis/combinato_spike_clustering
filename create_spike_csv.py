# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Paths
processed_data = Path("./processed_data")
patients = processed_data.glob("Patient*/")

# %% Process
dfs = []
for patient_path in patients:

    mat_files = (patient_path / "sliced_after_clustering/").glob("*.mat")

    for spike_path in mat_files:

        patient, visit, sensor, task = spike_path.stem.split("_")

        spike_data = loadmat(spike_path)

        spike_times = np.ravel(spike_data["times"])
        neurons = spike_data["labels"]
        waveforms = spike_data["spikes"]

        if not spike_times.shape[0] == neurons.shape[0] == waveforms.shape[0] :
            raise RuntimeError(
                f"{patient} {visit} {sensor} {task}: spike data shape mismatch")

        spike_times_df = pd.DataFrame(
            {"patient":         patient,
            "visit":            visit,
            "sensor":           sensor,
            "task":             task,
            "neuron_class":     neurons[:,0],
            "neuron_group":     neurons[:,1],
            "spike_time":       spike_times
            }
        )

        waveform_df = pd.DataFrame(
            spike_data["spikes"],
            columns=[f"s{i}" for i in range(64)]
        )

        df = pd.concat(
            [spike_times_df, waveform_df],
            axis=1
        )

        dfs.append(df)

full_df = pd.concat(
    dfs,
    axis=0
)

full_df.sort_values(
    by=[
        "patient",
        "visit",
        "sensor",
        "neuron_group",
        "neuron_class",
        "task",
        "spike_time"
    ],
    inplace=True
)

unique_values = pd.Series(
    {c: full_df[c].unique() for c in full_df.iloc[:,:5]}
)
print(unique_values)

full_df.to_csv(
    "all_spikes.csv",
    index=False
)