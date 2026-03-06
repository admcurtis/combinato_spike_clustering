#%% Dependencies 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% Constants
ppt = 1 
sensor = 5
unit = 2
stim = "Danny Dyer"

#%% Main
def main():
    waveforms = pd.read_csv("all_spike_waveforms.csv")
    raster_data = pd.read_pickle("raster_data.pkl")

    plt.figure(figsize=(12, 8))

    # Top-left
    plt.subplot(2, 2, 1)  # (rows, cols, index)
    plot_unit(waveforms, ppt, sensor, unit)

    # Top-right
    plt.subplot(2, 2, 2)
    plot_unit(waveforms, ppt, sensor, unit, stim)

    # Bottom-left
    plt.subplot(2, 2, 3)
    plot_raster(raster_data, stim)

    plt.tight_layout()
    plt.show()


#%% Plot functions
def plot_unit(waveforms, ppt=1, sensor=0, unit=2, stim=""):

    cluster = waveforms[
    (waveforms["ppt"] == ppt)  &
    (waveforms["sensor"] == sensor) &
    (waveforms["unit"] == unit) 
    ]

    if stim:
        cluster = cluster[cluster["stimulus"] == stim]

    cluster = cluster.iloc[:, 5:]

    n = cluster.shape[0]

    for _, row in cluster.iterrows():
        plt.plot(row.values)

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(f"{stim} (n spikes = {n})")


def plot_raster(raster_data, stim=""):

    cell_to_plot = raster_data[raster_data["stimulus"] == stim]
    cell_to_plot = [sorted(np.array(i)) for i in cell_to_plot["rel_spikes"]]

    for trial_idx, trial_spikes in enumerate(cell_to_plot):
        plt.vlines(
            trial_spikes,
            trial_idx + 0.5,
            trial_idx + 1.5
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Trial")
    plt.yticks(range(1, len(cell_to_plot) + 1))
    plt.title("Raster plot")
    plt.axvline(0, color="red")
    plt.axvline(1, color="red")

#%% Execute
if __name__ == "__main__":
    main()

