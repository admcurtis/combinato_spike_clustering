#%% Dependencies 
import matplotlib.pyplot as plt
import pandas as pd

#%% Constants
ppt = 1 
sensor = 5
unit = 2
stim = "Danny Dyer"

#%% functon
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

    plt.figure()

    for _, row in cluster.iterrows():
        plt.plot(row.values)

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(f"{stim} (n spikes = {n})")
    plt.show()

if __name__ == "__main__":
    waveforms = pd.read_csv("all_spike_waveforms.csv")
    plot_unit(waveforms, ppt, sensor, unit)

