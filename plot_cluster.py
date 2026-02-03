#%% Dependencies 
import matplotlib.pyplot as plt
import pandas as pd

#%% Constants
ppt = 1 
sensor = 0
unit = 2

#%% functon
def plot_unit(waveforms, ppt=1, sensor=0, unit=2):

    cluster = waveforms[
    (waveforms["ppt"] == ppt)  &
    (waveforms["sensor"] == sensor) &
    (waveforms["unit"] == unit)
    ]

    cluster = cluster.iloc[:, 5:]

    plt.figure()

    for _, row in cluster.iterrows():
        plt.plot(row.values)

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("All Waveforms")
    plt.show()

if __name__ == "__main__":
    waveforms = pd.read_csv("all_spike_waveforms.csv")
    plot_unit(waveforms, ppt, sensor, unit)

