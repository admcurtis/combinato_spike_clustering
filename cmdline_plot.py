#%% Dependencies
from plotting_functions import plot_raster, plot_unit
import matplotlib.pyplot as plt
import pandas as pd
import sys

#%% Constants
_, ppt, sensor, unit, stim = sys.argv

ppt = int(ppt)
sensor = int(sensor)
unit = int(unit)

#%% Plot
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

