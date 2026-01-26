#%% Dependencies 
import pandas as pd
import numpy as np
import tabularise

#%% Load data
waveform_data = pd.read_csv("./all_spike_waveforms.csv")
event_data = pd.read_csv("./all_events.csv")


#%% Script
# Count spikes for each particiapnt individually
spike_countss = []
for ppt in np.unique(waveform_data["ppt"]):
    temp_waveform = waveform_data[waveform_data["ppt"] == ppt]
    temp_events = event_data[event_data["ppt"] == ppt]

    spike_counts = tabularise.add_spike_counts(temp_events, temp_waveform)

    spike_countss.append(spike_counts)

spike_counts = pd.concat(spike_countss, ignore_index=True)


#%% Santiy Check
if sum(spike_counts["n_spikes"]) != waveform_data.shape[0]:

    print(f"Error: sum(n_spikes)={sum(spike_counts['n_spikes'])}, "
          f"waveform_df rows={waveform_data.shape[0]}")
    
    raise ValueError("Spike counts do not match waveform rows!")

