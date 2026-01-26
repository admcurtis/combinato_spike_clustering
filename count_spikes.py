#%% Dependencies 
import pandas as pd
import numpy as np

#%% Load data
waveform_data = pd.read_csv("./all_spike_waveforms.csv")
event_data = pd.read_csv("./all_events.csv")


#%% Function
# NOTE: merged can grow exponentially with large datasets, apply to each ppt individually
def add_spike_counts(events_df, waveform_df, s_min=0, b_min=0):

    """
    s_min: Time after stim onset to start counting spikes in seconds
    b_min: Time before stim onset to start counting spikes for the baseline period in seconds
    """

    stim_events = events_df[events_df["stimulus"] != "BASELINE"]
    stim_waveforms = waveform_df[waveform_df["stimulus"] != "BASELINE"]

    bl_events = events_df[events_df["stimulus"] == "BASELINE"]
    bl_waveforms = waveform_df[waveform_df["stimulus"] == "BASELINE"]

    # Merge spikes onto stimulus presentations
    merged_stim = stim_events.merge(
        stim_waveforms,
        on=['ppt', 'sensor', 'unit', 'stimulus'],
        how='left'
    )

    # Merge spikes onto stimulus presentations
    merged_bl = bl_events.merge(
        bl_waveforms,
        on=['ppt', 'sensor', 'unit', 'stimulus'],
        how='left'
    )

    # Keep only spikes that fall within the stimulus window
    stim_in_window = merged_stim[
        (merged_stim['spike_time'] >= merged_stim['start'] + s_min) &
        (merged_stim['spike_time'] <= merged_stim['end'])
    ]

    # Keep only spikes that fall within the stimulus window
    bl_in_window = merged_bl[
        (merged_bl['spike_time'] >= merged_bl['end'] - b_min) &
        (merged_bl['spike_time'] <= merged_bl['end'])
    ]

    in_window = pd.concat([stim_in_window, bl_in_window], ignore_index=True)

    # Count spikes per stimulus presentation
    spike_counts = (
        in_window
        .groupby(events_df.columns.tolist(), dropna=False)
        .size() # number of rows in each group, i.e., spike count
        .reset_index(name='n_spikes')
    )

    # Merge spike counts into events_df
    spike_count_df = events_df.merge(
        spike_counts,
        on=events_df.columns.tolist(),
        how='left'
    )

    spike_count_df['n_spikes'] = spike_count_df['n_spikes'].fillna(0).astype(int)

    return spike_count_df


#%% Script
spike_counts_list = []
for ppt in np.unique(waveform_data["ppt"]):
    temp_waveform = waveform_data[waveform_data["ppt"] == ppt]
    temp_events = event_data[event_data["ppt"] == ppt]

    spike_counts = add_spike_counts(temp_events, temp_waveform, s_min=0.3, b_min=0.3)

    spike_counts_list.append(spike_counts)

spike_counts = pd.concat(spike_counts_list, ignore_index=True)


#%% Santiy Check
# NOTE: only works correctly if spike_counts hasn't been filtered using t_min/t_max
if sum(spike_counts["n_spikes"]) != waveform_data.shape[0]:

    print(f"Error: sum(n_spikes)={sum(spike_counts['n_spikes'])}, "
          f"waveform_df rows={waveform_data.shape[0]}")
    
    raise ValueError("Spike counts do not match waveform rows!")


#%% Save spike counts
spike_counts.to_csv("spike_counts.csv", index=False)
