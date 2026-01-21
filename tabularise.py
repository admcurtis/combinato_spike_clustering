#%% Dependencies
import pandas as pd 
import numpy as np

#%%
def create_waveform_df(spikes_per_stimulus, sensor, ppt_num):

    rows = [
        {"unit": unit, "stimulus": stim, "spike_times": spikes}
        for unit, stim_dict in spikes_per_stimulus.items()
        for stim, spikes in stim_dict.items()
    ]

    df = pd.DataFrame(rows)
    df.insert(0, "sensor", sensor)
    df.insert(0, "ppt", ppt_num)

    df_long= (
        df
        .explode("spike_times")
        .dropna(subset=["spike_times"])
        .rename(columns={"spike_times": "spike_time"})
    )

    df_long["spike_time"] = pd.to_numeric(df_long["spike_time"], errors="coerce")

    return df_long


def create_event_df(stim_start_end, cluster_labels, sensor, ppt_num, tmin=0.3):

    rows = []
    for stim, intervals in stim_start_end.items():
        for i, (start, end) in enumerate(intervals):
            rows.append(
                 {'stimulus': stim, 'trial_num': i+1, 'start': start, 'end': end}
            )

    temp = []
    for unit in np.unique(cluster_labels):
        # rows for stimului
        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        temp.append(df)

        # rows for baseline
        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        df = df.assign(stimulus="BASELINE")
        df["end"] = df["start"]
        df["start"] = df["start"] - tmin
        df = df.sort_values(by=["start"])
        df["trial_num"] = list(range(1, 301))
        temp.append(df)

    return pd.concat(temp, ignore_index=True)


def add_spike_counts(events_df, waveform_df):

    # Merge spikes onto stimulus presentations
    merged = events_df.merge(
        waveform_df,
        on=['ppt', 'sensor', 'unit', 'stimulus'],
        how='left'
    )

    # Keep only spikes that fall within the stimulus window
    in_window = merged[
        (merged['spike_time'] >= merged['start']) &
        (merged['spike_time'] <= merged['end'])
    ]

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