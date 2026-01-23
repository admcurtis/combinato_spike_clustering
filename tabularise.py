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


def create_event_df(stim_start_end, cluster_labels, sensor, ppt_num, tmin=0.0):

    rows = []
    for stim, intervals in stim_start_end.items():
        for i, (start, end) in enumerate(intervals):
            rows.append(
                 {'stimulus': stim, 'exemplar_num': i+1, 'start': start, 'end': end}
            )

    temp = []
    for unit in np.unique(cluster_labels):
        # rows for stimului
        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        df = df.sort_values(by=["start"])
        df.insert(4, "trial_num", list(range(1, df.shape[0]+1)))
        df = df.sort_values(by=["ppt", "sensor", "unit", "stimulus"])
        temp.append(df)

        # rows for baseline
        df = pd.DataFrame(rows)
        df.insert(0, "unit", unit)
        df.insert(0, "sensor", sensor)
        df.insert(0, "ppt", ppt_num)
        df = df.assign(stimulus="BASELINE")
        df = df.sort_values(by=["start"])
        df.insert(4, "trial_num", list(range(1, df.shape[0]+1)))
        df["end"] = df["start"]
        df["start"] = df["start"].shift(1).fillna(0)
        df = df.sort_values(by=["ppt", "sensor", "unit", "stimulus"])
        temp.append(df)

    return pd.concat(temp, ignore_index=True)


def add_spike_counts(events_df, waveform_df):

    groups = ['ppt', 'sensor', 'unit', 'stimulus']
    spike_counts = events_df.copy()
    spike_counts['n_spikes'] = 0

    for keys, events_grp in events_df.groupby(groups):

        spikes = waveform_df.loc[
            (waveform_df['ppt'] == keys[0]) &
            (waveform_df['sensor'] == keys[1]) &
            (waveform_df['unit'] == keys[2]) &
            (waveform_df['stimulus'] == keys[3]),
            'spike_time'
        ].to_numpy()

        if spikes.size == 0:
            continue

        starts = events_grp['start'].to_numpy()
        ends = events_grp['end'].to_numpy()

        if events_grp["stimulus"].iloc[0] == "BASELINE":
            counts = (
                (spikes[:, None] >= ends - 0.3) &
                (spikes[:, None] < ends)
            ).sum(axis=0)
        else:
            counts = (
                (spikes[:, None] >= starts + 0.3) &
                (spikes[:, None] <= ends)
            ).sum(axis=0)

        spike_counts.loc[events_grp.index, 'n_spikes'] = counts

    return spike_counts




# NOTE: this is fast but the intial merge requires a tonne of memory for big datasets.
def add_spike_counts2(events_df, waveform_df):

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