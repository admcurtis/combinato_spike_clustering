#%% Dependencies
import pandas as pd 

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




    