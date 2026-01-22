#%% Dependencies
import matplotlib.pyplot as plt
import pandas as pd

#%% Load data
concept_cells = pd.read_csv("./detected_concepts.csv")
spike_data = pd.read_csv("./spike_waveforms.csv")
event_times = pd.read_csv("./spike_counts.csv")

#%%

concept_spikes = spike_data.merge(
    concept_cells,
    on=["ppt", "sensor", "unit", "stimulus"],
    how="inner"
)


