import numpy as np

def spikes_to_stimuli(spike_times, stim_start_end, t_min=0.3, t_max=0.0):
    """
    Returns a dictionary with each stimulus as a key and a list of spikes times to that
    stimulus. The list is collapsed across trials.
    
    spike_times: list of spikes in each cluster
    stim_start_end: dict of onset/offset of each stimulus
    t_min: time after stim onset to count spikes from
    t_max: time relative to stim offset to count spikes from. default = stim offset
    """
    stim_spikes = {
        stim: spike_times[np.any(
            (spike_times[:, None] >= (start_end[:, 0] + t_min)) & 
            (spike_times[:, None] <= (start_end[:, 1] + t_max)),
            axis=1
        )]
        for stim, start_end in stim_start_end.items()
    }

    return stim_spikes


def group_spikes_per_trial(spike_times, stim_duration=0.7) -> list[list]:
    """
    Takes list of all spikes to a given stimulus across trials
    and returns list of lists with each sublist concerning a single trial
    """
    if len(spike_times) == 0:
        return []
    diffs = np.diff(spike_times)
    split_indices = np.where(diffs > stim_duration)[0] + 1
    split_stims = np.split(spike_times, split_indices)
    return [stim.tolist() for stim in split_stims]


def count_spikes(spike_times, stim_start_end) -> dict[str, list]:
    """
    For a cluster (neuron), count the number of times the unit spiked for each stimulus 
    and during the baseline periods

    Returns a dict with stimuli as keys and a list of spikes as values
    The final key is BASELINE indicating when the neuron fired during
    the baseline period.
    """

    stim_spikes = spikes_to_stimuli(spike_times, stim_start_end, t_min=0.3)

    # group spikes by trial
    stim_spikes = {k: group_spikes_per_trial(v) for k,v in stim_spikes.items()}

    all_intervals = np.array([
        interval 
        for stim_interval in stim_start_end.values()
        for interval in stim_interval
    ])

    # Baseline: -700ms - 0ms 
    # all spikes that are less than stim onset but greater than onset - 0.7
    baseline_spikes = spike_times[np.any(
            (spike_times[:, None] <= all_intervals[:, 0]) &
            (spike_times[:, None] >= all_intervals[:, 0] - 0.7),
            axis=1
        )]
    
    # group baseline spikes per trial
    baseline_spikes = group_spikes_per_trial(baseline_spikes)
    
    stim_spikes["BASELINE"] = baseline_spikes
    
    return stim_spikes