import numpy as np

def spikes_to_stimuli(spike_times, stim_start_end, t_min=0.0):
    """
    Returns a dictionary with each stimulus as a key and a list of spikes times to that
    stimulus. The list is collapsed across trials.
    """
    stim_spikes = {
        stim: spike_times[np.any(
            (spike_times[:, None] >= (start_end[:, 0] + t_min)) & 
            (spike_times[:, None] <= (start_end[:, 1])),
            axis=1
        )]
        for stim, start_end in stim_start_end.items()
    }

    return stim_spikes


def spikes_at_baseline(spike_times, stim_start_end, t_min=0.0):
    """
    Returns a list a spikes that occured during the baseline.
    """

    # get all time time intervals collapsed across stimulus
    all_trial_times = np.array([
        trial
        for stimulus in stim_start_end.values()
        for trial in stimulus
    ])

    # Add a lagged column to get previous stim offset and next onset in same row
    all_trial_times = all_trial_times[np.argsort(all_trial_times[:, 0])]
    lagged_col = np.zeros(all_trial_times.shape[0])
    lagged_col[1:] = all_trial_times[:-1, 1]
    all_trial_times = np.column_stack((all_trial_times, lagged_col))

    baseline_spikes = spike_times[np.any(
            (spike_times[:, None] <= all_trial_times[:, 0]) &
            (spike_times[:, None] >= all_trial_times[:, 2] - t_min),
            axis=1
        )]
    
    return baseline_spikes


# DEPRECATED: NOT IN USE.
def group_spikes_per_trial(spike_times, stim_duration=1) -> list[list]:
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
