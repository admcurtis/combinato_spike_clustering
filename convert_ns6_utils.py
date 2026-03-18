#%% functions
def sort_data_chronologically(paths, visit_data):
    """
    Takes a list of paths and a list of loaded ns6 files and sorts them chronologically
    """
    paired = list(zip(paths, visit_data))
    paired_sorted = sorted(
        paired,
        key=lambda x: x[1].basic_header["TimeOrigin"]
    )   
    paths_sorted, visit_data_sorted = zip(*paired_sorted)

    return list(paths_sorted), list(visit_data_sorted)


def get_odd_chans(chan_ids):
    """
    Takes a list of lists where each inner list is the channel ids in each run in a visit
    Returns a list of lists with only odd channels. Even numbered channels are not
    brain data
    """
    odd_idxs = []
    for chan_lst in chan_ids:
        odd_idx = [i for i, chan in enumerate(chan_lst) if int(chan) % 2 == 1]
        odd_idxs.append(odd_idx)
    
    odd_chans = []
    for chan_id, odd_idx in zip(chan_ids, odd_idxs):
        odds = [chan_id[i] for i in odd_idx]
        odd_chans.append(odds)

    return odd_chans, odd_idxs


def get_odd_signals(signals, odd_idxs):
    """
    Takes a list of signals, and gets only the rows corresponding to odd numberd channels
    Returns a list of signals with only odd numbered ids. 
    """
    odd_signals = [sig[idx, :] for sig, idx in zip(signals, odd_idxs)]
    return odd_signals


def remove_stimulus_chan(chans, signals, stim_chans=(129, 257)):
    """
    Remove stimulus channels and corresponding rows from signals.
    Returns: new_chans, new_signals
    """
    new_chans = []
    new_signals = []
    for chan_list, sig_array in zip(chans, signals):
        keep_idx = [i for i, chan in enumerate(chan_list) if chan not in stim_chans]
        
        new_chans.append([chan_list[i] for i in keep_idx])
        new_signals.append(sig_array[keep_idx, :])

    return new_chans, new_signals