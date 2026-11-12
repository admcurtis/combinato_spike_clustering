# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# %%
data = pd.read_csv("all_spikes.csv")

# %%
data_mat = np.array(data[[f"s{i}" for i in range(64)]])
wf = data_mat[200,:]
patient = "Patient10"
visit = "Visit1"
sensor = "sensor1"
group = "1"


# %% Functions
def fwhm(wf, spike_idx=19, baseline=0.0):
    """
    Full width at half maximum of a negative-going spike, in samples.

    wf       : waveform
    baseline : value the waveform rests at
    """
    wf = np.asarray(wf, dtype=float).ravel()
    n = wf.size

    # Remove baseline and flip so the spike is positive
    wf = -(wf - baseline)

    half_amp = wf[spike_idx] / 2.0

    # Walk left from the peak to the half-max crossing
    left_idx = spike_idx
    while left_idx > 0 and wf[left_idx - 1] > half_amp:
        left_idx -= 1

    # Interpolate left crossing
    if left_idx == 0:
        left = np.nan
    else:
        # Last sample at or below half amplitude
        prior_idx = left_idx - 1

        # Fraction of the way from prior_idx to left_idx where wf crosses half_amp
        frac = (half_amp - wf[prior_idx]) / (wf[left_idx] - wf[prior_idx])
        left = prior_idx + frac

    # Walk right from the peak to the half-max crossing
    right_idx = spike_idx
    while right_idx < n - 1 and wf[right_idx + 1] > half_amp:
        right_idx += 1

    # Interpolate right crossing
    if right_idx == n - 1:
        right = np.nan
    else:
        # Fraction of the way from right_idx to right_idx + 1 where wf crosses half_amp
        frac = (wf[right_idx] - half_amp) / (wf[right_idx] - wf[right_idx + 1])
        right = right_idx + frac

    width = right - left

    if np.isnan(width):
        warnings.warn(
            "FWHM is NaN: waveform does not cross half-max on both sides",
            RuntimeWarning
        )

    half_level = baseline - half_amp

    return width, left, right, half_level


# %%
def hyperpol_peak(wf, spike_idx=19):
    """
    Finds the index of the peak hyperpolarisation point
    """

    after_spike = wf[spike_idx:]

    # Indices after the spike that are above zero
    hyperpolarised = np.flatnonzero(after_spike >= 0)

    # Make sure spike actually hyperpolarises
    if hyperpolarised.size == 0:
        return spike_idx, None                    

    # Find the first and last hyperpolarised points before returning to resting potential
    first_hyperpol = hyperpolarised[0]           
    below_rest_potential = np.flatnonzero(after_spike[first_hyperpol:] < 0)

    if below_rest_potential.size > 0:
        last_hyperpol = first_hyperpol + below_rest_potential[0]
    else:
        last_hyperpol = len(after_spike)   

    # Get peak hyperpolarisation
    hyperpole_peak_idx = (
        spike_idx + 
        first_hyperpol + 
        int(np.argmax(after_spike[first_hyperpol:last_hyperpol]))
    )

    return spike_idx, hyperpole_peak_idx


def peak_to_trough(wf):
    """
    Calculates the difference between the spike peak and peak hyperpolarisation
    """
    spike_idx, hyperpol_peak_idx = hyperpol_peak(wf)
    if hyperpol_peak_idx is None:
        return np.nan
    p2t_dist = wf[spike_idx] - wf[hyperpol_peak_idx]
    return p2t_dist


def plot_fwhm(wf, baseline=0.0):
    """
    Plot a waveform with its FWHM marked.
    """
    y = np.ravel(wf)

    width, left, right, half_level = fwhm(y, baseline=baseline)
    spike_idx, hyperpole_peak_idx = hyperpol_peak(wf)

    fig, ax = plt.subplots()
    ax.plot(y, ".-", label="waveform")
    ax.hlines(
        half_level,
        left,
        right,
        color="r",
        linewidth=2,
        label=f"FWHM = {width:.2f} samples")

    ax.hlines(
        wf[spike_idx],
        spike_idx,
        hyperpole_peak_idx,
        color="g",
        linewidth=2,
        linestyles="dashed"
    )

    ax.vlines(
        x=hyperpole_peak_idx,
        ymin=wf[spike_idx],
        ymax=wf[hyperpole_peak_idx],
        color="g",
        linewidth=2,
        label=f"P2T = {peak_to_trough(wf)}"
    )
    
    ax.axhline(
        baseline,
        color="grey",
        linestyle=":",
        linewidth=1)
    
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()

    return fig, ax

# %%
WF_COLS = [f"s{i}" for i in range(64)]

def add_waveform_stats(df):
    wf_array = df[WF_COLS].to_numpy()
    df = df.copy()
    df["widths"] = np.apply_along_axis(fwhm, axis=1, arr=wf_array)[:, 0]
    df["p2t"] = np.apply_along_axis(peak_to_trough, axis=1, arr=wf_array)
    df["hyperpolarises"] = df["p2t"].notna()
    print(f"{(~df['hyperpolarises']).sum()} waveforms with no hyperpolarisation")
    return df


def get_group(data, patient, visit, sensor, group):
    return data[
        (data["patient"] == patient) &
        (data["visit"] == visit) &
        (data["sensor"] == sensor) &
        (data["neuron_group"] == group)
    ].reset_index(drop=True)


# %% Plot 
def plot_group(group_df):
    fig, ax = plt.subplots()
    for cls, g in group_df.groupby("neuron_class"):
        ax.scatter(g["widths"], g["p2t"], s=12, alpha=0.7, label=cls)

    ax.set_xlabel("FWHM (samples)")
    ax.set_ylabel("Peak-to-trough amplitude")
    ax.legend(title="Neuron class")
    plt.show()


# %%
def plot_groups(*group_dfs):
    fig, ax = plt.subplots()
    for df in group_dfs:
        group = df["neuron_group"].iloc[0]
        ax.scatter(df["widths"], df["p2t"], s=12, alpha=0.7, label=f"Group {group}")

    ax.set_xlabel("FWHM (samples)")
    ax.set_ylabel("Peak-to-trough amplitude")
    ax.legend(title="Neuron group")
    plt.show()


# %%
data = add_waveform_stats(data)

# %%
group1 = get_group(data, patient, visit, sensor, 1)
group2 = get_group(data, patient, visit, sensor, 2)
group4 = get_group(data, patient, visit, sensor, 4)

plot_groups(group1, group4)
