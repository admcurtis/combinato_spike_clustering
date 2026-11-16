# %% Set interactive backend to open plots in a separate window
# type: ignore
%matplotlib qt

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% Constants
WF_COLS = [f"s{i}" for i in range(64)]
PATIENT = 10
VISIT = 1
SENSOR = 1

patient = f"Pateint{PATIENT}"
visit = f"Visit{VISIT}"
sensor = f"sensor{SENSOR}"

# %%
data = pd.read_csv("all_spikes.csv")

# %% Metrics
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
    peak = wf[spike_idx]
    if hyperpol_peak_idx is None:
        return np.nan
    p2t_dist = peak - wf[hyperpol_peak_idx]
    return p2t_dist, peak


def plot_fwhm(wf, baseline=0.0):
    """
    Plot a waveform with its FWHM marked.
    """
    y = np.ravel(wf)

    width, left, right, half_level = fwhm(y, baseline=baseline)
    spike_idx, hyperpole_peak_idx = hyperpol_peak(wf)

    p2t, peak = peak_to_trough(wf)

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
        label=f"P2T = {p2t}"
    )
    
    ax.axhline(
        baseline,
        color="grey",
        linestyle=":",
        linewidth=1)

    ax.plot(spike_idx, wf[spike_idx], "o", color="r", markersize=5,
        label=f"Peak = {peak:.2f}")
    
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()

    return fig, ax

# %% Filtering
def add_waveform_stats(df):
    wf_array = df[WF_COLS].to_numpy()
    df = df.copy()
    df["widths"] = np.apply_along_axis(fwhm, axis=1, arr=wf_array)[:, 0]
    df[["p2t", "peak"]] = np.apply_along_axis(peak_to_trough, axis=1, arr=wf_array)
    df["hyperpolarises"] = df["p2t"].notna()
    print(f"{(~df['hyperpolarises']).sum()} waveforms with no hyperpolarisation")
    return df

def filter_data(data, patient, visit, sensor):
    return data[
        (data["patient"] == patient) &
        (data["visit"] == visit) &
        (data["sensor"] == sensor)
    ].reset_index(drop=True)


def get_groups(data, patient, visit, sensor):
    filtered = filter_data(data, patient, visit, sensor)
    group_dfs = {label: group for label, group in filtered.groupby("neuron_group")}
    return group_dfs
    

# %% Plotting
def plot_classes(group_df):
    fig, ax = plt.subplots()
    for cls, g in group_df.groupby("neuron_class"):
        ax.scatter(g["widths"], g["p2t"], s=12, alpha=0.7, label=cls)

    ax.set_xlabel("FWHM (samples)")
    ax.set_ylabel("Peak-to-trough amplitude")
    ax.legend(title="Neuron class")
    plt.show()


def plot_groups_3d(*group_dfs, standardise=False, model="lasso"):
    cols = ["widths", "p2t", "peak"]
    arrays = [df[cols].to_numpy(dtype=float) for df in group_dfs]

    if standardise:
        pooled = np.vstack(arrays)
        mu, sd = np.nanmean(pooled, axis=0), np.nanstd(pooled, axis=0)
        arrays = [(a - mu) / sd for a in arrays]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    for df, a in zip(group_dfs, arrays):
        group = df["neuron_group"].iloc[0]
        ax.scatter(a[:, 0], a[:, 1], a[:, 2],
                   s=12, alpha=0.7, label=f"Group {group}")

    auc, auc_sd = neuron_auc_test(*group_dfs, model=model, standardise=standardise)

    suffix = " (z)" if standardise else ""
    ax.set_xlabel("FWHM" + ("" if standardise else " (samples)") + suffix)
    ax.set_ylabel("Peak-to-trough amplitude" + suffix)
    ax.set_zlabel("Peak amplitude" + suffix)
    ax.legend(title="Neuron group")
    ax.set_title(f"{model} AUC = {auc:.3f} ± {auc_sd:.3f}")

    plt.show()

# %% Models
def neuron_auc_test(
        *group_dfs,
        model="lasso",
        n_splits=5,
        seed=0,
        standardise=True,
    ):

    cols = ["widths", "p2t", "peak"]

    if model == "gradBoost":
        arrays = [df[cols].to_numpy(dtype=float) for df in group_dfs]
        clf = HistGradientBoostingClassifier()
    elif model == "lasso":
        arrays = [df[cols].dropna().to_numpy(dtype=float) for df in group_dfs]
        clf = LogisticRegression(penalty="l1", solver="saga", max_iter=5000)

    X = np.vstack(arrays)
    y = np.concatenate([np.full(len(a), i) for i, a in enumerate(arrays)])

    model = make_pipeline(StandardScaler(), clf) if standardise else clf

    scoring = "roc_auc" if len(arrays) == 2 else "roc_auc_ovr"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return scores.mean(), scores.std()


# %%
data_mat = np.array(data[WF_COLS])
wf = data_mat[200,:]
plot_fwhm(wf)

# %%
data = add_waveform_stats(data)
groups = get_groups(data, patient, visit, sensor)
plot_groups_3d(groups[1], groups[2])
