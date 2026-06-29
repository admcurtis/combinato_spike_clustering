# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Paths
root = Path("../ieeg_data")

# Patient x visit x path dictionary
behave_files = [
    p for p in root.rglob("*assMemData.mat")
    if len(p.parts) == 6
]

# %% Process Study data
study_dfs = []
test_dfs = []
for behave_path in behave_files:

    patient = next(
        part for part in behave_path.parts if "Patient" in part
    ).replace(" ", "")

    visit = next(
        part for part in behave_path.parts if "Visit" in part
    ).replace(" ", "")

    behave_data = loadmat(
        behave_path,
        squeeze_me=True,
        struct_as_record=False
    )

    behave_data = behave_data["eegData"]

    # Study df
    study_df = pd.DataFrame(
        {
            "patient": patient,
            "visit":   visit,
            "event":   behave_data.Study.Event,
            "pair":    behave_data.Study.Pair
        }
    )

    stim_df = pd.DataFrame(
        behave_data.Study.Stimuli,
        columns = ["stim1", "stim2"]
    )

    time_df = pd.DataFrame(
        behave_data.Study.Times,
        columns = ["fixation_onset", "stim_onset"]
    )

    # Subsequent results
    sub_rec = pd.DataFrame(
        behave_data.Study.subRec,
        columns=["subRec_stim1", "subRec_stim2"]
    )

    sub_ass = pd.DataFrame(
        behave_data.Study.subAss,
        columns=["subAss_stim1", "subAss_stim2"]
    )

    sub_dep = pd.Series(
        behave_data.Study.subDep,
        name="subDep"
    )

    df = pd.concat(
        [study_df, stim_df, time_df, sub_rec, sub_ass, sub_dep],
        axis=1
    )

    study_dfs.append(df)

    # Test df
    test_df = pd.DataFrame(
        {
            "patient": patient,
            "visit":   visit,
            "event": behave_data.Test.Event.flatten("F"),
            "old_new": behave_data.Test.OldNew.flatten("F"),
            "cue_type": behave_data.Test.CueType.flatten("F"),
            "cue": behave_data.Test.Cue.flatten("F")
        }
    )

    times_df = pd.DataFrame(
        behave_data.Test.Times,
        columns=[
            "fixation_onset",
            "cue_onset",
            "oldNew_onset",
            "associate1_onset",
            "associate2_onset"
        ]
    )

    responses_df = pd.DataFrame(
        {
            "recRT":    behave_data.Test.recRT.flatten("F"),
            "recAcc":   behave_data.Test.recAcc.flatten("F"),
            "Target1":  behave_data.Test.Target1.flatten("F"),
            "Target2":  behave_data.Test.Target2.flatten("F"),
            "assResp1": behave_data.Test.assResp1.flatten("F"),
            "assResp2": behave_data.Test.assResp2.flatten("F"),
            "assRT1":   behave_data.Test.assRT1.flatten("F"),
            "assRT2":   behave_data.Test.assRT2.flatten("F"),
            "assAcc1":  behave_data.Test.assAcc1.flatten("F"),
            "assAcc2":  behave_data.Test.assAcc2.flatten("F")
        }
    )

    df = pd.concat(
        [test_df, times_df, responses_df],
        axis=1
    )

    test_dfs.append(df)

full_study_data = pd.concat(
    study_dfs,
    axis=0
)

full_test_data = pd.concat(
    test_dfs,
    axis=0
)

# %% 
# Load unclustered data to get sample rate and number of samples per run data
concat_paths = Path(f"./processed_data/{patient}").glob("*.mat")
concat_data = loadmat(next(concat_paths))