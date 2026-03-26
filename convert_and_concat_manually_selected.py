#%% DEPENDENCIES
# type: ignore
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from brpylib import NsxFile
from convert_ns6_utils import sort_data_chronologically

#%%
data = pd.read_csv("./practice_sensor_selection.csv")

#%%
os.makedirs("processed_data", exist_ok=True)
log_file = "run_errors.txt"

# Glob paths to all .ns6 files
root = Path("../ieeg_data")
ns6_files = [
    p for p in root.rglob("*.ns6")
    if "Visit" in str(p)
    and "Baseline" not in str(p)
    and "Closed Loop" not in str(p)
]

# Patient x visit x path dictionary
groups = defaultdict(lambda: defaultdict(list))
for path in ns6_files:
    try:
        patient = next(part for part in path.parent.parts if "Patient" in part)
        visit = next(part for part in path.parent.parts if "Visit" in part)
    except StopIteration:
        continue
    groups[patient][visit].append(str(path))


#%% 
patient = "Patient 1"
visit = "Visit 2"
paths = groups[patient][visit]

patient = patient.replace(" ", "")
visit = visit.replace(" ", "")

# Load all runs for this visit
visit_data = [NsxFile(f) for f in paths]

sorted_paths, sorted_data = sort_data_chronologically(paths, visit_data)

full_data = [f.getdata() for f in sorted_data]

signals = [np.array(f["data"]).squeeze() for f in full_data]
chan_ids = [f["elec_ids"] for f in full_data]
samp_rates = [float(f["samp_per_s"]) for f in full_data]
samples = [data.shape[-1] for data in signals]

#%% 
data_filtered = data[
    (data["patient"] == patient) &
    (data["visit"] == visit)
]

