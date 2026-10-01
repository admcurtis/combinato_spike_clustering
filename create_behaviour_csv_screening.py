# %% Dependencies
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# %% Paths
root = Path("../ieeg_data")

# Patient x visit x path dictionary
behave_files = [
    p for p in root.rglob("*Screening*.mat")
    if len(p.parts) == 6
]
