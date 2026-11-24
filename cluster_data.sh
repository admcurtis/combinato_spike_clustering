#!/bin/bash/

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add to PATH and PYTHONPATH
PATH=$PATH:$SCRIPT_DIR/combinato
PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/combinato
export PATH PYTHONPATH

# Save plots straight to file without Qt. With a display available, matplotlib
# otherwise picks its Qt backend, and the combinato env has two Qt builds
# (conda qt-main and pip PyQt5-Qt5) that abort Python when both load.
export MPLBACKEND=Agg

# Loop over participants and perform clustering
for ppt_dir in */; do

    cd $ppt_dir

    # loop over .mat files and extract spikes
    for mat_file in *.mat; do
        css-extract --matfile $mat_file --matfile-scale-factor 0.25;
    done

    # Pass channel files explicitly: with no arguments these tools only find
    # folders matching options['folder_patterns'] in combinato/options.py,
    # which doesn't match PatientNN_VisitN_sensorN.
    for data_file in Patient*/data_*.h5; do
        css-mask-artifacts --file "$data_file"
    done
    css-plot-extracted --datafiles Patient*/data_*.h5

    for directory in Patient*/; do
        cd $directory
        h5_file=(*h5)

        # ---- Negative spikes ----
        css-prepare-sorting --neg --data "$h5_file"
        job_file=(sort_neg*txt)
        css-cluster --jobs "$job_file"
        css-combine --jobs "$job_file"
        css-plot-sorted --neg --datafile "$h5_file" --label sort_neg_ada

        # ---- Positive spikes ----
        css-prepare-sorting --data "$h5_file"
        job_file=(sort_pos*txt)
        css-cluster --jobs "$job_file"
        css-combine --jobs "$job_file"
        css-plot-sorted --datafile "$h5_file" --label sort_pos_ada

        cd ..
    done

    cd ..
done