#!/bin/bash/

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add to PATH and PYTHONPATH
PATH=$PATH:$SCRIPT_DIR/combinato
PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/combinato
export PATH PYTHONPATH

# Loop over participants and perform clustering
for ppt_dir in */; do

    cd $ppt_dir

    # loop over .mat files and extract spikes
    for mat_file in *.mat; do
        css-extract --matfile $mat_file --matfile-scale-factor 0.25;
    done

    css-mask-artifacts --no-concurrent 
    css-plot-extracted

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