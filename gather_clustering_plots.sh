#!/bin/bash

pattern="./processed_data/Patient*/Patient*/overview"
destination="./clustering_plots/"

mkdir -p "$destination"

find . -type d -path "$pattern" -print0 | while IFS= read -r -d '' dir; do
    echo "Processing $dir"
    find "$dir" -maxdepth 1 -type f -exec mv {} "$destination"/ \;
done


