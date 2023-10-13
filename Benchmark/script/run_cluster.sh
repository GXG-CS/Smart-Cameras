#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"/..

# Variables to store the default arguments
DATA="data/pi3b_tf_pose_estimation_results.csv"
METHOD="kmeans"
N_CLUSTERS=6
OUTPUT="results/pi3b/tf_pose_estimation/cluster/kmeans/6"

# Run the python script using the default arguments
python cluster.py --data $DATA --method $METHOD --n_clusters $N_CLUSTERS --output $OUTPUT

echo "Clustering completed. Clustered data saved to $OUTPUT."
