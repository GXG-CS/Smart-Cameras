#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"/..

# Run the python script using relative paths
python cluster.py --data data/pi3b_tf_pose_estimation_results.csv --method kmeans --n_clusters 4 --output results/cluster
