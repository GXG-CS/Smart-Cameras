#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"/..

# Variables to store the default arguments
INPUT_DATA="results/pi3b/tf_pose_estimation/cluster/kmeans/5/pi3b_tf_pose_estimation_results_20231012_201953.csv"
OUTPUT="results/pi3b/tf_pose_estimation/cluster/kmeans/5/"

# Run the python script using the default arguments
python evaluation.py --data $INPUT_DATA --output $OUTPUT

echo "Evaluation completed. Metrics saved to $OUTPUT."

