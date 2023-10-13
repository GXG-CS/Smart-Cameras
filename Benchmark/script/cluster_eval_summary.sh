#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"/..

# Variables to store the default arguments
DATA="data/pi3b_tf_pose_estimation_results.csv"
METHOD="kmeans"
BASE_OUTPUT="results/pi3b/tf_pose_estimation/cluster/kmeans"
EVAL_OUTPUT_FILE="results/pi3b/tf_pose_estimation/cluster_eval_summary.txt"

# Clear or create the EVAL_OUTPUT_FILE
echo "Evaluation Metrics Summary" > $EVAL_OUTPUT_FILE
echo "---------------------------" >> $EVAL_OUTPUT_FILE

# Function to get the latest generated csv file in a directory
get_latest_csv() {
    ls -t $1/*.csv | head -1
}

# Loop through the desired values of N_CLUSTERS
for N_CLUSTERS in {6..12}; do
    OUTPUT="$BASE_OUTPUT/$N_CLUSTERS"
    # Run clustering
    python cluster.py --data $DATA --method $METHOD --n_clusters $N_CLUSTERS --output $OUTPUT
    echo "Clustering completed for n_clusters=$N_CLUSTERS. Clustered data saved to $OUTPUT."
    
    # Determine the latest generated csv file for evaluation
    INPUT_DATA=$(get_latest_csv $OUTPUT)
    # Run evaluation
    python evaluation.py --data $INPUT_DATA --output $OUTPUT
    echo "Evaluation completed for n_clusters=$N_CLUSTERS. Metrics saved to $OUTPUT."

    # Append the results to EVAL_OUTPUT_FILE
    echo "Metrics for n_clusters=$N_CLUSTERS:" >> $EVAL_OUTPUT_FILE
    # Dynamically get the latest .txt file from the directory
    LATEST_EVAL_FILE=$(ls -t $OUTPUT/*.txt | head -1)
    cat $LATEST_EVAL_FILE >> $EVAL_OUTPUT_FILE
    echo "---------------------------" >> $EVAL_OUTPUT_FILE

done

echo "All evaluations are gathered into $EVAL_OUTPUT_FILE."
