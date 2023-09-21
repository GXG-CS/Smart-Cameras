#!/bin/bash

# Hard coded video path (This will be the default path for all applications unless you specify a unique path for each)
VIDEO_PATH="../output.avi"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Get the current timestamp

# Create the results folder if it doesn't exist
mkdir -p results

# Define configurations
CPU_SPEEDS=(1800 1700 1600 1500 1400 1300 1200 1100 1000 900 800 700 600)
CORE_CONFIGS=("0-3" "0-1" "0")

# Define applications and their respective python files
declare -A APPS=(
    ["image_classification"]="tf_image_classification/classify.py"
    ["image_segmentation"]="tf_image_segementation/segment.py"
    ["object_detection"]="tf_objection_detection/detect_fps.py"
    ["pose_estimation"]="tf_pose_estimation/pose_estimation.py"
)

# Loop through applications
for app in "${!APPS[@]}"; do

    OUTPUT_FILE="results/${app}_results_$TIMESTAMP.csv"

    # Ensure results file is empty or create it
    echo "CPU_Freq,CPU_Cores,FPS_Avg,FPS_Min,FPS_Max,Total_Time" > $OUTPUT_FILE  # CSV Header

    # Loop through configurations and run each python script
    for speed in "${CPU_SPEEDS[@]}"; do
        sudo cpufreq-set -f ${speed}MHz

        for cores in "${CORE_CONFIGS[@]}"; do

            # Run the current application's script and append the output to the results file
            FPS_RESULTS=$(taskset -c ${cores//[-]/,} python ${APPS[$app]} --videoPath $VIDEO_PATH)

            # Construct and print the CSV-style line to the file
            echo "$speed,$cores,$FPS_RESULTS" | tee -a $OUTPUT_FILE
        done
    done
done
