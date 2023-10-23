#!/bin/bash

VIDEO_PATH="../output.avi"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Get the current timestamp
OUTPUT_FILE="../results/image_classification_results_$TIMESTAMP.csv"

# Create the results folder if it doesn't exist
mkdir -p results

# Ensure results file is empty or create it
echo "CPU_Freq,CPU_Cores,FPS_Avg,FPS_Min,FPS_Max,Total_Time" > $OUTPUT_FILE  # CSV Header

# Define configurations
CPU_SPEEDS=(1800 1700 1600 1500 1400 1300 1200 1100 1000 900 800 700 600)
CORE_CONFIGS=("0-3" "0-1" "0")

# Loop through configurations and run classify.py
for speed in "${CPU_SPEEDS[@]}"; do
    sudo cpufreq-set -f ${speed}MHz
    
    for cores in "${CORE_CONFIGS[@]}"; do

        # Run the classify.py script and append the output to the results file
        FPS_RESULTS=$(taskset -c ${cores//[-]/,} python3 classify.py --videoPath $VIDEO_PATH)
        
        # Construct and print the CSV-style line to the file
        echo "$speed,$cores,$FPS_RESULTS" | tee -a $OUTPUT_FILE
    done
done
