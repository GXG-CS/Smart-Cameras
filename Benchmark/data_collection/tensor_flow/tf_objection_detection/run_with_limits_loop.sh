#!/bin/bash

# Hard coded video path
VIDEO_PATH="/home/pi/smartCam/examples/lite/examples/object_detection/raspberry_pi/output.avi"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Get the current timestamp
OUTPUT_FILE="results/fps_$TIMESTAMP.txt"

# Create the results folder if it doesn't exist
mkdir -p results

# Ensure results file is empty or create it
echo "CPU_Speed,CPU_Cores,Avg_FPS,Min_FPS,Max_FPS" > $OUTPUT_FILE  # CSV Header

# Define configurations
CPU_SPEEDS=(1800 1700 1600 1500 1400 1300 1200 1100 1000 900 800 700 600)
CORE_CONFIGS=("0-3" "0-1" "0")

# Loop through configurations and run detect_fps.py
for speed in "${CPU_SPEEDS[@]}"; do
    sudo cpufreq-set -f ${speed}MHz
    
    for cores in "${CORE_CONFIGS[@]}"; do

        # Run the detect_fps.py script and append the output to the results file
        FPS_RESULTS=$(taskset -c ${cores//[-]/,} python3 detect_fps.py --videoPath $VIDEO_PATH)
        
        # Construct and print the CSV-style line to the file
        echo "$speed,$cores,$FPS_RESULTS" | tee -a $OUTPUT_FILE
    done
done


