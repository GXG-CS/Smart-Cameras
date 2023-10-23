#!/bin/bash

VIDEO_PATH="../output.avi"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Get the current timestamp
OUTPUT_FILE="../results/obj_detection_results_$TIMESTAMP.csv"

# Create the results folder if it doesn't exist
mkdir -p results

# Ensure results file is empty or create it
echo "CPU_Freq,CPU_Cores,RAM_Limit_MB,Storage_Limit_MB,FPS_Avg,FPS_Min,FPS_Max,Total_Time" > $OUTPUT_FILE  # CSV Header

# Define configurations
CPU_SPEEDS=(1800 1700 1600 1500 1400 1300 1200 1100 1000 900 800 700 600)
CORE_CONFIGS=("0-3" "0-1" "0")
RAM_LIMITS=(1024 512 256)  # RAM limits in MB
STORAGE_LIMITS=(256 128 64)  # Storage limits in MB

# Loop through configurations and run detect_fps.py
for speed in "${CPU_SPEEDS[@]}"; do
    sudo cpufreq-set -f ${speed}MHz
    
    for cores in "${CORE_CONFIGS[@]}"; do
        for ram in "${RAM_LIMITS[@]}"; do
            for storage in "${STORAGE_LIMITS[@]}"; do

                # Create a mount point if it doesn't exist
                sudo mkdir -p /mnt/tmpfs

                # Create a temporary filesystem with limited storage
                sudo mount -t tmpfs -o size=${storage}M tmpfs /mnt/tmpfs

                # Set RAM Limit
                ram_kb=$((ram * 1024))
                ulimit -m $ram_kb

                # Run the detect_fps.py script and append the output to the results file
                FPS_RESULTS=$(taskset -c ${cores//[-]/,} python3 detect_fps.py --videoPath $VIDEO_PATH)

                # Dismount the temporary filesystem
                sudo umount /mnt/tmpfs

                # Construct and print the CSV-style line to the file
                echo "$speed,$cores,$ram,$storage,$FPS_RESULTS" | tee -a $OUTPUT_FILE
            done
        done
    done
done

