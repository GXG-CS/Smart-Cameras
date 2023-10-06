#!/bin/bash

source /home/pi/miniforge3/etc/profile.d/conda.sh

conda activate gxg

# Define the folder where sdram_freq.txt, counter.txt, and log.txt are located
folder_path="/home/pi/Smart-Cameras/sdram_freq"

# Initialize counter.txt if it doesn't exist
if [ ! -f "$folder_path/counter.txt" ]; then
  echo "0" > "$folder_path/counter.txt"
fi

# Initialize log.txt if it doesn't exist
if [ ! -f "$folder_path/log.txt" ]; then
  touch "$folder_path/log.txt"
fi

# Initialize results.csv if it doesn't exist
if [ ! -f "$folder_path/results.csv" ]; then
  echo "sdram_freq,cpu_cores,cpu_freq,avg_fps" > "$folder_path/results.csv"
fi

# Read the current counter value
counter=$(cat "$folder_path/counter.txt")

# Read the sdram_freq values into an array from the specified path
IFS=',' read -ra sdram_freq_list <<< "$(cat "$folder_path/sdram_freq.txt")"
sdram_freq_list_size=${#sdram_freq_list[@]}

# Define CPU configurations
CPU_SPEEDS=("1400" "1300")
CORE_CONFIGS=("4" "2")

# Log the current sdram_freq and counter value
echo "Current counter: $counter" >> "$folder_path/log.txt"
echo "Current sdram_freq from list: ${sdram_freq_list[$counter]}" >> "$folder_path/log.txt"

# Loop through CPU configurations
for speed in "${CPU_SPEEDS[@]}"; do
  # Set CPU speed and cores here
  sudo cpufreq-set -f ${speed}MHz

  for cores in "${CORE_CONFIGS[@]}"; do

    # Log the current CPU configuration to log.txt
    echo "Setting CPU cores to $cores and CPU frequency to ${speed}MHz" >> "$folder_path/log.txt"

    # Log the actual CPU configuration to log.txt
    # echo "Actual CPU Configuration:" >> "$folder_path/log.txt"
    # echo "Number of cores: $(nproc)" >> "$folder_path/log.txt"
    # sudo cpufreq-info | grep 'current CPU frequency is' >> "$folder_path/log.txt"
    echo "Current CPU frequency: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz" >> "$folder_path/log.txt"

    # Generate a comma-separated list of core numbers
    core_list=$(seq -s, 0 $((cores - 1)))

    # Run the pose_estimation.py script and capture the average FPS
    avg_fps=$(taskset -c $core_list python /home/pi/Smart-Cameras/tf_pose_estimation/pose_estimation.py --videoPath /home/pi/Smart-Cameras/output.avi | awk -F',' '{print $1}')

    # Log the results to CSV
    echo "${sdram_freq_list[$counter]},$cores,$speed,$avg_fps" >> "$folder_path/results.csv"
  done
done

# Increment the counter
((counter++))
echo $counter > "$folder_path/counter.txt"

# Check if counter is less than the size of sdram_freq list
if [ $counter -lt $sdram_freq_list_size ]; then
  # Set the next sdram_freq value
  next_sdram_freq=${sdram_freq_list[$counter]}
  echo "Setting next sdram_freq to $next_sdram_freq and rebooting" >> "$folder_path/log.txt"

  # Check if sdram_freq already exists in /boot/config.txt
  if grep -q "^sdram_freq=" /boot/config.txt; then
    # Update the existing sdram_freq value
    sudo sed -i "s/^sdram_freq=[0-9]\+/sdram_freq=$next_sdram_freq/" /boot/config.txt
  else
    # Append the new sdram_freq value
    echo "sdram_freq=$next_sdram_freq" >> /boot/config.txt
  fi

  # Reboot the system
  # sudo reboot
else
  echo "Counter has reached the end of the sdram_freq list. Not rebooting." >> "$folder_path/log.txt"
fi
