#!/bin/bash

#disable swap temporarily until the next reboot
sudo swapoff -a

# Start Xvfb on display :1
Xvfb :1 &
export DISPLAY=:1

# source /home/pi/miniforge3/etc/profile.d/conda.sh

# conda activate gxg

# Define the folder where sdram_freq.txt, counter.txt, and log.txt are located
folder_path="/home/pi/Smart-Cameras/Benchmark/data_collection/tensor_flow/tf_pose_estimation"
    
cd /home/pi/Smart-Cameras/Benchmark/data_collection/tensor_flow/tf_pose_estimation
python /home/pi/Smart-Cameras/Benchmark/data_collection/tensor_flow/tf_pose_estimation/pose_estimation.py --videoPath /home/pi/Smart-Cameras/output.avi

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
  # echo "sdram_freq,cpu_cores,cpu_freq,avg_fps,total_time" > "$folder_path/results.csv"
  echo "sdram_freq,cpu_cores,cpu_freq,avg_fps,total_time,mem_limit_kb" > "$folder_path/results.csv"
fi

# Read the current counter value
counter=$(cat "$folder_path/counter.txt")

# Read the sdram_freq values into an array from the specified path
IFS=',' read -ra sdram_freq_list <<< "$(cat "$folder_path/sdram_freq.txt")"
sdram_freq_list_size=${#sdram_freq_list[@]}

# Check if swap is enabled
swap_enabled=$(swapon -s | grep -c "Filename")

# Log swap status
if [ "$swap_enabled" -eq 0 ]; then
  echo "Swap is disabled on this system." >> "$folder_path/log.txt"
else
  echo "Swap is enabled on this system." >> "$folder_path/log.txt"
fi

# Output sdram_freq_list and sdram_freq_list_size for debugging
echo "Debug: sdram_freq_list: ${sdram_freq_list[*]}" >> "$folder_path/log.txt"
echo "Debug: sdram_freq_list_size: $sdram_freq_list_size" >> "$folder_path/log.txt"

# Define CPU configurations
CPU_SPEEDS=("1000" "900" "800" "700" "600")
# CPU_SPEEDS=("900")

CORE_CONFIGS=("4" "2" "1")
# CORE_CONFIGS=("4")

# Define Memory configurations (in kilobytes)
MEMORY_LIMITS=("262144" "524288" "1048576")  # 256MB, 512MB, 1024MB
# MEMORY_LIMITS=("262144")  # 256MB, 512MB, 1024MB

# Log the current sdram_freq and counter value
echo "Current counter: $counter" >> "$folder_path/log.txt"
echo "Current sdram_freq from list: ${sdram_freq_list[$counter]}" >> "$folder_path/log.txt"

# Loop through CPU configurations
for speed in "${CPU_SPEEDS[@]}"; do
  # Set CPU speed and cores here
  sudo cpufreq-set -f ${speed}MHz

  for cores in "${CORE_CONFIGS[@]}"; do

    for mem_limit in "${MEMORY_LIMITS[@]}"; do
      # Log the current CPU configuration to log.txt
      echo "Setting CPU cores to $cores and CPU frequency to ${speed}MHz" >> "$folder_path/log.txt"
      echo "Current CPU frequency: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz" >> "$folder_path/log.txt"

      # Log the current Memory configuration to log.txt
      echo "Setting memory limit to $mem_limit kB" >> "$folder_path/log.txt"

      # Generate a comma-separated list of core numbers
      core_list=$(seq -s, 0 $((cores - 1)))

      # Run the pose_estimation.py script and capture the average FPS
      cd /home/pi/Smart-Cameras/Benchmark/data_collection/tensor_flow/tf_pose_estimation
      # output=$(taskset -c $core_list python /home/pi/Smart-Cameras/tf_pose_estimation/pose_estimation.py --videoPath /home/pi/Smart-Cameras/output.avi 2>> "$folder_path/error_log.txt")
      # output=$(taskset -c $core_list python -c "import resource; resource.setrlimit(resource.RLIMIT_AS, ($((mem_limit * 1024)), $((mem_limit * 1024)))); import sys; sys.argv = ['--videoPath', '/home/pi/Smart-Cameras/output.avi']; sys.path.append('/home/pi/Smart-Cameras/tf_pose_estimation'); import pose_estimation; pose_estimation.main()" 2>> "$folder_path/error_log.txt")
      # output=$(taskset -c $core_list python /home/pi/Smart-Cameras/tf_pose_estimation/pose_estimation.py --videoPath /home/pi/Smart-Cameras/output.avi 2>> "$folder_path/error_log.txt")
      # output=$(taskset -c $core_list python -c "import resource; import sys; resource.setrlimit(resource.RLIMIT_AS, ($((mem_limit * 1024)), $((mem_limit * 1024)))); sys.argv = ['pose_estimation.py', '--videoPath', '/home/pi/Smart-Cameras/output.avi']; sys.path.append('/home/pi/Smart-Cameras/tf_pose_estimation'); import pose_estimation; pose_estimation.main()" 2>> "$folder_path/error_log.txt")
      output=$(taskset -c $core_list python /home/pi/Smart-Cameras/Benchmark/data_collection/tensor_flow/tf_pose_estimation/pose_estimation.py --videoPath /home/pi/Smart-Cameras/output.avi --memory_limit $mem_limit 2>> "$folder_path/error_log.txt")

      # Extract avg_fps and total_time from output
      avg_fps=$(echo $output | awk -F',' '{print $1}')
      total_time=$(echo $output | awk -F',' '{print $2}')

      # Log the average FPS and total_time to log.txt
      echo "Average FPS: $avg_fps" >> "$folder_path/log.txt"
      echo "Total time: $total_time" >> "$folder_path/log.txt" 

      # Construct the CSV line to include total_time and memory limit
      csv_line="${sdram_freq_list[$counter]},$cores,$speed,$avg_fps,$total_time,$mem_limit"

      # Log the results to CSV
      echo "$csv_line" >> "$folder_path/results.csv"
    done
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
    echo "s/^sdram_freq=[0-9]\+/sdram_freq=$next_sdram_freq/" | sudo sed -i --file=- /boot/config.txt
  else
    # Append the new sdram_freq value
    # echo "sdram_freq=$next_sdram_freq" >> /boot/config.txt
    echo "sdram_freq=$next_sdram_freq" | sudo tee -a /boot/config.txt > /dev/null
  fi

  # Reboot the system
  sudo reboot
else
  echo "Counter has reached the end of the sdram_freq list. Not rebooting." >> "$folder_path/log.txt"
fi