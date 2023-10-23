#!/bin/bash

# Initialize or read counter
COUNTER_FILE="/home/pi/Smart-Cameras/sdram_freq/counter.txt"
if [ ! -f "$COUNTER_FILE" ]; then
  echo "0" > $COUNTER_FILE
fi
COUNTER=$(cat $COUNTER_FILE)

# Initialize or read log file
LOG_FILE="/home/pi/Smart-Cameras/sdram_freq/log.txt"
if [ ! -f "$LOG_FILE" ]; then
  touch $LOG_FILE
fi

# Initialize or read CSV file
CSV_FILE="/home/pi/Smart-Cameras/sdram_freq/results_pose_estimation.csv"
if [ ! -f "$CSV_FILE" ]; then
  echo "cpu_cores,cpu_freq,sdram_freq,avg_fps" > $CSV_FILE
fi

# Set SDRAM frequency based on counter
if [ "$COUNTER" -eq "0" ]; then
  SDRAM_FREQ=450
elif [ "$COUNTER" -eq "1" ]; then
  SDRAM_FREQ=400
else
  echo "Counter reached its limit. Exiting."
  exit 1
fi

# Log the current SDRAM frequency
echo "Current SDRAM Frequency: $SDRAM_FREQ" >> $LOG_FILE

# Loop through CPU frequencies and cores
for CPU_FREQ in 1.4GHz 1.3GHz; do
  for CPU_CORES in 4 2 1; do
    # Set CPU frequency
    sudo cpufreq-set -f $CPU_FREQ

    # Enable the required number of CPU cores and disable the others
    for i in {0..3}; do
      if [ "$i" -lt "$CPU_CORES" ]; then
        echo 1 | sudo tee /sys/devices/system/cpu/cpu$i/online > /dev/null
      else
        echo 0 | sudo tee /sys/devices/system/cpu/cpu$i/online > /dev/null
      fi
    done

    # Log the current configuration
    echo "CPU Frequency: $CPU_FREQ, CPU Cores: $CPU_CORES" >> $LOG_FILE

    # Run the pose estimation script
    OUTPUT=$(python /home/pi/Smart-Cameras/tf_pose_estimation/pose_estimation.py --videoPath output.avi)

    # Parse the output to get avg_fps (assuming the script prints it as the first value)
    AVG_FPS=$(echo $OUTPUT | awk -F',' '{print $1}')

    # Append the results to the CSV file
    echo "$CPU_CORES,$CPU_FREQ,$SDRAM_FREQ,$AVG_FPS" >> $CSV_FILE
  done
done

# Increment the counter
COUNTER=$((COUNTER + 1))
echo $COUNTER > $COUNTER_FILE

# Reboot with new SDRAM frequency if counter <= 1
if [ "$COUNTER" -le "1" ]; then
  # Set the new SDRAM frequency here (you might need to use vcgencmd or write to /boot/config.txt)
  # Reboot the system
  sudo reboot
fi
