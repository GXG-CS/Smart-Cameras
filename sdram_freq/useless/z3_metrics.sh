#!/bin/bash

set -x  # for debugging

# Path to the counter file
counter_file="/home/pi/Smart-Cameras/sdram_freq/reboot_counter.txt"

# Path to the log file
log_file="/home/pi/Smart-Cameras/sdram_freq/sdram_freq_log.txt"

# Path to the results CSV file
results_csv="/home/pi/Smart-Cameras/sdram_freq/results_sdram.csv"

# Function to write to the log
log() {
    echo "$(date): $1" >> "$log_file"
}

# Initialize reboot counter
if [ ! -f "$counter_file" ]; then
    echo 0 > "$counter_file"
    log "Counter file initialized."
fi

# Initialize results CSV
if [ ! -f "$results_csv" ]; then
    echo "cpu_cores,cpu_freq,sdram_freq,fps_avg" > "$results_csv"
    log "Results CSV initialized."
fi

# Read the current counter value
counter=$(cat "$counter_file")

# List of SDRAM frequencies to cycle through
sdram_frequencies=(450 400)

# List of CPU cores and frequencies to cycle through
cpu_cores=("4" "2")
cpu_freq=("1.4Ghz" "1.3Ghz")  # Add more frequencies as needed

# Read the current SDRAM frequency from /boot/config.txt
current_sdram_freq=$(grep -oP 'sdram_freq=\K\d+' /boot/config.txt | tail -n 1)

# Log the current frequency and counter
log "Current SDRAM frequency: $current_sdram_freq MHz."
log "Current reboot counter: $counter."

# If max reboot count reached, stop the script
if [ "$counter" -ge ${#sdram_frequencies[@]} ]; then
    log "Max reboot counter reached. Exiting the script."
    exit 0
fi

# Your existing code for CPU cores and frequencies, TensorFlow Lite program, etc.

# Increment the reboot counter and save to file
echo $((counter + 1)) > "$counter_file"
log "Incremented reboot counter to $((counter + 1))."

# Comment out any existing sdram_freq lines in /boot/config.txt
sudo sed -i 's/^\(sdram_freq.*\)$/#\1/' /boot/config.txt

# Calculate the index of the next frequency to set
next_index=$((counter))

# Set the next SDRAM frequency
next_sdram_freq=${sdram_frequencies[$next_index]}
log "Setting next SDRAM frequency to $next_sdram_freq MHz."
echo "sdram_freq=$next_sdram_freq" | sudo tee -a /boot/config.txt

# Log the action and reboot
log "Rebooting to apply changes."
sudo reboot
