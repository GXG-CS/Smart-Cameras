#!/bin/bash

set -x  # for debugging

# Path to the counter file
counter_file="/home/pi/sdram_freq/reboot_counter.txt"

# Path to the log file
log_file="/home/pi/sdram_freq/sdram_freq_log.txt"

# Function to write to the log
log() {
    echo "$(date): $1" >> "$log_file"
}

# Initialize reboot counter
if [ ! -f "$counter_file" ]; then
    echo 0 > "$counter_file"
    log "Counter file initialized."
fi

# Read the current counter value
counter=$(cat "$counter_file")

# List of SDRAM frequencies to cycle through
frequencies=(450 400 350)

# Read the current SDRAM frequency from /boot/config.txt
current_freq=$(grep -oP 'sdram_freq=\K\d+' /boot/config.txt | tail -n 1)

# Find the index of the current SDRAM frequency in the frequencies array
current_index=-1
for i in "${!frequencies[@]}"; do
   if [[ "${frequencies[$i]}" == "${current_freq}" ]]; then
       current_index=$i
       break
   fi
done

# Log the current frequency and counter
log "Current SDRAM frequency: $current_freq MHz."
log "Current reboot counter: $counter."
log "Current index: $current_index"

# If max reboot count reached or the last frequency reached, stop the script
if [ "$counter" -ge 2 ] || [ "$current_freq" -eq 350 ]; then
    log "Max reboot counter reached or target SDRAM frequency achieved. Exiting the script."
    exit 0
fi

# Increment the reboot counter and save to file
echo $((counter + 1)) > "$counter_file"
log "Incremented reboot counter to $((counter + 1))."

# Comment out any existing sdram_freq lines in /boot/config.txt
sudo sed -i 's/^\(sdram_freq.*\)$/#\1/' /boot/config.txt

# Calculate the index of the next frequency to set
next_index=$((current_index + 1))
if [ "$next_index" -eq "${#frequencies[@]}" ]; then
    next_index=0
fi

# Set the next frequency
next_freq=${frequencies[$next_index]}
log "Setting next SDRAM frequency to $next_freq MHz."
echo "sdram_freq=$next_freq" | sudo tee -a /boot/config.txt

# Log the action and reboot
log "Rebooting to apply changes."
sudo reboot
