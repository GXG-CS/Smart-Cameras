Focus on tf lite obj detection on rpi 4b

Model:
EfficientDet-Lite 
efficientdet_lite0.tflite 

The model itself processes individual frames (images) and is agnostic to the source of these frames, whether they come from a live camera feed or a pre-recorded video.

Steps:

1. Know the obj performance metrics(y) and the results output module.
2. Modify the detect.py so that it can work with video file(instead of camera feed).
3. Modify the results module.
4. Limit the hardware ability.
5. Create a Script to run all configuration with specific dataset and specific model(inference is ok now). 


Metrics in tflite obj detection: 
Frames-per-second (FPS)
(Maybe incorporate other metrics: model accuracy, robustness, latency...)




Pre-trained Models:
1. Feature Transfer
2. Fine-tuning: on a smaller dataset specific to a particular task.


Image classification:

Image recognition: 

Image processing:



Facial Recognition:
1. https://github.com/ageitgey/face_recognition


Motion Detection:

Anomaly Detection:

License Plate Recognition:






Python obj detection on rpi
1. https://github.com/automaticdai/rpi-object-detection
2. https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
3. https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi
4. https://github.com/ppogg/YOLOv5-Lite
5. https://github.com/bitsy-ai/rpi-object-tracking
6. 


rpi-objection-detection
1. Face-detection   Estimated fps < 5
2. Motion-detection  Estimated fps < 100
3. Object-tracking-shape Estimated fps <50

YOLOv5-Lite
It can reach 10+ FPS on the Raspberry Pi 4B when the input size is 320×320~




Tensorflow-Lite-Object-Detection-Raspberry-Pi
1. obj_detection
2. pose_estimation
3. image_classification
4. image_segementation
5. video_classification
6. audio_classification
7. sound_classification


Integrate all available applications.
Use one single script to run all of them.

Each results.txt should add some description of model.

I think we can use cpu_freq, cpu_cores, fps_avg to train a simple model first.



modify sdram_freq to raspberry pi 4b




Adjust sdram_freq:
1. Write a shell scripts to change settings in the /boot/config.txt
2. Use cron jobs to execute these scripts at startup.


Shell scripts for pose_estimation:
1. CPU_Freq : [1.4 1.3 1.2 1.1 1.0 0.9 0.8 0.7 0.6] GHz
2. CPU_Cores : [4, 2, 1]
3. SDRAM_Freq : [450, 400, 350, 300, 250, 200, 150, 100] MHz


Simple try for cron job on pi 3b+ focusing on sdram_freq:
1. Limited reboot








Shell script:
Loop:
Begin with sdram_freq=450(default) & init counter.txt counter=0

Output the current sdram_freq to log.txt

Run cpu_freq [1.4Ghz, 1.3Ghz, ...]
    Run cpu_cores [4, 2, 1]
Ouput each line of configuration to log.txt

counter++

#switch to another sdram_freq
if counter <= 1 (450:0 400:1 if counter reaches 2, stop setting sdram_freq)
Set the sdram_freq [450, 400, ...] & reboot

command line to run tf_pose_estimation: "python pose_estimation.py --videoPath output.avi"

store the .csv with [cpu_cores, cpu_freq, sdram_freq, avg_fps]

each time execute 3_metrics.sh will add new content of results_pose_estimation.csv

path of 3_metrics.sh : /home/pi/Smart-Cameras/sdram_freq/3_metrics.sh
path of pose_estimation.py :: /home/pi/Smart-Cameras/tf_pose_estimation/pose_estimation.py



cron job: 
"@reboot /path/to/3_metrics.sh"



4 files:
1. counter.txt   counter = 0
2. log.txt
3. sdram_freq.txt manually create   content is 450,400
4. 3_metrics.sh

In 3_metrics.sh:
1. Create counter.txt if no exists and set the counter = 0 only the first time create it.
2. Create log.txt


Output the current to sdram_freq and counter into log.txt

counter++ in counter.txt

if (counter < sdram_freq_listSize)
  set the sdram_freq to next value & reboot


crontab -e   "@reboot /path/to/3_metrics."


After that run cpu confiugration loop



cpu_freq = [1400 1300 1200]
cpu_cores = [4 2 1]

# Loop through configurations and run pose_estimation.py
for speed in "${CPU_SPEEDS[@]}"; do
    sudo cpufreq-set -f ${speed}MHz
    
    for cores in "${CORE_CONFIGS[@]}"; do

        # Run the pose_estimation.py script and append the output to the results file
        FPS_RESULTS=$(taskset -c ${cores//[-]/,} python3 pose_estimation.py --videoPath $VIDEO_PATH)
        
        # Construct and print the CSV-style line to the file
        echo "$speed,$cores,$FPS_RESULTS" | tee -a $OUTPUT_FILE
    done
done




Hardware configurations:
1. cpu_cores (✅)
2. cpu_freq  (✅)
3. sdram_freq  (✅)
4. USB Boost
5. Disk I/O Scheduler(noop, deadline, cfq)
6. Swap Space
7. Storage: Different microSD card has different I/O speeds.







Benchmark:

Pi 3b+
1. Classification
2. Regression
3. Predication?
4.  











