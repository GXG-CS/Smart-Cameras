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









