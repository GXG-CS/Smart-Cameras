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


