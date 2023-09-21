#!/bin/bash

# Execute the image classification app
cd /home/pi/smartCam/tf_image_classification
bash run_image_classification.sh

# Execute the image segmentation app
cd /home/pi/smartCam/tf_image_segementation
bash run_image_segmentation.sh

# Execute the object detection app
cd /home/pi/smartCam/tf_objection_detection
bash run_objection_detection.sh

# Execute the pose estimation app
cd /home/pi/smartCam/tf_pose_estimation
bash run_pose_estimation.sh

echo "All applications have been executed."
