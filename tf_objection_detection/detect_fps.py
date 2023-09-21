# Modfied version focusing on FPS calculation
# 1.FPS calculation and display
# 2.Omit Visualization
# 3.Modle Inference
# 4.Optional - Avoid Flipping and the Image
# 5.Clean up the code
# 6.Modify the Main function

"""Main script to run the object detection routine."""
import argparse
# import sys
import time
import numpy as np

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


# def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
#         enable_edgetpu: bool) -> None:
def run(model: str, video_path: str, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    video_path: The path of the video to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  # cap = cv2.VideoCapture(camera_id)
  cap = cv2.VideoCapture(video_path)

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1


  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  
  # List to store the FPS values for each frame
  fps_list = []

  loop_start_time = time.time()
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      # sys.exit(
      #     'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      # )
      # print("Video playback completed.")
      break

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    cv2.imshow('object_detector', image)

    # Append the FPS value to the list
    fps_list.append(fps)

  total_time = time.time() - loop_start_time

  cap.release()
  cv2.destroyAllWindows()

  # Calculate average FPS
  avg_fps = np.mean(fps_list)
  min_fps = np.min(fps_list)
  max_fps = np.max(fps_list)

  print(f"{avg_fps:.2f},{min_fps:.2f},{max_fps:.2f},{total_time:.2f}")

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  # parser.add_argument(
  #     '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
    '--videoPath',
    help='Path to the video file.',
    required=True  
  )
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  # run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
  #     int(args.numThreads), bool(args.enableEdgeTPU))
  run(args.model, args.videoPath, args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
  main()
