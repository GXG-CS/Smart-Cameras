import cv2

# Use the default camera (0). Change the number if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'MJPG', 'AVC1', etc. depending on your need
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
