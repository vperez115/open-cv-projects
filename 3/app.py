import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the webcam, or replace with video file path
if not cap.isOpened():
    print("Error: Could not open video device or file. Try Again.")
    exit()
# Read the first frame and convert it to grayscale
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while cap.isOpened():
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale and blur it to reduce noise
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute the absolute difference between the current frame and the previous one
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference to obtain binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in small holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Filter out small contours to reduce noise
        if cv2.contourArea(contour) < 500:
            continue

        # Get the bounding box for each contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Extract the region of motion
        motion_region = frame2[y:y+h, x:x+w]

        # Apply artistic effects to the motion region (e.g., color map)
        motion_region = cv2.applyColorMap(motion_region, cv2.COLORMAP_JET)

        # Replace the motion region with the processed region
        frame2[y:y+h, x:x+w] = motion_region

        # Optionally, draw a rectangle around the motion region
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the artistic motion capture frame
    cv2.imshow('Motion Capture Artistic Display', frame2)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame
    gray1 = gray2.copy()

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
