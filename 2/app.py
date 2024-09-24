# dynamic_color_filter.py

import cv2
import numpy as np

def filter_color(image, lower_bound, upper_bound):
    """
    Isolate and manipulate specific colors in the image.
    
    Args:
        image: The image or frame in which to filter colors.
        lower_bound: The lower bound of the color to isolate (in HSV space).
        upper_bound: The upper bound of the color to isolate (in HSV space).

    Returns:
        The color-filtered image.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specific color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to get the filtered image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image

def process_video(video_path=None):
    """
    Process a video or webcam feed to dynamically filter and manipulate colors.

    Args:
        video_path: Path to a video file. If None, uses webcam feed.
    """
    # Open the video capture (either a video file or webcam)
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(1)  # Use webcam if no video path provided

    # Define color bounds (for example, to isolate red)
    lower_red = np.array([0, 120, 70])  # Lower bound of red in HSV
    upper_red = np.array([10, 255, 255])  # Upper bound of red in HSV

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Filter for the specific color (red in this case)
        filtered_frame = filter_color(frame, lower_red, upper_red)

        # Show the original and filtered frames side by side
        combined_frame = np.hstack((frame, filtered_frame))
        cv2.imshow('Original and Color Filtered', combined_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the function to process video or webcam
    process_video()  # Use webcam; for video file, pass the file path as argument
