import cv2
import numpy as np
from tkinter import Tk, filedialog, messagebox

def select_input_method():
    """Popup window to choose between webcam or video file."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    choice = messagebox.askquestion("Select Input Method", 
                                    "Do you want to use the webcam? (Click 'Yes' for webcam, 'No' to upload a video)")
    root.destroy()
    return choice

def upload_video_file():
    """Open a file dialog to upload a video file."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def main():
    # Choose between webcam or video file input
    choice = select_input_method()

    if choice == 'yes':
        cap = cv2.VideoCapture(0)  # Use the webcam
    else:
        video_path = upload_video_file()
        if not video_path:
            print("No file selected. Exiting...")
            return
        cap = cv2.VideoCapture(video_path)  # Use the uploaded video file

    if not cap.isOpened():
        print("Error: Could not open video device or file. Try Again.")
        return

    # Read the first frame and convert it to grayscale
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

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

if __name__ == "__main__":
    main()
