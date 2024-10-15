import cv2
import numpy as np
import mediapipe as mp
from tkinter import Tk, filedialog, simpledialog

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def nothing(x):
    """Callback function for trackbars (does nothing)."""
    pass

from tkinter import messagebox

def select_input_method():
    """Popup window to choose between webcam and video file."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    choice = messagebox.askquestion("Select Input Method", 
                                    "Do you want to use a webcam? (Click 'Yes' for webcam, 'No' for video file)")

    root.destroy()  # Close the Tkinter window

    if choice == 'yes':
        return '1'
    else:
        return '2'


def filter_color(image, lower_bound, upper_bound):
    """Apply color filtering in HSV space."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    return filtered_image

def select_camera_source():
    """Select camera index."""
    source = simpledialog.askinteger("Camera Source", "Enter camera index (e.g., 0 or 1):", minvalue=0)
    return source

def upload_video_file():
    """Upload a video file."""
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    return video_path

def create_controls():
    """Create trackbars for filters and controls."""
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 600, 600)
    cv2.moveWindow('Controls', 50, 50)

    # Trackbars for HSV values
    cv2.createTrackbar('Lower H', 'Controls', 0, 179, nothing)
    cv2.createTrackbar('Lower S', 'Controls', 0, 255, nothing)
    cv2.createTrackbar('Lower V', 'Controls', 0, 255, nothing)
    cv2.createTrackbar('Upper H', 'Controls', 179, 179, nothing)
    cv2.createTrackbar('Upper S', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('Upper V', 'Controls', 255, 255, nothing)

    # Trackbars for rotation and scaling
    cv2.createTrackbar('Rotation', 'Controls', 0, 360, nothing)
    cv2.createTrackbar('Scale', 'Controls', 10, 50, nothing)

    # Trackbars for RGB color adjustments
    cv2.createTrackbar('Blue', 'Controls', 100, 200, nothing)
    cv2.createTrackbar('Green', 'Controls', 100, 200, nothing)
    cv2.createTrackbar('Red', 'Controls', 100, 200, nothing)

    # Toggle buttons for grayscale, inversion, and blur
    cv2.createTrackbar('Grayscale', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('Invert', 'Controls', 0, 1, nothing)
    cv2.createTrackbar('Blur', 'Controls', 1, 1, nothing)  # 1 = Blur enabled, 0 = Blur disabled

def get_trackbar_values():
    """Get values from trackbars."""
    lower_h = cv2.getTrackbarPos('Lower H', 'Controls')
    lower_s = cv2.getTrackbarPos('Lower S', 'Controls')
    lower_v = cv2.getTrackbarPos('Lower V', 'Controls')
    upper_h = cv2.getTrackbarPos('Upper H', 'Controls')
    upper_s = cv2.getTrackbarPos('Upper S', 'Controls')
    upper_v = cv2.getTrackbarPos('Upper V', 'Controls')
    rotation = cv2.getTrackbarPos('Rotation', 'Controls')
    scale = cv2.getTrackbarPos('Scale', 'Controls') / 10.0

    blue = cv2.getTrackbarPos('Blue', 'Controls') / 100.0
    green = cv2.getTrackbarPos('Green', 'Controls') / 100.0
    red = cv2.getTrackbarPos('Red', 'Controls') / 100.0

    grayscale = cv2.getTrackbarPos('Grayscale', 'Controls')
    invert = cv2.getTrackbarPos('Invert', 'Controls')
    blur = cv2.getTrackbarPos('Blur', 'Controls')

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    return lower_bound, upper_bound, rotation, scale, (blue, green, red), grayscale, invert, blur

def apply_transformations(frame, angle, scale, grayscale, invert, color_adjustment):
    """Apply transformations: rotation, scaling, grayscale, inversion, and color adjustments."""
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    transformed = cv2.warpAffine(frame, M, (w, h))

    if grayscale:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

    if invert:
        transformed = cv2.bitwise_not(transformed)

    blue, green, red = color_adjustment
    transformed = cv2.multiply(transformed, np.array([blue, green, red]))

    return transformed

def detect_faces_mediapipe(frame):
    """Use MediaPipe to detect faces and return bounding boxes."""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            faces.append((x, y, w_box, h_box))
    return faces

def mask_background(frame, faces, blur_enabled):
    """Blur the background while keeping faces sharp."""
    if not faces and not blur_enabled:
        return frame  # No faces and blur disabled, return the original frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255

    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0) if blur_enabled else frame
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_inv)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    return cv2.add(background, foreground)

def process_video(video_path=None, camera_source=0):
    """Process video with dynamic controls."""
    cap = cv2.VideoCapture(video_path if video_path else camera_source)
    create_controls()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_mediapipe(frame)
        lower_bound, upper_bound, rotation, scale, color_adjustment, grayscale, invert, blur = get_trackbar_values()

        filtered_frame = filter_color(frame, lower_bound, upper_bound)
        transformed_frame = apply_transformations(filtered_frame, rotation, scale, grayscale, invert, color_adjustment)

        final_frame = mask_background(transformed_frame, faces, blur_enabled=bool(blur))

        combined_frame = np.hstack((frame, final_frame))
        cv2.imshow('Video Feed', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = select_input_method()  # Use the new popup window for selection

    if choice == '1':
        camera_index = select_camera_source()
        if camera_index is not None:
            process_video(camera_source=camera_index)
    elif choice == '2':
        video_file = upload_video_file()
        if video_file:
            process_video(video_path=video_file)
