import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tkinter import Tk, filedialog, messagebox, simpledialog
from datetime import datetime

DATA_DIR = "labeled_data"  # Directory to save labeled data

pretrained_model = MobileNetV2(weights='imagenet')

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_labeled_frame(frame, label):
    """Save labeled data to appropriate directories."""
    label_path = os.path.join(DATA_DIR, label)
    create_directory(label_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(label_path, f"{label}_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    print(f"Saved labeled frame: {file_path}")

def get_manual_label():
    """Prompt user for manual label input."""
    root = Tk()
    root.withdraw()
    label = simpledialog.askstring("Manual Label", "Enter the label for the detected object:")
    root.destroy()
    return label

def prepare_frame(frame):
    """Preprocess frame for MobileNetV2."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_preprocessed = preprocess_input(frame_resized)
    return np.expand_dims(frame_preprocessed, axis=0)

def predict_with_mobilenet(frame):
    """Predict with pre-trained MobileNetV2."""
    frame_batch = prepare_frame(frame)
    predictions = pretrained_model.predict(frame_batch)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    label, confidence = decoded_predictions[0][1], decoded_predictions[0][2]
    return label, confidence

def collect_data():
    """Ask the user to collect data via webcam or upload images."""
    choice = messagebox.askquestion(
        "Collect Data", "Do you want to use the webcam for data collection?"
    )

    if choice == 'yes':
        collect_data_with_webcam()
    else:
        upload_and_label_image()

def collect_data_with_webcam():
    """Collect labeled data using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = predict_with_mobilenet(frame)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Data Collection - Press Space to Confirm or q to Quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            accept = messagebox.askyesno("Confirm Prediction", f"Is this a {label}?")
            if accept:
                save_labeled_frame(frame, label)
            else:
                manual_label = get_manual_label()
                if manual_label:
                    save_labeled_frame(frame, manual_label)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_and_label_image():
    """Upload an image and label it."""
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            print("Failed to load the image.")
            return

        label, confidence = predict_with_mobilenet(img)
        print(f"Predicted: {label} with confidence {confidence:.2f}")
        accept = messagebox.askyesno("Confirm Prediction", f"Is this a {label}?")

        if accept:
            save_labeled_frame(img, label)
        else:
            manual_label = get_manual_label()
            if manual_label:
                save_labeled_frame(img, manual_label)

def train_model():
    """Train the model on collected data."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    train_generator = datagen.flow_from_directory(
        DATA_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical'
    )

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training the model...")
    model.fit(train_generator, epochs=5)
    model.save('custom_model.h5')
    print("Model trained and saved as 'custom_model.h5'.")

def test_model():
    """Test the trained model."""
    model = load_model('custom_model.h5')
    choice = messagebox.askquestion("Test Model", "Do you want to use the webcam to test the model?")
    if choice == 'yes':
        predict_from_webcam(model)
    else:
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                print("Failed to load the image.")
                return

            img_batch = prepare_frame(img)
            predictions = model.predict(img_batch)
            predicted_class = np.argmax(predictions, axis=1)[0]

            class_indices = load_class_indices()
            reverse_class_indices = {v: k for k, v in class_indices.items()}
            label = reverse_class_indices.get(predicted_class, "Unknown")
            confidence = predictions[0][predicted_class]

            print(f"Predicted: {label} with confidence {confidence:.2f}")
            cv2.imshow("Test Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def predict_from_webcam(model):
    """Test the trained model with webcam predictions."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_batch = prepare_frame(frame)
        predictions = model.predict(frame_batch)
        predicted_class = np.argmax(predictions, axis=1)[0]

        class_indices = load_class_indices()
        reverse_class_indices = {v: k for k, v in class_indices.items()}
        label = reverse_class_indices.get(predicted_class, "Unknown")
        confidence = predictions[0][predicted_class]

        height, width, _ = frame.shape
        cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Test Webcam - Press q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_class_indices():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(DATA_DIR, target_size=(224, 224), batch_size=1)
    return generator.class_indices

def main():
    """Main function to run the program."""
    choice = messagebox.askquestion(
        "Select Action", "Do you want to collect data? (Click 'Yes' for Data Collection, 'No' for other options)"
    )

    if choice == 'yes':
        collect_data()
    else:
        action = messagebox.askquestion("Train or Test", "Do you want to train the model?")
        if action == 'yes':
            train_model()
        else:
            test_model()

if __name__ == "__main__":
    main()
