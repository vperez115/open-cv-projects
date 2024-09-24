import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def prepare_image(file_path):
    # Load the image file using OpenCV
    img = cv2.imread(file_path)
    
    # Resize the image to 224x224 pixels (size expected by MobileNetV2)
    img_resized = cv2.resize(img, (224, 224))
    
    # Preprocess the image for the MobileNetV2 model
    img_preprocessed = preprocess_input(img_resized)
    
    # Add an extra dimension because the model expects a batch of images, not a single image
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

def predict_image(file_path):
    # Prepare the image
    img_batch = prepare_image(file_path)
    
    # Make a prediction
    predictions = model.predict(img_batch)
    
    # Decode the predictions to get human-readable labels
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    return decoded_predictions

# Replace 'path_to_your_image.jpg' with the path to the image you want to use
file_path = 'C:/Users/Victo/Desktop/open-cv-projects/6/DSC00712-1560x1075.jpg'
predictions = predict_image(file_path)

# Print out the predictions
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i + 1}: Label: {label}, Score: {score:.2f}")

# Load and display the image using OpenCV
image = cv2.imread(file_path)
cv2.imshow("Image", image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop when 'q' is pressed
        break
        
cv2.destroyAllWindows()
