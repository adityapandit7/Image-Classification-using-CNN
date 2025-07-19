from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('cnn_model.h5')

# Load and preprocess a new image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to match the model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    return img_array

image_path = 'path_to_image.jpg'  # Replace with the path to your image
img_array = preprocess_image('horse.jpg')

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f"Predicted Class: {predicted_class}")