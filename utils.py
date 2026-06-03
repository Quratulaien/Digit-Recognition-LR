import numpy as np
from PIL import Image
import pickle

def preprocess_image(image):
    """
    Preprocess an uploaded image to match the digits dataset format.
    - Convert to grayscale
    - Resize to 8x8
    - Flatten to 64 features
    - Normalize to [0,1]
    """
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to 8x8
    image = image.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to numpy array and flatten
    image_array = np.array(image).astype(np.float32)

    # Flatten to 1D array (64 features)
    image_flattened = image_array.flatten()

    # Normalize to [0,1] (digits dataset pixels are 0-16, so divide by 16)
    image_normalized = image_flattened / 16.0

    return image_normalized.reshape(1, -1)  # Reshape for prediction

def load_trained_model(filename='logistic_regression_model.pkl'):
    """
    Load the trained Logistic Regression model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_digit(model, preprocessed_image):
    """
    Predict the digit and return prediction with probabilities.
    """
    prediction = model.predict(preprocessed_image)[0]
    probabilities = model.predict_proba(preprocessed_image)[0]
    confidence = np.max(probabilities)

    return prediction, confidence, probabilities