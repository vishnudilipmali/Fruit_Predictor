import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# --- Load Model & Resources Globaly ---
print("Loading model resources... please wait.")
try:
    # Load the trained model
    model = tf.keras.models.load_model("fruit_mobilenet_model.h5")
    
    # Load class indices and invert them to get {0: 'apple', 1: 'banana', ...}
    class_indices = joblib.load("class_indices.pkl")
    class_names = {v: k for k, v in class_indices.items()}
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Ensure 'fruit_mobilenet_model.h5' and 'class_indices.pkl' are in the same folder as app.py")
    model = None

def preprocess_image(image):
    """
    Preprocesses the image to match MobileNetV2 requirements:
    1. Resize to 224x224
    2. Ensure RGB channels
    3. Scale pixel values (-1 to 1)
    """
    # Resize
    image = image.resize((224, 224))
    
    # Convert to array
    img_array = np.array(image)

    # Handle Grayscale (2D) or RGBA (4 channels)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # MobileNetV2 specific preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Add batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Error: Model not loaded.")

    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    try:
        # Open and process image
        image = Image.open(file)
        processed_img = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_img)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        predicted_class = class_names[class_index]
        confidence_percent = f"{confidence * 100:.2f}%"

        return render_template('index.html', 
                             prediction=predicted_class, 
                             confidence=confidence_percent,
                             filename=file.filename)

    except Exception as e:
        return render_template('index.html', prediction=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    # usage_reloader=False is CRITICAL for Windows/VS Code to prevent signal errors
    app.run(debug=True, use_reloader=False)