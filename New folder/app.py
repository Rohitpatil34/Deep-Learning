from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import io

# Load the trained ANN model
with open("ANNModel.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

def preprocess_image(image):
    """Preprocess the uploaded image to match MNIST format"""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    return jsonify({"prediction": int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
