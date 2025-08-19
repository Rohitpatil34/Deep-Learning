from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained CNN model
with open("CNNModel.pkl", "rb") as f:
    model = pickle.load(f)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make a prediction
    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"

    return jsonify({"prediction": result, "image_url": file_path})

if __name__ == '__main__':
    app.run(debug=True)
