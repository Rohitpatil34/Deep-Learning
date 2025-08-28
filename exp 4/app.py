from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("emotion_model.h5")

CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                img = img.reshape(1, 48, 48, 1) / 255.0
                preds = model.predict(img)
                prediction = CATEGORIES[int(np.argmax(preds))]
    return render_template("index.html", prediction=prediction, classes=CATEGORIES)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
