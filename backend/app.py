from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (150, 150)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/")
def home():
    return "API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        result = "Dog"
        confidence = float(prediction)
    else:
        result = "Cat"
        confidence = float(1 - prediction)

    return jsonify({
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)