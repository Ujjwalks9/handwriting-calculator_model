from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import base64
import io
import re
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

app = Flask(__name__)

# Load model and label map
model = load_model("math_symbol_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()
label_map_rev = {v: k for k, v in label_map.items()}
IMAGE_SIZE = 45

def preprocess_image(image_data):
    # Convert base64 to grayscale image
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    byte_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(byte_data)).convert('L')

    # Convert to numpy
    img = np.array(image)

    # Threshold the image
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to separate connected symbols
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # increase kernel size
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # ignore very small noise
            bounding_boxes.append((x, y, w, h))

    # Sort contours left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    for (x, y, w, h) in bounding_boxes:
        roi = thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        symbols.append(roi)

    print(f"Detected {len(symbols)} symbols")
    return symbols


def predict_expression(symbols):
    expression = ""
    for symbol in symbols:
        symbol = np.expand_dims(symbol, axis=0)
        pred = model.predict(symbol, verbose=0)
        class_idx = np.argmax(pred)
        class_label = label_map_rev[class_idx]

        # Debug print
        print(f"Predicted: {class_label} (Confidence: {np.max(pred):.4f})")

        # Fix common misclassifications
        if class_label.lower() in ["decimal", "dot"]:
            class_label = "."
        elif class_label.lower() in ["x", "times", "multiply"]:
            class_label = "*"
        elif class_label.lower() in ["div", "divide"]:
            class_label = "/"

        expression += class_label
    return expression

def safe_eval(expr):
    try:
        # Only allow basic arithmetic
        if not re.match(r'^[\d+\-*/(). ]+$', expr):
            return "Invalid Expression"
        return str(eval(expr))
    except:
        return "Error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    symbols = preprocess_image(image_data)
    if not symbols:
        return jsonify({"expression": "", "result": "No symbols detected"})

    expr = predict_expression(symbols)
    result = safe_eval(expr)
    return jsonify({"expression": expr, "result": result})

if __name__ == '__main__':
    app.run(debug=True)
