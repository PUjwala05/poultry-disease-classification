from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("poultry_disease_model.h5")

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

treatments = {
    "Coccidiosis": "Give Amprolium in water for 3–5 days.",
    "Healthy": "No disease detected. Maintain hygiene.",
    "Newcastle": "Isolate birds and apply ND vaccine immediately.",
    "Salmonella": "Use antibiotics like Enrofloxacin and sanitize environment."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict_page")
def predict_page():
    return render_template("predict_page.html")


@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        img_path = None

        if 'file' in request.files and request.files['file'].filename != '':
            f = request.files['file']
            img_path = os.path.join("static/uploads", f.filename)
            f.save(img_path)
            print("Saved image path (upload):", img_path)

        elif 'imageData' in request.form and request.form['imageData'].startswith("data:image"):
            data_url = request.form['imageData']
            header, encoded = data_url.split(",", 1)
            decoded = base64.b64decode(encoded)
            img = Image.open(BytesIO(decoded)).convert("RGB")
            img_path = os.path.join("static/uploads", "captured.jpg")
            img.save(img_path)
            print("Saved webcam image")

        else:
            return "No image received."

        # Preprocess and predict
        img = load_img(img_path, target_size=(224, 224))
        image_array = np.array(img) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        pred_index = np.argmax(predictions, axis=1)[0]
        labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
        prediction = labels[pred_index]

        prediction_key = prediction.replace("New Castle Disease", "Newcastle")
        treatment = treatments.get(prediction_key, "No treatment information available.")

        print("Predicted Label:", prediction)
        return render_template("predict_page.html", predict=prediction,treatment=treatment)

    return render_template("predict_page.html")


    
if __name__ == '__main__':
    app.run(debug=True)

