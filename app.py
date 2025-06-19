from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("poultry_disease_model.h5")

# Make sure these match the class_indices order used during training
classes = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']

# Suggested treatments
treatments = {
    "Coccidiosis": "Give Amprolium in water for 3–5 days.",
    "Healthy": "No disease detected. Maintain hygiene.",
    "Newcastle": "Isolate birds and apply ND vaccine immediately.",
    "Salmonella": "Use antibiotics like Enrofloxacin and sanitize environment."
}

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded.", treatment="")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", prediction="No file selected.", treatment="")

    try:
        img = Image.open(file.stream).convert('RGB')
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        predicted_index = np.argmax(prediction)
        predicted_class = classes[predicted_index]
        confidence = round(np.max(prediction) * 100, 2)

        # Get treatment
        treatment = treatments.get(predicted_class, "No treatment needed.")

        # Print for debugging
        print("Prediction:", prediction)
        print("Class:", predicted_class)
        print("Confidence:", confidence)

        return render_template(
            "index.html",
            prediction=f"{predicted_class} ({confidence}%)",
            treatment=treatment
        )

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e), treatment="")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
