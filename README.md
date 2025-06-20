# Poultry Disease Classification 

# 🐔 Poultry Disease Classification - ML Flask App

This project uses **Transfer Learning** with a deep learning model to classify poultry diseases from images. It provides treatment suggestions through a user-friendly web interface built using **Flask**.

---

## 🚀 Features

- 🖼️ Upload poultry images for disease prediction.
- 📊 Supports 4 classes:
  - Coccidiosis
  - Newcastle
  - Salmonella
  - Healthy
- 💊 Suggests treatment for detected diseases.
- 🌐 Web UI using Flask + HTML + CSS.

---

## 🛠️ Tech Stack

- Python
- TensorFlow & Keras
- Flask
- HTML, CSS
- PIL (Python Imaging Library)
- NumPy

---

## 📁 Folder Structure

poultry_web_app/
│
├── app.py # Flask web app
├── train_model.py # Model training script
├── poultry_disease_model.h5 # Trained CNN model
│
├── templates/
│ └── index.html # HTML template
│
├── static/
│ ├── css/
│ │ └── style.css # CSS styling
│ └── bg.jpg # Background image
│
├── poultry_diseases/ # Training dataset (local use only)
│ ├── Coccidiosis/
│ ├── Healthy/
│ ├── Newcastle/
│ └── Salmonella/
│
└── README.md # Project documentation

yaml
Copy
Edit

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/PUjwala05/poultry-disease-classification.git
cd poultry_web_app
2️⃣ (Optional) Create a virtual environment:
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On macOS/Linux
3️⃣ Install dependencies:
bash
Copy
Edit
pip install tensorflow flask pillow numpy
4️⃣ Train the model (only if you don’t have the .h5 file):
bash
Copy
Edit
python train_model.py
5️⃣ Run the web application:
bash
Copy
Edit
python app.py
Then open your browser and visit:
📍 http://127.0.0.1:5000/

🧠 How It Works
A CNN model trained on images of infected and healthy poultry.

Flask handles image uploads and serves predictions.

Preprocessing resizes the image to 224x224 and normalizes it.

The model predicts the class and displays treatment info on the same page.

💊 Treatment Mapping
Disease	Suggested Treatment
Coccidiosis	Give Amprolium in water for 3–5 days.
Newcastle	Isolate birds and apply ND vaccine immediately.
Salmonella	Use antibiotics like Enrofloxacin and sanitize the environment.
Healthy	No disease detected. Maintain hygiene.


🙋‍♀️ Author
Ujwala Perugu
📘 Project Title: Transfer Learning-Based Classification of Poultry Diseases for Enhanced Health Management

📌 Notes

All predictions are based on model performance; use results responsibly.

📎 License
This project is for academic and educational purposes.

