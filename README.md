# 🐔 Poultry Disease Classification Web App

A deep learning-based web application to classify poultry diseases using images and provide treatment suggestions.

---

## 📌 Project Overview

This project uses a **Convolutional Neural Network (MobileNetV2)** to classify poultry diseases into the following categories:

- **Coccidiosis**
- **Newcastle**
- **Salmonella**
- **Healthy**

It provides disease prediction from uploaded images and suggests corresponding treatment steps.

---

## 🗂️ Project Structure

poultry_web_app/
├── app.py # Flask web application
├── train_model.py # Model training script
├── poultry_disease_model.h5 # Trained model file
├── trim_dataset.py # (Optional) Reduce dataset size
├── templates/
│ └── index.html # Frontend HTML file
├── static/
│ └── style.css # CSS for styling
├── data/
│ └── data/
│ ├── train/ # Training images (4 folders)
│ ├── val/ # Validation images
│ └── test/ # Testing images


## ⚙️ Setup Instructions

### 1. 🔧 Install Dependencies

```bash
pip install tensorflow flask pillow numpy
2. 🧠 Train the Model
bash
python train_model.py
This generates poultry_disease_model.h5.

3. ✂️ Trim Dataset (Optional)
To reduce the training size for faster testing:

bash
python trim_dataset.py
Keeps 300 images per class in each folder.

4. 🚀 Run the Web Application
bash
python app.py
Visit: http://127.0.0.1:5000 in your browser.

🖼️ Image Classification & Treatment
The app allows users to upload a poultry image and get a prediction. Each prediction is accompanied by treatment suggestions:
[App Preview]static/Screenshot1 2025-06-23 163137.PNG
[after uploading photo]static/Screenshot 2025-06-23 162507.png


🐓 Disease	💊 Treatment Suggestion
Coccidiosis	Use anticoccidial drugs (e.g., amprolium). Keep litter dry and clean.
Newcastle	Isolate infected birds. Vaccinate healthy ones. Disinfect surroundings.
Salmonella	Administer antibiotics (e.g., enrofloxacin) under vet supervision.
Healthy	No treatment needed. Keep monitoring. ✅

🎨 UI Features
Centered prediction box

Blue-themed upload and submit buttons

Background image for a clean, appealing interface

Responsive and minimal layout

🧑‍💻 Author
Ujwala Perugu

📌 Notes
Trained on MobileNetV2 using image size 224x224.

You can improve model accuracy by training with the full dataset.

Dataset Source: https://www.kaggle.com/datasets/chandrashekarnatesh/poultry-diseases