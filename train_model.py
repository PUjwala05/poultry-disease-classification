import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# ✅ Define correct base directory
base_dir = os.path.join("poultry_diseases", "data", "data")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# 📏 Image parameters
img_height, img_width = 224, 224
batch_size = 32

# 🧪 Preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# 📦 Data loaders
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# 📐 Load MobileNetV2 base
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

# 🧠 Build final model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# ⚙️ Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🚀 Train with 1 epoch for speed
model.fit(
    train_data,
    validation_data=val_data,
    epochs=1
)

# 💾 Save the trained model
model.save("poultry_disease_model.h5")
print("✅ Model training complete and saved as 'poultry_disease_model.h5'")
