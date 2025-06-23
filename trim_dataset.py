import os
import random
import shutil

# ✅ Set the base path to the dataset inside poultry_diseases/data/data
base_path = os.path.join("poultry_diseases", "data", "data")
splits = ["train", "val", "test"]
images_to_keep = 300  # 🔁 Keep only 300 images per class

for split in splits:
    split_path = os.path.join(base_path, split)
    
    if not os.path.exists(split_path):
        print(f"❌ Folder not found: {split_path}")
        continue

    print(f"📂 Trimming images in: {split_path}")
    
    for disease_class in os.listdir(split_path):
        class_path = os.path.join(split_path, disease_class)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        if len(images) <= images_to_keep:
            print(f"✅ {disease_class}: Only {len(images)} images, skipping.")
            continue

        random.shuffle(images)
        images_to_delete = images[images_to_keep:]

        for img in images_to_delete:
            try:
                os.remove(os.path.join(class_path, img))
            except Exception as e:
                print(f"⚠️ Failed to delete {img}: {e}")

        print(f"🗑️ Trimmed {disease_class}: Kept {images_to_keep}, Deleted {len(images_to_delete)}")
