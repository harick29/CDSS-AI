import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Paths
base_dir = r"c:\Users\DELL\Downloads\train"
benign_dir = os.path.join(base_dir, "benign")
malignant_dir = os.path.join(base_dir, "malignant")

print("Selecting 30 images (15 benign, 15 malignant)...")
# Get 15 images from each
benign_images = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir) if f.endswith('.jpg')][:15]
malignant_images = [os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir) if f.endswith('.jpg')][:15]

# Load and preprocess
def load_images(image_paths, label):
    X = []
    y = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = img / 255.0  # Normalize between 0 and 1
            X.append(img)
            y.append(label)
    return X, y

print("Loading images into memory...")
X_benign, y_benign = load_images(benign_images, 0) # 0 for benign
X_malignant, y_malignant = load_images(malignant_images, 1) # 1 for malignant

X = np.array(X_benign + X_malignant)
y = np.array(y_benign + y_malignant)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Build a fast, simple Convolutional Neural Network (CNN)
print("Building AI model...")
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train extremely fast
print("Training AI model on the 30 images...")
model.fit(X, y, epochs=5, batch_size=4, verbose=1)

# Save the model
model.save("skin_model.h5")
print("\n✅ Success! Model trained and saved as 'skin_model.h5'!")
