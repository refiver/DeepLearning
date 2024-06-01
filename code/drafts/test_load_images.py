import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def load_images_from_directory(directory, target_size=(200, 200)):
    images = []
    labels = []
    class_names = os.listdir(directory)

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, target_size)
            images.append(image)
            labels.append(class_index)

    return np.array(images), np.array(labels), class_names


# Lade die Bilder und Labels
train_dir = '../../data/train_data'
images, labels, class_names = load_images_from_directory(train_dir)

# Normalisiere die Bilder
images = images / 255.0

# Teile die Daten in Trainings- und Validierungsdaten auf
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Erstelle TensorFlow-Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Shuffle und Batching der Daten
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Baue das Modell
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # oder 'sigmoid' für binäre Klassifikation
])

# Kompiliere das Modell
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # oder 'binary_crossentropy' für binäre Klassifikation
              metrics=['accuracy'])

# Trainiere das Modell
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
