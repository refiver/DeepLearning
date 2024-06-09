import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# load validation data by directory
def load_validation_data(validation_data_dir, img_size, batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return validation_generator

# save confusion matrix by data and
def save_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("result/"+path+".jpg")  # Speichern der Confusion Matrix als Bild
    plt.close()  # Schließen des Plots, um Speicher zu sparen

# Hauptfunktion
def evaluate(model_path, data_dir, batch_size=32):
    model = load_model("models/"+model_path)
    img_size = model.input_shape[1:3]  # Annahme, dass die Eingabeform [None, Höhe, Breite, Kanäle] ist
    validation_generator = load_validation_data("data/"+data_dir, img_size, batch_size)

    # Vorhersagen für die Validierungsdaten
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Wahre Klassen
    y_true = validation_generator.classes
    class_names = list(validation_generator.class_indices.keys())

    # Auswertung
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    save_confusion_matrix(y_true, y_pred_classes, class_names,(data_dir+"_"+model_path))

folders = ["normal", "blueer", "greener", "redder", "darker", "blurry", "horizontal", "vertical"]

# Skript ausführen
for data_dir in folders:
    models= os.listdir("./models")
    for model in models:
        model_path = model
        evaluate( model_path, data_dir)
