import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd

# Erstelle den CSVLogger-Callback
csv_logger = CSVLogger('./training_log.csv', append=True)

checkpoint = ModelCheckpoint(
    './best_model.keras',  # Pfad und Name der Datei, in der das Modell gespeichert wird
    monitor='val_accuracy',        # Metrik, die überwacht wird, um das Modell zu speichern
    save_best_only=True,           # Nur das beste Modell speichern
    save_weights_only=False,       # Das gesamte Modell speichern (nicht nur die Gewichte)
    mode='auto',                   # Speichermodus ('auto', 'min', 'max')
    verbose=1                      # Fortschritt anzeigen
)


# Verzeichnis der Trainings- und test_data
train_dir = './train_data'
test_dir = './test_data'

# Bilddaten-Generator
train_datagen = ImageDataGenerator(rescale=0.255)
test_datagen = ImageDataGenerator(rescale=0.255)

# Lade die Trainings- und Testbilder
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # oder 'binary' für weniger als 2 Klassen
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # oder 'binary' für weniger als 2 Klassen
)

# Baue das Modell
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(9, activation='softmax')  # oder 'sigmoid' für weniger als 2 Klassen
])

# Kompiliere das Modell
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # oder 'categorical_crossentropy' für mehr als 2 Klassen
              metrics=['accuracy'])


loaded_model = tf.keras.models.load_model('./best_model.keras')

# Lade das Training-Log
training_log = pd.read_csv('./training_log.csv')

# Bestimme die letzte abgeschlossene Epoche
initial_epoch = training_log['epoch'].max() + 1

# Trainiere das Modell
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=17,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[checkpoint, csv_logger],
    initial_epoch=initial_epoch
)

# Evaluieren des Modells
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')