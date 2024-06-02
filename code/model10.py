
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import os.path
import environments

csv_logger = CSVLogger(environments.PATH_TO_TRAINING_LOGS, append=True)

checkpoint = ModelCheckpoint(
    environments.PATH_TO_BEST_MODEL,  # Pfad und Name der Datei, in der das Modell gespeichert wird
    monitor='val_loss',  # Metrik, die überwacht wird, um das Modell zu speichern
    save_best_only=True,  # Nur das beste Modell speichern
    save_weights_only=False,  # Das gesamte Modell speichern (nicht nur die Gewichte)
    mode='auto',  # Speichermodus ('auto', 'min', 'max')
    verbose=1  # Fortschritt anzeigen
)


def create_image_generators():
    # Bilddaten-Generator
    train_datagen = ImageDataGenerator(rescale=0.255)
    validation_datagen = ImageDataGenerator(rescale=0.255)
    test_datagen = ImageDataGenerator(rescale=0.255)

    # Lade die Trainings- und Testbilder
    train_generator = train_datagen.flow_from_directory(
        environments.PATH_TO_TRAINING_DATA,
        target_size=(200, 400),
        batch_size=32,
        class_mode='categorical'  # oder 'binary' für weniger als 2 Klassen
    )

    validation_generator = validation_datagen.flow_from_directory(
        environments.PATH_TO_VALIDATION_DATA,
        target_size=(200, 400),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        environments.PATH_TO_TEST_DATA,
        target_size=(200, 400),
        batch_size=32,
        class_mode='categorical'  # oder 'binary' für weniger als 2 Klassen
    )

    return train_generator, validation_generator, test_generator


def initialize_model():
    if os.path.exists(environments.PATH_TO_TRAINING_LOGS):
        # Datei entfernen
        os.remove(environments.PATH_TO_TRAINING_LOGS)
        print("Die Datei {csv_file} wurde erfolgreich entfernt.".format(csv_file = environments.PATH_TO_TRAINING_LOGS))

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 400, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((3, 3)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(9, activation='softmax')  # oder 'sigmoid' für weniger als 2 Klassen
    ])

    # Kompiliere das Modell
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # oder 'categorical_crossentropy' für mehr als 2 Klassen
                  metrics=['accuracy'])

    return model


# safely loads a model if available, otherwise signals, that it has to be created first
def load_model():
    # checks if a model already exists
    if os.path.exists(environments.PATH_TO_BEST_MODEL) and os.path.exists(environments.PATH_TO_TRAINING_LOGS):
        # loads model
        loaded_model = tf.keras.models.load_model(environments.PATH_TO_BEST_MODEL)
        # loads training logs
        training_log = pd.read_csv(environments.PATH_TO_TRAINING_LOGS)

        # determine the last completed epoch and therefor where to start from
        initial_epoch = training_log['epoch'].max() + 1

        print(training_log['epoch'])

        if initial_epoch >= environments.EPOCHS:
            print(initial_epoch)
            # initial_epoch = 0
            print('HERE IS THE EPOCHS')

        return loaded_model, initial_epoch
    else:
        return initialize_model(), 0


def fit_and_evaluate_model(model, initial_epoch, train_generator, validation_generator, test_generator):
    # Trainiere das Modell
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=environments.EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[checkpoint, csv_logger],
        initial_epoch=initial_epoch
    )

    # Evaluieren des Modells
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')
