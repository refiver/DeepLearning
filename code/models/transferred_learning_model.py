import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.insert(1, '../../code')

import environments
import model_administration

# Hyperparameter
img_height, img_width = 200, 200
num_classes = 9
batch_size = 32
epochs = 10

train_data, val_data, test_data = model_administration.create_image_generators()

# Funktion zum Erstellen des Modells
def create_model(base_model):
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)  # Erster Dense-Layer
    predictions = Dense(num_classes, activation='softmax')(x)  # Ausgabeschicht für 9 Klassen
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Laden des vortrainierten VGG16 Modells
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model_vgg16.layers:
    layer.trainable = False  # Gefrorene Schichten

model_vgg16 = create_model(base_model_vgg16)

# Kompilieren des Modells
model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trainieren des Modells
history_vgg16 = model_vgg16.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs)

# Laden des vortrainierten VGG19 Modells
base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model_vgg19.layers:
    layer.trainable = False  # Gefrorene Schichten

model_vgg19 = create_model(base_model_vgg19)

# Kompilieren des Modells
model_vgg19.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trainieren des Modells
history_vgg19 = model_vgg19.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs)

# Optional: Modell speichern
model_vgg16.save('vgg16_transfer_learning_model.h5')
model_vgg19.save('vgg19_transfer_learning_model.h5')
