import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
sys.path.insert(1, '../../code')

import environments
import model_administration
from logger import ExtendedCSVLogger


def create_image_generators():
    """Creates and returns instances of image data generators that are used to generate the correct data format
     needed for the neural network"""

    train_datagen = ImageDataGenerator(rescale=0.255)
    validation_datagen = ImageDataGenerator(rescale=0.255)
    test_datagen = ImageDataGenerator(rescale=0.255)

    # loads training images
    train_generator = train_datagen.flow_from_directory(
        '../' + environments.PATH_TO_TRAINING_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    # loads validation images
    validation_generator = validation_datagen.flow_from_directory(
        '../' + environments.PATH_TO_VALIDATION_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    # loads test images
    test_generator = test_datagen.flow_from_directory(
        '../' + environments.PATH_TO_TEST_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


train_generator, validation_generator, test_generator = create_image_generators()

# initialization of the vgg19 model
base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
for layer in base_model_vgg19.layers:
    layer.trainable = False # freezing layers so the parameters will not change while training with own data

model = models.Sequential()
model.add(base_model_vgg19)

# adding dense layers in order to classify own data
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# initializes csv-logger
csv_logger = ExtendedCSVLogger('../' + environments.PATH_TO_TRAINING_LOGS + str(14) + ".csv", validation_generator, True)

# creates saving checkpoint for models
checkpoint = ModelCheckpoint(
    '../' + environments.PATH_TO_BEST_MODEL + str(14) + ".keras",  # path where the model should be saved to
    monitor='accuracy',  # metric that is watched to determine whether it is necessary to save the model
    save_best_only=True,  # saves the best model only
    save_weights_only=False,  # saves whole model (not only weights)
    mode='auto',  # saving-mode ('auto', 'min', 'max')
    verbose=1  # show progress
)

# trains and validates the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=environments.EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint, csv_logger],
    initial_epoch=0
)

# evaluates the model using the test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
