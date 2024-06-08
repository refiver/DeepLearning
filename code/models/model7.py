from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import environments

def create_image_generators():
    # creates augmented training data which is vertically flipped
    train_datagen = ImageDataGenerator(
        rescale=0.255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        channel_shift_range=0.2,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=0.255)
    test_datagen = ImageDataGenerator(rescale=0.255)

    train_generator = train_datagen.flow_from_directory(
        environments.PATH_TO_TRAINING_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        environments.PATH_TO_VALIDATION_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        environments.PATH_TO_TEST_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


def initialize_model():
    """This model is equal to the original model but uses augmented training data which has vertically flipped
    images."""

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(9, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

