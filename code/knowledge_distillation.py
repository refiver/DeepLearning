import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models
import pandas as pd
import os
import environments
from logger import ExtendedCSVLogger


def create_image_generators():
    """Creates and returns instances of image data generators that are used to generate the correct data format needed for the neural network"""
    train_datagen = ImageDataGenerator(rescale=0.255)
    validation_datagen = ImageDataGenerator(rescale=0.255)
    test_datagen = ImageDataGenerator(rescale=0.255)

    # loads training images
    train_generator = train_datagen.flow_from_directory(
        environments.PATH_TO_TRAINING_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    # loads validation images
    validation_generator = validation_datagen.flow_from_directory(
        environments.PATH_TO_VALIDATION_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    # loads test images
    test_generator = test_datagen.flow_from_directory(
        environments.PATH_TO_TEST_DATA,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


def initialize_teacher_model():
    """Initializes the teacher model based on VGG19"""
    base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    for layer in base_model_vgg19.layers:
        layer.trainable = False  # Freezing layers so the parameters will not change while training with own data

    model = models.Sequential()
    model.add(base_model_vgg19)

    # adding dense layers in order to classify own data
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def initialize_student_model():
    """Initializes the student model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ])
    return model


def load_model(model_number):
    """Safely loads a model if available, otherwise signals, that it has to be created first"""
    if os.path.exists(environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras") and \
            os.path.exists(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv"):
        loaded_model = tf.keras.models.load_model(environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras")
        training_log = pd.read_csv(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv")
        initial_epoch = training_log['epoch'].max() + 1
        return loaded_model, initial_epoch
    else:
        if model_number == 'teacher':
            return initialize_teacher_model(), 0
        else:
            return initialize_student_model(), 0


def distillation_loss(y_true, y_pred, teacher_logits, alpha=0.1, temperature=3):
    """Custom distillation loss function"""
    soft_targets_loss = tf.keras.losses.KLDivergence()(
        tf.nn.softmax(teacher_logits / temperature, axis=1),
        tf.nn.softmax(y_pred / temperature, axis=1)
    ) * (temperature ** 2)
    hard_targets_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return alpha * soft_targets_loss + (1. - alpha) * hard_targets_loss


def generator_with_teacher_predictions(generator, teacher_preds):
    """Generator that yields batches of images and teacher predictions"""
    while True:
        for i, (images, labels) in enumerate(generator):
            yield images, [labels, teacher_preds[i * generator.batch_size:(i + 1) * generator.batch_size]]


def fit_and_evaluate_model(model, initial_epoch, train_generator, validation_generator, test_generator, model_number):
    """Takes in a model that is then trained and evaluated"""
    info = "Model:{number}".format(number=model_number)
    print(info)
    # print(model.summary())

    csv_logger = ExtendedCSVLogger(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv",
                                   validation_generator, True)
    checkpoint = ModelCheckpoint(
        environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        verbose=1
    )

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=environments.EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[checkpoint, csv_logger],
        initial_epoch=initial_epoch
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')


if __name__ == "__main__":
    train_generator, validation_generator, test_generator = create_image_generators()

    teacher_model, initial_epoch = load_model('teacher')
    fit_and_evaluate_model(teacher_model, initial_epoch, train_generator, validation_generator, test_generator, 'teacher')

    teacher_predictions = []
    train_labels = []
    for images, labels in train_generator:
        preds = teacher_model.predict(images)
        teacher_predictions.append(preds)
        train_labels.append(labels)
        if len(teacher_predictions) * train_generator.batch_size >= train_generator.samples:
            break
    teacher_predictions = np.vstack(teacher_predictions)
    train_labels = np.vstack(train_labels)

    student_model, initial_epoch = load_model('student')
    student_model.compile(optimizer='adam',
                          loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, teacher_predictions))

    fit_and_evaluate_model(student_model, initial_epoch, train_generator, validation_generator, test_generator,
                           'student')
