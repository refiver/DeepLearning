
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import os.path
import environments
from models import model1
from models import model2
from models import model3
from models import model4
from models import model5
from models import model6
from models import model7
from models import model8
from models import model9
from models import model10
from logger import ExtendedCSVLogger


def create_image_generators():
    """Creates and returns instances of image data generators that are used to generate the correct data format
     needed for the neural network"""

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


def initialize_selected_model(model_number):
    """Initializes a specific model which can be selected by its number"""

    # checks if old log-file exists
    if os.path.exists(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv"):
        # removes the file
        os.remove(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv")

    # determines which model should be loaded
    match model_number:
        case 1:
            return model1.initialize_model()
        case 2:
            return model2.initialize_model()
        case 3:
            return model3.initialize_model()
        case 4:
            return model4.initialize_model()
        # case 5:
        #    return model5.initialize_model()
        # case 6:
        #    return model6.initialize_model()
        # case 7:
        #    return model7.initialize_model()
        case 8:
            return model8.initialize_model()
        case 9:
            return model9.initialize_model()
        case 10:
            return model10.initialize_model()


# safely loads a model if available, otherwise signals, that it has to be created first
def load_model(model_number):
    # checks if a model already exists if not then it initializes and returns a newly created one
    if os.path.exists(environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras") and \
            os.path.exists(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv"):

        # loads model
        loaded_model = tf.keras.models.load_model(environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras")

        # loads training logs
        training_log = pd.read_csv(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv")

        # determine the last completed epoch and therefor where to start from
        initial_epoch = training_log['epoch'].max() + 1

        return loaded_model, initial_epoch
    else:
        return initialize_selected_model(model_number), 0


def fit_and_evaluate_model(model, initial_epoch, train_generator, validation_generator, test_generator, model_number):
    """Takes in a model that is then trained and evaluated"""

    # initializes csv-logger
    csv_logger = ExtendedCSVLogger(environments.PATH_TO_TRAINING_LOGS + str(model_number) + ".csv", validation_generator, True)

    # creates saving checkpoint for models
    checkpoint = ModelCheckpoint(
        environments.PATH_TO_BEST_MODEL + str(model_number) + ".keras",  # path where the model should be saved to
        monitor='val_loss',  # metric that is watched to determine whether it is necessary to save the model
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
        initial_epoch=initial_epoch
    )

    # evaluates the model using the test data
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')
