import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=0.255)

test_generator = test_datagen.flow_from_directory(
    './data/normal',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

loaded_model = tf.keras.models.load_model("./models/best_model1.keras")

print(loaded_model.evaluate(test_generator))
