import tensorflow as tf

# Schritt 1: Laden des Keras-Modells
keras_model = tf.keras.models.load_model('model1.keras')

# Schritt 2: Konvertieren des Modells in das TFLite-Format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Schritt 3: Speichern des konvertierten TFLite-Modells
with open('treemodel.tflite', 'wb') as f:
    f.write(tflite_model)
