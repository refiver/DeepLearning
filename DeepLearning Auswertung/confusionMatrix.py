import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import zipfile
#version 10.6.2024
#names of the nine classes
class_names = ["Birke", "Buche","Eiche","Kastanie","Kiefer","Kirsche","Linde","Platane", "Robinie"]
#the modified data for testing
folders = ["turned90", "turned180","snipped","normal", "blueer", "brighter", "greener", "redder", "darker", "blurry", "horizontal", "vertical"]

#save confusion matrix as image
def save_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, vmin=0,
                vmax=200)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(-0.5, -0.5, f'Accuracy: {accuracy:.2f}', fontsize=12, color='red', va='center', ha='center',
             backgroundcolor='white')
    plt.savefig("result/"+path+".jpg")  # Speichern der Confusion Matrix als Bild
    plt.close()  # Schließen des Plots, um Speicher zu sparen

test_datagen = ImageDataGenerator(rescale=0.255)

#load model and test data to calculate the confusion matrix
def evaluate(model_path, data_dir):
    test_generator = test_datagen.flow_from_directory(
        './data/'+data_dir,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
    )
    loaded_model = tf.keras.models.load_model("./models/"+model_path)
    test_predict, test_targ = [],[]
    y_true, y_pred =  np.zeros(1800),np.zeros(1800)
    counter = 0
    for i in range(len(test_generator)):
        x_val, y_val = test_generator[i]
        test_predict.extend(np.argmax(loaded_model.predict(x_val), axis=1))
        predicteds = (np.argmax(loaded_model.predict(x_val), axis=1))
        test_targ.extend(np.argmax(y_val, axis=1))
        truevalues = (np.argmax(y_val, axis=1))
        for j in range(len(predicteds)):
            y_pred[counter] = predicteds[j]
            y_true[counter] = truevalues[j]
            counter+=1
    save_confusion_matrix(y_true, y_pred, class_names, data_dir+"_"+model_path[:-6])

def zip_dir(directory, zip_filename):
    # create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # all all roots into zip file
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                # add zip folder to path
                zipf.write(file_path, os.path.relpath(file_path, directory))

# test all models and all data modifications
for data_dir in folders:
    for modelpath in os.listdir("./models"):
        evaluate(modelpath, data_dir)
# add confusion matrizes to zip folder
zip_dir('result', 'result.zip')

