import tensorflow as tf
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import numpy as np
import os


class ExtendedCSVLogger(tf.keras.callbacks.Callback):
    """This class is based on the tensorflow keras callbacks library and implements an extended version of a csv-logger.
    It adds the more parameters that are later mentioned in the code to the log-file."""
    def __init__(self, filename, validation_generator, append=False, separator=','):
        super(ExtendedCSVLogger, self).__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.validation_generator = validation_generator
        self.file_writer = None

    def on_train_begin(self, logs=None):
        """Called at the beginning of training a model to create or append content to a log-file"""

        # checks if file already exists and if it contains data
        if self.append and os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            # if it does only append to it and do not write the header again
            self.file_writer = open(self.filename, 'a')
        else:
            # if it does not create a new file and initially write the header with the column names
            self.file_writer = open(self.filename, 'w')
            columns = [
                'epoch',
                'accuracy',
                'loss',
                'validation_accuracy',
                'validation_loss',
                'precision', 'recall',
                'learning_rate'
            ]
            self.file_writer.write(self.separator.join(columns) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch to calculate all necessary parameters and write them into the log-file"""
        logs = logs or {}

        # calculate the predictions and true labels for the entire validation dataset
        val_predict = []
        val_targ = []
        for i in range(len(self.validation_generator)):
            x_val, y_val = self.validation_generator[i]
            val_predict.extend(np.argmax(self.model.predict(x_val), axis=1))
            val_targ.extend(np.argmax(y_val, axis=1))

        val_predict = np.array(val_predict)
        val_targ = np.array(val_targ)

        # calculate precision and recall
        precision = precision_score(val_targ, val_predict, average='macro')
        recall = recall_score(val_targ, val_predict, average='macro')

        # capture learning rate
        learning_rate = \
            self.model.optimizer.learning_rate.numpy() if hasattr(self.model.optimizer, 'learning_rate') else 'N/A'

        row = [str(epoch),
               str(logs.get('accuracy', '')),
               str(logs.get('loss', '')),
               str(logs.get('val_accuracy', '')),
               str(logs.get('val_loss', '')),
               str(precision),
               str(recall),
               str(learning_rate)]

        self.file_writer.write(self.separator.join(row) + '\n')

    def on_train_end(self, logs=None):
        """Called at the end of the training process to close the log-file"""
        if self.file_writer:
            self.file_writer.close()
