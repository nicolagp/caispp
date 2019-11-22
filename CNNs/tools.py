# This Python file provides a helper function for loading
# in the Cifar-10 dataset
import numpy as np 
import tensorflow as tf

# Loads data from cifar-10, given the path to the dataset
# Undo the preprocessing to learn how to do it in the notebook
def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Convert y_train, y_test to be a list
    y_train = [y for arr in y_train for y in arr]
    y_test = [y for arr in y_test for y in arr]

    return X_train, y_train, X_test, y_test

# The following code was taken from https://github.com/caisplusplus/caispp/blob/master/caispp/ImageClassifier.py
def create_model(num_features=64, show_model=True):
     # Create inceptionv3 model
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape = (75, 75, 3))
        
        for layer in image_model.layers:
            layer.trainable = False
            
        # Add new head
        model = tf.keras.models.Sequential()
        model.add(image_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_features, activation='relu'))

        # Add layer of size num_classes (10 for cifar-10)
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        if show_model:
            model.summary()
        return model