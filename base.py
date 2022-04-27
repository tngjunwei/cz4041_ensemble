# Train the base Xception model

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit

from config import model_dir, img_dim
from base_classifier import BaseClassifier


# Hyperparameters
epochs = 100
learning_rate = 0.0001
batch_size = 32

datagen = ImageDataGenerator(height_shift_range=0.25,
                            width_shift_range=0.25,
                            horizontal_flip=True, 
                            vertical_flip=True,
                            rotation_range=90,
                            brightness_range=[0.2,1.0],
                            zoom_range=[0.5,1.0])

class BaseModel_Xception(BaseClassifier):

    def __init__(self):
        base_model = Xception(input_shape=(img_dim, img_dim, 3), include_top=False, weights='imagenet', pooling='avg') # Average pooling reduces output dimensions
        x = base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(12, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)


    def predict(self, X):
        pred = self.model.predict(X)
        return pred        


    def train(self, dataset, model_name=None):
        if model_name is None:
            model_name = "base_model"

        model_path = os.path.join(model_dir, model_name)

        x_train = dataset.X
        y_train = dataset.Y

        callbacks = [ EarlyStopping(monitor='loss', patience=5, verbose=0), 
                #ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0),
                ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

        self.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

        # ------ TRAINING ------
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train)/batch_size,
                                callbacks=callbacks,
                                epochs=epochs, 
                                verbose=1)


    def save(self, filepath=None):
        if filepath == None:
            filepath = os.path.join(model_dir, "base_model.h5")

        self.model.save_weights(filepath)
    

    def load(self, filepath=None):
        if filepath == None:
            filepath = os.path.join(model_dir, "base_model.h5")

        self.model.load_weights(filepath)