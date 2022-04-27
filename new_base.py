# Train the base EfficientNet model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import efficientnet
import tensorflow_addons as tfa

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

class BaseModel_EfficientNet(BaseClassifier):

    def __init__(self):
        base_model = efficientnet.EfficientNetB3(input_shape=(img_dim, img_dim, 3), include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        outputs = Dense(12, activation="softmax", name="pred")(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])


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