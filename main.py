import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data import get_all_train_data, eval_test, Dataset
from base import BaseModel
from new_base import BaseModel_EfficientNet
from model import Ensemble
from config import model_dir


def split_train_test(dataset, test_split=0.2, random_state=None):
    X = dataset.X
    Y = dataset.Y

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state) # Want a balanced split for all the classes
    for train_index, test_index in sss.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
    return Dataset(x_train, y_train), Dataset(x_test, y_test)

def sample_train_data(dataset, train_split=0.2):
    X = dataset.X
    Y = dataset.Y

    idx = np.arange(0, X.shape[0])
    num_train = int(train_split * idx.shape[0])
    sel_idx = np.choice(idx, size=num_train, replace=True)

    return Dataset(X[sel_idx], Y[sel_idx])

def train_base_model(dataset):
    #base_model = BaseModel_Xception()
    base_model = BaseModel_EfficientNet()
    base_model.train(dataset)
    random_name = str(np.random.randint(0, 999999999))
    base_model.save(f"./models/{random_name}.h5")
    
    return base_model


def train_base_estimators(dataset, n=5, test_split=0.6):
    estimators = []

    for i in tqdm(range(n)):
        train_dataset, _ = split_train_test(dataset, test_split=test_split)
        #train_dataset = sample_train_data(dataset, train_split=1-test_split) #bagging - select with replacement
        estimator = train_base_model(train_dataset)
        estimators.append(estimator)
    
    return estimators


def train_ensemble(dataset, classifiers):
    X = dataset.X
    Y = dataset.Y

    inputs = []
    for m in classifiers:
        pred = m.predict(X)
        inputs.append(pred)

    inputs = np.concatenate(inputs, axis=1)
    targets = Y

    # train meta learner
    epochs = 100
    learning_rate = 0.01
    batch_size = 32

    model = Sequential()
    model.add(Input(shape=(inputs.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(targets.shape[1], activation='softmax'))

    callbacks = [ EarlyStopping(monitor='loss', patience=5, verbose=0), 
              #ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0),
              ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    model.fit(inputs, targets,
            steps_per_epoch=len(X)/batch_size,
            callbacks=callbacks,
            epochs=epochs, 
            verbose=0)

    return model


def main():

    load_prev_estimators = False
    train_more = True
    num_estimators_to_train = 5
    test_split = 0.2
    val_split = 0.6

    classifiers = []

    if load_prev_estimators:
        model_names = os.listdir(model_dir)
        for m in model_names:
            model_path = os.path.join(model_dir, m)
            tmp = BaseModel()
            tmp.load(model_path)
            classifiers.append(tmp)
    
    original_dataset = get_all_train_data()
    train_dataset, test_dataset = split_train_test(original_dataset, test_split=test_split, random_state=13)

    if train_more:
        trained_classifiers = train_base_estimators(train_dataset, n=num_estimators_to_train, test_split=val_split)
        classifiers.extend(trained_classifiers)
    
    if len(classifiers) > 0:
        meta = train_ensemble(test_dataset, classifiers)
        model = Ensemble(classifiers, meta)
        eval_test(model)

    print("Done")


if __name__ == "__main__":
    main()