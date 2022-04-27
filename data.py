import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from config import label_map, img_dim, data_dir

def get_logits(one_hot_encoded):
    return np.argmax(one_hot_encoded, axis=1)


# Preparing training data
def get_all_train_data():
    X = []
    Y = []
    filepaths = []

    train_dir = os.path.join(data_dir, "train")
    dirs = os.listdir(train_dir)

    for k in tqdm(range(len(dirs))):    # Directory
        folder_path = os.path.join(train_dir, dirs[k])
        files = os.listdir(folder_path)

        for f in range(len(files)):     # Files
            filepath = os.path.join(train_dir, dirs[k], files[f])
            img = cv2.imread(filepath)
            targets = np.zeros(12)
            targets[label_map[dirs[k]]] = 1

            X.append(cv2.resize(img, (img_dim, img_dim)))
            Y.append(targets)
            filepaths.append(filepath)

    X = np.array(X, np.float32)
    Y = np.array(Y, np.uint8)
    dataset = Dataset(X, Y, filepaths)

    return dataset


# ------ TESTING ------

def eval_test(model, filename=None):
    if filename == None:
        filename = "submission.csv"

    x_test = []
    df_test = pd.read_csv('./data/sample_submission.csv')
    test_dir = os.path.join(data_dir, "test")

    for f, species in tqdm(df_test.values, miniters=100):
        img_path = os.path.join(test_dir, f)
        img = cv2.imread(img_path)
        x_test.append(cv2.resize(img, (img_dim, img_dim)))


    # test data augmentation
    x_test = np.array(x_test, np.float32)
    x_test2 = np.array([np.rot90(i, k=1) for i in x_test], np.float32)
    x_test3 = np.array([np.rot90(i, k=2) for i in x_test], np.float32)
    x_test4 = np.array([np.rot90(i, k=3) for i in x_test], np.float32)

    p_test = model.predict(x_test) # return logits
    p_test2 = model.predict(x_test2)
    p_test3 = model.predict(x_test3)
    p_test4 = model.predict(x_test4)

    p_test = np.argmax(p_test + p_test2 + p_test3 + p_test4, axis=1)

    preds = []
    reverse_lbl_map = {v:k for k,v in label_map.items()}
    for i in range(len(p_test)):
        class_name = reverse_lbl_map[p_test[i]]
        preds.append(class_name)
        
    df_test['species'] = preds
    df_test.to_csv(filename, index=False)
    print("Prediction completed")


class Dataset():
    def __init__(self):
        pass

    def __init__(self, X=None, Y=None, filepaths=None):
        self.X = np.array(X) # images
        self.Y = np.array(Y) # one-hot

        if filepaths is not None:
            self.filepaths = np.array(filepaths) #list of filepaths

    def save(self, filepath):
        f = open(filepath, 'wb')
        pickle.dump(self, f)
        f.close()
    
    def load(self, filepath):
        f = open(filepath, 'rb')
        dataset = pickle.load(f)
        f.close()

        self.X = np.array(dataset.X)
        self.Y = np.array(dataset.Y)
        self.filepaths = np.array(dataset.filepaths)

