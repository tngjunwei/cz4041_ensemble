import numpy as np

class Ensemble():
    def __init__(self, classifiers, meta):
        self.classifiers = classifiers
        self.meta = meta

    def predict(self, inputs):
        preds = []

        for classifier in self.classifiers:
            pred = classifier.predict(inputs)
            preds.append(pred)

        preds = np.concatenate(preds, axis=1)
        output = self.meta.predict(preds)

        return output