
class BaseClassifier:
    def __init__(self):
        raise NotImplementedError("Not implemented")
    
    def predict(self):
        raise NotImplementedError("Predict not implemented")

    def train(self):
        raise NotImplementedError("Train not implemented")
