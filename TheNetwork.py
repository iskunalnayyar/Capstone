"""
The Network is here
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense

from WorkinWithData import MessingWithData


class NeuralNetwork:

    def __init__(self):
        np.random.seed(10)

    def define_mode(self):
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def train(self, model):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=300, batch_size=256)
        _, acc = model.evaluate(X_test, y_test)
        print('Accuracy: %.2f' % (acc * 100))


if __name__ == "__main__":
    print("Reading file")
    md = MessingWithData('/Users/k.n./Downloads/microsoft-malware-prediction', 'median.csv')
    print("Pre-Processing")
    X_train, X_test, y_train, y_test = md.read_file()
    print("Defining Model")
    nn = NeuralNetwork()
    print("Defining Model")
    model = nn.define_mode()
    print("Launching the NN now")
    nn.train(model)
    # 32 12 1
    # 61.73 % on all features
    # 61.51 % on some deleted features

    # 32 24 12 1
    # % on all features
    # 61.38 % on some deleted features
