"""
The Network is here
"""

import numpy as np
from keras import Sequential
from keras.layers import Dense


class NeuralNetwork:

    def __init__(self):
        np.random.seed(10)

    def define_mode(self, shape):
        model = Sequential()
        model.add(Dense(32, input_dim=shape))
        model.add(Dense(24, activation='sigmoid'))
        model.add(Dense(12, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def train(self, model, X_train, X_test, y_train, y_test):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=4086)
        _, acc = model.evaluate(X_test, y_test)
        return 'Accuracy: %.2f' % (acc * 100)

    # 32 12 1
    # 61.73 % on all features
    # 61.51 % on some deleted features

    # 64 32 24 12 1
    # % on some deleted features
    # 61.38 % on some deleted features
