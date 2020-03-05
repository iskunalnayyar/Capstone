"""
The Network is here
"""
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from WorkinWithData import MessingWithData


class NeuralNetwork:

    def __init__(self):
        np.random.seed(10)

    def define_mode(self, shape):
        model = Sequential()
        model.add(Dense(32, input_dim=shape))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.summary()
        return model

    def train(self, model, X_train, X_test, y_train, y_test):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=700, verbose=2)
        _, acc = model.evaluate(X_test, y_test)
        # print(history.history.keys())
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['loss'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        return acc

    # 32 12 1
    # 61.73 % on all features
    # 61.51 % on some deleted features

    # 64 32 24 12 1
    # % on some deleted features
    # 61.38 % on some deleted features


md = MessingWithData('/Users/k.n./Downloads/', 'train.csv')
print("Pre-Processing")
X_train, X_test, y_train, y_test, cols_list = md.read_file()
print("Defining Model")
nn = NeuralNetwork()
print(X_train.shape)
model = nn.define_mode(X_train.shape[1])
print("Launching the NN now")
acc = nn.train(model, X_train, X_test, y_train, y_test)
print("Accuracy ", acc)
