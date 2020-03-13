# Imports
import numpy
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Embedding, \
    TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from .carac_extract import load_data
# from neural_network import carac_extract.load_data
import numpy as np
import os


class Perceptron():
    def __init__(self):
        # Configuration options
        self.num_classes = 12

        # self.input_shape = (50,6)
        self.input_shape = (14, 6)
        self.X_train = None
        self.Y_train = None
        self.model = None
        self.batch_size = 100

    def load_dataset(self, debug=False):
        # Load the data
        # (self.X_train, self.Y_train) = load_data(
        #   './SavedData/dataset_04-02-2020_15-06-02')

        (self.X_train, self.Y_train) = load_data(
            os.path.dirname(os.path.dirname(
                __file__)) + '/SavedData/dataset_20-02-2020_17-36-31')

        if debug:
            print(self.X_train.shape)
            print(self.Y_train.shape)

    def model_perceptron(self):
        # Create the model
        self.model = Sequential()
        self.model.add(
            Dense(100, input_shape=self.input_shape, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def model_rnn(self):
        # Create the model
        self.model = Sequential()
        # self.model.add(Embedding(input_dim = 50*6, output_dim=64))
        # self.model.add(Dense(100, input_shape=(50, 6), activation='relu'))
        # self.model.add(LSTM(14 * 6 * 2, return_sequences=True,
        #                     input_shape=self.input_shape, stateful=True,
        #                     batch_input_shape=(self.batch_size, 1, 14 * 6)))
        self.model.add(TimeDistributed(
            Dense(14 * 6 * 2, activation='relu',
                 ), batch_input_shape=(self.batch_size, 1, 14 * 6)))
        self.model.add(TimeDistributed(Dense(14 * 6, activation='relu')))
        self.model.add(TimeDistributed(Dense(14 * 6, activation='relu')))
        self.model.add(TimeDistributed(Dense(14 * 6*2, activation='relu')))
        self.model.add(TimeDistributed(Dense(14 * 6*2, activation='relu')))
        self.model.add(TimeDistributed(Dense(14 * 3, activation='relu')))
        self.model.add(TimeDistributed(Dense(14 * 3, activation='relu')))
        # self.model.add(LSTM(14 * 6, return_sequences=True, stateful=True,
        #                     batch_input_shape=(self.batch_size, 1, 14 * 6)))
        self.model.add(LSTM(14*6, stateful=True,
                            batch_input_shape=(self.batch_size, 1, 14 * 6)))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def train(self, debug=False):


        # Configure the model and start training
        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        if debug:
            print(self.model.optimizer.get_config())
        history = self.model.fit(self.X_train, self.Y_train, epochs=500,
                                 batch_size=1000, verbose=1, shuffle=False,
                                 validation_split=0.1)

    def trainRNN(self, debug=False):



        # train_size = int(len(dataset) * 0.67)
        # test_size = len(dataset) - train_size
        # train, test = dataset[0:train_size, :], dataset[
        #                                         train_size:len(dataset), :]
        #
        # print(len(train), len(test))

        self.X_train = self.X_train[:40000]
        self.Y_train = self.Y_train[:40000]
        self.X_train = self.X_train.reshape(self.X_train.shape[0],
                                            self.X_train.shape[1] *
                                            self.X_train.shape[2])

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1,
                                            self.X_train.shape[1])

        # Configure the model and start training
        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        if debug:
            print(self.model.optimizer.get_config())
        epoch = 200
        for i in range(epoch):
            print("epoch ", i + 1, "/", epoch)
            history = self.model.fit(self.X_train, self.Y_train, epochs=1,
                                     batch_size=self.batch_size, verbose=1,
                                     shuffle=False,
                                     validation_split=0.1)
            self.model.reset_states()

    def save_model(self):
        self.model.save('model ' + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + '.h5')  # creates a HDF5 file

    def load_model(self, model_path):

        # del self.model  # deletes the existing model

        # returns a compiled model
        # identical to the previous one
        self.model = load_model(model_path)

    def predict(self, framedata, debug=False):

        framedata = framedata.reshape(1, framedata.shape[0]*framedata.shape[1])

        scaler = MinMaxScaler(feature_range=(0, 1))
        framedata = scaler.fit_transform(framedata)

        prediction = self.model.predict(framedata)

        # prediction = self.model.predict(framedata.reshape(
        #     (1, self.X_train.shape[1], self.X_train.shape[2])))
        if debug:
            print('real prediction', prediction)

        arg_max = np.argmax(prediction)

        result = np.zeros(self.num_classes)
        result[arg_max] = 1
        # for i in range(len(prediction[0])):
        #   if prediction[0][i] > 0.2:
        #      result[i] = 1
        return result

    @staticmethod
    def test_predict_on_model(debug=False):
        perceptron = Perceptron()
        perceptron.model()
        perceptron.load_model(
            os.path.dirname(
                __file__) + '/model 19-02-2020_01-13-26_78_acc_23_val.h5')
        perceptron.load_dataset(debug=debug)
        result = perceptron.predict(framedata=perceptron.X_train[0],
                                    debug=debug)

        if debug:
            print(result)

        return result

    @staticmethod
    def test_launch_training():
        perceptron = Perceptron()
        perceptron.load_dataset(debug=True)
        perceptron.model_rnn()
        perceptron.trainRNN(debug=True)
        perceptron.save_model()


if __name__ == '__main__':
    # print(os.path.dirname(os.getcwd()))
    Perceptron.test_launch_training()
    # Perceptron.test_predict_on_model(True)
    # perceptron = Perceptron()
    # perceptron.load_dataset(debug=True)
    # print(perceptron.create_dataset(perceptron.X_train)[0])
    # test = [[[0, 0, 0],
    #          [1, 0, 0],
    #          [2, 0, 0]],
    #         [[0, 1, 0],
    #          [1, 1, 0],
    #          [2, 1, 0]],
    #         [[0, 2, 0],
    #          [1, 2, 0],
    #          [2, 2, 0]]]
    # test = numpy.array(test)
    # print(test.reshape(test.shape[0], test.shape[1] * test.shape[2]))
