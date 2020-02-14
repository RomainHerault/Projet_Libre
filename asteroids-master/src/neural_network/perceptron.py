# Imports
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
from .carac_extract import load_data
import numpy as np


class Perceptron():
    def __init__(self):
        # Configuration options
        self.num_classes = 4

    def load_dataset(self, debug=False):
        # Load the data
        (self.X_train, self.Y_train) = load_data(
            r'P:\Temp\Ecole\Projet_Libre\asteroids-master\src\SavedData\dataset_04-02-2020_15-06-02')

        if debug:
            print(self.X_train.shape)
            print(self.Y_train.shape)

    def model(self):
        # Create the model
        self.model = Sequential()
        self.model.add(Dense(100, input_shape=(50, 6), activation='relu'))
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

    def train(self, debug=False):

        # Configure the model and start training
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        if debug:
            print(self.model.optimizer.get_config())
        history = self.model.fit(self.X_train, self.Y_train, epochs=500, batch_size=1000, verbose=1, shuffle=True,
                                 validation_split=0.1)

    def save_model(self):
        self.model.save('model ' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.h5')  # creates a HDF5 file

    def load_model(self, model_path):

        # del self.model  # deletes the existing model

        # returns a compiled model
        # identical to the previous one
        self.model = load_model(model_path)

    def predict(self, framedata, debug=False):
        prediction = self.model.predict(framedata.reshape((1, self.X_train.shape[1], self.X_train.shape[2])))

        if debug:
            print('real prediction', prediction)

        arg_max = np.argmax(prediction)

        result = np.zeros(4)
        result[arg_max] = 1

        return result

    @staticmethod
    def test_predict_on_model(debug=False):
        perceptron = Perceptron()
        perceptron.model()
        perceptron.load_model(
            r'C:\Users\nath\PycharmProjects\Projet_Libre\asteroids-master\src\neural_network\model 04-02-2020_19-51-12_81_acc_29_val.h5')
        perceptron.load_dataset(debug=debug)
        result = perceptron.predict(framedata=perceptron.X_train[0], debug=debug)

        if debug:
            print(result)

        return result

    @staticmethod
    def test_launch_training():
        perceptron = Perceptron()
        perceptron.load_dataset(debug=True)
        perceptron.model()
        perceptron.train(debug=True)
        perceptron.save_model()


if __name__ == '__main__':
    # Perceptron.test_launch_training()
    Perceptron.test_predict_on_model(True)
