# Imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from datetime import datetime
from carac_extract import load_data

# Configuration options
num_classes = 4

# Load the data
(X_train, Y_train) = load_data(
    r'C:\Users\21506969t\PycharmProjects\Projet_Libre\asteroids-master\src\SavedData\dataset_04-02-2020_15-06-02')
print(X_train.shape)
print(Y_train.shape)

# Create the model
model = Sequential()
model.add(Dense(100, input_shape=(50, 6), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.optimizer.get_config())
history = model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, shuffle=True, validation_split=0.1)

model.save('model ' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.h5')  # creates a HDF5 file

# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
# model = load_model('model_70_acc.h5')
