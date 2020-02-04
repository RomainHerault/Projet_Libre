# Imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model


# Configuration options
num_classes = 4

# Load the data
(X_train, Y_train) = ...

# Create the model
model = Sequential()
model.add(Dense(50, input_shape=(50, 6), activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#optimizer
opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')

