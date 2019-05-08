#*****************************************
import numpy as np
import matplotlib.pyplot as plt
from HodaDatasetReader.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
plt.rcParams['figure.figsize'] = (7,9) # Make the figures a bit bigger
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical

#load training data
nb_classes = 10
x_train, y_train = read_hoda_dataset(dataset_path='HodaDatasetReader/DigitDB/Train 60000.cdb',
                                images_height=16,
                                images_width=16,
                                one_hot=False,
                                reshape=True)

x_test, y_test = read_hoda_dataset(dataset_path='HodaDatasetReader/DigitDB/Test 20000.cdb',
                              images_height=16,
                              images_width=16,
                              one_hot=True,
                              reshape=False)

test_labels=y_test

x_train = x_train.reshape((60000, 16 * 16))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((20000, 16 * 16))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
# print(len(y_train))
# print(y_train[2])
# print(x_test[0],y_test[0])

network = Sequential()
network.add(Dense(10, activation='softmax', input_shape=(256,)))

#network.add(Dense(256, activation='relu'))
#network.add(Dense(10, activation='softmax'))

# print ("x_Test is: ",x_test[0])
# print ("y_Test is: ",y_test[0])

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(x_train, y_train, epochs=2, batch_size=1)

print("x is : ",x_test[0]," *y is : ",y_test[0])
score = network.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = network.predict_classes(x_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == test_labels)[0]
incorrect_indices = np.nonzero(predicted_classes != test_labels)[0]

