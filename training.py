####### Do not run this file; unless you wanted to train another neural network just for fun ########
####### For demonstration purposes only #######
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras import optimizers
from PIL import Image
import os

def main():
	# Get a list of all the images of lightcurves in the specified directory
	imlist = os.listdir(os.getcwd() + "/inputdatarevised")
	imlist.sort()

	# Create one array for the data of all the images used
	immatrix = np.array([np.array(Image.open("inputdatarevised/" + image)).flatten() for image in imlist], 'f')

	# Create array of output values, assigning proper values to each image
	responses = np.ones((np.size(imlist)), dtype='int')
	label[0:16] = 0
	label[16:] = 1

	# Create our train_data 
	data,Responses = shuffle(immatrix,responses, random_state=4)

	train_data = [data,Responses]

	(X, y) = (train_data[0], train_data[1])

	# Use scikit-learn in order to split the train and test data accordingly.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.28)

	# Define some variables to be used below
	nb_classes = 2
	img_height, img_width = 500, 500

	# Reshape array out of flattened state to prepare for feeding into ConvNet later
	X_train = X_train.reshape(X_train.shape[0], 1, img_height, img_width)
	X_test = X_test.reshape(X_test.shape[0], 1, img_height, img_width)

	# Convert data to float
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# Scaling down the data to a range of 0 to 1, as floats
	X_train /= 255
	X_test /= 255

	# One-hot-encoding the train and test results.
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	# Defining the structure of the model, explained more in depth in the design document.
	model = Sequential()
	model.add(Conv2D(32, kernel_size=3, activation="relu", data_format="channels_first", input_shape=(1,500,500)))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, kernel_size=3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(nb_classes, activation='softmax'))

	# Define our custom optimizer. Had to play with some values a bit as the training loss was diverging at first start.
	sgd = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=False)

	# Compile our model using our optimizer, a loss function for categorical classification, and accuracy metric.
	model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

	# Fit our model to the data we have already split and processed, using 50 epochs due to small learning rate
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, verbose=1)

	# Just giving a last time confirmation of what the loss and accuracy is on training set.
	# Loss diminished from around 0.8 to 0.2, while accuracy was above 90%. (Caveat: possible overfitting, although
	# Test data still performed average at over 65% accuracy).
	print(model.evaluate(X_train, Y_train))

	# Load our weights into json and hdf5 formats
	model_json = model.to_json()
	with open("model2.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model3.h5")


if __name__ == '__main__':
	main()