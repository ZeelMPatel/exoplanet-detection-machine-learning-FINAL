import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
from keras.utils import to_categorical
from keras.models import model_from_json
from PIL import Image
import os


def main():
	# Get all images we have in our specified directory
	imlist = os.listdir(os.getcwd() + "/inputdatarevised")
	if '.DS_Store' in imlist:
		imlist.remove('.DS_Store')
	imlist.sort()

	# Open all photos in the array and place them in a numpy array
	immatrix = np.array([np.array(Image.open("inputdatarevised/" + image)).flatten() for image in imlist], 'f')

	# Initializes an array that will specify the classification of the lightcurve, 'yes' or 'no.'
	results = np.ones(np.size(imlist), dtype='int')

	# First 17 images we can test are no's, and the last 21 are yes's.
	results[0:16] = 0
	results[16:] = 1

	# Sets the X (input images) and y(actual output classifications) to the immatrix and results
	X,y = (immatrix, results)


	# Reshapes images to unflatten them, still storing in single array, and scaling all values to a float between 0,1
	X= X.reshape(X.shape[0], 1, 500, 500)
	X = X.astype('float32')
	X /= 255

	# One-hot encoding our categories 'yes' and 'no'
	y = np_utils.to_categorical(y, 2)

	# Open our stored model which was generated in our other file, now prepare it for use.
	json_file = open('model2.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model2.h5")

	# Demand a reponse from our user that corresponds to one of the images in our folder. Can add your own images to folder but it 
	# Is possible the data has been slightly overfitted so you might see diminishing returns (particularly if it is a noisier graph)
	# Our program does not also extend support (as 'yes', 'no' labels given implicity in code) for additional images currently.
	while True:
		response = input("Please input the name of the image in the given folder you would like to submit for prediction. ")
		if response in imlist:
			index = imlist.index(response)
			break

	# Now, our model predicts whether or not this is a lightcurve indicative of an exoplanet orbitting the star.
	prediction = loaded_model.predict_classes(X[index:index+1])
	actual = np.where(y[index] == 1)[0]

	# Some basic print statements to give the user feedback.
	if prediction == 0:
		print("Our model predicts that this is not an exoplanet.")
	else:
		print("Our model predicts that this is an exoplanet.")

	if actual == 0:
		print("From the given data, this is not the graph of a confirmed exoplanet.")
	else:
		print("From the given data, this is the graph of a confirmed exoplanet.")

	if actual == prediction:
		print("Based on the actual classification, our model is correct.")
	else:
		print("Based on the actual classification, our model is unfortunately incorrect.")


	# Plot the lightcurve to give visual confirmation that prediction was/was not correct.
	plt.imshow(X[index][0], cmap="gray")
	plt.show()


if __name__ == '__main__':
	main()



