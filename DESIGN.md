# Exoplanet Detection using Machine Learning by Zeel Patel & Jackson Quick – Designing Our Project

Most of the design process for our project involved decisions on the structuring of the convolutional neural network
(also known as CNN or ConvNet) that took in our lightcurves and, based on its weights in the synapses,
outputted the desired classification of the curve as evidence of an exoplanet or otherwise.

## Image Preprocessing
In order to reach this point, we first decided (per common practice) to take all of the images given, resize them to a 500x500 PNG file,
and convert them to grayscale in order to block out any additional noise that would not be important to the classification. We felt
confident in doing this since the images were all roughly the same size, making any unwanted relative distortions not too noticeable
on the model. This was done in a one-time process using the os library for Python3, although the script was not included here out of
triviality.

## Building the Model
With image preprocessing out of the way, it was now time to construct the first renditions of our CNN using Keras, a ML library for Python
built on top of the well-known Tensorflow. Both of us were familiar with ML before having done this project, but it is always nice to get
a refresher on such a deep topic (there is an entire course on it at Harvard, after all!). We decided on a CNN with two Convolution layers
(basically iterating over the image with a window of specified area, checking for certain patterns in the data), which were each followed
by pooling of the data ("generalizations" on the outputs of the convolutional layer: for example, pooling might reduce a 100x100 to a size
of 50x50, capturing more general features in the process). Finally, our data was flattened (2D to 1D) and sent to our output
layer, with two nodes ('yes' and 'no,' or whatever you want to call them – get creative). Other formats were attempted, but this seemed to be a pretty
standard pick that achieved some average to above average results.

## Tweaking the Model
This is not to say the model worked on the first go; rather, the training loss (the "error" in the training set data, if you will)
quickly diverged after a few epochs (or, iterations through the neural network where each image was tested), usually sending accuracy plummeting
with it. We thus created a customized optimization function (with the help of Keras, of course): a stochastic gradient descent (i.e.,
find the direction where loss decreases fastest and go in that direction) optimizer with learning rate (how fast we go in this direction)
of 0.002, compared to one standard value of 0.01. This caused the loss to converge to its minimum more slowly, but hey, at least it converged
at all. After this, the model was left to train for about 15 minutes before finishing, with an accuracy above 85% on all or our tested
data. Industry standards for ML classification vary, but are usually above 95% (and usually higher than that), but we attribute this
at least in part to a somewhat limited amount of data from which to work, which may have led to slight overfitting of the train data.

## Conclusion
All in all, this was a very educational project that allowed both of us to replenish our knowledge of common machine learning practices through
a shared interest in space exploration. We would say the results slightly exceeded our expectations (except in the web submission aspect),
although there is always room for improvement – but hey, that's why there's CS 181!

## Don't Forget
Make sure to check out this [link](https://www.exoplanets50.com/) for more details, and don't forget to
watch [this video](https://www.youtube.com/watch?v=6QvRw4B7ZWA) for a brief demonstration of our model in action.
