# Exoplanet Detection Using Machine Learning by Zeel Patel

Contact: zpatel@hpair.org

## Background Information for Implementation

The purpose of this project, at its core, is to analyze a light curve for periodic dips that are characteristic of exoplanets. To understand this, you will have to understand
a bit of the science behind it as well. When stars from galaxies far away are observed through our modern telescopes in low earth orbit (such as Hubble, TESS, Kepler to name a few), scientists track
the emission spectrum of that distant star and observe the brightness of the light coming from that star over a period of time. Thus, the graph can be modelled as brightness of a distant
star as a function of time. However, every now and then, a planet orbiting the distant star will pass by in front of it, creating an eclipse whereby it is in the middle of our telescope
and the star we hope to observe, casting a shadow and decreasing the intensity of the light we observe. This phenomenon is called a transit, and only occurs when the exoplanet is in between
the star and the telescope. On a graph, this will be a dip in the brightness of the light. Now, say a planet's orbit is once every 28 days around its star. Then, we will notice this
dip in the brightness of star occur periodically: once every 28 days. This recurring periodic dip in the brightness of an observed star is a property of exoplanets, and is how scientists
around the world find and detect them (this method of detection is called transit observation). Currently in space right now, there is one major satellite called Kepler that is commissioned
by NASA and is exploring hundreds of these distant stars in search of new planets. The pipeline is such that NASA's Kepler telescope will observe a star for just under a month
and then beam that data back down to NASA and Jet Propulsion Labs (JPL) and MAST to process that data and remove instrumentation noise and anomalies that may cause error before making the data public.
After that, the light curves (the graph of brightness vs time) will become open to the scientific community to use, and scientists can began discovering exoplanets in the universe!

I decided to tackle this problem from a machine learning perspective. My approach was to get the light curves of some of the exoplanets that have
been confirmed by NASA over the past 5 years whose light curves are available on the NASA Exoplanets Archive and MAST, and use that set to train a CNN model to recognize which light curve is for an exoplanet
or which is a false/non-exoplanet light curve. To train it on the false batch, I will also be using the validated non-exoplanet light curves from NASA that are known to be false positives or negatives.

To do this, I built a Convolutional Neural Network in Python through Keras (a Python ML library which is built on top of TensorFlow) which currently outputs the result with
greater than 65% accuracy during regulated testing rounds (neither terrible nor spectacular). 

In order to actually run the model developed, all you will need to do is execute my program as described below:

## Running my project:

1.	Navigate to/create an Exoplanet Project directory after download of files, making sure to keep hierarchy intact. Open terminal on Mac.
3.	Ensure all required files are present (model2.json, model2.h5, prediction.py, the inputdatarevised subdirectory, and training.py (last one need not be executed; unless you want to train another neural network)).
4.	Ensure that the following (rather lengthy, I apologize in advance) set of packages is installed for Python3 (hey, this just means you’ll have them if you need them later): Matplotlib/Pyplot `pip3 install matplotlib`,
5.	Numpy `pip3 install numpy`, Sklearn `pip3 install scikit-learn`, Tensorflow `pip3 install tensorflow`, Keras `pip3 install keras`, PIL `pip3 install pillow`, H5Py `pip3 install h5py`
5.	To see what images are available to test, navigate to the inputdatarevised subdirectory and open one of the lightcurves.
6.	Execute `python3 prediction.py` in terminal.
7.	When prompted to input an image, please give the name of one of the images in inputdatarevised (e.g., ‘yesa.png’). You will see a few print statements giving you the results as well
    as a visual representation of the image.
8.	That’s all there is to it!
