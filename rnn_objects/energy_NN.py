'''
Authors: Clay Elmore, Grace Kopp

This file will hold the basic python objects needed to make and train
various Recurrent Neural Networks for energy market price predictions.
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class manip_data:

	def import_prices(path):

		'''
		Function to import time series data as a matrix
		'''

		data = pd.read_csv(path,header=None).values
		return data

# class rnn:

# 	'''
# 	Class to hold all functions used for Recurrent Neural Network 
# 	training and testing.
# 	'''

# 	def 








if __name__ == '__main__':
	
	ex_plot = False
	np.random.seed(7)

	# data = manip_data.import_prices('/Users/ClayElmore/Desktop/neural_net_predictions/prices/prices.csv')
	# time_series = data[0]

	dataframe = pd.read_csv('/Users/ClayElmore/Desktop/neural_net_predictions/prices/international-airline-passengers.csv', \
		usecols=[1], engine='python', skipfooter=3)
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	

	# make a training and testing set
	frac_train = 0.67

	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print(len(train), len(test))
	
	# function to create a training and learning set
	def create_dataset(dataset, look_back=1):
		dataX, dataY = [], []
		for i in range(len(dataset)-look_back-1):
			a = dataset[i:(i+look_back)]
			dataX.append(a)
			dataY.append(dataset[i + look_back])
		return np.array(dataX), np.array(dataY)

	# make a real dataset that can be used
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
		
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	print('Reshaped training data:')
	print('X:',trainX.shape)
	print('Y:',trainY.shape)

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	print('Results Train Shape:',trainPredict.shape)
	testPredict = model.predict(testX)
	print('Results test Shape:',testPredict.shape)

	# revert the normalization
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)
	print('Non-normalized Results Train Shape:',trainPredict.shape)
	print('Non-normalized Results test Shape:',testPredict.shape)	

	# # calculate root mean squared error
	train_error = (trainPredict - trainY) / np.mean(trainY)
	print('Train Score: %.2f RMSE' % (train_error))
	test_error = (testPredict - testY) / np.mean(testY)
	print('Test Score: %.2f RMSE' % (test_error))

	# # shift train predictions for plotting
	# trainPredictPlot = numpy.empty_like(dataset)
	# trainPredictPlot[:, :] = numpy.nan
	# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# # shift test predictions for plotting
	# testPredictPlot = numpy.empty_like(dataset)
	# testPredictPlot[:, :] = numpy.nan
	# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# # plot baseline and predictions
	# plt.plot(scaler.inverse_transform(dataset))
	# plt.plot(trainPredictPlot)
	# plt.plot(testPredictPlot)
	# plt.show()

	# plot if desired
	if ex_plot:
		plt.plot(time_series)
		plt.show()
	


