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

class RNN(object):

    ''' initialize RNN object and its attributes'''
    def __init__(self):
        self.dataset = None
        self.test, self.train = None, None
        self.t_train, self.t_test = None, None
        self.X, self.y = None, None
        self.num_in, self.num_out = None, None
        self.num_predict = 24

    ''' load data into usable objects '''
    def load_data(self, data, frac_train, show_graph):

        i = 0
        for value in data:
            if i == 5:
                break
            print ('Node #',i)
            i += 1
            self.dataset = value
            self.dataset = self.dataset.astype('float32')
            self.train = self.dataset[: -self.num_predict]
            self.test = self.dataset[-self.num_predict :]
            self.split_train_test(frac_train)
            print('Training:',len(self.train), '\tTesting:',len(self.test))
            if show_graph:
                self.display_data()
            self.split_timesteps(24, 24);
            self.predict()
         

    ''' split ordered dataset into train and test sets '''
    def split_train_test(self, frac_train):
        self.t_train = np.arange(self.train.shape[0])
        self.t_test = np.arange(self.test.shape[0]) + self.train.shape[0]

    ''' visualize train and test data '''
    def display_data(self):
        plt.plot(self.t_train,self.train)
        plt.plot(self.t_test,self.test)
        plt.legend(['train','test'])
        plt.show()

    '''  Splits the time series into a sample with a specifed number of input and output components. '''
    def split_timesteps(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out

        # intialize vectors for the input and output into the NN
        in_vec = []
        out_vec = []

        # determine how many training examples can be made
        num_instances = self.train.shape[0] - num_in - num_out + 1
        print('# of training instances used:',num_instances)
    
        # loop over the timeseries
        for i in range(num_instances):

            # input and output data for each instance
            data_inp = self.train[i:num_in + i]
            data_out = self.train[num_in + i:num_in + num_out+i]

            # store values
            in_vec.append(data_inp)
            out_vec.append(data_out)

        self.X = np.array(in_vec)
        self.y = np.array(out_vec)

        # reshape data
        features = 1
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], features))
        print(self.X.shape)

    ''' define a model and use it to predict the next 10 timesteps '''
    def predict(self):

        # define model
        features = 1
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(self.num_in, features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(self.num_out))
        model.compile(optimizer='adam', loss='mse')

        # fit model 
        model.fit(self.X, self.y, epochs=100, verbose=0)

        # demonstrate prediction
        x_input = self.train[-24:]
        print('Input Price: ')
        print(x_input)
        x_input = x_input.reshape((1, self.num_in, features))
        yhat = model.predict(x_input, verbose=0)
        print("Predicted: ")
        print(yhat)
        self.calculate_error(yhat[0])

    def calculate_error(self, predicted):
        sum = 0
        correct = self.test
        print("Actual:")
        print(correct)
        for i in range(168):
            sum += 100*(abs(predicted[i] - correct[i]))/(correct[i])
        print("Average percent error: " + str(sum/168))
        print()



