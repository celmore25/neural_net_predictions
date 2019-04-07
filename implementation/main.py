import rnn

if __name__ == '__main__':

	time_rnn = rnn.RNN()
	time_rnn.load_data('../prices/prices.csv',.67, False);
	time_rnn.split_timesteps(24, 10);
	time_rnn.predict()
 
