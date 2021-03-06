import rnn_clay_revise as rnn
import pandas as pd

if __name__ == '__main__':

	time_rnn = rnn.RNN()
	dataframe = pd.read_csv('../prices/prices.csv',header = None).values
	for i in range(7):
		print('=========================================================')
		print('Testing Day',i)
		print('=========================================================')
		roll_data = dataframe.T[i*24:168 + i*24].T
		time_rnn.load_data(roll_data,num_locs = 50)
