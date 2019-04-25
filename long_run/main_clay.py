import rnn_clay_revise as rnn
import pandas as pd

if __name__ == '__main__':

	time_rnn = rnn.RNN()
	dataframe = pd.read_csv('../prices/all_prices.csv',header = None).values
	for i in range(340):
		print('=========================================================')
		print('Testing Day',i)
		print('=========================================================')
		train_num = 336
		roll_data = dataframe.T[i*24:336 + i*24].T
		time_rnn.load_data(roll_data,num_locs = 1)
