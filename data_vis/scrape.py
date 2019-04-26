def get_results(path,inp_days,verbose = False):

	''' print out clean version of data to show usage and verify accuracy '''
	def print_results(days):
		for day_num in range(len(days)):
			print()
			print("Testing Day " + str(day_num))
			for i in range(len(days[day_num])):
				print("    Node " + str(i) + " = " + str(days[day_num][i]))

	''' Scrape results file into meaningful format where error can be linked to
		day # and node #  '''

	# average percent error will be accessible as days[day #][node #]
	days = [dict() for i in range(inp_days)]

	# initialize indexing variables
	curr_day = None
	curr_node = None

	for line in open(path):
		line = line.strip()
		words = line.split()

		# only look at populated lines
		if not words:
			continue

		# keep track of current testing day for proper indexing
		if words[0] == "Testing":
			curr_day = int(words[2])

		# keep track of current node number for proper indexing
		if words[0] == "Node":
			curr_node = int(words[2])

		# associate average percent error with the current day and node        
		if words[0] == "Average":
			days[curr_day][curr_node] = float(words[3])

	# print results
	if verbose:
		print_results(days)

	return days

if __name__ == '__main__':
	path = '../implementation/log_clay.txt'
	inp_days = 7
	days = get_results(path,inp_days,verbose = True)












