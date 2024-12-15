import os
import random
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib

NUM_DATAPOINTS = 100
NUM_SAMPLES_PER_DP = 500
TIME_STEP = 0.1

def run():
	count = 0

	for k in range(NUM_DATAPOINTS):

		x_samples = []
		y_samples = []		

		split_str = 'train'
		if random.random() > 0.8:
			split_str = 'val'

		mu = 1.0
		sigma = 0.3
		frequency = np.random.normal(mu, sigma)

		for i in range(NUM_SAMPLES_PER_DP):
			x = i * TIME_STEP * frequency
			y = math.sin(x)

			x_samples.append(x)
			y_samples.append(y)

		x_samples = np.asarray(x_samples)
		y_samples = np.asarray(y_samples)

		out_path = 'datasets/sin_data/' + split_str + '/'

		if not os.path.exists(out_path):
			os.makedirs(out_path)					

		#Write data file
		file_name = out_path + str(count).zfill(4) + '.txt'
		file = open(file_name, "a")
		for i in range(len(x_samples)):
			s = str(x_samples[i]) + ' ' + str(y_samples[i]) + '\n'
			file.write(s)
		file.close()

		#Plot data 
		plt.clf()
		plt.plot(y_samples)
		plt.title('Sin Plot: ')
		plt.ylabel('Y')
		plt.xlabel('X')
		plt.legend(['Sin'], loc='upper left')
		plt.savefig(out_path + str(count).zfill(4) + '.png')

		count += 1

if __name__== "__main__":
  run()