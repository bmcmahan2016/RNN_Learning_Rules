'''
create the 3-bit flip flop task
'''

import numpy as np
#import utils
import torch
import pdb
import matplotlib.pyplot as plt

# note that whenever I call getInput, I'm drawing a sample from the distribution


class ThreeBitFlipFlop():
	def GetInput(self, sequence_len, show_plot=False):
		'''
		GetInput() will generate data for training
		reccurent neral networks (RNNs) on a discrimination
		task

		sequence_len: the number of data points to be used in the sequence

		show_plot: by default this is false. When this is set to true,
		the generated training data will be plotted

		**NOTE**
		as written pulses may co-occur superimposed on one another but all 
		pulses are guarenteed to be finished by the end fo the data 
		sequence
		'''

		#creates some zero centered noise with variance 0.1
		t_series = 0.1*torch.randn((3, sequence_len))
		#will store target training values
		y_train = torch.zeros((t_series.shape))

		#loop over three inputs
		for curr_input in range(3):
			#indices of pulses
			idx = torch.rand(3)*(sequence_len-25)
			idx = idx.int()
			#sort idx from smallest to largest
			idx, null = torch.sort(idx)
			#loop over each pulse in current input
			for _ in range(3):
				curr_idx = idx[_].item()
				if torch.randn(1).item() > 0:
					t_series[curr_input, curr_idx:curr_idx+25] += 1
					y_train[curr_input, curr_idx:] = 1
				else:
					t_series[curr_input, curr_idx:curr_idx+25] -= 1
					y_train[curr_input, curr_idx:] = -1

		if show_plot:
			#print(y_train)
			plt.subplot(311)
			plt.plot(np.array(t_series[0]))
			plt.plot(np.array(y_train[0]))
			plt.legend(['data', 'target'])
			plt.ylim([-2, 2])
			plt.subplot(312)
			plt.plot(np.array(t_series[1]))
			plt.plot(np.array(y_train[1]))
			plt.ylim([-2, 2])
			plt.subplot(313)
			plt.plot(np.array(t_series[2]))
			plt.plot(np.array(y_train[2]))
			plt.ylim([-2, 2])
			plt.show()
	
		return t_series, y_train

	# note that here the desired output is baked into the loss function
	# TODO: see if there's a better way so I don't need to specify the numpy and torch version
	# TODO: write a loss function for this specific task, current loss was copied from williams task
	def Loss(self, y, mu):
		pass

my_task = ThreeBitFlipFlop()
my_task.GetInput(1000, show_plot=True)
