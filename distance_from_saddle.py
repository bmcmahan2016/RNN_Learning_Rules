# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:39:25 2020

@author: Brandon McMahan
Description: This will plot the distance from the unstable fixed point of the RNNs
trajectory through time
"""


import numpy as np 
import numpy.linalg as LA
import torch
from rnn import RNN, loadRNN 
import rnntools as r
from FP_Analysis import FindFixedPoints, FindFixedPointsC, GetDerivatives, ComputeDistance, IsAttractor
import time
from task.williams import Williams
from task.contexttask import ContextTask
from sklearn.decomposition import PCA
import json
import sys

import matplotlib.pyplot as plt


def GetUnstableRoot(listOfRoots, dynamicsFunc):
	'''
	Will check a list of roots to see which one is unstable and return it
	'''
	for root in listOfRoots:
		if not IsAttractor(root, dynamicsFunc):
			return root
	# if an unstable root was never detected return False
	return False

# load in a model
model = loadRNN('models/bptt_061')
assert(model)
def GetDistanceFromSaddle(trajectories, saddle_node):
	distFromSaddle = trajectories - saddle_node
	distFromSaddle = np.square(distFromSaddle)
	distFromSaddle = np.sum(distFromSaddle, axis=-1)
	distFromSaddle = np.sqrt(distFromSaddle)
	return distFromSaddle


# get the fixed points for the model
test_inpt = 0.01
cs = ['r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b']
trial_data = r.record(model, \
	test_inpt*np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]), \
	title='fixed points', print_out=True, plot_recurrent=False)
# find the fixed points for the model using the PC axis found on the niave model
F = model.GetF()
input_values = [[1],[.9],[.8],[.7],[.6],[.5],[.4],[.3],[.2],[.1],[0],\
					[-.1],[-.2],[-.3],[-.4],[-.5],[-.6],[-.7],[-.8],[-.9],[-1]]
input_values = test_inpt * np.array(input_values)
#input_values = [[.1],[0],[-.1]]
roots, idx, pca = FindFixedPoints(F, input_values, embedding='', embedder=model._pca, Verbose=False)

# we need to determine which root is unstable
posSaddle = GetUnstableRoot(roots['[0.01]'], F(np.array([0.01])))
negSaddle = GetUnstableRoot(roots['[-0.01]'], F(np.array([-0.01])))

# now have the trajectories contained in trial_data
distFromSaddle = GetDistanceFromSaddle(trial_data[:5], posSaddle)
distFromSaddle = np.concatenate((distFromSaddle, GetDistanceFromSaddle(trial_data[5:], negSaddle)), axis=0) 

# loop over trials and plot distances to saddle
plt.figure()
for trial_num in range(10):
	plt.plot(distFromSaddle[trial_num], c=cs[trial_num])
plt.xlabel('Time in Trial')
plt.ylabel('Distance to Saddle')