# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:07:16 2020

@author: bmcma
"""
import numpy as np
from rnn import loadRNN, RNN
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def get_fp_structure(modelNum, learningRule="bptt"):
    input_levels = ['[0.2]', '[0.18]', '[0.16]', '[0.14]','[0.12]','[0.1]','[0.08]','[0.06]','[0.04]',
                    '[0.02]','[0.]','[-0.02]','[-0.04]','[-0.06]','[-0.08]','[-0.1]','[-0.12]','[-0.14]','[-0.16]','[-0.18]','[-0.2]']
    modelPath = "models/" + learningRule + "_0" + str(modelNum)
    model = loadRNN(modelPath)
    fp_structure = []

    for thing in input_levels:
        fp_structure.append(len(model._fixedPoints[thing]))
    return np.array(fp_structure)

fp_distances = np.zeros((10, 21))

fp_distances[0,:] = get_fp_structure(89, "bptt")
fp_distances[1,:] = get_fp_structure(90, "bptt")
fp_distances[2,:] = get_fp_structure(91, "bptt")

fp_distances[3,:] = get_fp_structure(67, "bptt")
fp_distances[4,:] = get_fp_structure(68, "bptt")
fp_distances[5,:] = get_fp_structure(69, "bptt")

fp_distances[6,:] = get_fp_structure(85, "GA")
fp_distances[7,:] = get_fp_structure(86, "GA")
fp_distances[8,:] = get_fp_structure(87, "GA")
fp_distances[9,:] = get_fp_structure(88, "GA")

fp_distances += 0.1*np.random.randn(10, 21)

clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(fp_distances)
plt.figure()
plt.scatter(clustered_data[:3,0], clustered_data[:3,1], c='r')
plt.scatter(clustered_data[3:6,0], clustered_data[3:6,1], c='b')
plt.scatter(clustered_data[6:,0], clustered_data[6:,1], c='g')
plt.legend(["BPTT (var=1.0)", "BPTT (var=0.5)", "GA (var=1.0)"])