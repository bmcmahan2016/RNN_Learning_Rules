# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:07:16 2020

@author: bmcma

Description: Clusters model fixed points based on input topology

"""
import numpy as np
from rnn import loadRNN, RNN
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import sys
import pdb
import argparse

def key2array(key):
    '''
    converts fixed point input key to an array with type float

    Parameters
    ----------
    key : string
        input key value in fixed points dictionary.

    Returns
    -------
    array (float)
        input value for fixed point associated with key.

    '''
    return np.array(float(key[1:-1]))




def getMDS(modelNum, learningRule="bptt"):
    '''
    Creates an embedding matrix from MDS based on input topology of fixed points
    for a single RNN. 

    Parameters
    ----------
    modelNum : int
        specifies the RNN model to load.
    learningRule : string, optional
        specifies which model to load based on the learning rule. The default is "bptt".

    Returns
    -------
    MDS_embedding : NumPy array
        embedding of networks fixed points based on input topology two nearest 
        neighbors. Has shape (num_fixed_points, 3).

    '''
    


    if modelNum < 100:
        modelPath = "models/" + learningRule + "_0" + str(modelNum)
    else:
        modelPath = "models/" + learningRule + "_" + str(modelNum)
    model = loadRNN(modelPath)
    print(modelPath)
    '''
    fixed_points = []
    counter = 0
    for key in model._fixedPoints:
        print(key)
        counter += 1
    print(counter)
    for inpt in model._fixedPoints:
        for fixed_point in model._fixedPoints[inpt]:
            # fixed_point is now a single fixed point
            fixed_points.append( np.hstack((key2array(inpt), fixed_point)) )
    fixed_points = np.array(fixed_points)
    '''
    #fixed_points = model._fixedPoints
    inpt_values = model._fixedPoints[:,1]
    fixed_points = model._fixedPoints[:,2:]
    
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fixed_points)
    distances, indices = nbrs.kneighbors(fixed_points)
    MDS_embedding = inpt_values[indices]
    num_fixed_points_found = 40
    start_idx = -(num_fixed_points_found//2) + (num_fixed_points_found // 2)
    end_idx = (num_fixed_points_found//2) + (num_fixed_points_found // 2)
    
    
    return MDS_embedding[start_idx:end_idx].reshape(num_fixed_points_found,3)


# determines what analysis to run
parser = argparse.ArgumentParser(description="Clusters RNNs by Topology of Fixed Points")
parser.add_argument("fname", help="name of file containing RNNs to analyze")
args = parser.parse_args()

a_file = open("models/"+args.fname, 'r')
list_of_lists = [(line.strip()).split() for line in a_file]
a_file.close()

embeddings = []
max_fixed_points = 0
counter = 0

bptt_list = list_of_lists[0]
bptt_start = 0
for num in bptt_list:
    num = int(num)
    embeddings.append(getMDS(num).reshape(1,-1))
    counter += 1
bptt_end = counter
ReLU_list = list_of_lists[1]
ReLU_start = counter
for num in ReLU_list:
    num = int(num)
    embeddings.append(getMDS(num).reshape(1,-1))
    counter += 1
ReLU_end = counter
ga_list = list_of_lists[2]
ga_start = counter
for num in ga_list:
    num = int(num)
    embeddings.append(getMDS(num, learningRule = "ga").reshape(1,-1))
    counter += 1
ga_end = counter

# ff_start = counter
# for num in [41, 42, 43, 44, 45]:
#     embeddings.append(getMDS(num, "FullForce").reshape(1,-1))
#     counter +=1 
# ff_end = counter



# embeddings.append(getMDS(67).reshape(1,-1))
# embeddings.append(getMDS(68).reshape(1,-1))
# embeddings.append(getMDS(69).reshape(1,-1))


embeddings = np.squeeze(np.array(embeddings))

# pad embeddings to account for differences in number of fixed points between RNNs
# padded_embeddings = -np.ones((len(embeddings), max_fixed_points))
# for rnn_num, embedding in enumerate(embeddings):
#     padded_embeddings[rnn_num:rnn_num+1, :embedding.shape[1]] = embedding


# plot MDS clusters
clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(embeddings)
plt.figure()
plt.scatter(clustered_data[:bptt_end,0], clustered_data[:bptt_end,1], c='r')                      # BPTT (tanh)
plt.scatter(clustered_data[ReLU_start:ReLU_end,0], clustered_data[ReLU_start:ReLU_end,1], c='b')  # BPTT (ReLU)
plt.scatter(clustered_data[ga_start:ga_end,0], clustered_data[ga_start:ga_end,1], c='g')          # GA
#plt.scatter(clustered_data[ff_start:ff_end,0], clustered_data[ff_start:ff_end,1], c='y')         # FF
plt.legend(["BPTT (tanh)", "BPTT (ReLU)", "GA", "FF", "Heb"])      
plt.show()