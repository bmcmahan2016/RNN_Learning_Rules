# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:07:16 2020

@author: bmcma
"""
import numpy as np
from rnn import loadRNN, RNN
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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
    


    if learningRule[0].lower() == 'f':
        modelPath = "models/" + learningRule + "0" + str(modelNum)
    else:
        modelPath = "models/" + learningRule + "_0" + str(modelNum)
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



embeddings = []
max_fixed_points = 0
counter = 0

bptt_start = 0
for num in [69, 68, 87, 88, 89, 90, 91, 92]:
    embeddings.append(getMDS(num).reshape(1,-1))
    counter += 1
bptt_end = counter
ga_start = counter
for num in [ 81, 82, 83, 84, 85, 86, 87, 88]:
    embeddings.append(getMDS(num, "ga").reshape(1,-1))
    counter += 1
ga_end = counter

# ff_start = counter
# for num in [80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]:
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
plt.scatter(clustered_data[:bptt_end,0], clustered_data[:bptt_end,1], c='r')         # BPTT
plt.scatter(clustered_data[-3:,0], clustered_data[-3:,1], c='b')       # BPTT
plt.scatter(clustered_data[ga_start:ga_end,0], clustered_data[ga_start:ga_end,1], c='g')         # GA
plt.scatter(clustered_data[ff_start:ff_end,0], clustered_data[ff_start:ff_end,1], c='y')
plt.legend(["BPTT (var=1.0)", "BPTT (var=0.5)", "GA (var=1.0)", "FF (var=1.0)"])       # FF
plt.show()