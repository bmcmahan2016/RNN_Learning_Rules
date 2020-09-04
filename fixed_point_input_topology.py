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
    

    input_levels = ['[0.1]','[0.95]','[0.09]','[0.085]','[0.08]','[0.075]','[0.07]','[0.065]','[0.06]','[0.055]','[0.05]',\
                        '[0.045]','[0.04]','[0.035]','[0.03]','[0.025]','[0.02]','[0.015]','[0.005]','[0]','[-0.005]','[-0.01]',\
                            '[-0.015]','[-0.02]','[-0.025]','[-0.03]','[-0.035]','[-0.04]','[-0.045]','[-0.05]','[-0.055]','[-0.06]','[-0.065]',\
                            '[-0.07]','[-0.075]','[-0.08]','[-0.085]','[-0.09]','[-0.095]','[-0.1]']
    modelPath = "models/" + learningRule + "_0" + str(modelNum)
    model = loadRNN(modelPath)
    print(modelPath)
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
    inpt_values = fixed_points[:,0]
    fixed_points = fixed_points[:,1:]
    
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(fixed_points)
    distances, indices = nbrs.kneighbors(fixed_points)
    MDS_embedding = inpt_values[indices]
    num_fixed_points_found = MDS_embedding.shape[0]
    start_idx = -20 + (num_fixed_points_found // 2)
    end_idx = 20 + (num_fixed_points_found // 2)
    
    
    return MDS_embedding[start_idx:end_idx].reshape(40,3)



embeddings = []
max_fixed_points = 0

embeddings.append(getMDS(89).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(90).reshape(1,-1))
min_fixed_points = min(min_fixed_points, embeddings[-1].shape)
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(91).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(88).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(87).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(86).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(67).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(68).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

embeddings.append(getMDS(69).reshape(1,-1))
min_fixed_points = embeddings[-1].shape
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])


embeddings.append(getMDS(80, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(81, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(82, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(83, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(84, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(85, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(86, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(87, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])
embeddings.append(getMDS(88, "ga").reshape(1,-1))
max_fixed_points = max(max_fixed_points, embeddings[-1].shape[1])

# pad embeddings to account for differences in number of fixed points between RNNs
padded_embeddings = -np.ones((len(embeddings), max_fixed_points))
for rnn_num, embedding in enumerate(embeddings):
    padded_embeddings[rnn_num:rnn_num+1, :embedding.shape[1]] = embedding


# plot MDS clusters
clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(padded_embeddings)
plt.figure()
plt.scatter(clustered_data[:6,0], clustered_data[:6,1], c='r')
plt.scatter(clustered_data[6:9,0], clustered_data[6:9,1], c='b')
plt.scatter(clustered_data[9:,0], clustered_data[9:,1], c='g')
plt.legend(["BPTT (var=1.0)", "BPTT (var=0.5)", "GA (var=1.0)"])
plt.show()