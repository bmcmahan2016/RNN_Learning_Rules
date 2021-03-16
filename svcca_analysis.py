# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:21:08 2020

@author: bmcma
"""

import rnn
import numpy as np
import torch
import matplotlib.pyplot as plt
import svcca.cca_core as cca_core
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import pdb
import argparse

def getNumSVs(singularValues):
    counter = 0
    explainedVariance = 0
    SVNorm = np.linalg.norm(singularValues)
    while (explainedVariance < 0.95):
        explainedVariance += (singularValues[counter] / SVNorm)**2
        counter+=1
    return counter

def get_rdm_inputs(inputSize):
    # returns inputs for the RDM task
    NUM_INPUT_CONDITIONS = 500
    static_inputs = np.linspace(-0.1857, 0.1857, NUM_INPUT_CONDITIONS).reshape(1, NUM_INPUT_CONDITIONS)
    static_inputs = np.matmul(np.ones((750, 1)), static_inputs)
    static_inputs = torch.tensor(static_inputs).float().cuda()
    static_inputs = torch.unsqueeze(static_inputs.t(), 1)
    
def get_static_inputs(inputSize):
    # will get inputs for the N = 1 context task
    N_INPUTS_PER_CONTEXT = 10
    N_CONTEXTS = int(inputSize / 2)
    MIN_INPUT = -0.1857
    MAX_INPUT = 0.1857
    static_inpt_arr = np.zeros((N_INPUTS_PER_CONTEXT*N_CONTEXTS, inputSize))
    # loop over contexts
    for context in range(N_CONTEXTS):
       strt_ix = context*N_INPUTS_PER_CONTEXT
       end_ix = (context+1)*N_INPUTS_PER_CONTEXT
       static_inpt_arr[strt_ix:end_ix, context] = np.linspace(MIN_INPUT, MAX_INPUT, N_INPUTS_PER_CONTEXT)  # set inputs
       static_inpt_arr[strt_ix:end_ix, context+N_CONTEXTS] = 1  # set GO signal
    # static_inpt_arr now filled in with shape (20, 8)
    static_inpt_arr = np.expand_dims(static_inpt_arr, -1)  # (20, 8, 1)
    assert static_inpt_arr.shape == (N_INPUTS_PER_CONTEXT*N_CONTEXTS, inputSize, 1)
    inpts = np.matmul(static_inpt_arr, np.ones((1, 750))) # (20, 8, 750)
    # feed batch of 20 inpts into network to get ativations 
    # cast static inputs to PyTorch Tensor
    static_inputs = torch.tensor(inpts).float().cuda() # 20, 8, 750
    return static_inputs
        
        
def getActivations(rnn_model):
    '''generates inputs for SVCCA'''
    switcher = {
            1:get_rdm_inputs,
            2:get_static_inputs,
            4:get_static_inputs,
            6:get_static_inputs,
            8:get_static_inputs
        }
    static_inputs = switcher[rnn_model._inputSize](rnn_model._inputSize)
    _, activations = rnn_model.feed(static_inputs, return_hidden=True)
    #finalActivations = activations[-1,:,:]       # keeps activation at end of trial only
    activations = np.swapaxes(activations, 0, 1).reshape(rnn_model._hiddenSize, -1)
    return activations #finalActivations    # 50 x 500  ----> numHidden x numInputs
        
# get a file of models to analyze
parser = argparse.ArgumentParser(description="Clusters RNNs by SVCCA")
parser.add_argument("fname", help="name of file containing RNNs to analyze")
args = parser.parse_args()

file_of_rnns = open("models/"+args.fname, 'r')
models = [(line.strip()).split() for line in file_of_rnns]
file_of_rnns.close()


modelActivations = []
NUM_DIMENSIONS = 2

#models = [bptt, ga]
numModelsOfType = {}
count = 0
for i in range(len(models)):
    numModelsOfType[models[i][0]] = len(models[i])-1
    count += len(models[i]) - 1
numModelsOfType["total"] = count

distances = np.zeros((numModelsOfType["total"], numModelsOfType["total"]))
for modelType in models:
    for count, model in enumerate(modelType):
        if (count == 0):
            print("model name:", model)
        else:
            print("name:", model)
            rnn_inst = rnn.loadRNN("models/"+model)
            activations = getActivations(rnn_inst)
            modelActivations.append(activations)
 
# compute the distance matrix
print("computing distance matrix ...\n")
for i in range(len(modelActivations)):
    print("    Computing row", int(i))
    for j in range(i, len(modelActivations)):   # ~2X speedup using symmetry
        activationsI = modelActivations[i]
        activationsJ = modelActivations[j]
        activationsI -= np.mean(activationsI, axis=1, keepdims=True)
        activationsJ -= np.mean(activationsJ, axis=1, keepdims=True)
        
        # take top 20 singular values
        U1, s1, V1 = np.linalg.svd(activationsI, full_matrices=False)
        U2, s2, V2 = np.linalg.svd(activationsJ, full_matrices=False)
        
        #print("Number of singular values needed to explain 95% of variance:", getNumSVs(s1))

        svacts1 = np.dot(s1[:NUM_DIMENSIONS]*np.eye(NUM_DIMENSIONS), V1[:NUM_DIMENSIONS])
        svacts2 = np.dot(s2[:NUM_DIMENSIONS]*np.eye(NUM_DIMENSIONS), V2[:NUM_DIMENSIONS])
        
        svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
        distances[i, j] = 1 - np.mean(svcca_results["cca_coef1"])
        distances[j, i] = distances[i, j]
        # distances are shape (numRNNs, numRNNs)



plt.imshow(distances)
plt.colorbar()
plt.title("SVCCA Distances Between Networks")

clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(distances)

################
# compute K-Means clusters
################
K_cluster = KMeans(n_clusters = 4)
kmeans = K_cluster.fit(distances)

plt.figure()
colors = ['r', 'b', 'g', 'y', 'k']
for i in range(len(clustered_data)):
    plt.scatter(clustered_data[i, 0], clustered_data[i,1], c = colors[kmeans.labels_[i]])

################


plt.figure()  # Scatter plot all the models
count = 0     # keeps track of how many models plotted so far
legends = []  # holds model names to be used in legend
for modelType in numModelsOfType:
    if modelType == "total": # no plotting for total since it isn't a valid model type
        continue
    value = numModelsOfType[modelType]
    plt.scatter(clustered_data[count:count+value,0], clustered_data[count:count+value,1])
    i+=1
    count += value  # increment the number of models plotted
    legends.append(modelType) # add this model type name to the legend
    
#plt.scatter(clustered_data[N_BPTT_MODELS:N_BPTT_MODELS+N_GA_MODELS,0], clustered_data[N_BPTT_MODELS:N_BPTT_MODELS+N_GA_MODELS,1], c='g')
#plt.scatter(clustered_data[N_BPTT_MODELS+N_GA_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS,0], clustered_data[N_BPTT_MODELS+N_GA_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS,1], c='b')
#plt.scatter(clustered_data[N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS+N_H_MODELS,0], clustered_data[N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS+N_H_MODELS,1], c='y')

plt.legend(legends)
plt.show()
#assert False

# Mean subtract activations
#cacts1 = activations1 - np.mean(activations1, axis=1, keepdims=True)
#cacts2 = activations2 - np.mean(activations2, axis=1, keepdims=True)

# Perform SVD
#U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
#U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

# take top 20 singular values
#svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
#svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])

#svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)

# Mean subtract baseline activations
#cb1 = b1 - np.mean(b1, axis=0, keepdims=True)
#cb2 = b2 - np.mean(b2, axis=0, keepdims=True)

# Perform SVD
#Ub1, sb1, Vb1 = np.linalg.svd(cb1, full_matrices=False)
#Ub2, sb2, Vb2 = np.linalg.svd(cb2, full_matrices=False)

#svb1 = np.dot(sb1[:20]*np.eye(20), Vb1[:20])
#svb2 = np.dot(sb2[:20]*np.eye(20), Vb2[:20])

#svcca_baseline = cca_core.get_cca_similarity(svb1, svb2, epsilon=1e-10, verbose=False)
#print("Baseline", np.mean(svcca_baseline["cca_coef1"]), "and MNIST", np.mean(svcca_results["cca_coef1"]))

#plt.plot(svcca_baseline["cca_coef1"], lw=2.0, label="baseline")
#plt.plot(svcca_results["cca_coef1"], lw=2.0, label="MNIST")
#plt.xlabel("Sorted CCA Correlation Coeff Idx")
#plt.ylabel("CCA Correlation Coefficient Value")
#plt.legend(loc="best")
#plt.grid()
