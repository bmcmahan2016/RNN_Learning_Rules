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

def getNumSVs(singularValues):
    counter = 0
    explainedVariance = 0
    SVNorm = np.linalg.norm(singularValues)
    while (explainedVariance < 0.95):
        explainedVariance += (singularValues[counter] / SVNorm)**2
        counter+=1
    return counter

def getActivations(rnn_model):
    NUM_INPUT_CONDITIONS = 500
    static_inputs = np.linspace(-0.1857, 0.1857, NUM_INPUT_CONDITIONS).reshape(1, NUM_INPUT_CONDITIONS)
    static_inputs = np.matmul(np.ones((750, 1)), static_inputs)
    static_inputs = torch.tensor(static_inputs).float().cuda()
    _, activations = rnn_model.feed(static_inputs, return_hidden=True)
    #finalActivations = activations[-1,:,:]       # keeps activation at end of trial only
    activations = np.swapaxes(activations, 0, 1).reshape(50, -1)
    return activations #finalActivations    # 50 x 500  ----> numHidden x numInputs
        

modelActivations = []
NUM_DIMENSIONS = 2
bptt = ["models/bptt_081", "models/bptt_082", "models/bptt_083", "models/bptt_084", "models/bptt_085", "models/bptt_086", "models/bptt_087", "models/bptt_088", "models/bptt_089", "models/bptt_090"]
ga = ["models/GA_080", "models/GA_081", "models/GA_082", "models/GA_083", "models/GA_084", "models/GA_085", "models/GA_086", "models/GA_087", "models/GA_088"]
ff = ["models/FullForce080", "models/FullForce081", "models/FullForce082", "models/FullForce083"]
models = [bptt, ga, ff]
N_BPTT_MODELS = len(bptt)
N_GA_MODELS = len(ga)
N_FF_MODELS = len(ff)
N_TOTAL_MODELS = N_BPTT_MODELS + N_GA_MODELS + N_FF_MODELS
distances = np.zeros((N_TOTAL_MODELS, N_TOTAL_MODELS))
for modelType in models:
    for model in modelType:
        rnn_inst = rnn.loadRNN(model)
        activations = getActivations(rnn_inst)
        modelActivations.append(activations)
    
for i in range(len(modelActivations)):
    for j in range(i, len(modelActivations)):
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

plt.imshow(distances)
plt.colorbar()
plt.title("SVCCA Distances Between Networks")

clustering_algorithm = MDS()
clustered_data = clustering_algorithm.fit_transform(distances)
plt.figure()
plt.scatter(clustered_data[:N_BPTT_MODELS,0], clustered_data[:N_BPTT_MODELS,1], c='r')
plt.scatter(clustered_data[N_BPTT_MODELS:N_BPTT_MODELS+N_GA_MODELS,0], clustered_data[N_BPTT_MODELS:N_BPTT_MODELS+N_GA_MODELS,1], c='g')
plt.scatter(clustered_data[N_BPTT_MODELS+N_GA_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS,0], clustered_data[N_BPTT_MODELS+N_GA_MODELS:N_BPTT_MODELS+N_GA_MODELS+N_FF_MODELS,1], c='b')
plt.legend(["BPTT", "GA", "FORCE"])
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
