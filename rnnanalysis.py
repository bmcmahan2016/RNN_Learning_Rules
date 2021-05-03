import numpy as np 
import numpy.linalg as LA
import torch
from rnn import RNN, loadRNN 
 
import rnntools as r
from FP_Analysis import Roots
import time
from task.williams import Williams
from task.context import context_task
from task.multi_sensory import multi_sensory
from task.dnms import DMC
from sklearn.decomposition import PCA
import json
import sys
import pdb
import matplotlib.pyplot as plt


def rdm_fixed_points(modelPath, inputs, save_fp=False):
    assert (inputs == 'large' or inputs == 'small'), "Must use either small or large inputs"
    model = loadRNN(modelPath)
    #AnalyzeLesioned(model, modelPath, xmin, xmax, ymin, ymax)
    
    inpts = {}  # two sets of inputs for solving fixed points
    inpts['large'] = np.array ( [[0.5], [0.2], [0], [-0.2], [-0.5]] )
    inpts['small'] = 0.03 * np.array( [[1], [0.75], [.5], [0.25], [0], [-0.25], 
                                [-.5], [-0.75], [-1]] )
    input_values = inpts[inputs]   # user specified input set

    model_roots = Roots(model)
    model_roots.FindFixedPoints(input_values)  # compute RNNs fixed points
   
    model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 50)
    plt.title("Early")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time=50, end_time=200)
    plt.title("Mid")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time=200)
    plt.title("Late")
    
    plt.figure(2)
    model_roots.plot(fixed_pts=True, slow_pts=False, plot_traj=False)
    plt.title("Model Attractors") 
    
    plt.figure()
    model_roots.plot(fixed_pts=False, slow_pts=False, plot_traj=False, plot_PC1=True)
    plt.title("PC1")
    plt.xlabel("Time")
    plt.ylabel("PC1")
            
    
    if save_fp:
        model_roots.save(modelPath)   


def context_fixed_points(modelPath, inputs, save_fp=False):
    model = loadRNN(modelPath, task="context")

    # construct inputs
    if inputs=='small':
        fixed_point_resolution = 21
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[:fixed_point_resolution, 0] = np.linspace(-0.02, 0.02, fixed_point_resolution)     # motion context
        static_inpts[:fixed_point_resolution, 2] = 1                                    # go signal for motion context
        static_inpts[fixed_point_resolution:, 1] = np.linspace(-0.02, 0.02, fixed_point_resolution)     # color context
        static_inpts[fixed_point_resolution:, 3] = 1                                    # go signal for color context
    elif inputs == 'large':
        fixed_point_resolution = 5
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[:fixed_point_resolution, 0] = np.linspace(-0.4, 0.4, fixed_point_resolution)     # motion context
        static_inpts[:fixed_point_resolution, 2] = 1                                    # go signal for motion context
        static_inpts[fixed_point_resolution:, 1] = np.linspace(-0.4, 0.4, fixed_point_resolution)     # color context
        static_inpts[fixed_point_resolution:, 3] = 1         

    model_roots = Roots(model)
    model_roots.FindFixedPoints(static_inpts)


    plt.figure(100)
    plt.title("PCA of Fixed Points For Contextual Integration Task")
    
    
    model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 100)
    plt.title("Early")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 100, end_time = 300)
    plt.title("Mid")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 400)
    plt.title("Late")
    
    model_roots.plot(fixed_pts=True, slow_pts=True)

    plt.figure()
    model_roots.plot(fixed_pts=True, slow_pts=False, plot_traj=False)
    plt.title("Model Attractors")

    plt.figure(123)
    plt.title('Evaluation of Model on Multisensory Task')   

    if save_fp:
        model_roots.save(modelPath)   

    plt.show()

def N_fixed_points(model_choice, save_fixed_points=False):
    model = loadRNN(model_choice, task="Ncontext")
    model_roots = Roots(model)

    model.plotLosses()
    
    
    ###########################################################################
    #TENSOR COMPONENT ANALYSIS
    ###########################################################################
    activity_tensor = model._activityTensor
    neuron_factor = r.plotTCs(activity_tensor, model._targets, 1)
    neuron_idx = np.argsort(neuron_factor)     # sorted indices of artificial neurons
    
    # find the index where neuron_factors changes sign
    # p is the index that partitions neuron_idx into two clusters
    sign_change_idx = np.diff(np.sign(neuron_factor[neuron_idx]))
    if not np.all(sign_change_idx==0):
        # executes when neuron factors are not all the same sign
        last_pos_neuron = np.nonzero(sign_change_idx)[0][0]
        print('number of positive neurons:', last_pos_neuron)
        print('number of negative neurons:', len(neuron_factor)-last_pos_neuron)
        p = np.nonzero(np.diff(np.sign(neuron_factor[neuron_idx])))[0][0] + 1
    else:
        # executes when neuron factors all have the same sign
        print('artifical neurons all have sign', np.sign(neuron_factor[0]))
        p = 25
    
    
    ###########################################################################
    #VISUALIZE RECURRENT WEIGHTS
    ###########################################################################
    model._neuronIX = neuron_idx
    model.VisualizeWeightMatrix()
    model.VisualizeWeightClusters(neuron_idx, p)
     

    fixed_point_resolution = 5
    static_inpts = np.zeros((3*fixed_point_resolution, 6))
    static_inpts[:fixed_point_resolution, 0] = np.linspace(-0.1857, 0.1857, fixed_point_resolution)     # motion context
    static_inpts[:fixed_point_resolution, 3] = 1                                    # go signal for motion context
    static_inpts[fixed_point_resolution:2*fixed_point_resolution, 1] = np.linspace(-0.1857, 0.1857, fixed_point_resolution)     # color context
    static_inpts[fixed_point_resolution:2*fixed_point_resolution, 4] = 1                                    # go signal for color context
    static_inpts[2*fixed_point_resolution:, 2] = np.linspace(-0.1857, 0.1857, fixed_point_resolution)     # color context
    static_inpts[2*fixed_point_resolution:, 5] = 1                                    # go signal for color context

    #roots = fp.FindFixedPoints(model, static_inpts, embedding='pca', embedder=model._pca, Verbose=False)
    print('Static Inputs \n\n', static_inpts)    # print to verify correct
    model_roots.FindFixedPoints(static_inpts)
    #context1_roots = fp.FindFixedPoints(model, context1_inpts, embedding='pca', embedder=model._pca, Verbose=False)


    plt.figure(100)
    plt.title("PCA of Fixed Points For Contextual Integration Task")
    
    
    model_roots.plot(fixed_pts=True, slow_pts=True, end_time = 100)
    plt.title("Early")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 100, end_time = 300)
    plt.title("Mid")
    model_roots.plot(fixed_pts=True, slow_pts=True, start_time = 400)
    plt.title("Late")
    
    model_roots.plot(fixed_pts=True, slow_pts=False)


    plt.figure(123)
    plt.title('Evaluation of Model on Multisensory Task')
    
    plt.figure()
    r.TestTaskInputs(model)     

    model_roots.save(model_choice)   

    plt.show()