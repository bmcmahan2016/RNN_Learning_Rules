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
from task.Ncontext import Ncontext


def rdm_fixed_points(modelPath, inputs, save_fp=False):
    assert (inputs == 'large' or inputs == 'small'), "Must use either small or large inputs"
    model = loadRNN(modelPath)
    #AnalyzeLesioned(model, modelPath, xmin, xmax, ymin, ymax)
    
    inpts = {}  # two sets of inputs for solving fixed points
    inpts['large'] = np.array ( [[0.5], [0.2], [0], [-0.2], [-0.5]] )
    inpts['small'] = 0.03 * np.array( [[0], [0.1], [-0.1], [0.2], [-0.2], [0.3], [-0.3], [0.4], [-0.4], [0.5], 
                                [-0.5], [0.6], [-0.6], [0.7], [-0.7], [0.8], 
                                [-0.8], [0.9], [-0.9], [1.0], [-1.0]] )
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
    model = loadRNN(modelPath)
    model._task = Ncontext(device="cpu", dim=2)

    # construct inputs
    if inputs=='small':
        fixed_point_resolution = 21
        tmp = np.linspace(-0.02, 0.02, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
                                 # go signal for color context
    elif inputs == 'large':
        fixed_point_resolution = 5
        tmp = np.linspace(-0.4, 0.4, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
       

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

def N_fixed_points(modelPath, inputs, save_fp=False):
    
    model = loadRNN(modelPath)
    n_contexts = int(model._inputSize / 2)
    model._task = Ncontext(device="cpu", dim=n_contexts)
    # construct inputs
    if inputs=='small':
        fixed_point_resolution = 21
        static_inpts = np.zeros((n_contexts*fixed_point_resolution, n_contexts*2))
        for context_count in range(n_contexts):
            tmp = np.linspace(-0.02, 0.02, fixed_point_resolution)    
            tmp = tmp[np.argsort(np.abs(tmp))]
            static_inpts[context_count::n_contexts, context_count + n_contexts] = 1  # skip n_context rows at a time
            static_inpts[context_count::n_contexts, context_count] = tmp
                                 # go signal for color context
    elif inputs == 'large':
        fixed_point_resolution = 5
        tmp = np.linspace(-0.4, 0.4, fixed_point_resolution)    
        tmp = tmp[np.argsort(np.abs(tmp))]
        static_inpts = np.zeros((2*fixed_point_resolution, 4))
        static_inpts[0::2, 2] = 1
        static_inpts[0::2, 0] = tmp
        static_inpts[1::2, 3] = 1
        static_inpts[1::2, 1] = tmp
       

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