# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:59:23 2021

@author: bmcma
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np 
import torch 
from rnn import RNN, loadRNN, loadHyperParams
from task.williams import Williams
from task.multi_sensory import multi_sensory
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from task.context import context_task
from task.Ncontext import Ncontext
import argparse
import pdb
import os
paths = ["models/analysis/fig1/cdi_models_BPTT_pschometrics.npy", "models/analysis/fig1/cdi_models_GA_pschometrics.npy",
         "models/analysis/fig1/cdi_models_HEB_pschometrics.npy", "models/analysis/fig1/cdi_models_FF_pschometrics.npy"]
color_ix = 0
for data_path in paths:
    psycho_data = np.load(data_path)
    psycho_statistics = np.zeros((2, 2, 8))
    psycho_statistics[0] = np.mean(psycho_data, 1)
    psycho_statistics[1] = np.std(psycho_data, 1)
    coherence_vals = 2*np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])
    
    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return (y)
    def linear(x, m, b):
        y = m*x + b
        return (y)
    
    
    def fit_sigmoid(coherence_vals, reach_data, p0):
        '''fits a sigmoid curve to psychometric data'''
        popt, pcov = curve_fit(sigmoid, coherence_vals, np.squeeze(reach_data), p0, method='dogbox')
        data_fit = sigmoid(np.linspace(min(coherence_vals), max(coherence_vals), 100), *popt)
        return data_fit
    
    def fit_linear(coherence_vals, reach_data, p0):
        '''fits a linear curve to psychometric data'''
        p0 = [max(reach_data), np.median(coherence_vals),1,min(reach_data)] 
        popt, pcov = curve_fit(linear, coherence_vals, np.squeeze(reach_data), method='dogbox')
        data_fit = linear(np.linspace(min(coherence_vals), max(coherence_vals), 100), *popt)
        return data_fit
    
    
    def Plot(coherence_vals, reach_data, marker="x", ymin=-.1, ymax=1.1, fit='sigmoid', newFig=True, cs='b'):
        '''Plots psychometric data'''
    
        if newFig:     # generate a new figure
            fig_object = plt.figure()   
            axis_object = fig_object.add_subplot(1,1,1)
            axis_object.spines["left"].set_position("center")
            axis_object.spines["bottom"].set_position("center")
            axis_object.spines["right"].set_color("none")
            axis_object.spines["top"].set_color("none")
            axis_object.xaxis.set_label("top")
    
        # plots RNN data
        plt.scatter(coherence_vals, reach_data, marker=marker, c=cs, s=100)
    
        p0 = [max(reach_data), np.median(coherence_vals),1,min(reach_data)] 
        if fit=="sigmoid":
            data_fit = fit_sigmoid(coherence_vals, reach_data, p0)
        elif fit == "linear":
            data_fit = fit_linear(coherence_vals, reach_data, p0)
        else:
            raise NotImplementedError()
    
        plt.plot(np.linspace(min(coherence_vals), max(coherence_vals), 100), data_fit, alpha=.5, c=cs)
        
        # adds plot lables
        plt.title("Psychometrics")
        plt.ylim([ymin, ymax])
    
    
    
    markers = ["o", "x"] 
    fit_type = ["sigmoid", "linear"]
    colors=['r', 'g', 'b', 'y']
    use_new_fig = [1, 0]
    for N in range(2):
        Plot(coherence_vals, psycho_statistics[0,N], marker=markers[N>=1], fit=fit_type[N>=1], newFig=(0), cs=colors[color_ix])    # plot in-context psychometric data
    color_ix += 1

plt.legend(['BPTT', 'BPTT', 'GA', 'GA', 'HEB', 'HEB', 'FF', 'FF'])
