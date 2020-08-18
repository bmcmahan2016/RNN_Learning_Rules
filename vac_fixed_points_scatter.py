# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:23:07 2020

@author: bmcma
"""

import numpy as np
import matplotlib.pyplot as plt
import com
import FP_Analysis as fp
from rnn import loadRNN
from scipy import stats

input_values = [[1],[.9],[.8],[.7],[.6],[.5],[.4],[.3],[.2],[.1],[0],\
                    [-.1],[-.2],[-.3],[-.4],[-.5],[-.6],[-.7],[-.8],[-.9],[-1]]
input_values = 0.1 * np.array(input_values)


# MODEL_ARRAY = [["models/GA_041", "models/GA_042", "models/GA_043", "models/GA_044", "models/GA_045", "models/GA_046", "models/GA_047", "models/GA_048", "models/GA_049", "models/GA_050", "models/GA_051", "models/GA_052", "models/GA_053", "models/GA_054", "models/GA_055", "models/GA_056", "models/GA_057", "models/GA_058", "models/GA_059" ],\
#                 ["models/GA_061", "models/GA_062", "models/GA_063", "models/GA_064", "models/GA_065", "models/GA_066", "models/GA_067", "models/GA_068", "models/GA_069", "models/GA_070", "models/GA_071", "models/GA_072", "models/GA_073", "models/GA_074", "models/GA_075", "models/GA_076", "models/GA_077", "models/GA_078", "models/GA_079"],\
#                 ["models/GA_081", "models/GA_082", "models/GA_083", "models/GA_084", "models/GA_085", "models/GA_086", "models/GA_087", "models/GA_088", "models/GA_089"]]    

    
#MODEL_ARRAY = [["models/bptt_061", "models/bptt_062", "models/bptt_063", "models/bptt_064", "models/bptt_065", "models/bptt_066", "models/bptt_067", "models/bptt_068", "models/bptt_069", "models/bptt_070", "models/bptt_071", "models/bptt_072", "models/bptt_073", "models/bptt_074", "models/bptt_075", "models/bptt_076", "models/bptt_077", "models/bptt_078", "models/bptt_079"],\
#            ["models/bptt_081", "models/bptt_082", "models/bptt_083", "models/bptt_084", "models/bptt_085", "models/bptt_086", "models/bptt_087", "models/bptt_088", "models/bptt_089", "models/bptt_089"]]    


MODEL_ARRAY=[["models/FullForce080", "models/FullForce081", "models/FullForce082", "models/FullForce083", "models/FullForce084", "models/FullForce085", "models/FullForce086"]]

#MODEL_ARRAY = [["models/ga_040", "models/ga_041", "models/ga_042", "models/ga_043", "models/ga_44"],["models/ga_080", "models/ga_081", "models/ga_082", "models/ga_083", "models/ga_84"]]
coherenceVals = np.array([0, 0.02, 0.03, 0.04, 0.05])
X_data = []
y_data = []


for MODELS in MODEL_ARRAY:
    x = []       # holds fraction of inputs for these models
    y = []       # holds percentage COM corrective for these models
    for MODEL_NAME in MODELS:
        model = loadRNN(MODEL_NAME)
        if (model==False):  # model does not exist
            continue
            print("model not loaded:", MODEL_NAME)
        F = model.GetF()
        x.append(fp.FindFixedPoints(F, input_values, embedding='', embedder=model._pca, Verbose=False, just_get_fraction=True))

        
        # get the fraction of trials wtih vacillations
        yTmp = []
        for _, coherence in enumerate(coherenceVals):
            taskData = com.generateTaskData(model._task, coherence)
            # get the decisions for COM trials
            rnnOutput = model.feed(taskData)
            rnnOutput = rnnOutput.detach().cpu().numpy()
            rnnOutput = rnnOutput.T
                
            yTmp.append(len(com.detectVacillationTrials(rnnOutput))/5_000)
        y.append(np.mean(np.array(yTmp)))
        #########################################################
        


    x = np.array(x)
    y = np.array(y)
    X_data.append(x)
    y_data.append(y)
    plt.scatter(x, y)      # plots current model ensemble

#plt.legend(["0.5", "0.75", "1.0"])
plt.ylim([0,1])
plt.xlim([0,1])

X_data = np.concatenate(np.array(X_data)).ravel()
y_data = np.concatenate(np.array(y_data)).ravel()
slope, intercept, r_value, p_value, std_err = stats.linregress(X_data, y_data)

plt.text(0.4, 0.6, "r: " + str(r_value))
plt.text(0.4, 0.5, "p: " + str(p_value))
plt.show()