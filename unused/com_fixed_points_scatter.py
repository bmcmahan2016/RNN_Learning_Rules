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

input_values = [[1],[.9],[.8],[.7],[.6],[.5],[.4],[.3],[.2],[.1],[0],\
                    [-.1],[-.2],[-.3],[-.4],[-.5],[-.6],[-.7],[-.8],[-.9],[-1]]
input_values = 0.05 * np.array(input_values)


MODEL_ARRAY = [["models/GA_041", "models/GA_042", "models/GA_043", "models/GA_044", "models/GA_045", "models/GA_046", "models/GA_047", "models/GA_048", "models/GA_049", "models/GA_050", "models/GA_051", "models/GA_052", "models/GA_053", "models/GA_054", "models/GA_055", "models/GA_056", "models/GA_057", "models/GA_058", "models/GA_059" ],\
               ["models/GA_061", "models/GA_062", "models/GA_063", "models/GA_064", "models/GA_065", "models/GA_066", "models/GA_067", "models/GA_068", "models/GA_069", "models/GA_070", "models/GA_071", "models/GA_072", "models/GA_073", "models/GA_074", "models/GA_075", "models/GA_076", "models/GA_077", "models/GA_078", "models/GA_079"],\
               ["models/GA_081", "models/GA_082", "models/GA_083", "models/GA_084", "models/GA_085", "models/GA_086", "models/GA_087", "models/GA_088", "models/GA_089"]]    


for MODELS in MODEL_ARRAY:
    x = []       # holds fraction of inputs for these models
    y = []       # holds percentage COM corrective for these models
    for MODEL_NAME in MODELS:
        model = loadRNN(MODEL_NAME)
        if (model==False):
            continue
            print("model not loaded:", MODEL_NAME)
        F = model.GetF()
        y.append(np.mean(com.fractionCorrective(model)))
        x.append(fp.FindFixedPoints(F, input_values, embedding='', embedder=model._pca, Verbose=False, just_get_fraction=True))

    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)      # plots current model ensemble

plt.legend(["0.5", "0.75", "1.0"])
plt.ylim([0,1])
plt.xlim([0,1])