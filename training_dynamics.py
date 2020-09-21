# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:18:46 2020

This will train a BPTT model and halt training at intermediate
points to assess how topology of fixed points evolve with training
progress

@author: bmcma
"""
import rnn as r
from bptt import Bptt

hyperParams = {                  # dictionary of all RNN hyper-parameters
   "inputSize" : 1,
   "hiddenSize" : 50,
   "outputSize" : 1,
   "g" : 1 ,
   "inputVariance" : 0.5,
   "outputVariance" : 0.5,
   "biasScale" : 0,
   "initScale" : 0.3,
   "dt" : 0.1,
   "batchSize" : 500,
   "taskMean" : 0.1857,
   "taskVar" : 1
   }

# initialize a BPTT model
initial_rnn_model = Bptt(hyperParams)
initial_rnn_model.save()

# loop over stopping conditions
stopping_conditions = [0.5, 0.65, 0.8, 0.9]
for termination_accuracy in stopping_conditions:
    # train a BPTT model to current stopping condition
    rnn_model = r.loadRNN(fName, optimizer="BPTT")
    
    
    # run the FP analysis
    # save this intermediate model
    