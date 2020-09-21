# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:18:46 2020

This will train a BPTT model and halt training at intermediate
points to assess how topology of fixed points evolve with training
progress

bptt_model_200 is the initial model before any training
bptt_model_201 is after a validation accuracy of 50% is reached
bptt_model_202 is after a validation accuracy of 65% is reached
bptt_model_203 is after a validation accuracy of 80% is reached
bptt_model_204 is after a validation accuracy of 90% is reached

@author: Brandon McMahan
"""
import rnn as r
from bptt import Bptt
from genetic import Genetic

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

# initialize a BPTT model and save it
initial_rnn_model = Genetic(hyperParams)
initial_rnn_model._MODEL_NAME = "models/ga_model_200"
initial_rnn_model.save()
fName = "models/ga_model_200"

# loop over stopping conditions
stopping_conditions = [0.5, 0.65, 0.8, 0.9]
for model_num, termination_accuracy in enumerate(stopping_conditions):
    # train a BPTT model to current stopping condition
    print("training model to validation accuracy of", termination_accuracy)
    rnn_model = r.loadRNN(fName, optimizer="GA")
    if (rnn_model == False):
        print("load failed!")
        assert False
    fName = "models/ga_model_20" + str(1+model_num)   # name for next model
    rnn_model.train(termination_accuracy=termination_accuracy)
    
    # TODO: run the FP analysis
    
    # saves the intermediate model
    rnn_model._MODEL_NAME = fName
    rnn_model.save()
    