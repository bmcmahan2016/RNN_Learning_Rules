# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:31:36 2020

@author: bmcma

This script trains RNN models using a specified learning rule
"""
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
   "taskVar" : 0.5
   }

def TrainRDM(name, hyperParams):
    if name.lower()[:2] == "bp":
        raise NotImplementedError()
    if name.lower()[:2] == "he":
        raise NotImplementedError()
    if name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams)
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
        
    # trains the  network according to learning rule specified by name    
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()

if __name__ == '__main__':
    
    # train an ensemble
    TrainRDM("GA_001", hyperParams)