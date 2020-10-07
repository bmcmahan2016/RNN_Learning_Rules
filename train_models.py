# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:31:36 2020

@author: bmcma

This script trains RNN models using a specified learning rule
"""
from genetic import Genetic
from bptt import Bptt
import argparse

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

def TrainRDM(name, hyperParams, use_ReLU=False):
    print(name.lower()[:2])
    if name.lower()[:2] == "bp":
        rnnModel = Bptt(hyperParams)
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams)
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    if use_ReLU:
        rnnModel._use_ReLU = True
        
    # trains the  network according to learning rule specified by name    
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()
    
def TrainContext(name, hyperParams):
    if name.lower()[:2] == "bp":
        print("begining training BPTT network")
        rnnModel = Bptt(hyperParams, task="context")
    elif name.lower()[:2] == "he":
        raise NotImplementedError()
    elif name.lower()[:2] == "ga" or name.lower()[:2] == "ge":      # use genetic learning rule
        rnnModel = Genetic(hyperParams, task="context", mutation=0.005, numPop=20)
    else:
        print("unclear which learning rule should be used for training")
        raise NotImplementedError()
    rnnModel.setName(name)
    rnnModel.train()
    rnnModel.save()


parser = argparse.ArgumentParser(description="Trains RNN models")
parser.add_argument("model_name", help="name of model")
parser.add_argument("-m", "--mean", type=float, help="input mean")
parser.add_argument("-v", "--variance", type=float, help="input variance")
task_choice = parser.add_mutually_exclusive_group()
task_choice.add_argument("--rdm", action="store_true", default=False)
task_choice.add_argument("--context", action="store_true", default=False)

args = parser.parse_args()
if args.rdm:
    TrainRDM(args.model_name, hyperParams)
elif args.context:
    pass
else:
    raise NotImplementedError()
    
    
    # hyperParams["taskVar"] = 0.5        # train an ensemble of GA models w/ var = 0.75
    # TrainRDM("GA_041", hyperParams)
    # TrainRDM("GA_042", hyperParams)
    # TrainRDM("GA_043", hyperParams)
    # TrainRDM("GA_044", hyperParams)
    # TrainRDM("GA_045", hyperParams)
    # TrainRDM("GA_046", hyperParams)
    # TrainRDM("GA_047", hyperParams)
    # TrainRDM("GA_048", hyperParams)
    # TrainRDM("GA_049", hyperParams)
    # TrainRDM("GA_050", hyperParams)
    # TrainRDM("GA_051", hyperParams)
    # TrainRDM("GA_052", hyperParams)
    # TrainRDM("GA_053", hyperParams)
    # TrainRDM("GA_054", hyperParams)
    # TrainRDM("GA_055", hyperParams)
    # TrainRDM("GA_056", hyperParams)
    # TrainRDM("GA_057", hyperParams)
    # TrainRDM("GA_058", hyperParams)
    # TrainRDM("GA_059", hyperParams)
    # TrainRDM("GA_060", hyperParams)
    
    
    
    # hyperParams["taskVar"] = 0.75        # train an ensemble of GA models w/ var = 0.75
    # TrainRDM("GA_061", hyperParams)
    # TrainRDM("GA_062", hyperParams)
    # TrainRDM("GA_063", hyperParams)
    # TrainRDM("GA_064", hyperParams)
    # TrainRDM("GA_065", hyperParams)
    # TrainRDM("GA_066", hyperParams)
    # TrainRDM("GA_067", hyperParams)
    # TrainRDM("GA_068", hyperParams)
    # TrainRDM("GA_069", hyperParams)
    # TrainRDM("GA_070", hyperParams)
    # TrainRDM("GA_071", hyperParams)
    # TrainRDM("GA_072", hyperParams)
    # TrainRDM("GA_073", hyperParams)
    # TrainRDM("GA_074", hyperParams)
    # TrainRDM("GA_075", hyperParams)
    # TrainRDM("GA_076", hyperParams)
    # TrainRDM("GA_077", hyperParams)
    # TrainRDM("GA_078", hyperParams)
    # TrainRDM("GA_079", hyperParams)
    # TrainRDM("GA_080", hyperParams)
    
    # hyperParams["taskVar"] = 0.75        # train an ensemble of bptt models w/ var = 0.75
    # TrainRDM("bptt_061", hyperParams)
    # TrainRDM("bptt_062", hyperParams)
    # TrainRDM("bptt_063", hyperParams)
    # TrainRDM("bptt_064", hyperParams)
    # TrainRDM("bptt_065", hyperParams)
    # TrainRDM("bptt_066", hyperParams)
    # TrainRDM("bptt_067", hyperParams)
    # TrainRDM("bptt_068", hyperParams)
    # TrainRDM("bptt_069", hyperParams)
    # TrainRDM("bptt_070", hyperParams)
    # TrainRDM("bptt_071", hyperParams)
    # TrainRDM("bptt_072", hyperParams)
    # TrainRDM("bptt_073", hyperParams)
    # TrainRDM("bptt_074", hyperParams)
    # TrainRDM("bptt_075", hyperParams)
    # TrainRDM("bptt_076", hyperParams)
    # TrainRDM("bptt_077", hyperParams)
    # TrainRDM("bptt_078", hyperParams)
    # TrainRDM("bptt_079", hyperParams)
    # TrainRDM("bptt_080", hyperParams)
    
    
    # hyperParams["taskVar"] = 1.0         # train an ensemble of GA models w/ var = 1.0
    # # TrainRDM("GA_081", hyperParams)
    # # TrainRDM("GA_082", hyperParams)
    # # TrainRDM("GA_083", hyperParams)
    # # TrainRDM("GA_084", hyperParams)
    # # TrainRDM("GA_085", hyperParams)
    # # TrainRDM("GA_086", hyperParams)
    # # TrainRDM("GA_087", hyperParams)
    # # TrainRDM("GA_088", hyperParams)
    # # TrainRDM("GA_089", hyperParams)
    # # TrainRDM("GA_090", hyperParams)
    # TrainRDM("GA_091", hyperParams)
    # TrainRDM("GA_092", hyperParams)
    # TrainRDM("GA_093", hyperParams)
    # TrainRDM("GA_094", hyperParams)
    # TrainRDM("GA_095", hyperParams)
    # TrainRDM("GA_096", hyperParams)
    # TrainRDM("GA_097", hyperParams)
    # TrainRDM("GA_098", hyperParams)
    # TrainRDM("GA_099", hyperParams)
    # # TrainRDM("GA_100", hyperParams)
    # assert False
    
    
    
    
    #hyperParams["taskVar"] = 1         # train an ensemble of bptt models w/ var = 1.0
    #TrainRDM("bptt_081", hyperParams)
    #TrainRDM("bptt_082", hyperParams)
    #TrainRDM("bptt_083", hyperParams)
    #TrainRDM("bptt_084", hyperParams)
    #TrainRDM("bptt_085", hyperParams)
    #TrainRDM("bptt_086", hyperParams)
    #TrainRDM("bptt_087", hyperParams)
    #TrainRDM("bptt_088", hyperParams)
    #TrainRDM("bptt_089", hyperParams)
    #TrainRDM("bptt_090", hyperParams)
    #TrainRDM("bptt_091", hyperParams)     # only recurrent weights trained
    #TrainRDM("bptt_092", hyperParams)     # only recurrent weights trained
    #TrainRDM("bptt_093", hyperParams)
    #TrainRDM("bptt_094", hyperParams)
    #TrainRDM("bptt_095", hyperParams)
    #TrainRDM("bptt_096", hyperParams)
    #TrainRDM("bptt_097", hyperParams)
    #TrainRDM("bptt_098", hyperParams)
    #TrainRDM("bptt_099", hyperParams)
    # print("training bptt models")
    # TrainRDM("bptt_111", hyperParams)
    # TrainRDM("bptt_112", hyperParams)
    # TrainRDM("bptt_113", hyperParams)
    # TrainRDM("bptt_114", hyperParams)
    # TrainRDM("bptt_115", hyperParams)
    # TrainRDM("bptt_116", hyperParams)
    # TrainRDM("bptt_117", hyperParams)
    # TrainRDM("bptt_118", hyperParams)
    # TrainRDM("bptt_119", hyperParams)
    # TrainRDM("bptt_120", hyperParams)
    # TrainRDM("bptt_121", hyperParams)
    # print("Done training BPTT models")
    # assert False
    # hyperParams["taskVar"] = 1
    # TrainRDM("ga_402", hyperParams, use_ReLU=True)
    # assert False
    
    #hyperParams["inputSize"] = 4
    # hyperParams["hiddenSize"] = 50
    
    # hyperParams["mean"] = 0.1857
    # hyperParams["taskVar"] = 1
    # print("Training on context task")
    # TrainContext("ga_1005", hyperParams)
    #TrainContext("ga_101", hyperParams)
    #TrainContext("ga_102", hyperParams)
    #TrainContext("ga_103", hyperParams)
    # TrainContext("ga_104", hyperParams)
    
  