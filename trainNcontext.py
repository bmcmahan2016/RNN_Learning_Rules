# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:08:11 2020

@author: bmcma
"""

import time
import FF_Demo
from task.Ncontext import Ncontext
import pickle          # will be used to save model parameters

var = 0.5
task = Ncontext(var=var, device="cpu")   # FULL FORCE doesn't use cuda
inps_and_targs = task.get_inps_and_targs

# create the network and set hyper-parameters
p = FF_Demo.create_parameters(dt=0.003)
p['g'] = 1.0
p['network_size'] = 50
p['tau'] = 0.03
p['test_init_trials']=10
p['test_trials'] = 2_000
p['ff_alpha'] = 10_000
p['ff_steps_per_update']=2
rnn = FF_Demo.RNN(p,6,1)   # hyper-params, num_inputs, num_outputs

# train the current model
num_train_attempts = 0
training_exit_status = False
while (not training_exit_status):
    num_train_attempts += 1
    training_exit_status = rnn.train_rdm(inps_and_targs, errorTrigger=200)



# create a dictionary containing the desired model data to be saved
model_data = {
        'input_weights': rnn.rnn_par['inp_weights'],
        'rec_weights' : rnn.rnn_par['rec_weights'],
        'output_weights' : rnn.rnn_par['out_weights'],
        'bias' : rnn.rnn_par['bias'],
        'activity_tensor' : rnn.activity_tensor,
        'trgets' : rnn.activity_targets,
        'losses' : rnn.losses,
        'w_rec_hist' : rnn.Wrec_hist,
        'w_i_hist' : rnn.Win_hist,
        'valHist' : rnn.valHist
        }
   

# converts to RNN object
p["inputSize"] = 6
p["hiddenSize"] = 50
p["outputSize"] = 1
p["inputVariance"] = 0.5
p["outputVariance"] = 0.5
p["biasScale"] = 0
p["initScale"] = 0.3
p["dt"] = 0.1
p["batchSize"] = 1
p["taskVar"] = var
p["taskMean"] = 0.1857
p["num_train_attempts"] = num_train_attempts

import sys
sys.path.insert(0,"C:/Users/bmcma/KaoLab/code/RNN_Learning_Rules")

from rnn import RNN
import torch


my_rnn = RNN(p)    # constructs RNN object
modelName = "FullForce_N_" + str(time.time())
# convert the model

	
inp_weights = rnn.rnn_par['inp_weights'].T
rec_weights = rnn.rnn_par['rec_weights'].T
out_weights = rnn.rnn_par['out_weights'].T
bias = rnn.rnn_par['bias']
bias2 = torch.from_numpy(bias).float()
my_rnn._activityTensor = rnn.activity_tensor
my_rnn.AssignWeights(inp_weights, rec_weights, out_weights)
my_rnn._J['bias'][:] = bias2.t()[:]
my_rnn._targets = rnn.activity_targets
my_rnn.losses = rnn.losses

	
model_name =  modelName
my_rnn._MODEL_NAME = model_name
my_rnn.save()

import shutil
#model_name += ".pt"
shutil.move(model_name+".pt", "C:/Users/bmcma/KaoLab/code/RNN_Learning_Rules/models/"+model_name+".pt")
shutil.move(model_name+".txt", "C:/Users/bmcma/KaoLab/code/RNN_Learning_Rules/models/"+model_name+".txt")
sys.path.remove("C:/Users/bmcma/KaoLab/code/RNN_Learning_Rules")
