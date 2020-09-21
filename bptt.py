# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:57:23 2020

@author: bmcma
"""


import numpy as np
import torch
from rnn import RNN
import torch.nn as nn
import matplotlib.pyplot as plt
import facilities as fac
from torch.autograd import Variable


class Bptt(RNN):
    '''create a trainer object that will be used to trian an RNN object'''
    def __init__(self, hyperParams, task="rdm"):
        super(Bptt, self).__init__(hyperParams, task=task)
        '''
        description of parameters:

        '''

        self._num_epochs = 20000
        self._learning_rate = 1e-3
        self._hParams["learning_rate"] = self._learning_rate
        
        # cast as PyTorch variables
        #self._J['in'] = Variable(self._J['in'], requires_grad=True)
        self._J['rec'] = Variable(self._J['rec'], requires_grad=True)
        #self._J['out'] = Variable(self._J['out'], requires_grad=True)
        
        #self._params = [self._J['in'], self._J['rec'], self._J['out']]
        self._params = [self._J['rec']]
        self._optimizer = torch.optim.Adam(self._params, lr=self._learning_rate)


        
        self.all_losses = []
        
        self._hidden = Variable(self._hidden, requires_grad=True)


    # hack for now, fix by casting to numpy and then back to Torch perhaps
    # TODO: fix this
    def my_loss(self, mu, output):
        np_loss = self._task.Loss(output, mu)  # here mu is condition
        # torch_loss = Variable(torch.from_numpy(np.array(np_loss)))
        #print("np_loss", np_loss)
        return np_loss  # torch.log(1 + torch.exp(-mu * yt))

    def trainBPTT(self, input, trial, condition):
        #create an activities tensor for the rnn_model
        self._activityTensor = np.zeros((self._num_epochs, int(self._task.N/10), self._hiddenSize))
        self._optimizer.zero_grad()
        self.StoreRecMag()

        output = torch.zeros((self._task.N, 500*self._outputSize))      #i am lazy and this is a hack
        output_temp = torch.Tensor([0])

        trial_length = self._task.N
        for i in range(trial_length): 
            inputNow = input[:,i,:].t()
            output_temp, hidden = self._forward(inputNow)           #I need to generalize this line to work for context task
            #output_temp, hidden = self.rnn_model.forward(input[:,i], hidden, dt)             #this incridebly hacky must improve data formatting accross all modules to correctly implement a context task that doesn't clash with DM task
            output[i] = np.squeeze(output_temp)
            if (i %10 == 0):
                activityIX = int(i/10)
                self._activityTensor[self.trial_count, activityIX, :] = np.squeeze(torch.tanh(self._hidden).cpu().detach().numpy())[:,0]
        self.trial_count += 1
        # self.activity_tensor[trial, i, :] = hidden.detach().numpy()  # make sure calling detach does not mess with the backprop gradients (I think .data does)
        # https://pytorch.org/docs/stable/autograd.html
        loss = self.my_loss(condition, output)
        loss.backward()
        self._optimizer.step()
        
        return output, loss.item()
    
    def getBatch(self):
        x_batch = torch.zeros((self._batchSize, self._task.N, self._inputSize))
        #x_batch = torch.zeros((750, self._inputSize, self._batchSize))
        y_batch = torch.zeros(self._batchSize)
        for dataPtIX in range(self._batchSize):
            inpt, condition = self._task.GetInput()
            x_batch[dataPtIX,:,:] = inpt
            y_batch[dataPtIX] = condition
        return x_batch, y_batch

    def train(self, termination_accuracy=0.9):
        self._startTimer()
        # create a validation dataset for the model
        self.createValidationSet()
        
        # inps_save = np.zeros((num_epochs, trial_length))
        self.trial_count=0
        self.targets = []   #will hold target output for each trial
        
        validation_accuracy = 0.0
        validation_acc_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        meanLossHist = 100*np.ones(20)
        # empty list that will hold history of validation accuracy
        loss_hist = []
        
        # for CUDA implementation
        inpt = Variable(torch.zeros(int(self._batchSize), self._task.N, self._inputSize).cuda())
        
        #trial = 0
        loss = np.inf
        
        # start main training loop
        while(validation_accuracy < termination_accuracy):
            self.SaveWeights()
            if self.trial_count%10 == 0:
                print('trial #:', self.trial_count)
                print('validation accuracy', validation_accuracy)
                print("validation history", validation_acc_hist)
            #print('loss', loss)\
            
            
            # get a current batch of input
            inpt[:], condition = self.getBatch()    # inpt has shape 750x1
            self.targets.append(condition[-1].item())
            condition = Variable(condition)
            
            # train model on this data    
            self._init_hidden()
            output, loss = self.trainBPTT(inpt, self.trial_count, condition)
            
            # append current loss to history
            self.all_losses.append(loss)

            validation_accuracy_curr = self.GetValidationAccuracy()
            loss_hist.append(1-validation_accuracy_curr)
            #print('loss hist', np.mean(np.diff(meanLossHist)))
            validation_acc_hist[:9] = validation_acc_hist[1:]
            validation_acc_hist[-1] = validation_accuracy_curr
            meanLossHist[:19] = meanLossHist[1:]
            meanLossHist[-1] = np.mean(self.all_losses[-20:])
            validation_accuracy = np.min(validation_acc_hist)
                 
            
            # save the model every 100 trials
            if self.trial_count %100 == 0:
                self.saveProgress()
            
        self._targets = np.array(self.targets)        #hacky
        self._losses = np.array(self.all_losses)#self.all_losses      #also hacky
        self._activityTensor = self._activityTensor[:self.trial_count,:,:]
        print('shape of activity tensor', self._activityTensor.shape)  
        print('trial count', self.trial_count)
        self._endTimer()


    # store stuff
    # TODO: make a folder if the existing folder does not exist

    def saveProgress(self):
        self._targets = self.targets        #hacky
        self._losses = np.array(self.all_losses)#self.all_losses      #also hacky
        #self.rnn_model.activity_tensor = self.rnn_model.activity_tensor[:self.trial_count,:,:]
        self.save()
        print('model back-ed up')



if __name__ == '__main__':
    
    # sets the appropriate system path
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    
    import numpy as np
    from rnn import RNN
    from task.williams import Williams
    import utils
    import matplotlib.pyplot as plt
    import time
    #from rnntools import plotTCs
    #from FP_Analysis import FindZeros2
    #from rnntools import plotMultiUnit, plotPSTH, plotWeights, plotTCs

    hyperParams = {       # dictionary of all hyper-parameters
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
    "taskVar" : 0.75
    }
    rnn_inst = Bptt(hyperParams)
    
    print("\nsuccesfully constructed bptt rnn object!")
    
    # rnn_inst.createValidationSet()
    # print("Validation accuracy:", rnn_inst.GetValidationAccuracy())
    
    rnn_inst.train()
    rnn_inst.save()
    print("rnn model succesfully saved!")
    
