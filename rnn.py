# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:26:23 2020

@author: bmcma

RNN code refactor

This is the base class from which the RNN training classes will be derived
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import time
from FP_Analysis import FindFixedPoints
from task.williams import Williams
import os


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
    }

class RNN(nn.Module):
    # recently changed var by normalizing it by N, before was 0.045 w/o normilazation
    def __init__(self, hyperParams):

        super(RNN, self).__init__()                                            # initialize parent class
        self._inputSize = int(hyperParams["inputSize"])
        self._hiddenSize = int(hyperParams["hiddenSize"])
        self._outputSize = int(hyperParams["outputSize"])
        self._g = hyperParams["g"]
        self._hiddenInitScale = hyperParams["initScale"]                       # sets tolerance for determining validation accuracy# initializes hidden layer                                                   # sets noise for hidden initialization
        self._dt= hyperParams["dt"]
        self._batchSize = hyperParams["batchSize"]
        self._hParams = hyperParams                                             # used for saving training conditions
        self._init_hidden()   
        self._totalTrainTime = 0                                               # accumulates training time
        self._timerStarted = False
        self._useForce = False            # if set to true this slightly changes the forward pass 
        self._fixedPoints = []
        
        self._task = Williams(N=750, mean=hyperParams["taskMean"], \
                              variance=hyperParams["taskVar"])                

        #create an activity tensor that will hold a history of all hidden states
        self._activityTensor = np.zeros((50))
        self._neuronIX = None    #will hold neuron sorting from TCA
        self._targets=[]
        self._losses = 0     #trainer should update this with a list of training losses
        self._MODEL_NAME = 'models/UNSPECIFIED MODEL'   #trainer should update this
        self._pca = []
        self._recMagnitude = []     #will hold the magnitude of reccurent connections at each step of training
        # will hold the previous weight values
        self._w_hist = []    # will hold the previous weight values

        self._valHist = []        # empty list to hold history of validation accuracies

        #self.fractions = []
        #weight matrices for the RNN are initialized with random normal
        self._J = {
        'in' : (torch.randn(self._hiddenSize, self._inputSize).cuda())*(1/2),
        'rec' : ((self._g**2)/50)*torch.randn(self._hiddenSize, self._hiddenSize).cuda(),
        'out' : (0.1*torch.randn(self._outputSize, self._hiddenSize).cuda()),
        'bias' : torch.zeros(self._hiddenSize, 1).cuda()*(1/2)
        }
        
    def _startTimer(self):
        '''
        starts timer for training purposes

        Returns
        -------
        None.

        '''
        self._tStart = time.time()
        self._timerStarted = True
    
    def _endTimer(self):
        '''
        stops training timer

        Returns
        -------
        None.

        '''
        if self._timerStarted == False:
            return
        else:
            self._totalTrainTime += time.time() - self._tStart
            self._timerStarted = False
    
    def setName(self, name):
        self._MODEL_NAME = "models/" + name
        
    # function to store recurrent magnitudes to a list    
    def StoreRecMag(self):
        return self._recMagnitude.append( np.mean( np.abs(self._J['rec'].cpu().detach().numpy()) ) )
        #self.rec_magnitude.append( LA.norm( self.J['rec'].detach().numpy() ) )

    def AssignWeights(self, Jin, Jrec, Jout):
        '''
        AssignWeights will manually asign the network weights to values specified by
        Jin, Jrec, and Jout. Inputs may either be Numpy arrays or Torch tensors. The 
        model weights are maintained as Torch tensors.
        '''
        if torch.is_tensor(Jin):
            self._J['in'][:] = Jin[:]
        else:
            self._J['in'][:] = torch.from_numpy(Jin).float()[:]
        if torch.is_tensor(Jrec):
            self._J['rec'][:] = Jrec[:]
        else:
            self._J['rec'][:] = torch.from_numpy(Jrec).float()[:]
        if torch.is_tensor(Jout):
            self._J['out'][:] = Jout[:]
        else:
            self._J['out'][:] = torch.from_numpy(Jout).float()[:]
        self.StoreRecMag()
        
    def createValidationSet(self, test_iters=2_000):
        '''
        DESCRIPTION:
        Creates the validation dataset which will be used to 
        decide when to terminate RNN training. The validation 
        dataset consists of means sampled uniformly from -0.1875 
        to +0.1875. The variance of all instances in the validation 
        dataset is equal to the variance of the trainign dataset 
        as specified by the task object.
        
        PARAMETERS:
        **valSize: specifies the size of the validation dataset to use
        **task: task object that is used to create the validation data 
        set
        '''
        # initialize tensors to hold validation data
        self.validationData = torch.zeros(test_iters,self._task.N).cuda()
        self.validationTargets = torch.zeros(test_iters,1).cuda()
        # means for validation data
        meanValues = np.linspace(0, 0.1875, 20)
        for trial in range(test_iters):
            # to get genetic and bptt different I divided by 30
            mean_overide = meanValues[trial %20]
            inpt_tmp, condition_tmp = self._task.GetInput(mean_overide=mean_overide)
            self.validationData[trial,:] = inpt_tmp[:,0]
            self.validationTargets[trial] = condition_tmp
        print('Validation dataset created!\n')

    def GetValidationAccuracy(self, test_iters=2000):
        '''
        Will get validation accuracy of the model on the specified task to 
        determine if training should be terminated
        '''
        accuracy = 0.0
        tol = 1

        inpt_data = self.validationData.t()
        condition = self.validationTargets
        condition = torch.squeeze(condition)
        output = self.feed(inpt_data)
        output_final = torch.sign(output[-1,:])
        # compute the difference magnitudes between model output and target output
        error = torch.abs(condition-output_final)
        # threshold this so that errors greater than tol(=0.1) are counted
        num_errors = torch.where(error>tol)[0].shape[0]
        accuracy = (test_iters - num_errors) / test_iters

        
        self._valHist.append(accuracy)                                          # appends current validation accuracy to history
    
        
        return accuracy

    def _UpdateHidden(self, inpt, use_relu=False):
        '''
        INPUTS
        inpt: an 1xN array corresponding to N samples of data
        to be feed to the network at the current timestep
        OUTPUTS
        hidden_next: the new activations of the hidden units as a torch tensor
        with shape (hidden_size, N)
        '''
        dt = self._dt
        if use_relu:
            hidden_floor = torch.zeros(self._hidden.shape).cuda()
            hidden_next = dt*torch.matmul(self._J['in'], inpt) + \
            dt*torch.matmul(self._J['rec'], (torch.max(hidden_floor, self._hidden))) + \
            (1-dt)*self._hidden + dt*self._J['bias']
        else:   # add nosie term
            noiseTerm=0
            hidden_next = dt*torch.matmul(self._J['in'], inpt) + \
            dt*torch.matmul(self._J['rec'], (1+torch.tanh(self._hidden))) + \
            (1-dt)*self._hidden + dt*self._J['bias'] + 0*noiseTerm

        self._hidden = hidden_next        # updates hidden layer
        return hidden_next

    def _forward(self, inpt):
        '''
        computes the forward pass activations

        Args:
            inpt (torch tensor): this is a torch tensor that contains inputs to
            the network at a given timestep. This should have shape (1, batch_size). 
            inpt will be reshaped to (1,-1)
            hidden (torch tensor): activity of the hidden units. Should have shape
            (hidden_size, batch_size)
            
            dt (TYPE): DESCRIPTION.

        Returns:
            output (torch tensor): output activity of the network for current timestep.
            output has shape (1, batch_size).
            hidden_next (torch tensor): activities of all artificial neurons. Has
            shape (hidden_size, batch_size)

        '''

        #ensure the input is a torch tensor
        if torch.is_tensor(inpt) == False:
            inpt = torch.from_numpy(inpt).float()                              # inpt must have shape (1,1)
        inpt = inpt.reshape(1, -1)
        
        # compute the forward pass
        self._UpdateHidden(inpt)
        if self._useForce:
            output = torch.tanh(torch.matmul(self._J['out'], torch.tanh(self._hidden)))
        else:
            output = torch.matmul(self._J['out'], self._hidden)

        return output, self._hidden.clone()

    def feed(self, inpt_data, return_hidden=False):
        '''
        feed is a method that can be used for feeding input data 
        into an RNN. Feed initializes the hidden state and passes inputs into the 
        RNN updating the hidden layer and outputs.
        
        INPUTS
        inpt_data: NxM torch tensor where N is the number of time steps and M is
        the number of 
        OUTPUTS
        output_trace: output of the network over all timesteps. Will have shape 
        (time_steps, num_samples) i.e. 40x1 for single sample inputs
        '''
        num_inputs = len(inpt_data[0])
        output_trace = torch.zeros(inpt_data.shape[0], inpt_data.shape[1]).cuda()
        hidden_trace = []
        self._init_hidden(numInputs=num_inputs)                                # initializes hidden state
        for t_step in range(len(inpt_data)):
            output, hidden = self._forward(inpt_data[t_step])
            if return_hidden:
                hidden_trace.append(hidden.cpu().detach().numpy())
                
            output_trace[t_step,:] = output
        if return_hidden:
            return output_trace, np.array(hidden_trace)
        #print('shape of output trace', len(output_trace[0]))
        return output_trace
    
    def Train(self):      # pure virtual method
        raise NotImplementedError() 
    
    def SaveWeights(self):
        self._w_hist.append(self._J['rec'].cpu().detach().numpy())

    def save(self, N="", tElapsed=0, *kwargs):
        '''
        saves RNN parameters and attributes. User may define additional attributes
        to be saved through kwargs
        '''
        print('valdiation history', self._valHist)
        if N=="":     # no timestamp
            model_name = self._MODEL_NAME+'.pt'
        else:         # timestamp
            model_name = self.MODEL_NAME + '_' + str(N) + '_.pt'
        if tElapsed==0:
            torch.save({'weights': self._J, \
                        'weight_hist':self._w_hist, \
                        'activities': self._activityTensor, \
                        'targets': self._targets, \
                        'pca': self._pca, \
                        'losses': self._losses, \
                        'rec_magnitude' : self._recMagnitude, \
                        'neuron_idx': self._neuronIX,\
                        'validation_history' : self._valHist,
                        'fixed_points': self._fixedPoints}, model_name)
                
            # save model hyper-parameters to text file
            f = open(self._MODEL_NAME+".txt","w")
            for key in self._hParams:
                f.write( str(key)+" : "+str(self._hParams[key]) + '\n')
            f.write( "total training time: " + str(self._totalTrainTime) + '\n')
            f.close()
            
        else:
            torch.save({'weights': self.J, \
                        'weight_hist':self.w_hist, \
                        'activities': self.activity_tensor, \
                        'targets': self.targets, \
                        'pca': self.pca, \
                        'losses': self.losses, \
                        'rec_magnitude' : self.rec_magnitude, \
                        'neuron_idx': self.neuron_idx, \
                        'fractions' : self.fractions, \
                        'validation_history' : self.valHist, \
                        'tElapsed' : tElapsed,
                        'fixed_points' : self._fixedPoints}, model_name)
        
        #torch.save({'weights': self.J, 'targets': self.targets,  'losses': self.losses,'validation_history' : self.valHist}, model_name)

    def load(self, model_name, *kwargs):
        '''
        Loads in parameters and attributers from a previously instantiated model.
        User may define additional model attributes to load through kwargs
        '''
        # add file suffix to model_name
        fname = model_name+'.pt'
        model_dict = torch.load(fname)
        # load attributes in model dictionary
        if 'weights' in model_dict:
            self._J = model_dict['weights']
        else:
            print('WARNING!! NO WEIGHTS FOUND\n\n')
        if 'activities' in model_dict:
            self._activityTensor = model_dict['activities']
        else:
            print('WARNING!! NO ACTIVITIES FOUND\n\n')
        if 'targets' in model_dict:
            self._targets = model_dict['targets']
        else:
            print('WARNING!! NO TARGETS FOUND\n\n')
        if 'pca' in model_dict:
            self._pca = model_dict['pca']
        else:
            print('WARNING!! NO PCA DATA FOUND\n\n')
        if 'losses' in model_dict:
            self._losses = model_dict['losses']
        else:
            print('WARNING!! NO LOSS HISTORY FOUND\n\n')
        if 'validation_history' in model_dict:
            self._valHist = model_dict['validation_history']
        else:
            print('WARNING!! NO VALIDATION HISTORY FOUND\n\n')
            
        if 'fixed_points' in model_dict:
            self._fixedPoints = model_dict['fixed_points']
            
        # try to load additional attributes specified for kwargs
        for key in kwargs:
            print('loading of', key, 'has not yet been implemented!')

        
        
        
        
        if 'rec_magnitude' in model_dict:
            self.rec_magnitude = model_dict['rec_magnitude']
        else:
            print('WARNING!! NO WEIGHT HISTORY FOUND\n\n')
        if 'neuron_idx' in model_dict:
            self.neuron_idx = model_dict['neuron_idx']
        else:
            print('WARNING!! NO NEURON INDEX FOUND\n\n')

        print('\n\n')
        print('-'*50)
        print('-'*50)
        print('RNN model succesfully loaded ...\n\n')


    # maybe I should consider learning the initial state?
    def _init_hidden(self, numInputs=1):
        self._hidden = self._hiddenInitScale*(torch.randn(self._hiddenSize, numInputs).cuda())

    def LearningDynamics(self):
        raise NotImplementedError()                                            # this function is not being used anywhere
        F = self.GetF()
        frac = FindFixedPoints(F, [[1],[0.9],[0.8],[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1],\
                        [-0.1],[-0.2],[-0.3],[-0.4],[-0.5],[-0.6],[-0.7],[-0.8],[-0.9],[-1]], num_hidden=50, just_get_fraction=True)
        self.fractions.append(frac)
        print('frac', frac)
        self.SaveWeights()
    
    def VisualizeWeightClusters(self, neuron_sorting, p):
        plt.figure()
        cmap=matplotlib.cm.bwr
        weight_matrix=self._J['rec'].cpu().detach().numpy()
        weights_ordered = weight_matrix[:,neuron_sorting]
        weights_ordered = weights_ordered[neuron_sorting, :]
        #average four clusters
        C11 = np.mean( weights_ordered[:p, :p] )
        C12 = np.mean( weights_ordered[:p, p:] )
        C21 = np.mean( weights_ordered[p:, :p] )
        C22 = np.mean( weights_ordered[p:, p:] )
        weight_clusters = np.array([[C11, C12],[C21, C22]])
        plt.imshow(weight_clusters, cmap=cmap, vmin=-0.1, vmax=0.1)
        plt.title('Clustered Weight Matrix')
        plt.ylabel("Presynaptic")
        plt.xlabel("Postsynaptic")

    def VisualizeWeightMatrix(self):
        #pass
        plt.figure()
        cmap=matplotlib.cm.bwr
        neuron_sorting=self._neuronIX
        MINIMUM_DISPLAY_VALUE = -0.5
        MAXIMUM_DISPLAY_VALUE = 0.5
        if not neuron_sorting is None:
            weight_matrix=self._J['rec'].cpu().detach().numpy()
            weights_ordered = weight_matrix[:,neuron_sorting]
            weights_ordered = weights_ordered[neuron_sorting, :]
            plt.imshow(weights_ordered, cmap=cmap, vmin=MINIMUM_DISPLAY_VALUE, vmax=MAXIMUM_DISPLAY_VALUE)
            #plt.imshow(weights_ordered, cmap=cmap, vmin=np.min(weight_matrix), vmax=np.max(weight_matrix))
            plt.title('Recurrent Weights (Ordered by Neuron Factor)')
        else:
            weight_matrix = self._J['rec'].cpu().detach().numpy()
            plt.imshow(weight_matrix, cmap=cmap, vmin=MINIMUM_DISPLAY_VALUE, vmax=MAXIMUM_DISPLAY_VALUE)
            plt.title('Network Weight Matrix (unsorted)')
        plt.ylabel('Postsynaptic Neuron')
        plt.xlabel('Presynaptic Neuron')

    def GetF(self):
        W_rec = self._J['rec'].data.cpu().detach().numpy()
        W_in = self._J['in'].data.cpu().detach().numpy()
        b = self._J['bias'].data.cpu().detach().numpy()

        def master_function(inpt, relu=False):
            dt = 0.1
            sizeOfInput = len(inpt)
            inpt = inpt.reshape(sizeOfInput,1)
            if relu:
                return lambda x: np.squeeze( dt*np.matmul(W_in, inpt) + dt*np.matmul(W_rec, (np.maximum( np.zeros((50,1)), x.reshape(50,1)) )) - dt*x.reshape(50,1) + b*dt)
            else:
                return lambda x: np.squeeze( dt*np.matmul(W_in, inpt) + dt*np.matmul(W_rec, (1+np.tanh(x.reshape(50,1)))) - dt*x.reshape(50,1) + b*dt)

        return master_function
        
    def plotLosses(self):
        plt.plot(self._losses)
        plt.ylabel('Loss')
        plt.xlabel('Trial')
        plt.title(self.model_name)

###########################################################
# AUXILLARY FUNCTIONS
###########################################################

def loadRNN(fName):
    '''
    loads an rnn object that was previously saved

    Returns
    -------
    model if it was succesfully loaded, otherwise false       

    '''
    if os.path.exists(fName+".pt"):
        f = open(fName+".txt", 'r')
        hyperParams = {}
        for line in f:
            key, value = line.strip().split(':')
            hyperParams[key.strip()] = float(value.strip())
        f.close()
        model = RNN(hyperParams)
        model.load(fName)      # loads the RNN object
        model._MODEL_NAME = fName
        return model
    else:       # file does not exist
        return False

def loadHeb():
    hyperParams = {       # dictionary of all hyper-parameters
    "inputSize" : 5,
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
    "taskVar" : 0.1
    }
    rnnModel = RNN(hyperParams)
    Jin = np.loadtxt("Win.txt")
    Jrec = np.loadtxt("Jrec.txt")
    Jout = np.zeros((1,50))
    rnnModel.AssignWeights(Jin, Jrec, Jout)
    
    return rnnModel

###########################################################
#DEBUG RNN CLASSS
###########################################################
if __name__ == '__main__':

    #hyper-parameters for RNN
    input_size=1
    hidden_size=50
    output_size=1

    #create an RNN instance
    rnn_inst = RNN(input_size, hidden_size, output_size)
    rnn_inst.createValidationSet()
    print("validation accuracy", rnn_inst.GetValidationAccuracy())
    rnn_inst.VisualizeWeightMatrix()

    
    # #verify network forward pass
    # inpt = np.random.randn(1)
    # print('\n\nCOMPUTING FORWARD PASS...\n')
    # hidden = rnn_inst.init_hidden()
    # output, hidden = rnn_inst.forward(inpt, hidden, 0.1)
    # print('network output:', output)
    # print('\nupdated network hidden state:\n', hidden)

    # print('\nTesting Master Function Generator...\n')
    # F = rnn_inst.GetF()
    # my_func = F(1)
    # x=np.random.rand(50,1)
    # print('output:', my_func(x).shape)




