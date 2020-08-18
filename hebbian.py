'''
Hebian learning rule

Hebian class allows for construction of an RNN that can then be trained based on a biologically 
plausable hebbian update rule

Brandon McMahan
May 22, 2019
Miconi 2017 paper
'''
import numpy as np
import torch
import matplotlib.pyplot as plt 
import time
from rnn import RNN
from task.williams import Williams


class Hebbian(RNN):
    def __init__(self, hyperParams): 
        '''
        Initialize training object that uses Hebian learning for RNN
        '''
        super(Hebbian, self).__init__(hyperParams)       # initialize parent class
        
        #initialize hebbian specific hyper-parameters
        #self.alpha_trace = 0.75
        self._alphaTrace = 0.33
        self.lr = 0.002
        self.ExpectedError = []
        
        self._task = Williams(N=50, mean=1, variance=0.1)
        
        print('INPUT SIZE', self._inputSize)

        #initialize neurons with random activity
        self.neurons = {
        'in' : 0.1*(torch.rand(self._inputSize, 1).cuda() - 0.5),
        'rec' : 0.1*(torch.rand(self._hiddenSize, 1).cuda() - 0.5),
        }


        self._avgActivity = torch.zeros(self._hiddenSize, 1).cuda()    # holds an average of past neural activity
        # accumulates hebbian potential during each trial
        self.eligabilityTrace = torch.zeros((self._hiddenSize, self._hiddenSize)).cuda()

        #initialize hebbian potentials to zero
        # self.potentials = {
        # 'in' : torch.zeros((hidden_size, input_size)).cuda(),
        # 'rec' : torch.zeros((hidden_size, hidden_size)).cuda(),
        # 'out' : torch.zeros((output_size, hidden_size)).cuda()
        # }


        self.spikes = torch.tanh(self._hidden.clone())    # spiking activity of network
        self.outputs = torch.zeros((self._task.N)).cuda()

        #initialize error stores to zero
        self.R_exp = torch.zeros((2, 1)).cuda()
        
        #initialize gradients
        self.dJ = {}
        
    def GetValidationAccuracy(self, test_iters=2000):     # overides base class
        '''
        Will get validation accuracy of the model on the specified task to 
        determine if training should be terminated
        '''
        accuracy = 0.0
        tol = 1

        inpt_data = self.validationData.t()
        condition = self.validationTargets
        condition = torch.squeeze(condition)
        _, hiddenState = self.feed(inpt_data, return_hidden=True)
        output_final = torch.tanh(torch.tensor(hiddenState[-1][20]).cuda())

        # compute the difference magnitudes between model output and target output
        error = torch.abs(condition-output_final)
        # threshold this so that errors greater than tol(=0.1) are counted
        num_errors = torch.where(error>tol)[0].shape[0]
        accuracy = (test_iters - num_errors) / test_iters

        
        self._valHist.append(accuracy)                                          # appends current validation accuracy to history
    
        
        return accuracy

    def ResetActivity(self):

        #traces are set equal to current neural activity
        '''
        self.traces = {
        'in' : torch.randn(1,1),
        'rec' : torch.randn(self.model.hidden_size, 1),
        'out' : torch.randn(1,1)
        }

        '''
        input_size = self._inputSize
        hidden_size = self._hiddenSize
        output_size = self._outputSize
        
        self._init_hidden()     # initializes the hidden layer
        
        self.neurons = {
        'in' : 0.1*(torch.rand(input_size, 1).cuda() - 0.5),
        'rec' : self._hidden.clone(),
        }
        
        self.eligabilityTrace = torch.zeros((self._hiddenSize, self._hiddenSize)).cuda()
        self.outputs = torch.zeros(self._task.N).cuda()


        # #potentials set back to zero between trials
        # self.potentials = {
        # 'in' : torch.zeros((self._hiddenSize, self._inputSize)).cuda(),
        # 'rec' : torch.zeros((self._hiddenSize, self._hiddenSize)).cuda(),
        # 'out' : torch.zeros((self._outputSize, self._hiddenSize)).cuda()
        # }

    def Loss(self, output, condition):
        loss = self._task.Loss(output, condition, errorTrigger=-10)    # train on last 100 timesteps
        return loss

    def S(self, x):
        '''monotonic non-decreasing function'''
        return torch.mul(torch.mul(x,x), x)
        #return np.power(x, 3)

    def UpdatePotentials(self):
        '''
        UpdatePotentials will compute the potential weight
        updates to accumulate during a given timestep of 
        training

        keywrods
        layer: indicates the layer for which to compute the 
        eligibilities/potentials
        '''
        dx = self.GetFastFluctuations()
        self.eligabilityTrace += self.S(torch.matmul(dx, self.spikes.t()))



    def GetSparsePerturb(self, perturbProb=0.95):
        '''
        GetSparsePerturb will return a numpy array
        of perturbation values and zeros. The probability of 
        perturbation is given by 1-perturbProb

        **perturbProb is the fraction of units to not perturb

        **NOTE: ALLOWING FOR NEGATIVE PERTURBATIONS RESULTS
        IN A NETWORK THAT MAY EXPLODE/DIVERGE DURING TRAINING
        '''
        perturbs = torch.rand(self._hiddenSize, 1).cuda() - 0.5    #50x1
        perturb_inds = torch.rand(self._hiddenSize)           #50
        perturbs[perturb_inds < perturbProb, :] = 0
        self._hidden += perturbs                       #50x1
        return perturbs

    def ClipGrads(self, thresh=10e-4):
        #for layer in ['in', 'rec', 'out']:
        for layer in ['rec']:
            grad_tmp = self.dJ[layer]
            grad_tmp[grad_tmp < -thresh] = -thresh
            grad_tmp[grad_tmp > thresh] = thresh
            self.dJ[layer] = grad_tmp
            grad_tmp = 0
            
    def ZeroGrads(self):
        self.dJ['rec'][:, :] = 0

    #use lr=0.1 when output layer is true
    def UpdateWeights(self, R, curr_trial, condition, thresh=10e-4):
        '''
        Description of UpdateWeights

        PARAMETERS
        alpha controls how much expected error depends on the past errors, an alpha=1 
        will effectively mean that the expected error is never updated from its initial 
        condition while an alpha=0 means the expected error is exactly equal to the 
        last error
        '''
        #alpha=self.alpha_trace
        #update neuron weights
        # condition specific expected reward (note earlier E was massive --> dJ massive)
        #layers = ['input', 'recurrent', 'output']
        layers = ['rec']
        if condition == 1:
            for layer in layers:
                #print("R-Rexp", R - self.R_exp[0])
                self.R_exp[0] = self._alphaTrace * self.R_exp[0] + (1 - self._alphaTrace) * R
                self.dJ[layer] = self.lr * self.eligabilityTrace * (R - self.R_exp[0])
                self.ExpectedError.append((R-self.R_exp[0]).item())
        else:
            for layer in layers:
                #print("R-Rexp", R - self.R_exp[1])
                self.R_exp[1] = self._alphaTrace * self.R_exp[1] + (1 - self._alphaTrace) * R
                self.dJ[layer] = self.lr * self.eligabilityTrace * (R - self.R_exp[1])
                self.ExpectedError.append((R-self.R_exp[1]).item())

        self.ClipGrads()

        #update the weights
        for layer in layers:
            self._J[layer] = self._J[layer] - self.dJ[layer]
        self.StoreRecMag()    #store the maginutde of recurrence at this timestep
        #store the rewards
        self.reward_store[curr_trial] = R


    def GetFastFluctuations(self):
        dx = self._hidden.clone() - self._avgActivity
        return dx

    def UpdateAvg(self, tStep):
        '''
        alpha_trace = 1 --> traces are constant
        alpha_trace = 0 --> traces are activity at previous time step
        setting alpha_trace closer to one makes traces change more slowly
        setting alpha_trace nearer to zero makes traces change more rapidly
        '''
        self._avgActivity = self._avgActivity + (1/(1+tStep)) * (self._hidden.clone() - self._avgActivity)
        #print("avg activity:", self._avgActivity[0])

    def _train(self, inpts, curr_trial):
        for t_step in range(len(inpts)):
            #holds spike rates from last time step
            self.spikes = 1+torch.tanh(self._hidden.clone())

            self._UpdateHidden(inpts[t_step].reshape(1,-1))            
            #self.neurons['out'], self.neurons['rec'] = self._forward(inpts[t_step])
            if t_step %100 == 0:
                self.activity_tensor[curr_trial, int(t_step/100), :] = np.squeeze(self._hidden.clone().cpu().detach().numpy())
            self.GetSparsePerturb()         #add perturbations to neuron activations

            self.UpdatePotentials()
            
            self.UpdateAvg(t_step)      # updates running average of neural activity
            self.outputs[t_step] = self._hidden.clone()[20]

            
    def train(self, num_trials=10_000):
        startTime = time.time()          # training start time
        self.createValidationSet()
        #initialize some tensors that will hold reward signal
        self.reward_store = np.zeros((num_trials))
        self.activity_tensor = np.zeros((num_trials, int(1+self._task.N/100), self._hiddenSize))
        self.targets = []
        


        validation_accuracy = 0.0
        validation_acc_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        curr_trial = 0
        condition = torch.zeros(1).cuda()
        # will store error rate on validation dataset
        loss_hist = []
        self._w_hist.append(self._J)
        while(validation_accuracy < 0.9):
            self.ResetActivity()                            #resets traces
            inpt, condition[:] = self._task.GetInput()
            self.targets.append(condition.cpu())   #store the target output for the current trial
            
            #print("trial:", curr_trial)
            self._train(inpt, curr_trial)       #train network on current trial
            #RNN output at the end of current trial
            #output = self._hidden.clone()[20]                       # use neuron 20 as network output   
            R = self.Loss(self.outputs, condition)                        #computes loss
            self.UpdateWeights(R, curr_trial, condition)            #updates weights
            # store a history of the new weights
            self._w_hist.append(self._J)
            curr_trial += 1
            validation_accuracy_curr = self.GetValidationAccuracy()
            loss_hist.append(validation_accuracy_curr)
            validation_acc_hist[:9] = validation_acc_hist[1:]
            validation_acc_hist[-1] = validation_accuracy_curr
            validation_accuracy = np.min(validation_acc_hist)
            
            if curr_trial %10 == 0:
                print(curr_trial,'\nvalidation accuracy:', validation_acc_hist)
                print('R', R)

            if curr_trial %1000 == 999:   # save the model every 1,000 iterations
                self.save(self.modelName, N=curr_trial, tElapsed=time.time()-startTime)


            '''
            #update the console every 100 trials
            if curr_trial %100 == 0:
                print('trial:', curr_trial, 'of', num_trials)
                #print('output:', output)
                print('loss:', R[0])
                self.model.GetValidationAccuracy(self.task)
            '''


        self.losses = np.array(loss_hist)#self.reward_store
        self._activityTensor = self.activity_tensor[:curr_trial,:,:]
        self._targets = self.targets
        

    def test(self, verbose=False):
        #grab input
        inpts, condition = self.task.GetInput()
        for t_step in range(len(inpts)):
            self.ForwardPass(inpts[t_step])    
        output = np.tanh(self.neurons['output'])
        if verbose:
            print('Output:', output, 'Condition:', condition, '\n')
        return output, condition
    
    def PlotRewards(self):
        plt.figure()
        plt.plot(self.reward_store, alpha=0.5)
        #plt.plot(np.array(self.ExpectedError), alpha=0.1)
        plt.ylabel('Loss')
        plt.xlabel('Trial #')
        plt.ylabel('Reward Signal')
        plt.title('Reward Signal During Trainign on Perceptual Discrimination Task')
        #plt.show()

#end of Hebian class

#debuging PyTorch Version of Hebian Class below
if __name__ == "__main__":
    hyperParams = {                  # dictionary of all RNN hyper-parameters
       "inputSize" : 1,
       "hiddenSize" : 50,
       "outputSize" : 1,
       "g" : 1 ,
       "inputVariance" : 1,
       "outputVariance" : 0.5,
       "biasScale" : 0,
       "initScale" : 0.3,
       "dt" : 0.1,
       "batchSize" : 500,
       "taskMean" : 1,
       "taskVar" : 0.1,
       "N" : 50
       }
    
    rnnModel = Hebbian(hyperParams)
    rnnModel.setName("HebbianDebug")
    rnnModel.train()
    rnnModel.save()
    