import rnn
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from rnn import RNN
#import pdb
# import williams as task  # ideally, I call this from another script
# as of now would have to uncomment williams and repplace with the appropriate task


class Genetic(RNN):
    def __init__(self, hyperParams, numPop = 50, numGenerations = 500, numParents=5, mutation=0.005, task="rdm"):
        super(Genetic, self).__init__(hyperParams, task=task)                             # initialize parent class

        self._numGenerations = numGenerations                                  # maximum allowed generations for convergence
        self._hParams["numPop"] = numPop                                       # population size
        self._hParams["numParents"] = numParents
        self._hParams["mutation"] =  mutation
        self.losses = np.zeros((self._numGenerations, ))


    def initSuperHidden(self, superModel, num_pop, batch_size):
        #torch.manual_seed(0)
        self._init_hidden()
        hidden = torch.zeros(superModel._hiddenSize, batch_size)
        for member in range(num_pop):
            hidden[member*num_pop:(member+1)*num_pop] = self._hidden.clone()
        hidden = hidden.cuda()
        superModel._hidden = hidden                                             # initializes hidden layer of concatned rnn
        
    def initializeSuperModel(self, superModel, num_pop, hiddenSize=50):
        '''
        Initializes the recurrent weights for the concatenated RNN model that 
        represents the current population
        '''
        g = 1
        Win = torch.zeros((2500,self._inputSize))
        Wrec = torch.zeros((num_pop*hiddenSize, num_pop*hiddenSize))
        Wout = torch.zeros((num_pop, num_pop*hiddenSize))
        for currChild in range(num_pop):
            #torch.manual_seed(currChild)
            Win[currChild*50:(currChild+1)*50] = 0.5*(torch.randn(self._hiddenSize, self._inputSize))
            Wrec[currChild*50:(currChild+1)*50, currChild*50:(currChild+1)*50] = ((g**2)/50) * torch.randn((50,50))
            Wout[currChild,currChild*50:(currChild+1)*50] = 0.3*torch.randn((1,50))
        superModel.AssignWeights(Win, Wrec, Wout)

    
    def CreateDescendants(self, parentSet, superModel, num_pop, hiddenSize, mutation):
        '''
        adds noise to appropriate matrix elements to create children for next 
        generation
        '''
        # initialize new parameters for the next generation
        Win = torch.zeros((num_pop*hiddenSize, self._inputSize))
        Wrec = torch.zeros((num_pop*hiddenSize, num_pop*hiddenSize))
        Wout = torch.zeros((num_pop, num_pop*hiddenSize))
        
        numParents = len(parentSet)
        
        # loop over each child for the next generation
        for newChild in range(num_pop):   
            # extract the parent
            if newChild == 0:
                parentIX = parentSet[0]
            else:
                #np.random.seed(newChild)
                parentIX = parentSet[int(np.random.rand()*numParents)]
            # if newChild < 5:
            #     print("parent selected for next gen:", parentIX)
            
            parentWin = superModel._J['in'][parentIX*num_pop:(parentIX+1)*num_pop]
            parentWrec = superModel._J['rec'][parentIX*num_pop:(parentIX+1)*num_pop, parentIX*num_pop:(parentIX+1)*num_pop]
            parentWout = superModel._J['out'][parentIX, parentIX*num_pop:(parentIX+1)*num_pop]
            
            # mutate the parent to produce the child
            #torch.manual_seed(newChild)
            if newChild == 0:
                # keep the parent in the next generation
                Win[newChild*num_pop:(newChild+1)*num_pop, :] = parentWin 
                Wrec[newChild*num_pop:(newChild+1)*num_pop, newChild*num_pop:(newChild+1)*num_pop] = parentWrec 
                Wout[newChild, newChild*num_pop:(newChild+1)*num_pop] = parentWout 
            else:
                # mutate the parent
                Win_noise = torch.randn(parentWin.shape[0], parentWin.shape[1]).cuda()
                Wrec_noise = torch.randn(parentWrec.shape[0], parentWrec.shape[1]).cuda()
                Wout_noise = torch.randn(parentWout.shape[0]).cuda()
                Win[newChild*num_pop:(newChild+1)*num_pop, :] = parentWin + mutation * Win_noise
                Wrec[newChild*num_pop:(newChild+1)*num_pop, newChild*num_pop:(newChild+1)*num_pop] = parentWrec + mutation * Wrec_noise
                Wout[newChild, newChild*num_pop:(newChild+1)*num_pop] = parentWout + mutation * Wout_noise
            
        # update the model parameters for the next generation
        superModel.AssignWeights(Win, Wrec, Wout)
    
    def computeSuperLoss(self, modelOutputs, num_pop):
        # create an array that will hold losses for each model
        lossArray = torch.zeros((num_pop))
        batchSize = modelOutputs.shape[-1]
        # get the batch output for each model in the population
        for modelIX in range(num_pop):
            modelOutput = modelOutputs[:,modelIX,:]
            # compute the loss on the current models output
            tmp = self._task.Loss(modelOutput, self.batch_labels.cpu())
            tmp = torch.sum(tmp) / batchSize
            lossArray[modelIX] = tmp
            #print("(new) Loss:", tmp)
        return lossArray
        

    def train(self):
        '''
        trainGenetic will train the RNN using a high efficiency genetic algorithm
        '''
        print('\n\n\n')
        
        self._startTimer()
        num_pop = self._hParams["numPop"]
        num_parents = self._hParams["numParents"]
        mutation = self._hParams["mutation"]
        # create a super model that will hold the entire population
        superModel_params = self._hParams.copy()                               # copy constructs hyper-parameters for super model
        superModel_params["inputSize"] = self._inputSize
        superModel_params['hiddenSize'] = num_pop*50
        superModel_params["outputSize"] = num_pop
        superModel = RNN(superModel_params)
        self.initializeSuperModel(superModel, num_pop)
        
        self.createValidationSet()
        # for CUDA implementation
        self.batch_data = torch.zeros(self._batchSize, self._inputSize, self._task.N).cuda()
        self.batch_labels = torch.zeros(self._batchSize,1).cuda()
        self.activity_tensor = np.zeros((500, 75, 50))  # MAX_GENERATIONS x TIMESTEPS x HIDDEN_UNITS
        neuronActivities = np.zeros((75, 50*50))        # holds activity of all population members through trial
        self.activity_targets = np.zeros((500))

        
        
        generationCounter = 0
        lossHist = []      # stores best loss on each generation
        validation_accuracy = 0
        validation_acc_hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #while(generationCounter < self.num_generations):
        while(validation_accuracy < 0.9):
            print("Current Generation:", generationCounter)
        # create a training batch
            #torch.manual_seed(41)
            for b in range(self._batchSize):  # batch_size
                inpt_tmp, condition_tmp = self._task.GetInput()  # this reflects the williams decision making task, drawing samples from the distribution
                self.batch_data[b,:] = inpt_tmp[:].T
                self.batch_labels[b] = condition_tmp.item()
            
             # try passing some data through superModel
            inpt = self.batch_data.permute(1,0,2)
            
            self.initSuperHidden(superModel, num_pop, self._batchSize)
            outList = torch.zeros((self._task.N, self._hiddenSize, self._batchSize))
            for i in range(inpt.shape[1]):
                output_temp, hidden = superModel._forward(inpt[:, :, i])  # in the other case it was input[i, :]... just be careful\
                # output_temp is shape (superModel._outputSize, batchSize)
                if (i %100 == 0):
                    activityIX = int(i/100)
                    neuronActivities[activityIX, :] = hidden.detach().cpu().numpy()[:,-1]   # activty from last trial in batch saved
                outList[i,:,:] = output_temp.cpu()     # shape (Time, sueprModel._outputSize, batchSize)    
            
            # compute losses
            lossArray = self.computeSuperLoss(outList, 50)
            lossArray = lossArray.detach().numpy()
            parentSet = np.argsort(lossArray)
            
            lossArray = lossArray[parentSet]
            print("loss:", lossArray[0])
            parentSet = torch.from_numpy(parentSet).cuda()
            
            # generate the parent set
            lossArraySorted = lossArray
            parentSet = parentSet[:5]       # truncated to 5 best parents

            # update RNN model weights
            bestIX = parentSet[0]
            Win = superModel._J['in'][bestIX*num_pop:(bestIX+1)*num_pop]
            Wrec = superModel._J['rec'][bestIX*num_pop:(bestIX+1)*num_pop, bestIX*num_pop:(bestIX+1)*num_pop]
            Wout = superModel._J['out'][bestIX, bestIX*num_pop:(bestIX+1)*num_pop]
            self.AssignWeights(Win, Wrec, Wout)   
            
            # update validation accuracy 
            validation_accuracy_curr = self.GetValidationAccuracy()
            validation_acc_hist[:9] = validation_acc_hist[1:]
            validation_acc_hist[-1] = validation_accuracy_curr
            validation_accuracy = np.min(validation_acc_hist)
            print('validation accuracy', validation_accuracy)
            print('validation history', validation_acc_hist)
  
            # rebuilding bug from older version
            # if generationCounter == 0:
            #     hiddenSize=50
            #     # zero all parents --  bug from old version
            #     Win = torch.zeros((2500,1))
            #     Wrec = torch.zeros((num_pop*hiddenSize, num_pop*hiddenSize))
            #     Wout = torch.zeros((num_pop, num_pop*hiddenSize))
            #     superModel.AssignWeights(Win, Wrec, Wout)
            
            # now try to create the next generation 
            self.CreateDescendants(parentSet, superModel, num_pop, 50, mutation)
            #print("Generation", generationCounter, "update completed, best loss was:", lossArraySorted[0].item())
            lossHist.append(lossArraySorted[0].item())
            
            self.activity_tensor[generationCounter, :, :] = neuronActivities[:, bestIX*50:(bestIX+1)*50]
            self.activity_targets[generationCounter] = self.batch_labels[-1]
            generationCounter += 1
            
          
        # update RNN model
        self._losses = np.array(lossHist)
        self._activityTensor = self.activity_tensor[:generationCounter,:,:]
        self._targets = self.activity_targets[:generationCounter]
        self._endTimer()
       
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
    "batchSize" : 500
    }
    rnn_inst = Genetic(hyperParams)
    
    print("\nsuccesfully constructed genetic rnn object!")
    
    rnn_inst.createValidationSet()
    print("Validation accuracy:", rnn_inst.GetValidationAccuracy())
    
    rnn_inst.train(50)
    rnn_inst.save()
    print("rnn model succesfully saved!")
    
    # num_pop=50
    # sigma=0.01
    # start_time = time.time()
    # trainer.trainGenetic(num_pop, sigma, batch_size=500, num_parents=5, mutation=0.005)
    # #trainer.plotLosses()
    # end_time = time.time()
    # print('Training took', end_time-start_time, 'seconds\n')
