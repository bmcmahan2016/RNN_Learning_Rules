# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:43:24 2020

@author: bmcma

implementation of auxillary functions for performing psychometric and com 
analysis based on the work by Piexoto et al., 2019 found at:
    https://www.biorxiv.org/content/10.1101/681783v1
    
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from rnn import RNN
from task.williams import Williams

def isCOM(rnnOutput, zeroIX, showPlots=False):
    '''
    determines if a putative change of mind occured in an RNNs output at the specified
    zero index

    Args:
        rnnOutput (TYPE): DESCRIPTION.
        zeroIX (int): index of the sign change.

    Returns:
        TYPE: DESCRIPTION.

    '''
    # some constants for the COM detection algorithm
    stabilityDuration = 200
    thresh = 0.5
    
    rnnOutputSign = np.sign(rnnOutput)
    signBeforeZero = rnnOutputSign[zeroIX-stabilityDuration:zeroIX]
    signBeforeZero = np.unique(signBeforeZero)
    signAfterZero = rnnOutputSign[zeroIX+2:zeroIX+1+stabilityDuration]
    signAfterZero = np.unique(signAfterZero)
    
    if ( (len(signBeforeZero) != 1) or (len(signAfterZero) != 1) or signAfterZero == signBeforeZero):
        return False     # minimum duration of sign stability not met for this crossing to be considered a putative COM
    if zeroIX > 750-stabilityDuration:
        return False     # happened too late in trial
    
    timeAhead = 1
    while(True):
        if (zeroIX+timeAhead >= len(rnnOutput)):
            return False      # we exhausted the entire trial trying to satisfy minimum value requriement
        if (rnnOutput[zeroIX+timeAhead] == 0):
            return False      # we hit another zero crossing before minimum value requirement satisfied
        if (rnnOutput[zeroIX+timeAhead] > thresh*signAfterZero):
            break             # minimum value after zero crossing satisfied
        timeAhead += 1
            
    timeBack = 1
    while(True):
        if (zeroIX-timeBack < 0):
            return False      # we exhausted the entire trial trying to satisfy minimum value requriement
        if (rnnOutput[zeroIX-timeBack] == 0):
            return False      # we hit another zero crossing before minimum value requirement satisfied
        if (rnnOutput[zeroIX-timeBack] > thresh*signBeforeZero):
            break             # minimum value after zero crossing satisfied
        timeBack += 1
            
    # if we make it this far without returning this qualifies as a putative COM
    if (showPlots == True):
        if np.random.rand(1) < 0.01:
            plt.figure()
            plt.plot(rnnOutput[:zeroIX+stabilityDuration])
            plt.plot(np.linspace(0,0,zeroIX+stabilityDuration), c='k', ls='--')
    return True

def detectCOM(rnnOutput):
    '''
    detects if a putative change of mind occured in an RNNs output and the final 
    decision made by the RNN on the current trial
    
    Args:
        rnnOutput (TYPE): DESCRIPTION.

    Returns:
        boolean: True if a COM was detected otherwise False.
        float: The final decision made by the network for this trial.

    '''
    signChangeIXs = np.nonzero(np.diff(np.sign(rnnOutput[250:])))   # ignore first 250ms
    for putativeCOM in signChangeIXs[0]:
        if (isCOM(rnnOutput, putativeCOM+250)):
            return True, np.sign(rnnOutput[putativeCOM+251])
        
    return False, np.sign(rnnOutput[-1])
    

def Plot(xData, yData):
    '''
    Plot will take a data matrix and plot the mean with standard deviation
    shaded

    Args:
        xData (NumPy array): x-values to align each sample, must have shape (numSamplesPerTrials)
        yData (NumPy array): NumPy array of data with shape (numTrials, numSamplesPerTrial)
        to be plotted.

    Returns:
        True if succesful.

    '''
    if xData.shape[0] != yData.shape[1]:     # not enough x-labels for samples
        print("Warning not enough x values for number of samples recieved !")
        return False
    
    yMean = np.mean(yData, axis=0)
    yStd = np.std(yData, axis=0)
    plt.plot(xData, yMean)
    plt.fill_between(xData, yMean-yStd, yMean+yStd, alpha=.5)
    return True


def fractionCorrective(model):
    # generate the task data
    task = Williams(N=750, variance=0.2)
    coherence_vals = np.array([0, 0.001, 0.005, 0.008, 0.01, 0.02, 0.04])
    task_data = torch.zeros(task.N, 5_000).cuda()
    com_correct = np.zeros((7))
    com_wrong = np.zeros((coherence_vals.shape[0]))
    
    #model = RNN(1,50,1)
    #model.load(MODEL_NAME)
    #network_decisions = torch.zeros(5_000, 1).cuda()
    for repeatExperiment in range(5):     # repeats the experiment to reduce noise
        print("New Experiment !", repeatExperiment)
        for _, coherence in enumerate(coherence_vals):
            comDecisions = []
            targetOutputs = []
            # generate the task data
            for trial_num in range(5_000):
                task_data[:, trial_num] = task.PsychoTest(coherence).t()
            # get the decisions for COM trials
            output = model.feed(task_data)
            output = output.detach().cpu().numpy()
            for currTrial in range(output.shape[1]):
                comFlag, tmpDecision = detectCOM(output[:,currTrial])
                if comFlag:
                    comDecisions.append(tmpDecision)
                    if coherence == 0:
                        targetOutputs.append(1)
                    else:
                        targetOutputs.append(np.sign(coherence))
        
            comDecisions = np.array(comDecisions)
            targetOutputs = np.array(targetOutputs)
    
            num_com_wrong = np.nonzero(comDecisions-targetOutputs)[0].shape[0]
            num_com_trials = comDecisions.shape[0]
            num_com_corrective = num_com_trials - num_com_wrong
            
            com_correct[_] = com_correct[_] + num_com_corrective
            com_wrong[_] = com_wrong[_] + num_com_wrong
            
            print("com_correct", com_correct)
            print("com_wrong", com_wrong)
            
            # repeat the analysis for negative coherence values
            # generate the task data
            
            
        for _, coherence in enumerate(coherence_vals):
            comDecisions = []
            targetOutputs = []
            # generate the task data
            for trial_num in range(5_000):
                task_data[:, trial_num] = task.PsychoTest(-coherence).t()
            # get the decisions for COM trials
            output = model.feed(task_data)
            output = output.detach().cpu().numpy()
            for currTrial in range(output.shape[1]):
                comFlag, tmpDecision = detectCOM(output[:,currTrial])
                if comFlag:
                    comDecisions.append(tmpDecision)
                    if coherence==0:
                        targetOutputs.append(-1)
                    else:
                        targetOutputs.append(np.sign(coherence))
        
            comDecisions = np.array(comDecisions)
            targetOutputs = np.array(targetOutputs)
            #com_idxs = COM_Flag(output)
            # com_idxs contain the trial #s for all trials where a COM occured
            # contains the decision made on each com trial
            #com_decisions = torch.sign(output[-1, com_idxs])
            #target_output = np.sign(coherence)
            #if (coherence==0): target_output = 1    # randomly pick a direction
            num_com_wrong = np.nonzero(comDecisions-targetOutputs)[0].shape[0]
            num_com_trials = comDecisions.shape[0]
            num_com_corrective = num_com_trials - num_com_wrong
            #print("chorence:", coherence, "corrective COM:", num_com_corrective*100 / num_com_trials, "%" )
            
            com_correct[_] = com_correct[_] + num_com_corrective
            com_wrong[_] = com_wrong[_] + num_com_wrong
            
           
        # end loop over all coherence values
    # end loop over all experiments
    
    return com_correct / (com_correct + com_wrong)   # total fraction of corrective coms accross all input levels tested