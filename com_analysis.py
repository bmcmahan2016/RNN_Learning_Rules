# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:58:55 2020

@author: bmcma
DESCRIPTION:
    Performs analysis similar to Piexoto on changes of mind. 
"""


import numpy as np 
import torch
from rnn import RNN, loadRNN
from task.williams import Williams
import matplotlib.pyplot as plt
import com

def COM_Flag(rnn_output):
    '''
    COM_Flag returns a flag indicating if a change of mind occured in the 
    trial
    Args:
        rnn_output (N*M torch tensor): A tensor of rnn output data where N is the 
        number of timesteps (default is 40) and M is the number of sample trials                            .

    Returns:
        num_vacillations: returns the number of trials for which the network 
        'changes mind'.

    '''
    thresh = -.8
    # bin outputs to -1, 0, and +1
    rnn_output1 = rnn_output[300:, :]
    rnn_output2 = rnn_output1 > thresh
    rnn_output2 = 1*rnn_output2
    rnn_output3 = rnn_output1 < -thresh
    rnn_output3 = -1*rnn_output3
    rnn_output4 = rnn_output2 + rnn_output3
    #rnn_output = torch.round(rnn_output)
    binned_output = torch.sign(rnn_output4)   # binary RNN output
	
    # determine the stability of trials
    output_velocity = binned_output[1:,:] - binned_output[:-1,:]
    trial_stabilit = (output_velocity==0)                           # 0 at every time step when velocity changes
    trial_stability = torch.sum(trial_stabilit, axis=0)             # total number of time steps where velocity is constant
    
    # get the minimum and maximum binned output for each trial
    trial_maxs, _ = torch.max(binned_output[10:,:], axis=0)
    trial_mins, _ = torch.min(binned_output[10:,:], axis=0)
    trial_ranges = trial_maxs - trial_mins
    
    #if binned output changes by more than 2 within a trial, a change of mind occured
    idx1 = torch.where(trial_ranges>1)[0]
    idx2 = torch.where(trial_stability>200)[0]                        # imposes that that velocity must be constant at more than 35 timesteps
    vacillation_trials = np.intersect1d(idx1.cpu(), idx2.cpu())
    #vacillation_trials = torch.where(trial_ranges>1)[0]
    print('number of vacillations', len(vacillation_trials))
    return vacillation_trials

def plotCOMTrials(comIX, rnnOutput):
    PLOT_OFFSET = 2     # offsets curves for better visualization
    rnnComOutputs = rnnOutput[:,com_idxs].detach().cpu().numpy()
    numComTrials = len(rnnComOutputs)
    plt.figure()
    for currTrial in range(numComTrials):
        plt.plot(rnnComOutputs[:,currTrial] + PLOT_OFFSET*currTrial)
        plt.plot(PLOT_OFFSET*currTrial+np.linspace(0,0,750), c='k', ls='--')
        #plt.plot(np.linspace(0.09, 0.09,750), c='k', '--')
    plt.xlabel("Time in Trial")
    plt.ylabel("Trial #")
    plt.title("Detected COM Trials")
    plt.ylim([-1,6])


#task = Williams(N=750, variance=0.2)

MODEL_NAME = 'models/bptt_089' 
model = loadRNN(MODEL_NAME)
task = model._task
# generate the task data
coherence_vals = np.array([0, 0.001, 0.005, 0.008, 0.01, 0.02, 0.04])
task_data = torch.zeros(task.N, 5_000).cuda()
com_correct = np.zeros((7))
com_wrong = np.zeros((7))
num_models = 2
#for model_num in range(8,9):

model.load(MODEL_NAME)
network_decisions = torch.zeros(5_000, 1).cuda()
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
            comFlag, tmpDecision = com.detectCOM(output[:,currTrial])
            if comFlag:
                comDecisions.append(tmpDecision)
                if coherence == 0:
                    targetOutputs.append(1)
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
            comFlag, tmpDecision = com.detectCOM(output[:,currTrial])
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
        
        for trial_num in range(0):
            task_data[:, trial_num] = task.PsychoTest(-coherence).t()
        # get the decisions for COM trials
        output = model.feed(task_data)
        com_idxs = COM_Flag(output)
        # com_idxs contain the trial #s for all trials where a COM occured
        # contains the decision made on each com trial
        com_decisions = torch.sign(output[-1, com_idxs])
        target_output = np.sign(coherence)
        num_com_wrong = torch.nonzero(com_decisions-target_output).shape[0]
        num_com_trials = com_decisions.shape[0]
        num_com_corrective = num_com_trials - num_com_wrong
        
        com_correct[_] = com_correct[_] + num_com_corrective
        com_wrong[_] = com_wrong[_] + num_com_wrong
        
        # end loop over current model
    # end loop over all models
com_correct = com_correct / (num_models-1)
com_wrong = com_wrong / (num_models-1)
    
#plotCOMTrials(com_idxs, output)
    
labels = ['0', '0.001', '0.005', '0.008', '0.01', '0.02', '0.04']
#labels = ['0', '0.0001', '0.001', '0.005', '0.01']
men_means = com_correct
women_means = com_wrong

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Corrective')
rects2 = ax.bar(x + width/2, women_means, width, label='Erroneous')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#COM')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = np.round(rect.get_height(), decimals=1)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
# now we just need to count how many decisions were correct and how many were incorrect