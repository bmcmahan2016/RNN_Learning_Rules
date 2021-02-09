'''
Author: Brandon McMahan
Date: September 24, 2019
Will produce psychometric curves for RNN models
trained on the context task. Want
'''
import numpy as np 
import torch 
from rnn import RNN, loadRNN
from task.williams import Williams
from task.multi_sensory import multi_sensory
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from task.context import context_task
import argparse
import pdb


global coherence_vals
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)
def linear(x, m, b):
    y = m*x + b
    return (y)

def Plot(coherence_vals, reach_data, plt_title="", ymin=-.1, ymax=1.1, fit='sigmoid', newFig=True):
    '''Plots psychometric data'''

    if newFig:     # generate a new figure
        fig_object = plt.figure()   
        axis_object = fig_object.add_subplot(1,1,1)
        axis_object.spines["left"].set_position("center")
        axis_object.spines["bottom"].set_position("center")
        axis_object.spines["right"].set_color("none")
        axis_object.spines["top"].set_color("none")
        axis_object.xaxis.set_label("top")

    # plots RNN data
    plt.scatter(coherence_vals, reach_data)

    if not args.nofit:  # fit a curve to data
        p0 = [max(reach_data), np.median(coherence_vals),1,min(reach_data)] 
        if fit=="sigmoid":
            data_fit = fit_sigmoid(coherence_vals, reach_data, p0)
        elif fit == "linear":
            data_fit = fit_linear(coherence_vals, reach_data, p0)
        else:
            raise NotImplementedError()

        plt.plot(np.linspace(min(coherence_vals), max(coherence_vals), 100), data_fit, alpha=.5)
    
    # adds plot lables
    plt.title(plt_title)
    plt.ylim([ymin, ymax])

def fit_sigmoid(coherence_vals, reach_data, p0):
    '''fits a sigmoid curve to psychometric data'''
    popt, pcov = curve_fit(sigmoid, coherence_vals, np.squeeze(reach_data), p0, method='dogbox')
    data_fit = sigmoid(np.linspace(min(coherence_vals), max(coherence_vals), 100), *popt)
    return data_fit

def fit_linear(coherence_vals, reach_data, p0):
    '''fits a linear curve to psychometric data'''
    p0 = [max(reach_data), np.median(coherence_vals),1,min(reach_data)] 
    popt, pcov = curve_fit(linear, coherence_vals, np.squeeze(reach_data), method='dogbox')
    data_fit = linear(np.linspace(min(coherence_vals), max(coherence_vals), 100), *popt)
    return data_fit

def CountReaches(network_decisions, tol=1):
    
    '''
    CountReaches will count how many rightward reaches were made. 
    RNN outputs of +1 are defined as rightward reaches while RNN outputs 
    of -1 are defined as leftward reaches. CountReaches returns rightward 
    reaches as a fraction of total reaches (outputs) made by the network.
    
    PARAMETERS
    --network_decisions: 1D NumPy with length equal to the total number of trials/
    decisions made by the network. Each element of array corresponds to a network 
    output in [-1,1]
    
    RETURNS
    
    '''
    print("network decisions:", network_decisions)
    num_reaches = len(network_decisions)
    total_right = 0
    for _ in range(num_reaches):
        if network_decisions[_] > 0:#1-network_decisions[_] <= tol:
            total_right += 1
            #print(network_decisions[_])
        elif network_decisions[_] == 0: #torch.abs(network_decisions[_]) < 1-tol:
            # omit trials where a clear reach was not made
            #print('reach omitted...\n')
            num_reaches -= 1
        else:
            pass
            #print(" "*10, network_decisions[_])
    #print('num reaches', num_reaches)
    if num_reaches == 0:
        fraction_right = 0.5
    else:
        fraction_right = total_right / num_reaches
        
    return fraction_right

def TestCoherence(rnn, task, context_choice=0):
    '''Tests the coherence of a trained on RNN

    Parameters
    ----------
    rnn : rnn object
        trained rnn object that will be tested
    task : task object
        used to generate data for testing RNN
    context_choice : int
        specifies which input channel/context to test

    Returns
    -------
    numRightReaches : NumPy array
        array of shape (coherence_levels, 1) with the number of rightward reaches for each coherence level
    '''

    num_trials = 2_000
    # will hold the number of rightward reaches for each coherence level
    num_right_reaches = []
    num_right_reaches_com = []

    task_data = torch.zeros(num_trials, rnn._inputSize, 750).cuda()
    #task_data = torch.unsqueeze(task_data.t(), 1)
    
    for _, coherence in enumerate(coherence_vals):   # loop over coherence vals
        print('\n\ncoherence:', coherence)
        for trial_num in range(num_trials):
            if context_choice != "":
                task_data[trial_num,:,:] = task.PsychoTest(coherence, context=context_choice).t()#, context=context_choice).t()
            else:
                task_data[trial_num,:,:] = task.PsychoTest(coherence).t()
        print('shape of task data', task_data.shape)
        
        output = rnn.feed(task_data)
        network_decisions = output[-1,:]         # outputs are (t_steps, num_trials)
        if args.multi:                           # scales outputs from [0,1] -> [-1,1]
            print("Multisensory RNN scaling applied to network outputs")
            # TODO: uncomment below line to if network outputs [0, 1]
            #network_decisions = 2 * (network_decisions - 0.5)

        # computes fraction of reaches that were to the right
        ReachFraction = CountReaches(network_decisions)
        num_right_reaches.append(ReachFraction)
    # end loop over coherence vals

    # format and returns number of rightward reaches
    num_right_reaches = np.array(num_right_reaches).reshape(-1,1)
    assert(num_right_reaches.shape == (len(coherence_vals), 1))
    return num_right_reaches

###############################################################################
# Specify Analysis Here
###############################################################################
# determines what analysis to run
parser = argparse.ArgumentParser(description="Generates psychometric curve for RNN")
task_type = parser.add_mutually_exclusive_group()
task_type.add_argument("--rdm", action="store_true")
task_type.add_argument("--context", action="store_true")
task_type.add_argument("--multi", action="store_true")

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--nofit", action="store_true", default=False)

args = parser.parse_args()
# sets the model to be analyzed
model_name = "models/" + args.model_name     

print('evaluating model #', model_name)
# set the task (either context or Williams)
if args.rdm:
    rnn, hyperParams = loadRNN(model_name, load_hyper=True)
    task = Williams()
elif args.context:
    rnn, hyperParams = loadRNN(model_name, load_hyper=True)
    task = context_task()
elif args.multi:
    rnn, hyperParams = loadRNN(model_name, load_hyper=True, task="multi")
    task = multi_sensory(var=hyperParams["taskVar"])
###############################################################################
# End Analysis Specification
###############################################################################

if args.multi:    # generate pscyhometric curves for the multisensory task
    coherence_vals = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
    print("Generating psychometric curves for RNN trained on multisensory integration task...\n")

    num_right_reaches_auditory = TestCoherence(rnn, task, context_choice=0)
    num_right_reaches_visual = TestCoherence(rnn, task, context_choice=1)
    num_right_reaches_congruent = TestCoherence(rnn, task, context_choice=2)

    Plot(coherence_vals, num_right_reaches_auditory)
    Plot(coherence_vals, num_right_reaches_visual, newFig=False)
    Plot(coherence_vals, num_right_reaches_congruent, newFig=False, plt_title="Multisensory Psychometric Curves")
    plt.legend(["Auditory", "Visual", "Multisensory"])   # legend for generated plots  
elif args.context:# generate psychometric curves for the context task
    coherence_vals = 2*np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])
    num_right_reaches = TestCoherence(rnn, task, context_choice=0)
    Plot(coherence_vals, num_right_reaches, plt_title="in context", fit="sigmoid")    # plot in-context psychometric data

    num_right_reaches = TestCoherence(rnn, task, context_choice=1)
    Plot(coherence_vals, num_right_reaches, plt_title="out context", fit="linear", \
         newFig=False)   # plot out-context pyschometric data on same axis
elif args.rdm:    # generate psychometric curves for the rdm task
    coherence_vals = 2*np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])
    num_right_reaches = TestCoherence(rnn, task)
    Plot(coherence_vals, num_right_reaches, plt_title="in context")
plt.show()  