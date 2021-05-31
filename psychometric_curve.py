'''
Author: Brandon McMahan
Date: September 24, 2019
Will produce psychometric curves for RNN models
trained on the context task. Want
'''
import numpy as np 
import torch 
from rnn import RNN, loadRNN, loadHyperParams
from task.williams import Williams
from task.multi_sensory import multi_sensory
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from task.context import context_task
from task.Ncontext import Ncontext
import argparse
import pdb
import os


global coherence_vals
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)
def linear(x, m, b):
    y = m*x + b
    return (y)

def Plot(coherence_vals, reach_data, plt_title="", ymin=-.1, ymax=1.1, fit='sigmoid', newFig=True, cs='b'):
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
    plt.scatter(coherence_vals, reach_data, c=cs)

    if not args.nofit:  # fit a curve to data
        p0 = [max(reach_data), np.median(coherence_vals),1,min(reach_data)] 
        if fit=="sigmoid":
            data_fit = fit_sigmoid(coherence_vals, reach_data, p0)
        elif fit == "linear":
            data_fit = fit_linear(coherence_vals, reach_data, p0)
        else:
            raise NotImplementedError()

        plt.plot(np.linspace(min(coherence_vals), max(coherence_vals), 100), data_fit, alpha=.5, c=cs)
    
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

    num_trials = 2000
    # will hold the number of rightward reaches for each coherence level
    num_right_reaches = []
    num_right_reaches_com = []

    task_data = torch.zeros(num_trials, rnn._inputSize, 750)
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
task_type.add_argument("--N", type=int, default=0)

parser.add_argument("fname", help="filename of model to analyze")
parser.add_argument("--nofit", action="store_true", default=False)
parser.add_argument("--accumulate", type=int, default=0)  # adds data to existing curve plot

args = parser.parse_args()  

a_file = open("models/"+args.fname, 'r')
nested_models_list = [(line.strip()).split() for line in a_file]

a_file.close()

# loop over all models
names = []
start_ix = []
end_ix = []
counter = 0
new_line = True
num_model_types = len(nested_models_list)
numModelsOfType = {}   # indicates how many models of each type we have
count = 0
coherence_vals = 2*np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])
for model_ix in range(num_model_types):
    start_ix.append(counter)
    for model_num in nested_models_list[model_ix]:
        if new_line:
            names.append(model_num)
            if os.path.isfile("models/"+args.fname[:-4] + "_" + names[-1] + "_pschometrics.npy"):  # file exists
                new_line = True
                model_ix += 1
                print("skipped ", model_num, " models!")
                break
                #psycho_statistics = np.load("models/"+args.fname[:-4] + "_" + names[-1] + "_pschometrics.npy")
            new_line = False
            n_models = len(nested_models_list[model_ix])-1  # models of this type
            n_inputs = np.max((args.N, 1))                  # model input dimension
            n_coherence_vals = 8                            # coherence values
            psycho_data = np.zeros((n_inputs, n_models, n_coherence_vals))
            model_count = 0
            continue
        # load the current RNN
        model_name = "models/" + nested_models_list[model_ix][model_count+1]
        if args.N != 0:  # CDI task
            hyperParams = {}
            loadHyperParams(model_name, hyperParams)
            task = Ncontext(var=hyperParams["taskVar"], dim=args.N, device="cpu")
            rnn, hyperParams = loadRNN(model_name, load_hyper=True, task=task)
            
            titles = ["in context", "out context"] 
            fit_type = ["sigmoid", "linear"]
            use_new_fig = [1, 0]
            for N in range(args.N):
                psycho_data[N, model_count, :] = TestCoherence(rnn, task, context_choice=N)[:,0]
            #Plot(coherence_vals, num_right_reaches, plt_title=titles[N>=1], fit=fit_type[N>=1], newFig=(N==0))    # plot in-context psychometric data

        else:  # RDM task
            rnn, hyperParams = loadRNN(model_name, load_hyper=True)
            task = Williams(device="cpu")
            psycho_data[0, model_count, :] = TestCoherence(rnn, task)[:,0]
        # end load the current RNN
        #num = int(model_num)
        #embeddings.append(getMDS(model_num, learningRule=names[-1]).reshape(1,-1))
        counter += 1
        model_count += 1
    if not new_line:
        psycho_statistics = np.zeros((2, args.N, n_coherence_vals))
        psycho_statistics[0] = np.mean(psycho_data, 1)
        psycho_statistics[1] = np.std(psycho_data, 1)
        save_name = args.fname[:-4] + "_" + names[-1] + "_pschometrics"
        np.save("models/" + save_name, psycho_data)
    end_ix.append(counter)
    new_line = True
start_ix.append(counter)  # last element of start ix is the total


    
###############################################################################
# End Analysis Specification
###############################################################################
plt.show()  