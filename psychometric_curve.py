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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from task.context import context_task
import argparse
import pdb

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
    print(network_decisions)
    num_reaches = len(network_decisions)
    total_right = 0
    for _ in range(num_reaches):
        if 1-network_decisions[_] <= tol:
            total_right += 1
            #print(network_decisions[_])
        elif torch.abs(network_decisions[_]) < 1-tol:
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
d=0

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
    
    # bin outputs to -1, 0, and +1
    #rnn_output[:10, :] = 0
    rnn_output2 = rnn_output > .4
    rnn_output2 = 1*rnn_output2
    rnn_output3 = rnn_output < -.4
    rnn_output3 = -1*rnn_output3
    rnn_output4 = rnn_output2 + rnn_output3
    #rnn_output = torch.round(rnn_output)
    binned_output = torch.sign(rnn_output4)
	
    # determine the stability of trials
    output_velocity = binned_output[1:,:] - binned_output[:-1,:]
    trial_stabilit = output_velocity==0
    trial_stability = torch.sum(trial_stabilit, axis=0)
    
    # get the minimum and maximum binned output for each trial
    trial_maxs, _ = torch.max(binned_output[10:,:], axis=0)
    trial_mins, _ = torch.min(binned_output[10:,:], axis=0)
    trial_ranges = trial_maxs - trial_mins
    
    #if binned output changes by more than 2 within a trial, a change of mind occured
    idx1 = torch.where(trial_ranges>1)[0]
    idx2 = torch.where(trial_stability>35)[0]
    vacillation_trials = np.intersect1d(idx1.cpu(), idx2.cpu())
    #vacillation_trials = torch.where(trial_ranges>1)[0]
    print('number of vacillations', len(vacillation_trials))
    return vacillation_trials


def TestCoherence(rnn, task, context_choice="in"):
    global coherence_vals
    num_trials = 2_000
    coherence_vals = 2*np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])#np.linspace(-.2,.2,6)
    # will hold thenumber of rightward reaches for each coherence level
    num_right_reaches = []
    num_right_reaches_com = []
    # will hold the number of rightward reaches for coherence level in wrong context
    num_right_reaches_wrong = []
    task_data = torch.zeros(num_trials, rnn._inputSize, 750).cuda()
    #task_data = torch.unsqueeze(task_data.t(), 1)
    
    for _, coherence in enumerate(coherence_vals):
        print('\n\ncoherence:', coherence)
        # generate trials for this coherence value
        for trial_num in range(num_trials):
            if context_choice != "":
                task_data[trial_num,:,:] = task.PsychoTest(coherence, context=context_choice).t()#, context=context_choice).t()
            else:
                task_data[trial_num,:,:] = task.PsychoTest(coherence).t()
        print('shape of task data', task_data.shape)
        
        output = rnn.feed(task_data)
        com_idxs = COM_Flag(output)
        #output_out = rnn.feed(task_data_out_of_context)
        # we are only interested in the final network output
        #network_decisions.append( output[-1].cpu().detach().numpy() )
        network_decisions = output[-1,:]#[0,:]#[-1,:]                                  This controls if we do late v early in trial
        com_outputs = output[:,com_idxs]
        #if len(com_idxs)>0:
        #    plt.figure(11)
        #    plt.plot(com_outputs[:,0].cpu().detach().numpy())
        '''
        if len(com_idxs) >0:
            plt.figure()
            plt.plot(com_outputs.cpu().detach().numpy()[:,0])
            plt.show()
        '''
        network_com_decisions = network_decisions[com_idxs]
        print('network_com_decisions', network_com_decisions.shape)
        #if com_flag:
            # these are network outputs when the network changed its mind at least once
        #    network_com_decisions.append( output[-1].cpu().detach().numpy() )
        #network_decisions_out.append( output_out[-1].detach().numpy() )
        # at the end of 100 trials we want to compute how many rightward reaches occured
        ReachFraction = CountReaches(network_decisions)
        #print('network com decisions', len(network_com_decisions))
        if len(network_com_decisions) != 0:
            ReachFractionCOM = CountReaches(network_com_decisions)
        else:
            print('WARNING!! NO COM TRIALS!!')
            #assert False
            if coherence > 0:
                ReachFractionCOM=.5#1 
            else: 
                ReachFractionCOM=.5#0
        #ReachFractionOut = CountReaches(network_decisions_out)
        num_right_reaches.append(ReachFraction)
        num_right_reaches_com.append(ReachFractionCOM)
    return num_right_reaches, num_right_reaches_com

###############################################################################
# Specify Analysis Here
###############################################################################
# determines what analysis to run
parser = argparse.ArgumentParser(description="Generates psychometric curve for RNN")
task_type = parser.add_mutually_exclusive_group()
task_type.add_argument("--rdm", action="store_true")
task_type.add_argument("--context", action="store_true")
task_type.add_argument("--dnms", action="store_true")

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--nofit", action="store_true", default=False)

args = parser.parse_args()

# sets the model to be analyzed
model_name = "models/" + args.model_name     
rnn = loadRNN(model_name)
print('evaluating model #', model_name)
rnn.load(model_name)
# set the task (either context or Williams)
if args.rdm:
    task = Williams()
elif args.context:
    task = context_task()
elif args.dnms:
    raise NotImplementedError()
###############################################################################
# End Analysis Specification
###############################################################################

num_right_reaches = []
num_right_reaches_com = [] 

if str(type(task)) == "<class 'task.context.context_task'>":
    print("This line should print")
    context_choice = "in"
else:
    context_choice = ""
#### do in context first
num_right_reaches_tmp, num_right_reaches_com_tmp = TestCoherence(rnn, task, context_choice=context_choice)
num_right_reaches.append(num_right_reaches_tmp)
num_right_reaches_com.append(num_right_reaches_com_tmp)

# generate a figure to plot 
num_right_reaches = np.array(num_right_reaches)
num_right_reaches_com = np.array(num_right_reaches_com)
num_right_reaches_mean = np.mean(num_right_reaches, 0)
num_right_reaches_var = np.sqrt(np.var(num_right_reaches, 0))
num_right_reaches_com_mean = np.mean(num_right_reaches_com, 0)
num_right_reaches_com_var = np.sqrt(np.var(num_right_reaches_com, 0))


# fit a sigmoid curve to the data
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

xdata = coherence_vals
ydata = num_right_reaches_mean
p0 = [max(ydata), np.median(xdata),1,min(ydata)] 

if not args.nofit:
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    ydataa = sigmoid(np.linspace(-0.2, 0.2, 100), *popt)

fig_object = plt.figure(4)
axis_object = fig_object.add_subplot(1,1,1)

axis_object.spines["left"].set_position("center")
axis_object.spines["bottom"].set_position("center")
axis_object.spines["right"].set_color("none")
axis_object.spines["top"].set_color("none")

axis_object.xaxis.set_label("top")

print(coherence_vals.shape)
print(num_right_reaches[0,:].shape)
plt.scatter(coherence_vals, num_right_reaches_mean)
if not args.nofit:
    plt.plot(np.linspace(-0.2, 0.2, 100), ydataa, c='k', alpha=.5)

if context_choice != "":
    plt.title('in context')
else:
    plt.title("Psychometric Curve")
plt.ylim([-0.1, 1.1])


if str(type(task)) == "<class 'task.context.context_task'>":  # repeat analysis for out of context
    num_right_reaches = []
    num_right_reaches_com = []  
    num_right_reaches_tmp, num_right_reaches_com_tmp = TestCoherence(rnn, task, context_choice="out")
    num_right_reaches.append(num_right_reaches_tmp)
    num_right_reaches_com.append(num_right_reaches_com_tmp)
    
    # generate a figure to plot 
    num_right_reaches = np.array(num_right_reaches)
    num_right_reaches_com = np.array(num_right_reaches_com)
    num_right_reaches_mean = np.mean(num_right_reaches, 0)
    num_right_reaches_var = np.sqrt(np.var(num_right_reaches, 0))
    num_right_reaches_com_mean = np.mean(num_right_reaches_com, 0)
    num_right_reaches_com_var = np.sqrt(np.var(num_right_reaches_com, 0))
    
    def linear(x, m, b):
        y = m*x + b
        return (y)
    
    xdata = coherence_vals
    ydata = num_right_reaches_mean
    p0 = [max(ydata), np.median(xdata),1,min(ydata)] 
    popt, pcov = curve_fit(linear, xdata, ydata, method='dogbox')
    
    
    ydataa = linear(np.linspace(-0.2, 0.2, 100), *popt)
    
    fig_object = plt.figure(3)
    axis_object = fig_object.add_subplot(1,1,1)
    
    axis_object.spines["left"].set_position("center")
    axis_object.spines["bottom"].set_position("center")
    axis_object.spines["right"].set_color("none")
    axis_object.spines["top"].set_color("none")
    
    axis_object.xaxis.set_label("top")
    
    print(coherence_vals.shape)
    print(num_right_reaches[0,:].shape)
    plt.scatter(coherence_vals, num_right_reaches_mean)
    plt.plot(np.linspace(-0.2, 0.2, 100), ydataa, c='k', alpha=.5)
    plt.title('out of context')
    plt.ylim([-0.1, 1.1])


'''
# choice of context to test
for context_choice in range(2):
	# we will test the network under these values of coherence for the
	# simulated random dots motion 
    num_right_reaches = TestCoherence(rnn, task, context_choice=context_choice)
    context_out = ( context_choice+1 ) %2
    num_right_reaches_out = TestCoherence(rnn, task, context_out)

    # plot results in a 4 pannel figure for context RNN        
    i = int(2*context_choice)+1
    plt.subplot(2,2,i)
    plt.plot(coherence_vals, num_right_reaches)
    plt.title('Psychometrics Context'+ str(context_choice))
    plt.xlabel('Coherence')
    plt.ylabel('Fraction of Reaches to the Right (+1 output)')
    plt.ylim([-.15, 1.15])
	
    plt.subplot(2,2,i+1)
    plt.plot(coherence_vals, num_right_reaches_out)
    plt.title('Psychometrics Out of Context' + str(context_choice))
    plt.xlabel('Coherence')
    plt.ylabel('Fraction of Reaches to the Right (+1 output)')
    plt.ylim([-.15,1.15])
'''
plt.show()
