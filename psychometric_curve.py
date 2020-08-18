'''
Author: Brandon McMahan
Date: September 24, 2019
Will produce psychometric curves for RNN models
trained on the context task. Want
'''
import numpy as np 
import torch 
from rnn import RNN, loadRNN
from task.contexttask import ContextTask
from task.williams import Williams
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
            print(network_decisions[_])
        elif torch.abs(network_decisions[_]) < 1-tol:
            # omit trials where a clear reach was not made
            print('reach omitted...\n')
            num_reaches -= 1
        else:
            print(" "*10, network_decisions[_])
    #print('num reaches', num_reaches)
    if num_reaches == 0:
        fraction_right = 0.5
    else:
        fraction_right = total_right / num_reaches
        
    return fraction_right
d=0


# def COM_Flag(rnn_output):
#  	'''updated to use new algorithm'''
#  	# first compute velocity of rnn outputs
#  	output_velocity = rnn_output[1:,:] - rnn_output[:-1,:]
#  	#threshold velocities so small velocities are zero
#  	thresh = .15
#  	output_velocity[torch.abs(output_velocity)<thresh] = 0
#  	output_velocity[output_velocity>thresh] = 1
#  	output_velocity[output_velocity<-thresh] = -1
#  	trial_maxs, _ = torch.max(output_velocity, axis=0)
#  	trial_mins, _ = torch.min(output_velocity, axis=0)
#  	trial_ranges = trial_maxs - trial_mins
#  	vacillation_trials = torch.where(trial_ranges>1)[0]
#  	print('number of vacillations', len(vacillation_trials))
#  	return vacillation_trials


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

'''

    # get the first non-zero output sign
    t=0
    initial_sign = 0
    while (initial_sign == 0 and t<40):
        initial_sign = signed_output[t]
        t += 1
        
    #print('initial sign', initial_sign)
    for t_step in range(len(signed_output)):
        if initial_sign == 1 and signed_output[t_step] == -1:
            # a vacillation occured !
            return True
        elif initial_sign == -1 and signed_output[t_step] == 1:
            # a vacillation occured !
            return True
    # if we finish the loop then there must not have been any vacillations so 
    # return false
    return False'''


def TestCoherence(rnn, task, context_choice=-1):
    global coherence_vals
    num_trials = 2_000
    coherence_vals = np.array([-0.009, -0.09, -0.036, -0.15, 0.009, 0.036, 0.09, 0.15])#np.linspace(-.2,.2,6)
    # will hold thenumber of rightward reaches for each coherence level
    num_right_reaches = []
    num_right_reaches_com = []
    # will hold the number of rightward reaches for coherence level in wrong context
    num_right_reaches_wrong = []
    task_data = torch.zeros(750, num_trials).cuda()
    
    for _, coherence in enumerate(coherence_vals):
        print('\n\ncoherence:', coherence)
        # generate trials for this coherence value
        for trial_num in range(num_trials):
            task_data[:,trial_num] = task.PsychoTest(coherence).t()
        print('shape of task data', task_data.shape)
        # will hold network decisions for each trial
        #network_decisions = []
        network_decisions = torch.zeros(num_trials,1).cuda()
        #network_com_decisions = torch....
        # want to run the network 100 times under this level of coherence
        #for trial_num in range(200):
            # generate some input data
            #if context_choice == -1:
                # this is 1D task
                #task_data[:] = task.PsychoTest(coherence)
            #else:
                # this is context task
                #task_data[:] = task.PsychoTest(coherence, context_choice)
            #task_data_out_of_context = task.PsychoTestOut(coherence, context_choice)
            # now feed this data to the network and get the output
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


#load in model
#rnn = RNN(4,50,1)
#rnn.load('models/bptt_context_model0')
#task = ContextTask()
model_name = 'models/bptt_090'
rnn = loadRNN(model_name)
num_right_reaches = []
num_right_reaches_com = []  
num_models = 1
#for model_num in range(9,10):
print('evaluating model #', model_name)
rnn.load(model_name)
task = Williams(N=750, variance=0.75)
num_right_reaches_tmp, num_right_reaches_com_tmp = TestCoherence(rnn, task)
num_right_reaches.append(num_right_reaches_tmp)
num_right_reaches_com.append(num_right_reaches_com_tmp)

# generate a figure to plot 
num_right_reaches = np.array(num_right_reaches)
num_right_reaches_com = np.array(num_right_reaches_com)
num_right_reaches_mean = np.mean(num_right_reaches, 0)
num_right_reaches_var = np.sqrt(np.var(num_right_reaches, 0))/np.sqrt(num_models-1)
num_right_reaches_com_mean = np.mean(num_right_reaches_com, 0)
num_right_reaches_com_var = np.sqrt(np.var(num_right_reaches_com, 0))/np.sqrt(num_models-1)


# fit a sigmoid curve to the data
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

xdata = coherence_vals
ydata = num_right_reaches_mean
p0 = [max(ydata), np.median(xdata),1,min(ydata)] 
popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')


ydataa = sigmoid(np.linspace(-0.2, 0.2, 100), *popt)
plt.figure(3)
print(coherence_vals.shape)
print(num_right_reaches[0,:].shape)
plt.scatter(coherence_vals, num_right_reaches_mean)
plt.plot(np.linspace(-0.2, 0.2, 100), ydataa, c='k', alpha=.5)
#plt.plot(coherence_vals, num_right_reaches_com_mean)
#plt.fill_between(coherence_vals, num_right_reaches_mean-num_right_reaches_var, num_right_reaches_mean+num_right_reaches_var, alpha=.5)
#plt.fill_between(coherence_vals, num_right_reaches_com_mean-num_right_reaches_com_var, num_right_reaches_com_mean+num_right_reaches_com_var, alpha=.5)
plt.title('Psychometric')
plt.xlabel('Coherence')
plt.ylabel('Fraction of Reaches to the Right (+1 output)')
#plt.legend(['all trials', 'COM trials'])
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
