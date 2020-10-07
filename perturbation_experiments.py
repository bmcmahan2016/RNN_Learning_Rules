import numpy as np 
import numpy.linalg as LA
import torch
from rnn import RNN, loadRNN 
 
import rnntools as r
import FP_Analysis as fp
import time
from task.williams import Williams
from task.context import context_task
from sklearn.decomposition import PCA
import json
import sys

import matplotlib.pyplot as plt

#####################################################
#SELECT WHAT ANALYSIS TO PERFORM
#####################################################
#model to analyze
model_choice = 'force' 

#fixed point analysis
fixed_points = True

#trajectory analysis
trajectories = True

#energy landscape analysis
energy_landscape = True

def a_dummy_func():
    x = 5
    x = 3
    x = 2
    return x


def plotPCTrajectories(trial_data, trial_labels, pca, average=True):
    '''
    plots the mean PC trajectories for positive and negative trials for an RNN
    trained on the RDM task.

    Returns:
        None.

    '''
    plot_traj_from = 0
    plot_traj_up2 = -1
    
    if not average:    # do not average trials
        for curr_trial in range(10):
            transformedData = pca.transform(trial_data[curr_trial]-pca.offset_)
            if trial_labels[curr_trial] > 0:
                pltC = 'r'
            else:
                pltC = 'b'
            plt.plot(transformedData[plot_traj_from:plot_traj_up2,0], transformedData[plot_traj_from:plot_traj_up2,1], alpha=0.2, c=pltC)

        minX = -10
        maxX = 10
        minY = -10
        maxY = 10
        
    else:              # plots trajectory averaged over pos/neg trials
        meanPlusTrajectory = np.zeros((750, 3))       # saves top 3 PC components for 750 timesteps
        for curr_trial in range(5):
            N = curr_trial+1
            a = 1/N
            b = 1 - a
            meanPlusTrajectory = a * meanPlusTrajectory + b * pca.transform(trial_data[curr_trial]-pca.offset_)[:,:3]
        plt.plot(meanPlusTrajectory[plot_traj_from:plot_traj_up2,0], meanPlusTrajectory[plot_traj_from:plot_traj_up2,1], alpha=0.5, c='r')
        
        meanNegTrajectory = np.zeros((750, 3))
        for curr_trial in range(5,10):
            N = curr_trial-4
            a = 1/N
            b = 1 - a
            meanNegTrajectory = a * meanNegTrajectory + b * pca.transform(trial_data[curr_trial]-pca.offset_)[:,:3]
        plt.plot(meanNegTrajectory[plot_traj_from:plot_traj_up2,0], meanNegTrajectory[plot_traj_from:plot_traj_up2,1], alpha=0.5, c='b')
        
        # set the limits for the fixed point plot
        minX = np.min( (np.min(meanPlusTrajectory[:,0]), np.min(meanNegTrajectory[:,0])) )
        maxX = np.max( (np.max(meanPlusTrajectory[:,0]), np.max(meanNegTrajectory[:,0])) )
        minY = np.min( (np.min(meanPlusTrajectory[:,1]), np.min(meanNegTrajectory[:,1])) )
        maxY = np.max( (np.max(meanPlusTrajectory[:,1]), np.max(meanNegTrajectory[:,1])) )
    windowScale = 1.5
    
    plt.xlim([windowScale*minX, windowScale*maxX])
    plt.ylim([windowScale*minY, windowScale*maxY])

def AnalyzeLesioned(model, fig_name, PC1_min=-10, PC1_max=10, PC2_min=-10, PC2_max=10, test_inpt=0.2):
    '''
    AnalyzedLesioned performs a standard set of analysis on a fully trained RNN
    model and then plots the results as multiple figures. The following analyses 
    are carried out by AnalyzeLesioned and plotted:
        1. Tensor Component Analysis 
        2. Recurrent weight matrix is plotted (sorted by neuron factor)
        3. output of network in response to sample inputs is plotted
        4. the fixed points of the network are plotted

    Args:
        model (rnn object): This is an RNN object that has already been trained
        fig_name (string): I am unsure where this is being used
        PC1_min (TYPE, optional): DESCRIPTION. Defaults to -10.
        PC1_max (TYPE, optional): DESCRIPTION. Defaults to 10.
        PC2_min (TYPE, optional): DESCRIPTION. Defaults to -10.
        PC2_max (TYPE, optional): DESCRIPTION. Defaults to 10.
        test_inpt (int, optional): This will scale the test inputs. Defaults to 1.

    Returns:
        None.

     '''
    ###########################################################################
    #TENSOR COMPONENT ANALYSIS
    ###########################################################################
    activity_tensor = model._activityTensor
    neuron_factor = r.plotTCs(activity_tensor, model._targets, 1)
    neuron_idx = np.argsort(neuron_factor)     # sorted indices of artificial neurons
    
    # find the index where neuron_factors changes sign
    # p is the index that partitions neuron_idx into two clusters
    sign_change_idx = np.diff(np.sign(neuron_factor[neuron_idx]))
    if not np.all(sign_change_idx==0):
        # executes when neuron factors are not all the same sign
        last_pos_neuron = np.nonzero(sign_change_idx)[0][0]
        print('number of positive neurons:', last_pos_neuron)
        print('number of negative neurons:', len(neuron_factor)-last_pos_neuron)
        p = np.nonzero(np.diff(np.sign(neuron_factor[neuron_idx])))[0][0] + 1
    else:
        # executes when neuron factors all have the same sign
        print('artifical neurons all have sign', np.sign(neuron_factor[0]))
        p = 25
    
    
    ###########################################################################
    #VISUALIZE RECURRENT WEIGHTS
    ###########################################################################
    model._neuronIX = neuron_idx
    model.VisualizeWeightMatrix()
    model.VisualizeWeightClusters(neuron_idx, p)
     
    
    ###########################################################################
    # SAMPLE OUTPUTS
    ###########################################################################
    
    cs = ['r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b']
    trial_data, trial_labels = r.record(model, \
        title='fixed points', print_out=True, plot_recurrent=False, cs=cs)
    model._pca = PCA()
    model_trajectories = model._pca.fit_transform(trial_data.reshape(-1, 50)).reshape(10,-1,50)

    # find the fixed points for the model using the PC axis found on the niave model
    if True: #model._fixedPoints == []:
        #F = model.GetF()
        print("trial_data", trial_data)
        input_values = [[1],[.95],[.9],[.85],[.8],[.75],[.7],[.65],[.6],[.55],[.5],\
                        [.45],[.4],[.35],[.3],[.25],[.2],[.15],[.05],[0],[-.05],[-.1],\
                            [-.15],[-.2],[-.25],[-.3],[-.35],[-.4],[-.45],[-.5],[-.55],[-.6],[-.65],\
                            [-.7],[-.75],[-.8],[-.85],[-.9],[-.95],[-1]]
        input_values = [[0]]
        input_values = 3 * test_inpt * np.array(input_values)
        #input_values = [[.1],[0],[-.1]]
        roots = fp.FindFixedPoints(model, input_values)
        model.updateFixedPoints(roots, model._pca)      # fixed points now saved
        roots_embedded = fp.embed_fixed_points(roots, model._pca)
        plt.figure()
        fp.plotFixedPoints(roots_embedded)
        

    else:
        pca = model._pca
        roots = model._fixedPoints
    # partition PC space using Wout
    # W_out = model._J['out'].cpu().detach().numpy()
    # W_out_PC = pca.transform(W_out-pca.offset_)
    # PC1_axis = np.linspace(PC1_min, PC1_max, 100)
    # Wout_partition = -(W_out_PC[:, 0]/W_out_PC[:, 1])*PC1_axis
    # plt.plot(PC1_axis, Wout_partition, c='k', linestyle='dashed')
    
    # partition PC space using Win
    # W_in = model._J['in'].cpu().detach().numpy()
    # W_in_PC_pos = pca.transform(0.1857*W_in.T-pca.offset_)
    # W_in_PC_neg = pca.transform(-0.1857*W_in.T-pca.offset_)
    # W_in_PC = np.zeros((2,2))
    # W_in_PC[0,:] = W_in_PC_pos[0,:2]
    # W_in_PC[1,:] = W_in_PC_neg[0,:2]

    #plt.plot(np.linspace(0, W_in_PC[0,0], 100), np.linspace(0, W_in_PC[0,1], 100), 'r--', linewidth=3)
    #plt.plot(np.linspace(0, W_in_PC[1,0], 100), np.linspace(0, W_in_PC[1,1], 100), 'b--', linewidth=3)
    
    # this will plot trajectories of artificial neurons on top of fixed points
    #plotPCTrajectories(trial_data, trial_labels, pca, average=False)

    for i in range(10):
        plt.plot(model_trajectories[i,:,0], model_trajectories[i,:,1], c = cs[i], alpha=0.25)
    
    # plot the output of this lesioned network when feed a noisy input with mean +/-1
    plt.figure()
    plt.title('Evaluation of Model on 1D Decision-Making Task')
    task=Williams()
    r.TestTaskInputs(model, task)
    #plt.show()
    
    '''    This is not implemented yet
    ####
    #GET JACOBIAN
    ####
    jacobian = np.zeros((50, 50))
    
    hStars = model._fixedPoints["[0.02]"]
    plt.figure()
    cs = ["r", "b", "g"]
    for pointNum in range(3):
        jacobian = np.zeros((50, 50))
        hStar = hStars[pointNum].reshape(50,1)
        for j in range(50):
            mask = np.zeros((50,1))
            mask[j, 0] = 1
            jacobian[:,j] = model._dt*np.squeeze(np.matmul(model._J["rec"].detach().cpu().numpy(), 1-np.tanh(hStar*mask)**2)) + np.squeeze((1-model._dt)*mask)
        w, v = np.linalg.eig(jacobian)
        plt.bar(np.linspace(0,len(w),len(w))+0.2*pointNum, np.real(w), color=cs[pointNum])
    '''


def AnalyzePerturbedNetwork(model, model_name, test_inpt=1):
    # plot model losses
    model.plotLosses()
    plt.title('Training Loss')
    plt.figure()
    
    # peroform TCA on model data
    activity_tensor = model.activity_tensor
    neuron_factor = r.plotTCs(activity_tensor, model.targets, 1)
    neuron_idx = np.argsort(neuron_factor)
    # p is the index that partitions neuron_idx into two clusters
    sign_change_idx = np.diff(np.sign(neuron_factor[neuron_idx]))
    if not np.all(sign_change_idx==0):
        print('this shoudln\'t execute')
        p = np.nonzero(np.diff(np.sign(neuron_factor[neuron_idx])))[0][0] + 1
    else:
        p = 25
    
    #plot the sorted weight matrix
    plt.figure()
    model.VisualizeWeightMatrix(neuron_idx)
    plt.figure()
    model.VisualizeWeightClusters(neuron_idx, p)
    
    
    #GET TRAJECTORIES
    
    cs = ['r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b']
    trial_data = r.record(model, \
        test_inpt*np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]), \
        title=model_name, print_out=True, plot_recurrent=False, cs=cs, add_noise=False)
    
    '''This will plot 10 PSTHs
    for _ in range(len(trial_data)):
        plt.figure()
        r.plotMultiUnit(trial_data[_][:,neuron_idx], normalize_cols=True)
    #################################################################################################
    '''
    
    
    F = model.GetF()#func_master
    #roots, pca = FindFixedPoints(F, [0.1,-0.1], embedding='custom', embedder=model.pca)
    
    roots, idx, pca = FindFixedPoints(F, [[1],[0.9],[0.8],[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1],[0],\
                        [-0.1],[-0.2],[-0.3],[-0.4],[-0.5],[-0.6],[-0.7],[-0.8],[-0.9],[-1]], embedding='custom', embedder=model.pca, verbose=False)
    print('Variance explained by PCs:', pca.explained_variance_ratio_)
    
    W_out = model.J['out'].detach().numpy()
    W_out_PC = pca.transform(W_out)
    # the immediate next line can be uncommented to plot Wout in PC space as a vector at the origin
    #plt.quiver([0], [0], W_out_PC[:,0], W_out_PC[:,1], color='k')
    #plt.scatter(W_out_PC[:,0], W_out_PC[:,1], marker='+', s=20, c='k')
    # want to compute a seperating line from Wout in PC space
    PC1_axis = np.linspace(-10, 10, 100)
    Wout_partition = -(W_out_PC[:, 0]/W_out_PC[:, 1])*PC1_axis
    plt.plot(PC1_axis, Wout_partition, c='k', linestyle='dashed')
    
    # this will plot trajectories of artificial neurons on top of fixed points
    '''
    for curr_trial in range(len(trial_data)):
        curr_trajectory = pca.transform(trial_data[curr_trial]-np.mean(roots))
        plt.scatter(curr_trajectory[:,0], curr_trajectory[:,1], marker='x', s=3, alpha=0.3, c=cs[curr_trial])
    '''
    # saved the fixed point figure with trajectories
    
    
    plt.figure()
    plt.title('Evaluation of Model on 1D Decision Making Task')
    task=Williams()
    r.TestTaskInputs(model, task)
    plt.show()
    
    '''I uncomented the code to plot energy surfaces as I am not using it
    #####################################################
    #ENERGY SURFACE CODE
    #####################################################
    resolution=200
    #create the grid in PC space
    x_grid, y_grid = np.meshgrid(np.linspace(-20, 20, resolution), np.linspace(10,-10,resolution))
    landscapes = [np.zeros((resolution, resolution)) for i in range(6)]    #will hold the energy landscapes for 6 input conditions
    
    #we are interested in landscapes under the following input conditions
    energy_functions = [F([1]), F([0.6666]), F([0.3333]), F([-0.3333]), F([-0.6666]), F([-1])]
    
    x_comp = np.array(pca.components_[0])
    y_comp = np.array(pca.components_[1])
    #loop over each input
    for condition in range(len(energy_functions)):
        #print('Input Condition#', condition+1, '...')
        for x_idx in range(resolution):
            #start = time.time()
            for y_idx in range(resolution):
                #transform point into state space
                X = x_grid[x_idx, y_idx]*x_comp + y_grid[x_idx, y_idx]*y_comp #(50,)
                #evaluate derivative at this point
                landscapes[condition][x_idx, y_idx] = LA.norm(energy_functions[condition](X))
            #stop = time.time()
            #print('x index', x_idx,'(computed in',stop-start,'seconds)')
    
    plt.figure()
    inpts = [1, 0.6666, 0.3333, -0.3333, -0.6666, -1]
    for _ in range(6):
        plt.subplot(2,3,_+1)
        plt.contour(x_grid, y_grid, np.log(landscapes[_]), levels=20)
        #add trajectories for 0.3333 input
        #if _ == 2:
        #    for curr_trial in range(len(trial_data)):
        #        curr_trajectory = pca.transform(trial_data[curr_trial])
        #        plt.scatter(curr_trajectory[:,0], curr_trajectory[:,1], marker='x', s=3, alpha=0.5, c=cs[curr_trial])
        plt.title('When Input Is '+ str(inpts[_]))
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(x_grid, y_grid, landscape)
    #plt.title('When Input is 1')
                           #linewidth=0, antialiased=False)
    #Axes3D.plot_surface(x_grid, y_grid, landscape)
    plt.show()
    '''

def remove_inhibition(model_choice, xmin=-10, xmax=10, ymin=-10, ymax=10, test_inpt=1):
    #####################################################
    ############################
    #ANALYSIS CODE
    ############################
    #####################################################
    model = RNN(1, 50, 1)
    model.load(model_choice)
    model_choice+='_no_inhibition'
    
    #zero all negative weights
    W_rec = model.J['rec'].data.numpy()
    W_rec[W_rec<0]=0                    #remove all inhibitory connections
    W_in = model.J['in'].data
    model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)
    outputs, targets = r.TestTaskInputs(model, Williams())
    print('outputs', outputs.shape)
    print('targets', targets.shape)
    # compute the MSE
    network_final_decisions = outputs[:,-1].reshape(-1, 1)    # column vector
    # reshape conditions as column vector
    target_outs = targets.reshape(-1, 1)
    error = MeasureAccuracy(network_final_decisions, target_outs)
    print('Accuracy', error)
    
    return error
    AnalyzeLesioned(model, model_choice, xmin, xmax, ymin, ymax)

def MeasureAccuracy(network_choices, target_choices, tol=1):
    '''
    MeasureAccuracy takes as its input two arrays of shape (N,1)
    and will determine the accuracy of a model of a dataset. 
    Only the sign matters for this function, such that, an output
    of 0.2 when the desired output is +1 will be correct. Vice versa 
    -0.2 is correct when the target is -1
    
    PARAMETERS
    -network_choices: contains the final output of the network on
     N different trials
    -target_choices: contains the desired/target output for each of 
     N different trials
    -tol: specifies how close the network output must be to the target 
     output to be considered correct
    '''
    # determine number of trials in data
    num_trials = len(network_choices)
    # initialize a counter to keep track of how many outputs are correct
    num_correct = 0
    for curr_trial in range(num_trials):
        if np.abs(target_choices[curr_trial] - network_choices[curr_trial]) <= tol:
            num_correct += 1
    accuracy = num_correct / num_trials
    return accuracy

def lesion_control(model_choice, xmin=-10, xmax=10, ymin=-10, ymax=10, test_inpt=1):
    #####################################################
    ############################
    #ANALYSIS CODE
    ############################
    #####################################################
    model = RNN(1, 50, 1)
    model.load(model_choice)
    model_choice+='_no_excitation'
    
    # we don't wish to alter any weights for the control experiment
    outputs, targets = r.TestTaskInputs(model, Williams())
    print('outputs', outputs.shape)
    print('targets', targets.shape)
    # compute the MSE
    network_final_decisions = outputs[:,-1].reshape(-1, 1)    # column vector
    # reshape conditions as column vector
    target_outs = targets.reshape(-1, 1)
    error = MeasureAccuracy(network_final_decisions, target_outs)
    error = num_correct / total_num
    print('Accuracy', error)
    
    return error

def remove_excitation(model_choice, xmin=-10, xmax=10, ymin=-10, ymax=10, test_inpt=1):
    #####################################################
    ############################
    #ANALYSIS CODE
    ############################
    #####################################################
    model = RNN(1, 50, 1)
    model.load(model_choice)
    model_choice+='_no_excitation'
    
    #zero all negative weights
    W_rec = model.J['rec'].data.numpy()
    W_rec[W_rec>0]=0                    #remove all excitatory connections
    W_in = model.J['in'].data
    model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)
    outputs, targets = r.TestTaskInputs(model, Williams())
    print('outputs', outputs.shape)
    print('targets', targets.shape)
    # compute the MSE
    network_final_decisions = outputs[:,-1].reshape(-1, 1)    # column vector
    # reshape conditions as column vector
    target_outs = targets.reshape(-1, 1)    
    error = MeasureAccuracy(network_final_decisions, target_outs)
    print('Accuracy', error)
    
    return error
    AnalyzeLesioned(model, model_choice, xmin, xmax, ymin, ymax)

def niave_network(modelPath, xmin=-10, xmax=10, ymin=-10, ymax=10, test_inpt=.1):
    model = loadRNN(modelPath)
    if modelPath[7] == 'F':
        model._useForce = True
        print("Using force forward pass")
    #if modelPath[7].lower() == 'h':
    #    print("using hebbian RDM version")
    #    model._task._version = "Heb"
    modelPath+='_control'
    model._use_ReLU = True
    #AnalyzePerturbedNetwork(model, model_choice, test_inpt=test_inpt)
    AnalyzeLesioned(model, model_choice, xmin, xmax, ymin, ymax)

def GetNeuronIdx(model_choice):
    model = RNN(1,50,1)
    model.load(model_choice)
    activity_tensor = model.activity_tensor
    neuron_factor = r.plotTCs(activity_tensor, model.targets, 1)
    neuron_idx = np.argsort(neuron_factor)
    model.neuron_idx= neuron_idx
    model.save(model_choice)
    print('neuron indices saved !')

def QuantifyRecurrence(model_choice):
    model = RNN(1,50,1)
    model.load(model_choice)
    # create ideal bistable weight matrix
    w_bistable = torch.zeros(50,50)
    w_bistable[:25,:25] += 1
    w_bistable[25:,25:] += 1
    w_bistable[:25,25:] -= 1
    w_bistable[25:,:25] -= 1
    # normalize actual model weights
    w_actual = model.J['rec'].detach()
    w_actual /= torch.max(torch.abs(w_actual))
    # sort the neurons by their neuron factors
    if not model.neuron_idx is None:
        neuron_idx = model.neuron_idx
        w_ordered = w_actual[:,neuron_idx]
        w_ordered = w_ordered[neuron_idx, :]
        w_actual = w_ordered
        print(w_actual)
    #compute the difference between ideal and actual
    w_diff = w_bistable-w_actual
    w_diff *= w_diff
    structure_metric = torch.sum(w_diff)
    
    return structure_metric, w_diff


def PlotRecMag(model_choice):
    model = RNN(1, 50, 1)
    model.load(model_choice)
    ts = np.linspace(0,100,len(model.rec_magnitude))
    plt.plot(ts, model.rec_magnitude)
    #plt.ylim([0,0.3333])


def remove_lo_mag(model_choice, test_inpt=1):
    #####################################################
    ############################
    #ANALYSIS CODE
    ############################
    #####################################################
    model = RNN(1, 50, 1)
    model.load(model_choice)


    #zero all negative weights
    thresh=100
    W_rec = model.J['rec'].data.numpy()
    W_new = np.zeros(W_rec.shape)
    W_new[W_rec>thresh] = W_rec[W_rec>thresh]
    W_new[W_rec<-thresh] = W_rec[W_rec<-thresh]
    W_rec = W_new                    
    W_in = model.J['in'].data
    model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)
    AnalyzePerturbedNetwork(model, model_choice, test_inpt=test_inpt)

def remove_hi_mag(model_choice, test_inpt=1):
    #####################################################
    ############################
    #ANALYSIS CODE
    ############################
    #####################################################
    model = RNN(1, 50, 1)
    model.load(model_choice)


    #zero all negative weights
    W_rec = model.J['rec'].data.numpy()
    W_rec[W_rec>0.1] = 0
    W_rec[W_rec<-0.1] = 0                
    W_in = model.J['in'].data
    model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)
    AnalyzePerturbedNetwork(model, model_choice, test_inpt=test_inpt)

def clip_genetic(model_choice):
    model = RNN(1, 50, 1)
    model.load(model_choice)
    print('genetic model loaded ...')

    trialsXbatches = len(model.activity_tensor[:,0,0])
    num_neurons = len(model.activity_tensor[0,:,0])
    num_tsteps = len(model.activity_tensor[0,0,:])
    print('number of trials currently in model ', trialsXbatches)

    trials = trialsXbatches // 50
    print('number of trials that will remain after clipping ', trials)

    activity_tensor = np.zeros((trials, num_neurons, num_tsteps))
    targets = []
    for _ in range(trials):
        activity_tensor[_,:,:] = model.activity_tensor[50*_,:,:]
        targets.append(model.targets[50*_])

    # save new model as 997
    model.activity_tensor = activity_tensor
    model.targets = np.array(targets)
    model.save('genetic_model0')


def TestModel(model_choice):
    model = RNN(1, 50, 1)
    model.load(model_choice)
    task = Williams()
    r.TestTaskInputs(model, task)

def PlotRoots(all_roots, roots_embedded, idxs, colors, marker='o'):
    '''Plot roots color coded by input value'''
    start_idx = 0
    stop_idx = 0

    for _ in range(len(idxs)):
        stop_idx += idxs[_]
        # grab the raw roots corresponding to current input value
        curr_set_of_roots = all_roots[start_idx:stop_idx]
        # grab the first coordinate of these roots in PC space
        curr_set_of_roots_x = roots_embedded[start_idx:stop_idx,0]
        # grab the second coordinate of these roots in PC space
        curr_set_of_roots_y = roots_embedded[start_idx:stop_idx,1]
        # grab the third coordinate of these roots in PC space
        curr_set_of_roots_z = roots_embedded[start_idx:stop_idx,2]

        # loop over all roots corresponding to current input
        for curr_point in range(len(curr_set_of_roots)):
            # determine stability of current root
            #if IsAttractor(curr_set_of_roots[curr_point], functions[_]):
            plt.scatter(curr_set_of_roots_x[curr_point], curr_set_of_roots_y[curr_point], s=10, c=colors[_], marker=marker)
            #else:
            #plt.scatter(curr_set_of_roots_x[curr_point], curr_set_of_roots_y[curr_point], facecolors='none', edgecolors=colors[_], alpha=alpha)
        start_idx += idxs[_]

def ContextFixedPoints(model_choice):
    model = loadRNN(model_choice)
    #model.load(model_choice)
    task = context_task()

    model.plotLosses()

    '''
    model.plotLosses()
    plt.title('Training Loss')
    plt.figure()


    activity_tensor = model.activity_tensor
    neuron_factor = r.plotTCs(activity_tensor, model.targets, 1)
    neuron_idx = np.argsort(neuron_factor)
    #p is the index that partitions neuron_idx into two clusters
    p = np.nonzero(np.diff(np.sign(neuron_factor[neuron_idx])))[0][0] + 1
    #plot the sorted weight matrix
    plt.figure()
    model.VisualizeWeightMatrix(neuron_idx)
    plt.figure()
    model.VisualizeWeightClusters(neuron_idx, p)
    '''

    F = model.GetF()#func_master
    #roots, pca = FindFixedPoints(F, [0.1,-0.1], embedding='custom', embedder=model.pca

    context1_inpts = np.zeros((20, 4))
    context1_inpts[:,0] = np.linspace(-0.1857, 0.1857, 20)
    context1_inpts[:,2] = 1
    context1_inpts[:,1] = 0.1*np.random.randn(20)
    
    context2_inpts = np.zeros((20, 4))
    context2_inpts[:,1] = np.linspace(-0.1857, 0.1857, 20)
    context2_inpts[:,3] = 1
    context2_inpts[:,0] = 0.1*np.random.randn(20)
     

    # switch input to be the ignored context
    #inpts[:, 2] = 1
    #inpts[:, 3] = 0

    # add gaussian noise to ignored context
    #inpts[:, 0] = np.random.randn(19)-1

    context2_roots, context2_idxs, pca = FindFixedPoints(model, context2_inpts, embedding='pca', embedder=model._pca, Verbose=False)

    # create inputs for context 1 by transposing the first and second columns of inputs for 
    # context 2
    #context1_inpts = context2_inpts
    #context1_inpts[:,0] = context2_inpts[:,1]
    #context1_inpts[:,1] = 0

    print('Inputs for context 1 \n\n', context1_inpts)    # print to verify correct

    context1_roots, context1_idxs, pca = FindFixedPoints(model, context1_inpts, embedding='pca', embedder=model._pca, Verbose=False)

    print('\n'*5)
    print('shape of roots found for context 1: ', len(context1_roots))
    print('shape of roots found for context 2: ', len(context2_roots))

    # compute the number of roots in each context
    num_roots_context1 = len(context1_roots)
    num_roots_context2 = len(context2_roots)
    total_num_roots = num_roots_context1 + num_roots_context2

    # initialize an array to hold all the roots
    all_roots = np.zeros((total_num_roots, 50))

    # fill this array with the roots found in both contexts
    all_roots[:num_roots_context1, :] = context1_roots
    all_roots[num_roots_context1:, :] = context2_roots

    # perform PCA on these roots
    pca = PCA()
    roots_embedded=pca.fit_transform(all_roots)

    # save fixed points
    print('Saving fixed point data ...\n')
    fixed_point_data = {
    'context1_roots' : context1_roots.tolist(),
    'context2_roots' : context2_roots.tolist()
    }
    with open('bptt_context_data.json', 'w') as fp:
        json.dump(fixed_point_data, fp)
    print('Fixed point data succesfully saved! \n')

    # plot results in a figure
    colors = [[[1, 0, 0]], [[0.9, 0, 0]], [[0.8, 0, 0]], [[0.7, 0, 0]], [[0.6, 0, 0]], [[0.5, 0, 0]],\
            [[0.5, 0, 0.1]], [[0.5, 0, 0.2]], [[0.5, 0, 0.3]], [[0.5, 0, 0.4]], [[0.5, 0, 0.5]], [[0.4, 0, 0.5]], [[0.3, 0, 0.5]],\
            [[0.2, 0, 0.5]], [[0.1, 0, 0.5]], [[0, 0, 0.5]], [[0, 0, 0.6]], [[0, 0, 0.7]], [[0, 0, 0.8]],\
            [[0, 0, 0.9]], [[0, 0, 1]]]
    plt.figure()
    PlotRoots(context1_roots, roots_embedded[:num_roots_context1,:], context1_idxs, colors, marker='x')
    PlotRoots(context2_roots, roots_embedded[num_roots_context1:,:], context2_idxs, colors, marker='o')
    plt.show()
    # we want to add Wout to the plot
    W_out = model.J['out'].detach().numpy()
    W_out_PC = pca.transform(W_out)
    PC1_axis = np.linspace(-10, 10, 100)
    Wout_partition = -(W_out_PC[:, 0]/W_out_PC[:, 1])*PC1_axis
    plt.plot(PC1_axis, Wout_partition, c='k', linestyle='dashed')

    # add inputs to the graph
    # compute the forward pass for an input
    inpt1 = np.array([10,0,1,0])
    inpt2 = np.array([-10,0,1,0])
    inpt3 = np.array([0,10,1,0])
    inpt4 = np.array([0,-10,1,0])

    # compute the forward pass
    output, hidden1 = model.forward(inpt1, model.init_hidden(), 0.1)
    output, hidden2 = model.forward(inpt2, model.init_hidden(), 0.1)
    output, hidden3 = model.forward(inpt3, model.init_hidden(), 0.1)
    output, hidden4 = model.forward(inpt4, model.init_hidden(), 0.1)

    print('hidden1')
    print(hidden1.t().shape)
    # now transform hidden representations of input to PC space

    inpt1_vec = pca.transform(hidden1.t().detach().numpy())
    inpt2_vec = pca.transform(hidden2.t().detach().numpy())
    inpt3_vec = pca.transform(hidden3.t().detach().numpy())
    inpt4_vec = pca.transform(hidden4.t().detach().numpy())
    ax = plt.axes()
    print('vec1')
    print(inpt1_vec.shape)
    # plot vectors
    ax.arrow(0, 0, inpt1_vec[0][0], inpt1_vec[0][1], head_width=0.2, head_length=0.3, fc='grey', ec='grey')
    ax.arrow(0, 0, inpt2_vec[0][0], inpt2_vec[0][1], head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    ax.arrow(0, 0, inpt3_vec[0][0], inpt3_vec[0][1], head_width=0.2, head_length=0.3, fc='red', ec='red')
    ax.arrow(0, 0, inpt4_vec[0][0], inpt4_vec[0][1], head_width=0.2, head_length=0.3, fc='green', ec='green')

    plt.title('All Roots for Contextual Decision-Making Task')

def ComputeStepSize(model_choice):
    # load in the trained RNN model
    model = RNN(4, 50, 1)
    model.load(model_choice)
    task = ContextTask()

    # create an input to feed the network
    inpt1 = np.array([1, 0, 1, 0])        #go signal of 1 in signaled context (context 1)
    inpt2 = np.array([.1, 0, 1, 0])        #go signal of 0.1 in signaled context (context 1)
    inpt3 = np.array([-1, 0, 1, 0])        #go signal of -1 in signaled context (context 1)
    inpt4 = np.array([-.1, 0, 1, 0])    #go signal of -0.1 in signaled context (context 1)

    inpts = [inpt1, inpt2, inpt3, inpt4]

    # get the initial hidden state
    hidden_old = model.init_hidden()

    # initialize an empty list that will hold distances
    dX_dt_ = []

    # loop over all input conditions
    for inpt in inpts:
        dX_dt = []
        # loop over this input 40 times
        for _ in range(40):
            output, hidden_new=model.forward(inpt, hidden_old, 0.1)
            hidden_new=hidden_new.detach()
            # compute distance between old and new hidden layers
            dX_dt.append(ComputeDistance(hidden_old, hidden_new))
            # update the old hidden layer before updating the new hidden layer
            hidden_old = hidden_new
        # end current input condition
        dX_dt_.append(dX_dt)

    # plot results
    plt.figure()
    plt.plot(np.array(dX_dt_[0]))
    plt.plot(np.array(dX_dt_[1]))
    plt.plot(np.array(dX_dt_[2]))
    plt.plot(np.array(dX_dt_[3]))
    plt.legend(['input: 1', 'input: 0.1', 'input: -1', 'input: -0.1'])
    plt.ylabel('dX/dt')
    plt.xlabel('Time')

def PlotModelComplexities(models):
    # create a list that will hold the complexities for each model
    model_complexities = []
    # loop over each model
    for model_choice in models:
        model = RNN(1, 50, 1)
        model.load(model_choice)
        # get the PCA for the current model
        explained_var = model.pca.explained_variance_ratio_
        print(model_choice)
        print(explained_var)
        print('\n\n\n')
        total_var = 0
        num_comps_needed = 0
        # compute the number of components needed to account for 95% of variance
        while(total_var<=.99):
            num_comps_needed += 1
            total_var += explained_var[num_comps_needed-1]

        # append the results to the list of model complexities
        model_complexities.append(num_comps_needed)

    # plot the complexeties of each model
    model_complexities = np.array(model_complexities)
    plt.figure()
    plt.bar([1, 2, 3, 4], model_complexities, tick_label=['BPTT', 'Genetic', 'Hebbian', 'FORCE'])
    plt.ylabel('PCs Needed To Explain 99% of Variance')
    
def QuantifyVacillation(rnn_inst, task):
    vacillations = []
    vacillations_tmp = []
    
    # loop over different coherence levels to test the network
    coherence_levels = np.linspace(-1, 1, 50)
    for coherence in coherence_levels:
        # test the network several times
        for itr in range(100):
            # generate input conditions to feed the model
            inpt = task.PsychoTest(coherence)
            network_outputs = rnn_inst.feed(inpt)
            # count the number of vacillations in the network output
            signed_output = np.sign(network_outputs)
            sign_changes = np.diff(signed_output)
            sign_change_locations = np.nonzero(sign_changes)
            vacillations_tmp.append( (len(sign_change_locations[0])) )
        vacillations.append( np.mean(np.array(vacillations_tmp)) )
        vacillations_tmp = [] 
    return vacillations, coherence_levels
        

def QuantifyVascillation(model_choice, inpt_coef, lesion=0):
    '''
    QuantifyVascillation will count the number of times a network output changes 
    sign during the course of a single trial at test time

    PARAMETERS
    inpt_coef: will determine what is feed to the model to test vascillations
    lesion: determines if network should be lesioned prior to testing for 
        vascillations. Default value of 0 will not lesion network. Value 
        of +1 will remove all excitatory connections from Wrec. Value of 
        -1 will remove all inhibitory connections from Wrec. 

    RETURNS
    vascillations: The average number of vascillations performed by the network as 
    a function of input noise to the network. NumPy array with length equal to the 
    number of noise levels tests (should be 50)
    '''
    model = RNN(1,50,1)
    model.load(model_choice)

    # executes if we are testing an excitatory lesion
    if lesion == 1:
        # remove excitatory weights
        W_rec = model.J['rec'].data.numpy()
        W_rec[W_rec>0]=0                # zero all positive weights    
        model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)

    # executes if we are testing an inhibitory lesion
    if lesion == -1:
        # remove inhibitory weights
        W_rec = model.J['rec'].data.numpy()
        W_rec[W_rec<0]=0                # zero all positive weights    
        model.AssignWeights(model.J['in'].data, W_rec, model.J['out'].data)

    # will hold the number of vascillations
    vascillations = []
    vascillations_tmp = []    # will hold musltiple vascillations to be averaged for each trial

    # generate 100 input conditions to feed the model
    conditions = inpt_coef*np.ones(100)
    conditions[50:] *= -1

    # create noise levels
    noise_levels = np.linspace(0,300, 50)

    # loop over different noise levels
    for _ in range(len(noise_levels)):
        print(_)

        # test the network on 100 trials
        network_outputs = r.record(model, conditions,\
                    print_out=False, plot_recurrent=False, add_in_noise=noise_levels[_], only_out=True)

        # count the number of times the output changes sign for each trial
        for trial in range(len(network_outputs)):# loop over trial
            signed_output = np.sign(network_outputs[trial])
            sign_changes = np.diff(signed_output)
            sign_change_locations = np.nonzero(sign_changes)
            vascillations_tmp.append( len(sign_change_locations[0]) )
        # take the average numpber of vascillations for past 100 trials
        vascillations.append( np.mean(np.array(vascillations_tmp)) )
        # reset the temporary vascillations
        vascillations_tmp = []
    plt.plot(noise_levels/10, vascillations)
    plt.xlabel('Input Noise')
    plt.ylabel('Number of Vascillations')
    return vascillations, noise_levels/10

def GetModelRobustness(model_choice, inpt_coef=1, tol=0.1):
    '''
    Will determine the robustness of a model as the magnitude of inputs 
    are varied and/or noise is added to the model. The robustness here is
    defined as the proportion of outputs that are all-or-none. Furthermore, 
    noise may be added directly to the input values that are fed to then 
    network or the recurrent activations

    PARAMETERS

    '''
    model = RNN(1, 50, 1)
    model.load(model_choice)

    # will hold robustness scores
    robustness = []

    # generate 100 input conditions to feed the model
    conditions = inpt_coef*np.ones(100)
    conditions[50:] *= -1

    # test network at different noise levels
    noise_levels = np.linspace(0,100, 50)
    for _ in range(len(noise_levels)):

        # test this network
        network_outputs = r.record(model, conditions,\
                print_out=False, plot_recurrent=False, add_rec_noise=noise_levels[_], only_out=True)
        network_final_decisions = network_outputs[:,-1].reshape(-1, 1)    # column vector

        # compute the robustness
        robust_count = 0
        total_trials = len(network_final_decisions)
        for _ in range(total_trials):
            # is this near +/-1
            if 1 - np.abs(network_final_decisions[_]) < tol:
                robust_count += 1
        robustness.append( robust_count / total_trials )
    print('robustness:', robustness)
    accuracy = np.array(robustness)
    plt.plot(noise_levels/10, accuracy)
    plt.xlabel('Recurrent Noise')
    plt.ylabel('Network robustness')

def GetModelAccuracy(model_choice, noise, layer='input', inpt_coef=1, tol=1):
    '''
    Will determine the accuracy of the model in the presence of either 
    recurrent noise or input noise

    PARAMETERS
    model_choice: the choice of model to be analyzed. specified as 'model_name' 
            where model_name.pt exists in the current folder
    noise: level of noise to use for when evaluating network accuracy
    inpt_coef: will multiply test inputs to the network by this coefficient.
            by default, test inputs to the network are similar to that seen 
            during training: +/-1. Inputs to the network during noise testing 
            can be scaled with inpt_coef
    layer: specifies at what point in the network to add noise. May be either 
            'input' or 'recurrent'. Other values for this argument are not 
            accepted.
    '''
    # load in the desired model
    model = RNN(1, 50, 1)
    model.load(model_choice)

    # will hold MSE for network
    accuracy = []

    # generate 100 inputs to feed model
    conditions = inpt_coef*np.ones(100)
    # make the last half of conditions negative
    conditions[50:] *= -1

    # test network at different noise levels
    noise_levels = np.linspace(0,100, 50)
    for _ in range(len(noise_levels)):
        sys.stdout.write('\r')
        sys.stdout.write('[%-50s] %.2f%%' %('='*_, 2*_))
        # feed inptus to network and record outputs
        if layer == 'input':
            sys.stdout.write('   Adding input noise')
            network_outputs = r.record(model, conditions,\
            print_out=False, plot_recurrent=False, add_in_noise=noise_levels[_], only_out=True)
        elif layer == 'recurrent':
            sys.stdout.write('   Adding recurrent noise')
            network_outputs = r.record(model, conditions,\
            print_out=False, plot_recurrent=False, add_rec_noise=noise_levels[_], only_out=True)
        else:
            sys.stdout.write('   no noise added')
            network_outputs = r.record(model, conditions,\
            print_out=False, plot_recurrent=False, only_out=True)

        sys.stdout.flush()
        # now determine how accurate the model was
        network_final_decisions = network_outputs[:,-1].reshape(-1, 1)    # column vector
        # reshape conditions as column vector
        target_outs = conditions.reshape(-1, 1)
        accuracy.append( MeasureAccuracy(network_final_decisions, target_outs, tol=tol) )
    #print some new lines before moving on to next part of analysis
    #print('\n\n')

    accuracy = np.array(accuracy)
    plt.plot(noise_levels/10, accuracy)
    plt.xlabel('Recurrent Noise')
    plt.ylabel('Network Accuracy')

    return accuracy










