'''
implementation of fixed point analysis

Author: Brandon McMahan
Date: May 30, 2019
'''

import numpy as np
#import tensorflow as tf
from scipy.optimize import fsolve
from scipy.optimize import root
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
import matplotlib.pyplot as plt 
import numpy.linalg as LA
import time
from sklearn.decomposition import PCA
import sys
from AnhilationPotential import *            #CHANGE THIS 
import torch
import pdb
import numpy as np 

def GetUnique(points, tol=1e-3, verbose=False):
    '''find all unique points in a set of noisy points
    returns a list of unique fixed points'''
    #there must be at least one unique point
    unique_pts = [ [] ]        #list of lists (each list is a list of points corresponding to a unique root)
    clf_pts = []    #will hold indices of points that have already been grouped
    unclf_pts = []

    #first pass--point is classified identically to self
    # check if there were no points
    if isinstance(points, int):
        print('\nWarning! No points were found for this input level!!')
        return 0
    for idx in range(len(points)):
        pt1 = points[0]
        pt2 = points[idx]
        curr_distance = ComputeDistance(pt1, pt2)
        #if points are close enough together to be considered identical
        if curr_distance < tol:
            #store point idx in classified points
            clf_pts.append(idx)
            #if point not sufficiently far to be considered unique
            unique_pts[0].append(pt2)
        else:
            unclf_pts.append(idx)    #holds the indices of all points that are different from first point

    #keep classifying points untill all points have been classified
    while len(clf_pts) != len(points):
        unique_pts.append([])    #new sub-list of identical points
        unclf_pts_tmp = []        #reset to hold this round of unclassified pts
        pt1 = points[unclf_pts[0]]
        for idx in unclf_pts:
            pt2 = points[idx]
            curr_distance = ComputeDistance(pt1, pt2)
            if curr_distance < tol:
                unique_pts[-1].append(pt2)    #start at zero so pt2=pt1 --> pt1 is included if it is truly unique to all other points
                clf_pts.append(idx)
            else:
                unclf_pts_tmp.append(idx)
        unclf_pts = unclf_pts_tmp    #reset unclassified points
    if verbose:
        print('number unique:',len(unique_pts))

    for _ in range(len(unique_pts)):
        unique_pts[_] = np.mean(unique_pts[_], axis=0)    #want to average over rows to get one point

    return unique_pts

def ComputeDistance(point1, point2):
    '''Computes euclidean distance between two points'''
    point_dimension = len(point1)
    squared_distances = 0
    for idx in range(point_dimension):
        squared_distances += (point1[idx] - point2[idx])**2
    distance = np.sqrt(squared_distances)
    return distance


def IsAttractor(fixed_point, F, NumSimulations=25):   #NumSimulations=2500
    '''
    IsAttractor will determine if a fixed point is stable or unstable

    returns True if point is stable and False if point is unstable
    '''
    num_stable_iters = 0
    num_unstable_iters = 0

    #reformat fixed point
    original_shape = fixed_point.shape
    fixed_point = fixed_point.reshape(-1,1)

    for simulation in range(NumSimulations):
        epsilon = 10e-5 * np.random.randn(len(fixed_point), 1)                 #10e-3
        nearby_point = fixed_point+epsilon
        initial_distance = ComputeDistance(nearby_point, fixed_point)
        for iterator in range(100):
            nearby_point += F(nearby_point).reshape(-1,1)
        #end of iterations
        final_distance = ComputeDistance(nearby_point, fixed_point)
        if final_distance > initial_distance:
            num_unstable_iters += 1
            #print('iteration diverged...')
        else:
            num_stable_iters += 1
            #print('iteration converged...')
    #end of simulations
    if num_unstable_iters >= 1:
        return False
    else:
        return True

def FindZeros2(F, num_iters=100, visualize=True, num_hidden=50, inpts=False, Embedding='t-SNE', norm=False, debug=False):
    '''
    FindZeros takes a function F and will search for zeros
    using randomly generated initial conditions
    '''
    roots_found = []
    #print('\n\nSearching for zeros...\n')
    if debug:
        tag = []
    for _ in range(num_iters):
        #random activations on U[-1,1]
        x0 = 10_000*(np.random.rand(num_hidden,1)-0.5)
        # tolerance changed from 1e-8
        sol = root(F, x0, tol=1e-8)
        if sol.success == True:
            if norm:
                #if not a zero vector
                if LA.norm(sol.x) != 0:
                    curr_root = np.round( sol.x/LA.norm(sol.x), decimals=3 )
                #else don't take the norm (because it is a zero vector)
                else:
                    curr_root = np.round( sol.x, decimals=3 )
            else:
                curr_root = sol.x
            if debug:
                #first element of curr_root will mark its class (1, 0, or -1)
                tag.append(np.round(curr_root[0]))
            roots_found.append(curr_root)
            #print(curr_root)
            #print('root found!')
        #if ier == 1:
        #    print(curr_root)
        #time.sleep(1)
        #if LA.norm(curr_root) != 0:
        #    curr_root /= LA.norm(curr_root)
        #roots_found[_, :] = curr_root
    roots_found = np.array(roots_found)
    #print('roots found:', roots_found.shape)
    if len(roots_found) == 0:
        return 0
    if visualize:
        print('\n\nPerforming Local Linear Embedding for data visualization...\n')
        plt.figure()
        plt.title('t-SNE')
        zeros_embedded = TSNE().fit_transform(roots_found)
        if debug:
            for _ in range(len(tag)):
                if tag[_] == 1:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='r')
                elif tag[_] == 0:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='k')
                elif tag[_] == -1:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='b')
        else:
            plt.scatter(zeros_embedded[:, 0], zeros_embedded[:, 1])
        plt.figure()
        plt.title('Local Linear Embedding')
        zeros_embedded = LLE(n_neighbors=5).fit_transform(roots_found)
        if debug:
            for _ in range(len(tag)):
                if tag[_] == 1:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='r')
                elif tag[_] == 0:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='k')
                elif tag[_] == -1:
                    plt.scatter(zeros_embedded[_, 0], zeros_embedded[_, 1], c='b')
        else:
            plt.scatter(zeros_embedded[:, 0], zeros_embedded[:, 1])
    return roots_found
#end find zeros2


def updateStatusBar(progress_fraction):
    '''
    updates the status of a task and prints it to console

    Parameters
    ----------
    progress_fraction : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    sys.stdout.write('\r')
    sys.stdout.write('[%-19s] %.2f%% ' %('='*progress_fraction, 5.26*progress_fraction))
    sys.stdout.flush()
    
def concatenate_roots(roots, static_inpts, stability_flag):
    '''

    Parameters
    ----------
    roots : Nested list
        roots is a list of length num_static_inputs where each element is itself 
        a list that contains the models fixed points (as a NumPy array) for the 
        given input level.

    Returns
    -------
    concatenated_roots : NumPy array
        array with shape (total_num_roots, hidden_size+2) containing all the models 
        roots in state space. First column is a stability flag, second column is 
        the static_input value associated with each root.
    '''
    if len(roots) > 1:
        all_roots = np.concatenate((roots[0], roots[1]))
        idxs = [len(roots[0]), len(roots[1])]
    else:
        all_roots = np.array(roots[0])
        idxs = [len(roots[0])]
    for _ in range(2, len(roots)):
        try:
            all_roots = np.concatenate((all_roots, roots[_]))
        except ValueError:
            continue
        #keep a list detailing how many roots for each function was found
        #in order to delinate input conditions for plotting
        idxs.append(len(roots[_]))
        
    static_inpts = np.array(static_inpts).reshape(-1, 1)
    stability_flag = np.array(stability_flag).reshape(-1, 1)
    concatenated_roots = np.hstack((stability_flag, static_inpts, all_roots))
    return concatenated_roots

def cmap(static_inpt, max_inpt=0.1857):
    '''
    generates a color for plotting fixed point found under static_input. Colors 
    go from red (positive inputs) to blue (negative inputs)

    Parameters
    ----------
    static_inpt : float
        input value to network for which current fixed point was found.

    Returns
    -------
    list
        r,g,b color that should be used to plot current fixed points.

    '''
    
    # clamps input so RGGB values remain in range
    if static_inpt > max_inpt:
        static_inpt = max_inpt
    elif static_inpt < -max_inpt:
        static_inpt = -max_inpt

    m_r = 0.5 / max_inpt
    m_b = - 0.5 / max_inpt
    
    r = m_r * static_inpt + 0.5
    b = m_b * static_inpt + 0.5
    g = 0

    return [[r, g, b]]

def plotFixedPoints(roots_embedded):
    '''
    plots the embedded fixed points in two dimensions

    Parameters
    ----------
    roots_embedded : NumPy array
        contains the embedded roots along with stability flag (first column) and 
        static_input (second column) has shape (2+num_roots, hidden_size).

    Returns
    -------
    None.

    '''
    for _ in range(roots_embedded.shape[0]):
        if roots_embedded[_, 0] == 1:      # checks stability flag of current root
            #pdb.set_trace()
            plt.scatter(roots_embedded[_, 2], roots_embedded[_, 3], c=cmap(roots_embedded[_,1]), alpha=1)
        else:
            plt.scatter(roots_embedded[_, 2], roots_embedded[_, 3], marker='x', c=cmap(roots_embedded[_,1]), alpha=1)


def embed_fixed_points(all_roots, pca):
    roots_centered = all_roots[:, 2:]-0*np.mean(all_roots[:, 2:], axis=0).reshape(1, -1)
    roots_embedded = pca.transform(roots_centered)
    roots_embedded = np.hstack((all_roots[:,:2], roots_embedded))
    return roots_embedded


def FindFixedPoints(model, inpts, embedding='PCA', embedder=[], Verbose=True):
    '''
    
    finds the fixed points for model
    
    functions is a list of functions for which we desire to find the roots
    most likley, each function in the list corresponds to a recurrent neural
    network update function, (dx/dt) = F(x), under a different input condition

    Parameters
    ----------
    model : RNN object
        trained model for which we want to find the fixed points.
    inpts : list
        List of static inputs that will be used for finding fixed points, where
        each element is itself a list of length equal to the RNNs input dimension. 
    just_get_roots : bool, optional
        DESCRIPTION. The default is False.
    just_get_fraction : bool, optional
        DESCRIPTION. The default is False.
    embedding : TYPE, optional
        DESCRIPTION. The default is 'PCA'.
    model_name : TYPE, optional
        DESCRIPTION. The default is ''.
    embedder : TYPE, optional
        DESCRIPTION. The default is [].
    num_hidden : TYPE, optional
        DESCRIPTION. The default is 50.
    new_fig : TYPE, optional
        DESCRIPTION. The default is True.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    Verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    master_function = model.GetF()
    num_static_inpts = len(inpts)
    inpts = np.array(inpts)                                          # inpts is array with shape (num_static_inputs, input_dim)
    num_roots = np.zeros((len(inpts), 1+len(inpts[0])))              # use this line for non-context tasks
    
    functions = []         # holds RNN update functions under different static inputs
    for static_input in inpts:
        print(static_input)
        functions.append(master_function(static_input))
    
    roots = []       # each element of roots will be a numpy array of roots corresponding to a static input
    inpt_levels = []
    
    #find the roots of each desired function
    print('\nSEARCHING FOR ZEROS ... ')
    labels = []
    stability_flag = []                 # denotes stability of fixed point
    for IX, static_input in enumerate(inpts):                     # loop over static inputs
        num_roots[IX,:len(inpts[0])] = static_input               # update summary table
        
        updateStatusBar(IX)    # reports progress to console
        roots.append(GetUnique(FindZeros2(functions[IX], visualize=False, num_hidden=model._hiddenSize)))
        # update our dictionary with the roots we found
        curr_roots = roots[-1]
        num_roots[IX,-1] = len(roots[IX])
        for _ in range(len(roots[IX])):
            # appends static input value to roots
            if len(static_input) == 4:  # indicates context task network
                if (static_input[0] == 0 and static_input[1] == 0):   # both contexts are zero
                    inpt_levels.append(static_input[0])
                elif (static_input[0] == 0):                          # context 1 is zero
                    inpt_levels.append(static_input[1])
                elif (static_input[1] == 0):                          # context 2 is zero
                    inpt_levels.append(static_input[0])
                    
            else: # this is not the context task
                inpt_levels.append(static_input[0])
            if (IsAttractor(roots[-1][_], functions[IX])):
                stability_flag.append(1)
            else:
                stability_flag.append(0)
    # end loop over static inputs
    
    print('\nFixed Point Search Summary\n')
    print('Static Input   |    Roots Found\n')
    for _ in range(len(num_roots)):
        print("{0:.2f}      |      {1:.1f}".format(num_roots[_,0], num_roots[_,1]))
    
    all_roots = concatenate_roots(roots, inpt_levels, stability_flag)
    assert(all_roots.shape[0] == len(stability_flag))

    # embed all the roots found with a t-SNE (or LLE)
    # prior to embedding we want to center points around their mean
    return all_roots 

def numInputsWithThreeFixedPoints(roots):
    '''
    returns the number of inputs that resulted in the three fixed point mechanism

    Args:
        roots (dictionary): dictionary containing the roots for a model. Each key
            in the dictionary corresponds to an input value for which roots were
            found. Each value in the dictionary is a list of unique roots found for
            that input value.

    Returns:
        float: Fraction of tested inputs that yielded 3 unique fixed points.

    '''
    numInputs = 0.0
    inputsWithThree = 0.0
    for key in dict:
        if len(dict[key])==3:
            inputsWithThree += 1
    
    return inputsWithThree / numInputs


if __name__ == '__main__':
    '''below code used to debug above functions in this file'''
    #3x3 matrix with all 0 eigenvalues
    J = np.array([
        [0, -3, 8],
        [0, 0, -11],
        [0, 0, 0]])

    F = lambda x: np.matmul(J, x)
    #zeros are [1, 0, 0] and [0, 0, 0]
    #def master(J):
    #    ans = lambda x: np.matmul(J, x)
    #    return ans
    #g = master

    zeros = FindZeros2(F, num_hidden=3, num_iters=5_000, visualize=True, norm=True, debug=True)
    print(zeros[:20])
    #plt.figure()
    #plt.scatter(zeros[:,0], zeros[:,1])
    plt.show()
    #print(zeros)