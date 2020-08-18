'''
implementation of fixed point analysis

Author: Brandon McMahan
Date: May 30, 2019
'''

import numpy as np
import tensorflow as tf
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

import numpy as np 

def GetUnique(points, tol=1e-3, verbose=False):
    '''find all unique points in a set of noisy points'''
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


def IsAttractor(fixed_point, F, NumSimulations=2500):
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

def GetDerivatives(F):
    '''
    returns first order and second order derivatives for 
    F w.r.t. x as callable functions
    '''
    def YPrime(x, verbose=False):
        '''returns first and second order derivatives for F w.r.t. x'''
        x = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float64)
        with tf.GradientTape() as t:
            t.watch(x)
            #y = F(x)
            y = tf.norm(F(x))
            y = tf.pow(y, 2)
            y *= 0.5
            dy_dx = t.gradient(y, x)
        with tf.Session():
            if verbose:
                print('\nFirst derivative:\n', dy_dx.eval(), '\n\n')
            return dy_dx.eval()

    def Y2Prime(x, verbose=False):
        '''returns first and second order derivatives for F w.r.t. x'''
        x = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float64)
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                y = tf.norm(F(x))
                y = tf.pow(y, 2)
                y *= 0.5
                dy_dx = t1.gradient(y, x)
            d2y_dx2 = t2.gradient(dy_dx, x)
        with tf.Session():
            if verbose:
                print('\nSecond derivative:\n', d2y_dx2.eval(), '\n\n')
            return d2y_dx2.eval()

    return YPrime, Y2Prime
#end GetDerivatives

def FindZeros(F, num_iters=10_000, visualize=True, num_hidden=50):
    '''
    FindZeros takes a function F and will search for zeros
    using randomly generated initial conditions
    '''
    roots_found = np.empty((num_iters, num_hidden))
    #roots_found[:, :] = np.nan
    print('\n\nSearching for zeros...\n')
    for _ in range(num_iters):
        #random activations on U[-1,1]
        
        x0 = 2*(np.random.rand(num_hidden)-0.5)
        sol = root(F, x0)
        print(sol.success)
        #if ier == 1:
        #    print(curr_root)
        #time.sleep(1)
        #if LA.norm(curr_root) != 0:
        #    curr_root /= LA.norm(curr_root)
        #roots_found[_, :] = curr_root

    #if visualize:
    #    print('\n\nPerforming 2-D t-SNE for data visualization...\n')
    #    zeros_embedded = TSNE().fit_transform(roots_found)
    #    plt.figure()
    #    plt.scatter(zeros_embedded[:, 0], zeros_embedded[:, 1])
        

    return roots_found
#end FindZeros



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
        x0 = 1_000*(np.random.rand(num_hidden,1)-0.5)
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

def FindFixedPoints(master_function, inpts, just_get_roots=False, just_get_fraction=False, embedding='PCA', model_name='', embedder=[], num_hidden=50, new_fig=True, alpha=1, Verbose=True):
    '''
    functions is a list of functions for which we desire to find the roots
    most likley, each function in the list corresponds to a recurrent neural
    network update function, (dx/dt) = F(x), under a different input condition
    '''
    #master_function = model.GetF()
    #num_roots = np.zeros((len(inpts), len(inpts[0])+1))    #holds the number of roots under each condition found
    inpts = np.array(inpts)
    num_roots = np.zeros((len(inpts), 1+len(inpts[0])))                    #use this line for non-context tasks
    functions = []
    #each element of roots will be a numpy array of roots for the corresponding function
    #in functions
    roots = []
    # roots_ is a dictionary that will hold the coordinates of fixed points in R^50
    roots_ = {}
    #loop over input condition to create a unique function from the master function
    for _ in range(len(inpts)):
        print(inpts[_])
        functions.append(master_function(inpts[_]))
    #find the roots of each desired function
    num_deleted = 0        #will keep track of number of inputs that failed to converge 
    print('\nSEARCHING FOR ZEROS ... ')
    labels = []
    for _ in range(len(inpts)):
        num_roots[_,:len(inpts[0])] = inpts[_]    #update summary table
        #update the console with progress bar
        if True:
            sys.stdout.write('\r')
            sys.stdout.write('[%-19s] %.2f%% ' %('='*_, 5.26*_))
            #sys.stdout.write('INPUT = %f' % (inpts[_]))
            sys.stdout.flush()
        roots.append(GetUnique(FindZeros2(functions[_], visualize=False, num_hidden=num_hidden)))
        # update our dictionary with the roots we found
        roots_[str(inpts[_])] = roots[-1]
        curr_roots = roots[-1]
        #print('input to network:', inpts[_])
        # I am not sure I am usin gthis and it looks computationally expensive
        if False:
            if np.all(inpts[_][:] >= 0):
                if (np.all(inpts[_] == 0.08)) or (np.all(inpts[_] == 0.06)) or (np.all(inpts[_] == 0.04)):
                    potentials, success = PlotPotential(curr_roots, functions[_])
                    if success:
                        labels.append(inpts[_])
                        print('potentials calculated:', success)
        
        #time.sleep(5)
        #if no roots where found delete this element from list
        #print('empty roots:', roots[_])
        if isinstance(roots[_-num_deleted], int):
            del roots[_-num_deleted]
            num_deleted += 1
            num_roots[_,-1] = 0
        else:
            num_roots[_,-1] = len(roots[_-num_deleted])
        #print('last root found for input:', inpts[_], '\n', roots[-1][0])
    plt.legend(labels)
    #concatenate all the roots into one numpy array
    print('\nSummary\n')
    print('Input condition         |            Roots Found\n')
    for _ in range(len(num_roots)):
        print(num_roots[_,:])
    if just_get_fraction:
        # get a column vector containing the number of times each root appears
        num_of_pts = num_roots[:,-1]
        total_num_roots = len(num_of_pts)
        print(np.where(num_of_pts==3))
        num_three_pts = len(np.where(num_of_pts==3)[0])
        frac_three = num_three_pts / total_num_roots
        return frac_three
    all_roots = np.concatenate((roots[0], roots[1]))
    idxs = [len(roots[0]), len(roots[1])]
    for _ in range(2, len(roots)):
        #concatenation causes runtime error if roots[_] is of length zero
        #b/c no roots were found. I need to program an exception in to 
        #handle this situation
        try:
            all_roots = np.concatenate((all_roots, roots[_]))
        except ValueError:
            continue
        #keep a list detailing how many roots for each function was found
        #in order to delinate input conditions for plotting
        idxs.append(len(roots[_]))

    # return the roots if we don't want any further analysis done
    if just_get_roots:
        return all_roots, num_roots

    # embed all the roots found with a t-SNE (or LLE)
    # prior to embedding we want to center points around their mean
    roots_centered = all_roots-np.mean(all_roots, axis=0).reshape(1, 50)
    assert(roots_centered.shape == all_roots.shape)
    
    print('-'*50)
    if embedding == 't-SNE':
        print('\nusing t-SNE to visualize roots ...')
        roots_embedded = TSNE().fit_transform(roots_centered)
    elif embedding == 'LLE':
        print('\nusingsing Local Linear Embedding to visualize roots ...')
        roots_embedded = LLE().fit_transform(roots_centered)
    elif embedding == 'custom':
        print('\nusing custom embedding to visualize roots ...')
        pca = embedder
        roots_embedded = pca.transform(roots_centered)
    else: 
        print('\nusing PCA to visualize roots ...')
        pca = PCA()
        roots_embedded=pca.fit_transform(roots_centered)
        pca.offset_ = np.mean(all_roots, axis=0).reshape(1,50)
        print(pca.explained_variance_ratio_)


    if new_fig:
        plt.figure()
    start_idx = 0
    stop_idx = 0
    '''
    colors = ['black', 'grey', 'rosybrown', 'maroon', 'mistyrose', 'sienna',\
              'sandybrown', 'darkseagreen', 'lawngreen', 'darkolivegreen', \
              'yellow', 'gold', 'orange', 'lightseagreen', 'aqua', 'deepskyblue',\
              'midnightblue', 'darkslateblue', 'indigo', 'deeppink']
    '''
    '''
    colors = [(1, 0, 0), (0.9, 0, 0), (0.8, 0, 0), (0.7, 0, 0), (0.6, 0, 0), (0.5, 0, 0),\
                (0.5, 0, 0.1), (0.5, 0, 0.2), (0.5, 0, 0.3), (0.5, 0, 0.4), (0.5, 0, 0.5), (0.4, 0, 0.5), (0.3, 0, 0.5),\
                (0.2, 0, 0.5), (0.1, 0, 0.5), (0, 0, 0.5), (0, 0, 0.6), (0, 0, 0.7), (0, 0, 0.8),\
                (0, 0, 0.9), (0, 0, 1)]'''
    # use these colors when plotting -1 to +1 with 20 fixed points
    colors = [[[1, 0, 0]], [[0.9, 0, 0]], [[0.8, 0, 0]], [[0.7, 0, 0]], [[0.6, 0, 0]], [[0.5, 0, 0]],\
                [[0.5, 0, 0.1]], [[0.5, 0, 0.2]], [[0.5, 0, 0.3]], [[0.5, 0, 0.4]], [[0.5, 0, 0.5]], [[0.4, 0, 0.5]], [[0.3, 0, 0.5]],\
                [[0.2, 0, 0.5]], [[0.1, 0, 0.5]], [[0, 0, 0.5]], [[0, 0, 0.6]], [[0, 0, 0.7]], [[0, 0, 0.8]],\
                [[0, 0, 0.9]], [[0, 0, 1]]]
    '''
    colors = ['r','m','b']
    '''
    model_name += ' model'
    #xmin=2*np.min(roots_embedded[:,0])
    #xmax=10*np.max(roots_embedded[:,0])
    #ymin=5*np.min(roots_embedded[:,1])
    #ymax=5*np.max(roots_embedded[:,1])
    #plt.xlim([xmin, xmax])
    #plt.ylim([ymin, ymax])
    for _ in range(len(roots)):
        stop_idx += idxs[_]
        #this code terifies me for how hacky it is
        curr_set_of_roots = all_roots[start_idx:stop_idx]
        curr_set_of_roots_x = roots_embedded[start_idx:stop_idx,0]
        curr_set_of_roots_y = roots_embedded[start_idx:stop_idx,1]
        for curr_point in range(len(curr_set_of_roots)):
            if IsAttractor(curr_set_of_roots[curr_point], functions[_]):
                plt.scatter(curr_set_of_roots_x[curr_point], curr_set_of_roots_y[curr_point], c=colors[_], alpha=alpha)
            else:
                plt.scatter(curr_set_of_roots_x[curr_point], curr_set_of_roots_y[curr_point], marker='x', c=colors[_], alpha=alpha)
        '''

        if IsAttractor(all_roots[start_idx], functions[_]):        #this line of code scares me
            plt.scatter(roots_embedded[start_idx:stop_idx,0], roots_embedded[start_idx:stop_idx,1], c=colors[_])
        else:
            plt.scatter(roots_embedded[start_idx:stop_idx,0], roots_embedded[start_idx:stop_idx,1], facecolors='none', edgecolors=colors[_])
        '''
        start_idx += idxs[_]
        #plot labels
        plt.title('All Roots for '+model_name+' Using '+embedding)
        #plt.legend(['Fixed Points (input='+str(inpts[_])+')' for _ in range(len(inpts))])
        #plt.savefig('fixed_points.eps')
    if not Verbose:
        return roots_, idxs, pca #replaced all_roots with roots_
    else:
        return roots_embedded, pca

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


def FindFixedPointsC(master_function, inpts, embedding='PCA', model_name='', embedder=[], num_hidden=50, new_fig=True, alpha=1):
    '''
    functions is a list of functions for which we desire to find the roots
    most likley, each function in the list corresponds to a recurrent neural
    network update function, (dx/dt) = F(x), under a different input condition
    '''
    num_roots = np.zeros((len(inpts), len(inpts[0])+1))    #holds the number of roots under each condition found
    functions = []
    #each element of roots will be a numpy array of roots for the corresponding function
    #in functions
    roots = []
    #loop over input condition to create a unique function from the master function
    for _ in range(len(inpts)):
        functions.append(master_function(inpts[_]))
    #find the roots of each desired function
    num_deleted = 0        #will keep track of number of inputs that failed to converge 
    print('\nSEARCHING FOR ZEROS ... ')
    labels = []
    for _ in range(len(inpts)):
        num_roots[_,:len(inpts[0])] = inpts[_]    #update summary table
        #update the console with progress bar
        if True:
            sys.stdout.write('\r')
            sys.stdout.write('[%-19s] %.2f%%' %('='*_, 5.26*_))
            #sys.stdout.write('INPUT = %f' % (inpts[_]))
            sys.stdout.flush()
        roots.append(GetUnique(FindZeros2(functions[_], visualize=False, num_hidden=num_hidden)))
        curr_roots = roots[-1]
        print('input to network:', inpts[_])
        if np.all(inpts[_] >= 0):
            if (np.all(inpts[_] == 0.1)) or (np.all(inpts[_] == 0.3)) or (np.all(inpts[_] == 0.6)):
                potentials, success = PlotPotential(curr_roots, functions[_])
                if success:
                    labels.append(inpts[_])
                    print('potentials calculated:', success)
        
        #time.sleep(5)
        #if no roots where found delete this element from list
        #print('empty roots:', roots[_])
        if isinstance(roots[_-num_deleted], int):
            del roots[_-num_deleted]
            num_deleted += 1
            num_roots[_,-1] = 0
        else:
            num_roots[_,-1] = len(roots[_-num_deleted])
        #print('last root found for input:', inpts[_], '\n', roots[-1][0])
    plt.legend(labels)
    #concatenate all the roots into one numpy array
    print('\nSummary\n')
    print('Input condition         |            Roots Found\n')
    for _ in range(len(num_roots)):
        print(num_roots[_,0], '       |            ', num_roots[_,1])
    all_roots = np.concatenate((roots[0], roots[1]))
    idxs = [len(roots[0]), len(roots[1])]
    for _ in range(2, len(roots)):
        #concatenation causes runtime error if roots[_] is of length zero
        #b/c no roots were found. I need to program an exception in to 
        #handle this situation
        try:
            all_roots = np.concatenate((all_roots, roots[_]))
        except ValueError:
            continue
        #keep a list detailing how many roots for each function was found
        #in order to delinate input conditions for plotting
        idxs.append(len(roots[_]))

    #embed all the roots found with a t-SNE (or LLE)
    if embedding == 't-SNE':
        print('\nusing t-SNE to visualize roots ...')
        roots_embedded = TSNE().fit_transform(all_roots)
    elif embedding == 'LLE':
        print('\nusingsing Local Linear Embedding to visualize roots ...')
        roots_embedded = LLE().fit_transform(all_roots)
    elif embedding == 'custom':
        print('\nusing custom embedding to visualize roots ...')
        pca = embedder
        roots_embedded = pca.transform(all_roots)
    else: 
        print('\nusing PCA to visualize roots ...')
        pca = PCA()
        roots_embedded=pca.fit_transform(all_roots)
        print(pca.explained_variance_ratio_)

    if new_fig:
        plt.figure()
    start_idx = 0
    stop_idx = 0
    '''
    colors = ['black', 'grey', 'rosybrown', 'maroon', 'mistyrose', 'sienna',\
              'sandybrown', 'darkseagreen', 'lawngreen', 'darkolivegreen', \
              'yellow', 'gold', 'orange', 'lightseagreen', 'aqua', 'deepskyblue',\
              'midnightblue', 'darkslateblue', 'indigo', 'deeppink']
    '''
    '''
    colors = [(1, 0, 0), (0.9, 0, 0), (0.8, 0, 0), (0.7, 0, 0), (0.6, 0, 0), (0.5, 0, 0),\
                (0.5, 0, 0.1), (0.5, 0, 0.2), (0.5, 0, 0.3), (0.5, 0, 0.4), (0.5, 0, 0.5), (0.4, 0, 0.5), (0.3, 0, 0.5),\
                (0.2, 0, 0.5), (0.1, 0, 0.5), (0, 0, 0.5), (0, 0, 0.6), (0, 0, 0.7), (0, 0, 0.8),\
                (0, 0, 0.9), (0, 0, 1)]'''
    colors = [[[1, 0, 0]], [[0.9, 0, 0]], [[0.8, 0, 0]], [[0.7, 0, 0]], [[0.6, 0, 0]], [[0.5, 0, 0]],\
                [[0.5, 0, 0.1]], [[0.5, 0, 0.2]], [[0.5, 0, 0.3]], [[0.5, 0, 0.4]], [[0.5, 0, 0.5]], [[0.4, 0, 0.5]], [[0.3, 0, 0.5]],\
                [[0.2, 0, 0.5]], [[0.1, 0, 0.5]], [[0, 0, 0.5]], [[0, 0, 0.6]], [[0, 0, 0.7]], [[0, 0, 0.8]],\
                [[0, 0, 0.9]], [[0, 0, 1]]]
    model_name += ' model'
    #xmin=2*np.min(roots_embedded[:,0])
    #xmax=10*np.max(roots_embedded[:,0])
    #ymin=5*np.min(roots_embedded[:,1])
    #ymax=5*np.max(roots_embedded[:,1])
    #plt.xlim([xmin, xmax])
    #plt.ylim([ymin, ymax])
    for _ in range(len(roots)):
        stop_idx += idxs[_]
        #this code terifies me for how hacky it is
        curr_set_of_roots = all_roots[start_idx:stop_idx]
        curr_set_of_roots_x = roots_embedded[start_idx:stop_idx,0]
        curr_set_of_roots_y = roots_embedded[start_idx:stop_idx,1]
        for curr_point in range(len(curr_set_of_roots)):
            if IsAttractor(curr_set_of_roots[curr_point], functions[_]):
                plt.scatter(roots_embedded[_,0], roots_embedded[_,1], c=colors[_], alpha=alpha)
            else:
                plt.scatter(roots_embedded[_,0], roots_embedded[_,1], facecolors='none', edgecolors=colors[_], alpha=alpha)
        '''

        if IsAttractor(all_roots[start_idx], functions[_]):        #this line of code scares me
            plt.scatter(roots_embedded[start_idx:stop_idx,0], roots_embedded[start_idx:stop_idx,1], c=colors[_])
        else:
            plt.scatter(roots_embedded[start_idx:stop_idx,0], roots_embedded[start_idx:stop_idx,1], facecolors='none', edgecolors=colors[_])
        '''
        start_idx += idxs[_]
        #plot labels
        plt.title('All Roots for '+model_name+' Using '+embedding)
        #plt.legend(['Fixed Points (input='+str(inpts[_])+')' for _ in range(len(inpts))])
        #plt.savefig('fixed_points.eps')

    return roots_embedded, pca


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