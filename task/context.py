# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:05:52 2020

@author: bmcma
context dependent integration task
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class context_task():
    def __init__(self, N=750, mean=0.5, var=1):
        self.N = N
        self._mean = mean
        self._var = var
        
    def GetInput(self, mean_overide=1):
        '''
        

        Parameters
        ----------
        mean_overide : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        inpts : PyTorch CUDA Tensor
            DESCRIPTION.
        target : TYPE
            DESCRIPTION.

        '''
        inpts = torch.zeros((self.N, 4)).cuda()
        
        # randomly generates context 1
        if torch.rand(1).item() < 0.5:
            inpts[:,0] = self._mean*torch.ones(self.N)
        else:
            inpts[:,0] = -self._mean*torch.ones(self.N)
            
        # randomly generates context 2
        if torch.rand(1).item() < 0.5:
            inpts[:,1] = self._mean*torch.ones(self.N)
        else:
            inpts[:,1] = -self._mean*torch.ones(self.N)
            
        # randomly sets GO cue
        if torch.rand(1).item() > 0.5:
            inpts[:, 2] = 1
            target = torch.sign(torch.mean(inpts[:,0]))
        else:
            inpts[:,3] = 1
            target = torch.sign(torch.mean(inpts[:,1]))
        
        # adds noise to inputs
        inpts[:,:2] += self._var*torch.randn(750, 2).cuda()
        
        return inpts, target
    
    def Loss(self, y, target, errorTrigger=-1):
        if (errorTrigger != -1):
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            yt = y[-1]
        ys = y[0]
        if type(y) is np.ndarray:
            return (yt-np.sign(mu))**2
        else:
            # use loss from Mante 2013
            squareLoss = (yt-torch.sign(target.T))**2 + (ys - 0)**2
            meanSquareLoss = torch.sum( squareLoss, axis=0 )
            return meanSquareLoss


if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    from rnn import loadRNN, RNN
    
    model = loadRNN("bptt_100")
    
    
    task = context_task()
    inpts, target = task.GetInput()
    model_output = model.feed(torch.unsqueeze(inpts.t(),0))
    inpts = inpts.cpu().detach().numpy()
    print("target:", target.item())
    plt.figure(1)
    plt.plot(inpts[:,0])
    plt.plot(inpts[:,1])
    plt.legend(["Context 1", "Context 2"])
    
    
    plt.figure(2)
    plt.plot(inpts[:,2])
    plt.plot(inpts[:,3])
    plt.legend(["Go 1", "Go 2"])
    
    plt.figure()
    plt.plot(model_output.detach().cpu().numpy())