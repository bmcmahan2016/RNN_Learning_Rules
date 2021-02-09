# -*- coding: utf-8 -*-
"""
Created on Thursday December 10, 2020

@author: bmcma
MULTISENSORY INTEGRATION TASK (adapted from Song et al., 2016)

RNN is presented with two sources of information, both of which
should be integrated. 

This task is implemented with two input channels. One input channel
conveys visual information while the other input channel conveys 
auditory information. 

Input channels convey the rate of auditory or visual clicks. The 
network is tasked with determining if the combined rate was above a 
certain threshold

each input channel is presented at rates between 9events/sec and 16events/sect
threshold is set to 12.5 events/sec

When both inputs are presented they are at the same rate

therefore there are three cases:
CASE #1: only provide auditory information
CASE #2: only provide visual information
CASE #3: provide congruent auditory and visual information

"""

import torch
import numpy as np
import pdb


class multi_sensory():
    def __init__(self, N=750, peak=16, minimum=9, thresh=12.5, var=0.05):
        self.N = N
        
        if minimum <= 0:
            raise valueError("minimum must be above 0")
        self._min = minimum     # lowest event frequency  -- must be above zero
        
        if peak < minimum:
            raise valueError("peak must be above the minimum")
        self._peak = peak       # maximum event frequency -- must be above minimum
        self._var = var
        self._version = ""
        self._name = "multi"    # name of task type
        self._t_start = 200     # start of stimulus onset in trial
        self._thresh = thresh   # theshold for network output

        self.check_valid()      # checks constraints

    def check_valid(self):
        # checks that parameters are valid
        assert(self._peak > self._min)
        assert(self._min > 0)
        assert(self._t_start < self.N)
        
    def GetInput(self, mean_overide=-1, var_overide=1):
        '''Generates input for training an RNN on the multisensory integration task. 

        Parameters
        ----------
        mean_overide : float, optional
            This is a vestigal dummy argument without a purpose.

        Returns
        -------
        inpts : PyTorch CUDA Tensor
            2 channel input for training the network. Has shape (t_steps, 2)
        target : float64
            Network target outputs for current input, either zero or one. 

        '''
        inpts = torch.zeros((self.N, 2)).cuda()   # inputs are timesteps by 2 channels
        
        
        # randomly choose event frequency
        event_freqx_range = self._peak - self._min
        event_freqx = self._min + torch.rand(1).item()*event_freqx_range
        target = 2*np.floor(event_freqx/self._thresh)-1   # temporarily changed target range
        # normalize event frequency to be between zero and one
        event_freqx /= self._peak
        trial_duration = self.N - self._t_start 

        # randomly choose case
        trial_type = torch.rand(1).item()
        if trial_type < 1.0/3.0:     # only auditory
            inpts[self._t_start:,0] = event_freqx*torch.ones(trial_duration)
        elif trial_type < 2.0/3.0:   # only visual
            inpts[self._t_start:,1] = event_freqx*torch.ones(trial_duration)
        else:                        # congruent auditory and visual stimuli
            inpts[self._t_start:,0] = event_freqx*torch.ones(trial_duration)
            inpts[self._t_start:,1] = event_freqx*torch.ones(trial_duration)

        # normalize the noise and then add to the inputs
        inpts += (self._var/self._peak)*torch.randn(self.N, 2).cuda()
        
        return inpts, target

    def PsychoTest(self, coherence, context=0):
        '''
        Generates an input for generating a psychometric curve. There are three cases:
        case 1: only auditory   (first channel)
        case 2: only visual     (second channel)
        case 3: congruent       (both channels)
        '''

        assert(coherence >= 0)                    # coherence must be non-negative
        inpts = torch.zeros((self.N, 2)).cuda()   # inputs are timesteps by 2 channels
        
        
        # event frequency is equal to specified coherence
        event_freqx_range = self._peak - self._min
        event_freqx = self._min + coherence*event_freqx_range
        target = np.floor(event_freqx/self._thresh)
        event_freqx /= self._peak            # normalizes event freqx to (0,1]
        trial_duration = self.N - self._t_start 

        # choose channel specified by context
        if context == 0:   # auditory only
            inpts[self._t_start:,0] = event_freqx*torch.ones(trial_duration)
        elif context == 1: # visual only
            inpts[self._t_start:,1] = event_freqx*torch.ones(trial_duration)
        elif context == 2: # congruent stimulus
            inpts[self._t_start:,0] = event_freqx*torch.ones(trial_duration)
            inpts[self._t_start:,1] = event_freqx*torch.ones(trial_duration)
        else:
            raise NameError("This is not a supported context")

        # add normalized noise to the inputs
        inpts += (self._var/self._peak)*torch.randn(self.N, 2).cuda()

        assert(inpts.shape[1] == 2)   # check that we have two channels
        assert(inpts.shape[0] == self.N)   # check that trial is of correct length
        return inpts

    def Loss(self, y, target, errorTrigger=-1):
        '''
        network outputs are required to hold low prior to stimulus onset
        '''
        if (errorTrigger != -1):
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            y_stimulus = y[self._t_start:]              # output during stimulus
        y_fixation = y[:self._t_start]    # output prior to stimulus onset
        if type(y) is np.ndarray:
            assert False, "input to loss must be a PyTorch Tensor"
        else:
            #pdb.set_trace()
            #pdb.set_trace()
            # use loss from Mante 2013
            squareLoss = torch.sum((y_stimulus-torch.sign(target.T))**2, axis=0) + torch.sum((y_fixation - 0)**2, axis=0)
            meanSquareLoss = torch.sum( squareLoss, axis=-1 ) / 500   # this assumes a batch size of 500

            #print("MSE:", meanSquareLoss.item())
            return meanSquareLoss

# construct

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pdb
    print("Multisensory Integration Task Called Directly! \n")
    dnms = multi_sensory()

    for i in range(10):
        signal, target = dnms.GetInput()
        signal = signal.detach().cpu().numpy()
        plt.figure(i)
        plt.plot(signal[:,0])
        plt.plot(signal[:,1])
        plt.legend(["channel 1", "channel 2"])
        plt.ylim([-0.5 ,1.5])
        if target==0:
            plt.title("Below Threshold")
        else:
            plt.title("Above Threshold")
    plt.show()