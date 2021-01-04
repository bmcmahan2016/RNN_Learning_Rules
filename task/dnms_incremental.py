'''
delayed non-match to sample
----segment of RDM-----delay-----segment of rdm----
Network must output if the two RDM segments have the same net sign

lets start with an easier version of this task, say 0.5 input mean

standard trial:
1s rest
500ms fixation
650ms of sample
1s rest
250ms test
1s choice period

we will shorten this to
500ms sample, 500ms rest, 250ms test
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

class DMC():
    def __init__(self, N=750, mean=0.5, var=1):
        self.N = N
        self._mean = mean
        self._var = var
        self._version = ""
        self._name = "dnms"
        
    def GetInput(self, mean_overide=-1, var_overide=1):
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
        
        #randomly draw mean from distribution
        mean = torch.rand(1).item() * self._mean
        if mean_overide != -1:
            mean = mean_overide
        
        # randomly generates context 1
        if torch.rand(1).item() < 0.5:
            inpts[:600,0] = torch.ones(600)
            target = torch.tensor(1)
        else:
            inpts[:,0] = 0*torch.ones(self.N)
            target = torch.tensor(-1)
            
        # randomly generates context 2
        if torch.rand(1).item() < 0.5:
            inpts[:,1] = torch.ones(self.N)
        else:
            inpts[:,1] = -torch.ones(self.N)
            
        '''    
        if torch.mean(inpts[:750,0]) == torch.mean(inpts[:750,1]):
            target = torch.tensor(1)
        else:
            target= torch.tensor(-1)
        '''
        # adds noise to inputs
        #inpts[:,:2] += self._var*torch.randn(750, 2).cuda()
        
        return inpts[:,:2], target
    
    def PsychoTest(self, coherence, context="in"):
        inpts = torch.zeros((self.N, 4)).cuda()
        inpts[:,0] = coherence*torch.ones(self.N)                # attended signal       changed 0->2
        inpts[:,1] = 2*(torch.rand(1)-0.5)*0.1857*torch.ones(self.N)   # ignored signal  changed 1->3
        if context=="in":  # signals attended signal
             inpts[:, 2] = 1    # changed 2 - >0
             
        else: # attends to the ignored signal
             inpts[:, 3] = 1    # changed 3 -> 1
         
        inpts[:,:2] += self._var*torch.randn(750, 2).cuda()    # adds noise to inputs
        
        assert(inpts.shape[1] == 4)
        return inpts
    
    def Loss(self, y, target, errorTrigger=-1):
        if (errorTrigger != -1):
            yt = y[errorTrigger:]
            print("y", torch.mean(yt[:]).item())
        else:
            yt = y[-1]
        ys = y[0]
        if type(y) is np.ndarray:
            assert False, "input to loss must be a PyTorch Tensor"
        else:
            # use loss from Mante 2013
            squareLoss = (yt-torch.sign(target.T))**2 + (ys - 0)**2
            meanSquareLoss = torch.sum( squareLoss, axis=0 )
            return meanSquareLoss


# construct

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pdb
    print("Delayed Categorical Match Task Called Directly! \n")
    dnms = DMC()

    for i in range(10):
        signal, target = dnms.GetInput()
        signal = signal.detach().cpu().numpy()
        plt.figure(i)
        plt.plot(signal[:,0])
        plt.plot(signal[:,1])
        plt.legend(["channel 1", "channel 2"])
        plt.ylim([-5 ,5])
        if target==1:
            plt.title("target=1")
        else:
            plt.title("target=-1")
    plt.show()