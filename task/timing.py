#i use output being that of a written letter; see Goudar and Buonomano (2018)
# they open sourced some of their handwritten outputs
import scipy.io as sio
import numpy as np
import pdb
import torch
from torch.autograd import Variable


class Timing():
    def GetInput(self, length=852):
        input = np.ones((length, 1)) + (0.1 * np.random.randn(length, 1) -0.5)
        input_param = 0
        return input, input_param

    def Loss(self, y, blank=None):
        target = self.GetDesired()
        return ((y - Variable(torch.tensor(target).float())) ** 2).mean()

    def GetDesired(self):
        loaded = sio.loadmat('data/timing/three.mat')
        output = loaded['three']
        return output
        # return torch.from_numpy(output).float()
