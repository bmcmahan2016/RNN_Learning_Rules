# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:36:26 2020

@author: bmcma
"""
import rnn
import numpy as np
import torch
import matplotlib.pyplot as plt

model = rnn.loadHeb()
inpt = np.zeros((750, 5))
inpt[:,1] = 0 + 0.1*np.random.randn(750)     # ignored context
inpt[:,2] = 0.5 + 0.1*np.random.randn(750)    # attended context
inpt[:,4] = 1
inpt = torch.tensor(inpt).float().cuda()

outputs = []
dt = 0.1
for t in range(750):
    
    model._hidden = dt*torch.matmul(model._J["in"], inpt[t].reshape(5,1)) + \
            dt*torch.matmul(model._J['rec'], torch.tanh(model._hidden)) + \
            (1-dt)*model._hidden 
    outputs.append(np.tanh(model._hidden.detach()[0].item()))
    
outputs = np.array(outputs)
plt.plot(outputs)