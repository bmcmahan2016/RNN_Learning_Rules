import numpy as np 
import torch 
import utils
import matplotlib.pyplot as plt

class ContextTask():
    def __init__(self, N=40, mean=1, variance=1):
        self.N = N
        self.mean = mean
        self.variance = variance

    def PsychoTest(self, mean, context):
        '''
        DESCRIPTION
        PsychoTest will return data with specified mean in the specified context and random mean in the non-specified context
        PARAMETERS
        mean: specifies the mean (i.e. coherence of random dots) value of the signal 
                in the attended context. Signal for out of context will be generated 
                randomly
        context: specifies which context to be tested. This value must be either
                0 (for first context) or 1 (for second context)
        '''
        # create a zero filled tensor that will eventually hold our test data
        inpts = torch.zeros((4,40))
        inpts[0,:] = (utils.GetGaussianVector(torch.randn(1), self.variance, self.N))[:,0]  # changed from 0.5
        inpts[1,:] = (utils.GetGaussianVector(torch.randn(1), self.variance, self.N))[:,0]  # changed from 0.5
        # set the desired context to one
        inpts[2+context,:] = 1
        inpts[context,:] = (utils.GetGaussianVector(mean, self.variance, self.N))[:,0] 

        return inpts.t()

    def PsychoTestOut(self, mean, context):
        '''
        DESCRIPTION
        PsychoTestOut will return data with specified mean in the unspecified context and random mean in the specified context
        PARAMETERS
        mean: specifies the mean (i.e. coherence of random dots) value of the signal 
                in the attended context. Signal for out of context will be generated 
                randomly
        context: specifies which context to be tested. This value must be either
                0 (for first context) or 1 (for second context)
        '''
        # create a zero filled tensor that will eventually hold our test data
        inpts = torch.zeros((4,40))
        inpts[0,:] = (utils.GetGaussianVector(mean, self.variance, self.N))[:,0]  # changed from 0.5
        inpts[1,:] = (utils.GetGaussianVector(mean, self.variance, self.N))[:,0]  # changed from 0.5
        # set the desired context to one
        inpts[2+context,:] = 1
        # set specified context to be random
        inpts[context,:] = (utils.GetGaussianVector(torch.randn(1), self.variance, self.N))[:,0] 

        return inpts.t()

    def GetInput(self):
        '''
        GetInput will randomly generate a positive or negative 
        data sequence of length N for each of two contexts

        mean: the data sequence will be centered around plus or
        minus this value

        variance: variance of data sequence
        DEFAULTS
        N=40
        mean=1
        variance=1

        GetInput will return a torch tensor with shape 40x4. The first two columns contain the two different input signals while
        the last two columns contain the context zero. One of the last two columns will be all ones while the other will be all zeros

        '''
        num_contexts = 2
        inpts = torch.zeros((4,40))
        conditions = []

        #generate a context signal for each context (only one can be 1, all others are 0)
        context_chosen = False
        for _ in range(num_contexts, 2*num_contexts): #_=2,3
            '''#OVERWRITE TO ALWAYS CHOOSE FIRST CONTEXT
            if _ == num_contexts:
                inpts[_,:] = (torch.ones([1]).float())
                condition = _-num_contexts
                context_chosen=True
                print('first context overide!')
            #END OVERWRITE'''

            #if this is the last context and a context has not yet been selected force this one to get selected
            if (_ == (2*num_contexts-1)) and context_chosen==False:
                inpts[_,:] = (torch.ones([1]).float())
                context_chosen=True
                condition=_-num_contexts
                #print('last context chosen')

            if torch.rand(1) < 1/num_contexts:
                if context_chosen==False:
                    inpts[_,:]=(torch.ones(40).float())
                    context_chosen=True
                    condition=_-num_contexts
                    #print('context chosen')
            else:
                pass
                #don't overwrite last context
                #if not _ == 2*num_contexts-1:
                #    inpts[_,:]=(torch.zeros(40).float())
                #    print('context not selected')

        #generate an input for each context
        for _ in range(num_contexts):
            if torch.rand(1) < 0.5:
                inpts[_,:]=(utils.GetGaussianVector(-self.mean, self.variance, self.N))[:,0]  # changed from 0.5
                if condition == _:
                    target=(torch.tensor([-1]).float())
            else:
                inpts[_,:]=(utils.GetGaussianVector(self.mean, self.variance, self.N))[:,0]  # changed from 0.5
                if condition == _:
                    target=(torch.tensor([1]).float())

            #ensures a PyTorch Tensor object is returned
            if not torch.is_tensor(inpts[_]):
                inpts[-1] = torch.from_numpy(inpts[-1]).float()
        
        
        return inpts.t(), target

    def GetDesired(self):
        '''
        some classess attempt to make a call to GetDesired instead of GetInput. GetDesired 
        is only a wrapper fo the GetInput function
        '''
        inp, condition = self.GetInput()
        return inp, condition

    # note that here the desired output is baked into the loss function
    # TODO: see if there's a better way so I don't need to specify the numpy and torch version
    def Loss(self, y, mu):
        yt = y[-1]
        if type(y) is np.ndarray:
            return np.log(1 + np.exp(-yt * mu))
        else:
            return torch.log(1 + torch.exp(-yt * mu))

        # return log(1 + exp(-yt * mu))


#below code is used to debug this file
if __name__ == '__main__':
    task = ContextTask()
    inpt, condition = task.GetDesired()
    #print(inpt[:,:])
    #print(condition)
    plt.figure()
    plt.title('Context Decision Making Task\n(correct output is '+str(condition)+ ')')
    plt.plot(inpt[0].detach().numpy())
    plt.plot(inpt[1].detach().numpy())
    plt.plot(inpt[2].detach().numpy())
    plt.plot(inpt[3].detach().numpy())
    plt.legend(['Context 1', 'Context 2', 'Context 1 Signal', 'Context 2 Signal'])


    plt.show()
