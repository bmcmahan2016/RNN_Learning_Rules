'''
'''
from perturbation_experiments import *
from task.williams import Williams
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import argparse
import FP_Analysis as fp


# determines what analysis to run
parser = argparse.ArgumentParser(description="Analyzes trained RNNs")
task_type = parser.add_mutually_exclusive_group()
task_type.add_argument("--rdm", action="store_true")
task_type.add_argument("--context", action="store_true")
task_type.add_argument("--dnms", action="store_true")

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--save_fp", action="store_true", default=False)

args = parser.parse_args()



# perform the standard analysis
if True:
    print('performing standard analysis ... \n\n')
    if args.rdm:
        niave_network('models/'+args.model_name, xmin=-10, xmax=10, ymin=-10, ymax=10)
    elif args.context:
        plt.figure()
        ContextFixedPoints('models/'+args.model_name, save_fixed_points=args.save_fp)
    elif args.dnms:
    	print("Analyzing DNMS network")
    	DNMS_fixed_points('models/'+args.model_name)
    else:
        raise NotImplementedError
    


# # run analysis on RNNs trained to perform the Mante Susillo task
# if ContextFlag:
#     print('Analyzing RNNs trained on contextetual decision making task ...\n')
#     if BpttFlag:
#         plt.figure()
#         ContextFixedPoints('models/bptt_103')
#         #ComputeStepSize('models/bptt_context_model76')
#     if HebbianFlag:
#         plt.figure()
#         ContextFixedPoints('hebbian_context_model76')
#         ComputeStepSize('hebbian_context_model76')
#     if ForceFlag:
#         plt.figure()
#         ContextFixedPoints('models/force_context_model77')
#         ComputeStepSize('models/force_context_model77')
#     if GeneticFlag:
#         plt.figure()
#         print('Analyzing RNN trained on contextual decision-making task with Genetic learning algorithm')
#         ContextFixedPoints('models/ga_102')
#         #ComputeStepSize('genetic_context_model76')

# # analysis to perform get neuron factors
# if TCAFlag:
#     GetNeuronIdx('bptt_model0')
#     GetNeuronIdx('genetic_model0')
#     GetNeuronIdx('force_model0')
#     GetNeuronIdx('hebian_model0')

# # determine how structured weight matrices are
# if WeightStructureFlag:
#     # list to hold structure of each RNN
#     weight_struct = []
#     # obtain the structure metric for each trained network
#     tmp, w_diff = QuantifyRecurrence('bptt_model0')
#     plt.figure()
#     plt.imshow(w_diff)
#     plt.title('BPTT')
#     weight_struct.append(1/tmp)
#     tmp, w_diff=QuantifyRecurrence('force_model0')
#     plt.figure()
#     plt.imshow(w_diff)
#     plt.title('force')
#     weight_struct.append(1/tmp)
#     tmp, w_diff=QuantifyRecurrence('genetic_model0')
#     plt.figure()
#     plt.imshow(w_diff) 
#     print('Analyzing Genetic RNN ...')
#     GetModelAccuracy('genetic_model0', 1)
#     print('Analyzing Hebbian RNN ...')
#     GetModelAccuracy('hebian_model0', 1)
#     print('Analyzing FORCE RNN ...')
#     GetModelAccuracy('force_model0', 1)
#     plt.legend(['BPTT', 'Genetic', 'Hebbian', 'FORCE'])
#     plt.savefig('Robustness2Noise.eps')


plt.show()