'''
This script is a high level API to perform fixed point analysis 
on RNNs trained on one of 3 tasks using 1 of four learning rules.

'''


from perturbation_experiments import *
from task.williams import Williams
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import argparse
import FP_Analysis as fp

##############################################################
# determines what analysis to run
##############################################################
parser = argparse.ArgumentParser(description="Analyzes trained RNNs")
task_type = parser.add_mutually_exclusive_group()
task_type.add_argument("--rdm", action="store_true")
task_type.add_argument("--context", action="store_true")
task_type.add_argument("--multi", action="store_true")
task_type.add_argument("--N", action="store_true")

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--save_fp", action="store_true", default=False)

args = parser.parse_args()
##############################################################

if args.rdm:
    rdm_fixed_points('models/'+args.model_name)
elif args.context:
    plt.figure()
    context_fixed_points('models/'+args.model_name, save_fixed_points=args.save_fp)
elif args.multi:
	print("Analyzing multisensory network network")
	multi_fixed_points('models/'+args.model_name)
elif args.N:
    print("Analyzing N dimensional context")
    N_fixed_points('models/'+args.model_name)
else:
    raise NotImplementedError

plt.show()