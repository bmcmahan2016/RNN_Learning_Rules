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

parser.add_argument("model_name", help="filename of model to analyze")
parser.add_argument("--save_fp", action="store_true", default=False)

args = parser.parse_args()
##############################################################

if args.rdm:
    niave_network('models/'+args.model_name, xmin=-10, xmax=10, ymin=-10, ymax=10)
elif args.context:
    plt.figure()
    ContextFixedPoints('models/'+args.model_name, save_fixed_points=args.save_fp)
elif args.multi:
	print("Analyzing DNMS network")
	multi_fixed_points('models/'+args.model_name)
else:
    raise NotImplementedError

plt.show()