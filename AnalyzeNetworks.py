
from perturbation_experiments import *
from task.williams import Williams
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
'''
NOTES


'''
#######################################################
#SPECIFY WHAT ANALYSIS TO RUN 
#######################################################
#######################################################
'''Use these flags to control what analysis is performed
when this script is executed'''

# these flags will determine what analysis to run
StandardFlag = True    # will run basic analysis of network
EIFlag = False			 # will run excitation/inhibition analysis
AdvantageFlag = False	 # will run advantages of recurrence analysis
ContextFlag = False      # will run analysis of RNNs trained on contextual decision-making task
WeightStructureFlag = False
TCAFlag = False

# these flags will determine what RNNs to perform above analysis on
ForceFlag = False       # will analyze FORCE trained RNN
BpttFlag = False          # will analyze BPTT trained RNN
GeneticFlag = True       # will analyze Genetic trained RNN
HebbianFlag = False       # will analyze Hebbian trained RNN
ComplexityFlag = False     # will run analysis to determine model complexity
NoiseFlag = False          # will run analysis to determine how robust model is to noise

#######################################################
#######################################################
#######################################################

# perform the standard analysis
if StandardFlag:
	print('performing standard analysis ... \n\n')
	if GeneticFlag:
		niave_network('models/GA_001', xmin=-30, xmax=30, ymin=-6, ymax=6)
	if HebbianFlag:
		niave_network('models/hebian_model1')
	if ForceFlag:
		niave_network('models/FullForce0.75', xmin=-150, xmax=150, ymin=-6, ymax=6)
	if BpttFlag:
		niave_network('models/bptt_model8084', xmin=-10, xmax=10, ymin=-10, ymax=10)

# analysis to perform get neuron factors
if TCAFlag:
	GetNeuronIdx('bptt_model0')
	GetNeuronIdx('genetic_model0')
	GetNeuronIdx('force_model0')
	GetNeuronIdx('hebian_model0')

# determine how structured weight matrices are
if WeightStructureFlag:
	# list to hold structure of each RNN
	weight_struct = []
	# obtain the structure metric for each trained network
	tmp, w_diff = QuantifyRecurrence('bptt_model0')
	plt.figure()
	plt.imshow(w_diff)
	plt.title('BPTT')
	weight_struct.append(1/tmp)
	tmp, w_diff=QuantifyRecurrence('force_model0')
	plt.figure()
	plt.imshow(w_diff)
	plt.title('force')
	weight_struct.append(1/tmp)
	tmp, w_diff=QuantifyRecurrence('genetic_model0')
	plt.figure()
	plt.imshow(w_diff) 
	print('Analyzing Genetic RNN ...')
	GetModelAccuracy('genetic_model0', 1)
	print('Analyzing Hebbian RNN ...')
	GetModelAccuracy('hebian_model0', 1)
	print('Analyzing FORCE RNN ...')
	GetModelAccuracy('force_model0', 1)
	plt.legend(['BPTT', 'Genetic', 'Hebbian', 'FORCE'])
	plt.savefig('Robustness2Noise.eps')

# Perform the excitation/inhibition analysis
if EIFlag:
	print('Performing excitation-inhibition experiments ... \n\n')
	no_excit_acc = []
	no_inhib_acc = []
	cont_acc = []

	# perform control experiments
	tmp = lesion_control('force_model0', xmin=-150, xmax=150, ymin=-6, ymax=6)
	cont_acc.append(tmp)
	tmp = lesion_control('bptt_model0', xmin=-10, xmax=10, ymin=-5, ymax=5)
	cont_acc.append(tmp)
	tmp = lesion_control('genetic_model0', xmin=-30, xmax=30, ymin=-6, ymax=6)
	cont_acc.append(tmp)

	# remove excitatory connections
	tmp = remove_excitation('force_model0', xmin=-150, xmax=150, ymin=-6, ymax=6)
	no_excit_acc.append(tmp)
	tmp = remove_excitation('bptt_model0', xmin=-10, xmax=10, ymin=-5, ymax=5)
	no_excit_acc.append(tmp)
	tmp = remove_excitation('genetic_model0', xmin=-30, xmax=30, ymin=-6, ymax=6)
	no_excit_acc.append(tmp)


	# remove inhibitory connections
	tmp = remove_inhibition('force_model0', xmin=-150, xmax=150, ymin=-6, ymax=6)
	no_inhib_acc.append(tmp)
	tmp = remove_inhibition('bptt_model0', xmin=-10, xmax=10, ymin=-5, ymax=5)
	no_inhib_acc.append(tmp)
	tmp = remove_inhibition('genetic_model0', xmin=-30, xmax=30, ymin=-6, ymax=6)
	no_inhib_acc.append(tmp)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cont_results = ax.bar([-.27, .73, 1.73], cont_acc, 0.27, color='g')
	no_excit_results = ax.bar([0, 1, 2], no_excit_acc, 0.27, color='b')
	no_inhib_results = ax.bar([.27, 1.27, 2.27,], no_inhib_acc, 0.27, color='r')
	ax.set_ylabel('Model Accuracy')
	ax.set_xticks([0,1,2])
	ax.set_xticklabels( ('FORCE', 'BPTT', 'Genetic') )
	ax.legend(['Control', 'No Excitation', 'No Inhibition'])
	plt.show()

	# get the control networks

# run advantages of recurrence analysis
if AdvantageFlag:
	pass

# run analysis on RNNs trained to perform the Mante Susillo task
if ContextFlag:
	print('Analyzing RNNs trained on contextetual decision making task ...\n')
	if BpttFlag:
		plt.figure()
		ContextFixedPoints('models/bptt_context_model76')
		ComputeStepSize('models/bptt_context_model76')
	if HebbianFlag:
		plt.figure()
		ContextFixedPoints('hebbian_context_model76')
		ComputeStepSize('hebbian_context_model76')
	if ForceFlag:
		plt.figure()
		ContextFixedPoints('models/force_context_model77')
		ComputeStepSize('models/force_context_model77')
	if GeneticFlag:
		plt.figure()
		print('Analyzing RNN trained on contextual decision-making task with Genetic learning algorithm')
		ContextFixedPoints('genetic_context_model76')
		ComputeStepSize('genetic_context_model76')

# ensure that all figures get printed after analysis completes
plt.show()