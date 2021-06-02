# RNN_Learning_Rules

### Dependencies
The following should be installed

NumPy

SciPy (version must be 1.2.0)

matplotlib


### Model Training
To train an RNN model on the CDI-task, run the following:

python model_name -v 0.5 --N 2

alternatively, to train on the RDM-task,

python model_name -v 0.5 --rdm

where model_name can be any string designating the name to save your model after training. -v 0.5 specifies to train the model with an input variance of 0.5. You can also try setting -v to 0.1 or 1.0.
--N 2 specifies the CDI-task while --rdm specifies the RDM-taks.
Training on the N-CDI tasks is also possible by specifying a value of N greater than 2.
convergence is not guarenteed for N greater than or equal to 7.

### Visualizing Attractor States
To visualize the attractor states and PC trajectories for a specific model, run:
python analyze_dynamics.py model_name large

To view the attractors near zero, run
python analyze_dynamics.py model_name sparse

### Clustering RNNs
To cluster RNNs based on attractor topologies or representational geometries you will need a text file listing all the models to be analyzed.
The text file should be formatted such that each line in the file corresponds to a group of RNNs (e.g. each line could correspond to a different learning rule).
Then each model name on that line should be seperated by spaces.
If you were comparing 5 BPTT and 5 GA models this might look like:
BPTT bmodel_name_1 bmodel_name_2 bmodel_name_3 bmodel_name_4 bmodel_name_5
GA gmodel_name_1 gmodel_name_2 gmodel_name_3 gmodel_name_4 gmodel_name_5

Assume this file is saved as textfile.txt
Then, to analyze attractor topologies run,
python fixed_point_input_topology.py textfile.txt

To analyze representational geometry clusters run,
python svcca_analysis.py textfile.tx -reload
