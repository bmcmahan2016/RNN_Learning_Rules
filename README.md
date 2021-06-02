# RNN_Learning_Rules

### Dependencies
To install dependencies please run <br/>
pip install -r requirements.txt

You can also manually install packages as needed.
This list may not be complete but some of the notable packages are:<br/>
NumPy<br/>
SciPy (version must be 1.2.0)<br/>
matplotlib<br\>
PyTorch<br\>
scikit-learn

If you are manually installing dependencies, you will also need to install tensortools by running the following command: <br/>
pip install git+https://github.com/ahwillia/tensortools


### Model Training
To train an RNN model on the CDI-task, run the following:<br/>
python train_models.py XXXX_name -v 0.5 --N 2<br/>
where XXXX must be either 'bptt', 'ga', 'ff'.
Note for Hebbian models we used to Github repo previously published at: https://github.com/ThomasMiconi/BiologicallyPlausibleLearningRNN
We followed the directions found in this repo and just manually changed input variance values as desired in the code.

alternatively, to train on the RDM-task,<br/>
python train_models.py XXXX_model_name -v 0.5 --rdm

where model_name can be any string designating the name to save your model after training. -v 0.5 specifies to train the model with an input variance of 0.5. You can also try setting -v to 0.1 or 1.0.<br/>
--N 2 specifies the CDI-task while --rdm specifies the RDM-taks.<br/>
Training on the N-CDI tasks is also possible by specifying a value of N greater than 2.
convergence is not guarenteed for N greater than or equal to 7.

### Visualizing Attractor States
To visualize the attractor states and PC trajectories for a specific model (which must be contained in the models directory), and run:<br/>
python analyze_dynamics.py model_name large<br\>
using the included sample models, you can run <br\>
python analyze_dynamics.py bptt_0050 large

To view the attractors near zero, run<br/>
python analyze_dynamics.py bptt_0050 sparse <br/>

### Clustering RNNs
To cluster RNNs based on attractor topologies or representational geometries you will need a text file listing all the models to be analyzed.
The text file should be formatted such that each line in the file corresponds to a group of RNNs (e.g. each line could correspond to a different learning rule).
Then each model name on that line should be seperated by spaces.
This text file must be saved in the models directory.

If you were comparing 5 BPTT and 5 GA models this might look like:<br/>
BPTT bmodel_name_1 bmodel_name_2 bmodel_name_3 bmodel_name_4 bmodel_name_5<br/>
GA gmodel_name_1 gmodel_name_2 gmodel_name_3 gmodel_name_4 gmodel_name_5<br/>
<br/>
A sample text file comparing 3 BPTT RNNs to 3 GA RNNs is included.
To analyze attractor topologies run,<br/>
python fixed_point_input_topology.py sample.txt<br/>

To analyze representational geometry clusters run,<br/>
python svcca_analysis.py sample.tx -reload<br/>
Note: after running once, the -reload keyword argument can be ommitted.
