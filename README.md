# Part II Project - The Impact of Non-IID Data on Federated Learning
This is the Github Repository for my Cambridge Part II project on the impact of
 non-iid data on federated learning.
 
The following Readme will contain notes on setup, information about the
 experiments and acknowledgements for reused code.
 
## Setup

### Package Management
To initialise the project, first initialse a new virtual environment. The git
repository contains a requirements.txt file which contains all necessary
 packages.

```
$ virtualenv <env_name> 
$ source <env_name>/bin/activate 
(<env_name>)$ pip install -r requirements.txt 
```

### Dataset Setup
Then fetch and preprocess the datasets you are interested into with the
 following script:
```
python initialise_datasets.py [--celeba] [--femnist] [--shakespeare] 
```
This might take a lot of memory (up to 35GB), so ensure that you initialise this
repository on a drive with enough storage.

### GPU Acceleration

This project uses Tensorflow 2.x to run the code. Therefore, ensure that you
 have installed the latest version of CUDA and the necessary NVIDIA drivers
  on your system. (https://www.tensorflow.org/install/gpu)


## Experiments

###TODO 

## Acknowledgements

All code which I have copied fully or partially is acknowledged below
. All license statements have been copied and compiled into the LICENSES File
 (in the base folder).

In particular I would like to acknowledge the following two projects from
 which I have reused code in my project: 
 - LEAF (https://leaf.cmu.edu/) - BSD License 2.0
 - Flower (https://flower.dev/) - Apache License 2.0
 
 ### (Partial) Code Reuse
 
 - The leaf_root subfolder contains a clone of the LEAF Github repository
   (https://github.com/TalwalkarLab/leaf), which I use to download and
    preprocess data according to their [paper](https://arxiv.org/abs/1812.01097). 
    (BSD 2 License)
  
 - The flower github contains some scripts for preprocessing the data from
  LEAF. I could completely reuse two of those 
  [scripts](https://github.com/adap/flower/tree/ada3e12622187ae6f4b1f23aef576e23faa19674/baselines/flwr_baselines/scripts/leaf) (Shakespeare, Femnist
  ) and I adapted them for use with Celeba. (https://github.com/adap/flower)
  
 - Since I am using the Flower Framework for Federated Learning, I have also
  taken inspiration from their examples. In particular, I have tried looked
   at the way to use Tensorflow in their [advanced_tensorflow script](https://github.com/adap/flower/tree/f772993df9212b3e96b8fa916fdc1ecbe96500c2/examples/advanced_tensorflow)
   and the way the team creates [simulations](https://github.com/adap/flower/tree/f772993df9212b3e96b8fa916fdc1ecbe96500c2/examples/simulation) with flwr.
 