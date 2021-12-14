# Part II Project - The Impact of Non-IID Data on Federated Learning
This is the Github Repository for my Cambridge Part II project on the impact of
 non-iid data on federated learning.
 
The following Readme will contain notes on setup, information about the
 experiments and acknowledgements for reused code. The project is supposed to
  be executed on a unix based operating system since it uses shell scripts
   in some parts (for setting up the data pipelines). There is
    a workaround for Windows users which is described in Section 
    [Execute the Dataloader Script on Windows](#windows_script_fix).
 
## Setup

### Package Management
To initialise the project, first initialse a new virtual environment. The git
repository contains a requirements.txt file which contains all necessary
 packages. The minimum version of Python required is **Python 3.8**.

```
$ virtualenv <env_name> 
$ source <env_name>/bin/activate (<env_name>)
$ pip install -r requirements.txt 
```

### Dataset Setup
Then fetch and preprocess the datasets you are interested into with the
 following script:
```
$ python initialise_datasets.py [--celeba] [--femnist] [--shakespeare] 
```
This might take a lot of memory (up to 35GB), so ensure that you initialise this
repository on a drive with enough storage. This might take multiple hours since
a lot of data needs to get in-/deflated. I recommend initialising each
 dataset individually and in parallel (open a terminal for each). This will make
  the process finish far quicker since we can make use of multiple processors.

### GPU Acceleration

This project uses Tensorflow 2.x to run the code. Therefore, ensure that you
 have installed the latest version of CUDA and the necessary NVIDIA drivers
  on your system. (https://www.tensorflow.org/install/gpu)


### Execute the Dataloader Script on Windows
 
<a id="windows_script_fix">
   
</a>

The initialise_datasets.py script may fail on windows since it tries to
execute shell scripts as subprocesses. To circumvent this issue please
install e.g. Ubuntu on the WSL. You can find the instructions 
[here](https://ubuntu.com/wsl). 
   
First, ensure that you have both python3 and pip3 installed on the
 WSL. (On Ubuntu 20.04+ python3 is preinstalled. You can check if python3/pip3
  is preinstalled by typing `python3 -v`.)
 
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install python3-pip
```

Should you need to install python beforehand; use the deadsnakes ppa: 

```
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.8
```

Check that the installation succeeded by executing `python3 -v` and `pip3 -v`

Now install virtualenv on the environment:

```
$ pip3 install virtualenv
or
$ python3 -m pip install virtualenv
```

Note that the WSL allows accessing the Windows filesystem on /mnt/path/to
/project. Now, navigate to the project root folder on windows (by going
 through /mnt/drive...). We need to create a new virtual environment on linux 
 to load all necessary packages to run the dataset loading script.

```
$ python3 -m venv linux_venv        #Create virtual environment
$ source ./linux_venv/bin/activate  #Activate new Virtual Environment
$ pip install -r requirements.txt   #Install all necessary packages
```

Finally, we need to change the line endings from CRLF (Windows) to LF
 (Linux), so that the scripts will execute correctly. For this we will use a
  tool called dos2unix.

```
$ sudo apt install dos2unix
```

Then we need to change all python and shell scripts within leaf_root and
 data_loader folders to use LF.
 
```
$ find leaf_root data_loaders -maxdepth 4 -type f -name "*.sh" | xargs dos2unix
$ find leaf_root data_loaders -maxdepth 4 -type f -name "*.py" | xargs dos2unix
```

or in a more succinct manner:

```
$ find leaf_root data_loaders -maxdepth 4 -type f -regex ".*/.*\.\(\(py\)\|\(sh\)\)" | xargs dos2unix
```

Now you can finally run the dataloading script:

```
python initialise_datasets.py [--celeba] [--shakespeare] [--femnist]
```

You might want to consider reverting the change in line endings after
 completion. For this use the tool unix2dos similarly (this tool will get
  installed together with dos2unix). 
 
```
$ find leaf_root data_loaders -maxdepth 4 -type f -name "*.sh" | xargs unix2dos
$ find leaf_root data_loaders -maxdepth 4 -type f -name "*.py" | xargs unix2dos
```

or in a more succinct manner: 

```
$ find leaf_root data_loaders -maxdepth 4 -type f -regex ".*/.*\.\(\(py\)\|\(sh\)\)" | xargs unix2dos
```

## Experiments

### Notes on Execution

Note that executing a

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
 