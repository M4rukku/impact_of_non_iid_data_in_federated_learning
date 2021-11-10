# impact_of_non_iid_data_in_federated_learning
This is the Github for my Cambridge Part II project on the impact of non iid data on federated learning.


## Data

First ensure you have installed pillow, numpy and googledrivedownloader. (or the requirements.txt)

To initialise all client datasets, execute "python initialise_datasets.py".
This will try to automatically fetch all data from well-known URLs. It might fail for Celeba (if the drive link expired). 
In that case, please do *1.

This will create a folder "data" and will store the downloaded data inside.

*1
First, download the Celeba Images from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or directly from their Google Drive (https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ). Store this dataset in the leaf_root/data/celeba/data/raw folder (as a zip). Then restart the python script.
