# Tcga data

For this project, we are interested only in the diagnostic slides; overall around 1000.

### Location of the data

- /mnt/netcache/pathology/datasets/Lung/TCGA_LUAD
- /mnt/netcache/pathology/datasets/Lung/TCGA_LUSC

the tissue background masks are already created, e.g.

- /mnt/netcache/pathology/datasets/Lung/TCGA_LUAD/tissue_masks_diagnostic
- /mnt/netcache/pathology/datasets/Lung/TCGA_LUSC/tissue_masks_diagnostic​


### Nueral Compression Method

The neural image compression method has the following steps:

0. Train Encoder (for later)
For the BIGAN encoder: extract\_patches, train\_bigan\_encoder
There are already trained encoder which you can start to use; you can find them in the github repo
There is also a new supervised encoder:
/mnt/netcache/pathology/projects/pathology-proacting/neoadjuvant\_nki/nic/encoder\_zoo/supervsied\_enc\_2019\_4tasks.h5

1. Vectorize: We work not directly on the slide tifs, but on a more compact numpy representation
   (currently patches 128x128 from spacing 05)
2. Featurize: Apply the encoder to create the compressed representation for each slide
3. Train wsi classifier 
for this you will need a yaml-config for train/valid/test, here is an example:
/mnt/netcache/pathology/projects/pathology-proacting/neoadjuvant\_nki/configs/clf/data\_2cl\_03\_03\_03\_split1.yaml
4. Test wsi classifier

### Examples

There should be examples for these in the notebook in the nic github repo. I have also scripts for those steps, but I use an older version of the nic code (which has a lot of additional unnecessary/obsolete stuff, so it will be probably better to make new scripts (and you will get to know the code))...

you can put the vectorized and featurized and other files in the project directory on chansey, e.g. for tcga:
/mnt/netcache/pathology/projects/pathology-lung-cancer-weak-growth-pattern-prediction/tcga/tcga\_lusc/vectorized\_128
The datasets directory on chansey (will be renamed later to archives) is primarily for the slide files (and perhaps tissue background masks).
The project-directory is for project-specific files like the vectorized versions, configs, etc.

One more thing: you will run those steps on the cluster and here its important to remember to copy the necessary files (e.g. for training the featurized slides) to the cluster node first and not just use the files on chansey directly (otherwise it will be transfering the data all the time over the network - slow). Some of the scripts have that already built in (cache_dir argument). (also please remember to be cautious when deleting something on chansey since its not backed up)


### Libraries

- Tensorflow : 1.13.1
- Jupyter Notebook: `conda install jupyter notebook`
- matplotlib
- tqdm: `pip install tqdm`
- pandas: `pip install pandas`
- PIL : `pip install Pillow`
- Keras 2.2.4: 
- Add **multiresolutionimageinterface**  package to PYTHONPATH: **In case you are using Windows:** `conda-develop /path/to/`module/, i.e : `conda-develop  "D:\Program Files\ASAP 1.9\bin"`
####  INSTALLATION GUIDE:

To install Tensorflow, we advise you to use the conda package manager, obtainable from [https://www.anaconda.com/distribution](https://www.anaconda.com/distribution). For compatibility with NIC algorithm we use the python 3.6 version. 

**Create python environment**

- Make a separate environment (called tensorflow in this example)  by executing:
  
	`$ conda create -n 'tensorflow' python=3.6`

- Activate this environment using:

	` $ conda activate tensorflow`

Note: in older version of conda this was done with
   `$ source activate tensorflow`

**Install Tensorflow 1.13.1**

- Install Tensorflow version 1.13.1 (version 2.0 has just been released, and things work a little differently in that version) and matplotlib using 

	`$conda install tensorflow=1.13.1 matplotlib`
 
- Check if you have installed the right packages by opening a python shell and executing `import tensorflow as tf`, then try whether the provided `perceptron.py` script works.
- Make sure you have the ML_week6 environment enabled when you open the python
shell (the activate command above). 

**Note**: If you have a CUDA-capable GPU and want to use that, you can follow the guide on the  Tensorflow website [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu). But we won't be able to help with installing it this way.


**ERROR: tensorflow-estimator 1.13.0 requires mock>=2.0.0, which is not installed.**
#### Common Installation Problems

**tensorflow + numpy compatibility?**: [See github issue](https://github.com/tensorflow/tensorflow/issues/31249)

- Downgrade numpy version to 1.16.4 versions
  
	`$ pip uninstall numpy`

	`$ pip install numpy==1.16.4`

- After doing this, an error will pop up : ERROR: 
**tensorflow 1.14.0 requires google-pasta>=0.1.6, which is not installed.** Simply install google-pasta as follows: 

	`$ pip install google-pasta`


### Some issues:
    pip install pypiwin32


#### Removing an environment
To remove an environment, in your terminal window or an Anaconda Prompt, run:

    conda remove --name tensorflow --all