# Pathology Weakely Supervised Lung Cancer Classification

Evaluation of [Neural Image Compression for Gigapixel Histopathology Image Analysis](https://arxiv.org/abs/1811.02840) on lung cancer classification using TCGA data.

![method](https://github.com/computationalpathologygroup/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/blob/master/images/pipeline.PNG)


### General Overview:

1. Convert .SVS files to TIF files
1. Generate Mask files
1. Vectorize data
1. Featurize vecorized data
1. Data Analysis
1. Model
1. Train Model
1. Evaulation of the Model
1. GradCam Analysis
10. Evaluate model using data from different datasets.

### Files

1. Generate csv file with labels and slide names using the featurize folders with `python preprocessing.py`


### Preprocessing
1. Compressed Whole Slides
2. Crop images to set a fixed size to input the network. 


### python scripts:

1. `python preprocessing.py`: creates train, validation, test csv files.
2. 
### Folder structure

- data<br>
- model<br>
- neural-image-compression-private: cloned from https://github.com/DIAGNijmegen/neural-image-compression-private but not include in the repo.