# Pathology Weakely Supervised Lung Cancer Classification

Evaluation of [Neural Image Compression for Gigapixel Histopathology Image Analysis](https://arxiv.org/abs/1811.02840) on lung cancer classification using TCGA data.

![method](https://github.com/computationalpathologygroup/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/blob/master/images/pipeline.PNG)

Compressed whole slide images using NIC.


![compressed wsi](https://github.com/computationalpathologygroup/pathology-weakly-supervised-lung-cancer-growth-pattern-prediction/blob/master/images/sample_data_compressed.png)

## TCGA 

### Steps:

1. Vectorize data
2. Featurize vecorized data
3. Generate csv file with labels and slide names using the featurize folders with `python preprocessing.py`


### python scripts:

1. `python preprocessing.py`: creates train, validation, test csv files.
2. 
### Folder structure

- data<br>
- model<br>
- neural-image-compression-private: cloned from https://github.com/DIAGNijmegen/neural-image-compression-private but not include in the repo.