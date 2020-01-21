# Pathology Weakely Supervised Lung Cancer Growth pattern Prediction

## Approaches

1. Lung cancer classification using TCGA data
2. Lung cancer classification subtypes using NLST data



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