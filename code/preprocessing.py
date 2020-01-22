import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_csv(dir_class1, ddir_class0, csv_path, ext='.png'):
    """
    Creates csv file with slide names and labels.
    Class 1 is assinged to data from data_dir_class1

    Parameters
    ----------
    data_dir1 : TCGA LUAD featurized directory
    data_dir0 : TCGA LUSC featurized directory
    csv_path  : csv is exported to this path

    Output
    ----------
    csv file with labels and slide names

    Examples : create_csv(dir_luad, dir_lusc, csv_path)
    -----
    labesl 1 correspond to class 1 (LUAD)
    labesl 0 correspond to class 0 (LUSC)
    """

    files_class_1 = sorted([(os.path.basename(file)).split('.')[0] for file in tqdm(os.listdir(dir_class1)) if file.endswith(ext)])
    files_class_0 = sorted([(os.path.basename(file)).split('.')[0] for file in tqdm(os.listdir(ddir_class0)) if file.endswith(ext)])
    labels1 = np.ones(len(files_class_1), dtype=np.int8)
    labels0 = np.zeros(len(files_class_0), dtype=np.int8)

    df1 = pd.DataFrame(list(zip(files_class_1, labels1)), columns=['slide_id', 'label'])
    df0 = pd.DataFrame(list(zip(files_class_0, labels0)), columns=['slide_id', 'label'])

    # conacatenate dataframes
    data = pd.concat([df1, df0], ignore_index=True, )
    data.to_csv(csv_path, index=None, header=True)
    print('Csv file sucessfully exported!')


def split_csv_file(csv_dir, csv_train_dir, csv_test_dir, split=0.2):
    """
    Split a csv file into random train and test subset with for a given split

    Parameters
    ----------
    csv_dir : csv data file containing labels and slide names
    csv_train_dir : csv with training data
    csv_test_dir :  csv with testing data
    split: data split percentage

    Output
    ----------
    csv files are save in given paths
    """

    df = pd.read_csv(csv_dir)
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df['slide_id'], df['label'], test_size=split, random_state=0)
    # create train csv files
    df = pd.DataFrame(pd.concat([X_train, y_train], axis=1))
    df.to_csv(csv_train_dir, index=None, header=True)
    # create test csv files
    df = pd.DataFrame(pd.concat([X_test, y_test], axis=1))
    df.to_csv(csv_test_dir, index=None, header=True)


def generate_csv_files(csv_path, csv_train, csv_valid, csv_test, test_size=0.2, validation_size = 0.3):
    """
    Creates a complete test, train, validation split given the main csv data file.

    Inputs
    ----------
    csv_path: the original data set file

    Output
    ----------
    Splits save in csv_train, csv_valid, csv_test
    """

    # split data into train and test of test_size split
    split_csv_file(csv_path, csv_train, csv_test, split=test_size)
    # split data into train and test of test_size split
    split_csv_file(csv_train, csv_train, csv_valid, split=validation_size)
    print('Train/validation/test csv files sucessfully exported!')


if __name__ == '__main__':
    """
    Generates csv files to train the network given the paths where the featurize images are.
    """
    cluster = 0

    if cluster:
        # featurize wsi directory for TCGA dataset
        dir_luad =  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction\results\tcga\featurized\tcga_luad\normal'
        dir_lusc =  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction\results\tcga\featurized\tcga_lusc\normal'
        # featurize wsi with augmentations directory for TCGA dataset
        # dir_luad_aug =  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction\results\tcga\featurized\tcga_luad\augmented'
        # dir_lusc_aug =  r'Z:\projects\pathology-lung-cancer-weak-growth-pattern-prediction\results\tcga\featurized\tcga_lusc\augmented'

    else:
        # local test
        dir_luad =  r'E:\pathology-weakly-supervised-lung-cancer-growth-pattern-prediction\results\tcga_luad\featurized'
        dir_lusc =  r'E:\pathology-weakly-supervised-lung-cancer-growth-pattern-prediction\results\tcga_lusc\featurized'

    ######### Adjust this directories according where your project is ##########
    root_dir=  r'E:\pathology-weakly-supervised-lung-cancer-growth-pattern-prediction'

    csv_path =  os.path.join(root_dir,'data/slide_list_tcga.csv')
    csv_path_aug =  os.path.join(root_dir,'data/slide_list_tcga_aug.csv')

    csv_train =  os.path.join(root_dir,'data/train_slide_list_tcga.csv')
    csv_val =  os.path.join(root_dir,'data/validation_slide_list_tcga.csv')
    csv_test =  os.path.join(root_dir,'data/test_slide_list_tcga.csv')

    print('Creating main csv data files ...')
    create_csv(dir_luad, dir_lusc, csv_path)
    #create_csv(dir_luad_aug, dir_lusc_aug, csv_path_aug)

    print('Creating split train/validation/test csv files with no augmentations ...')
    generate_csv_files(csv_path, csv_train, csv_val, csv_test, test_size=0.2, validation_size = 0.3)

    # read files to check shapes
    df = pd.read_csv(csv_train);  df2 = pd.read_csv(csv_val);   df3 = pd.read_csv(csv_test)
    print(f'Files were read with shapes: Training: {df.shape}, Validation {df2.shape}, Testing {df3.shape}')
